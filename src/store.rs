//! SQLite index — derived state, never the source of truth.
//!
//! Everything here can be thrown away and rebuilt from the event log by
//! `brain reindex`. That is the property that makes the log the only thing
//! that has to sync, and it is worth protecting: nothing may be stored here
//! that cannot be recomputed from a log line.
//!
//! Concurrency: several CLIs write to one project at once, and each writer is
//! a short-lived process rather than a shared connection. WAL plus a busy
//! timeout is what makes that safe without a server serialising writes.

use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::event::{Event, EventKind, Source};

/// How long a writer waits for a competing writer before giving up.
const BUSY_TIMEOUT_MS: u32 = 5_000;

/// One search hit.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hit {
    pub id: String,
    pub ts: String,
    pub cli: String,
    pub kind: String,
    pub title: String,
    /// FTS5-generated excerpt with matches marked.
    pub snippet: String,
}

/// The derived index.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (creating if needed) the index and apply the schema.
    ///
    /// # Errors
    /// Returns an error when the database cannot be opened or migrated.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("open database {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_millis(u64::from(BUSY_TIMEOUT_MS)))
            .context("set busy timeout")?;
        // `journal_mode` is persistent, but setting it every open costs
        // nothing and keeps a hand-copied database correct.
        conn.pragma_update(None, "journal_mode", "WAL").context("enable WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL").context("set synchronous")?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Open a purely in-memory index.
    ///
    /// # Errors
    /// Returns an error when the schema cannot be applied.
    #[cfg(test)]
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("open in-memory database")?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS events (
                    id           TEXT PRIMARY KEY,
                    ts           TEXT NOT NULL,
                    workspace    TEXT NOT NULL,
                    project      TEXT NOT NULL,
                    session      TEXT NOT NULL,
                    cli          TEXT NOT NULL,
                    hook         TEXT NOT NULL,
                    kind         TEXT NOT NULL,
                    title        TEXT NOT NULL,
                    body         TEXT NOT NULL,
                    files        TEXT NOT NULL DEFAULT '[]',
                    topic        TEXT,
                    invocation   TEXT,
                    -- Set by a later tombstone or correction event. Derived:
                    -- replaying the log in ULID order reproduces both, because
                    -- a correction always sorts after what it corrects.
                    forgotten    INTEGER NOT NULL DEFAULT 0,
                    corrected_by TEXT,
                    -- How often an agent pulled this in full after seeing a
                    -- pointer to it. The only evidence we have that a memory
                    -- was worth keeping, as opposed to merely present.
                    read_count   INTEGER NOT NULL DEFAULT 0,
                    -- Lowered when a human says an entry is stale or wrong.
                    -- Nothing is destroyed; it just stops crowding the primer.
                    confidence   INTEGER NOT NULL DEFAULT 0,
                    consolidated INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS events_project_ts ON events(project, ts);
                CREATE INDEX IF NOT EXISTS events_session ON events(session);
                CREATE INDEX IF NOT EXISTS events_unconsolidated
                    ON events(project, consolidated);

                -- External-content FTS: the text lives once, in `events`.
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                    title,
                    body,
                    content='events',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                    INSERT INTO events_fts(rowid, title, body)
                    VALUES (new.rowid, new.title, new.body);
                END;
                CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
                    INSERT INTO events_fts(events_fts, rowid, title, body)
                    VALUES ('delete', old.rowid, old.title, old.body);
                END;
                CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
                    INSERT INTO events_fts(events_fts, rowid, title, body)
                    VALUES ('delete', old.rowid, old.title, old.body);
                    INSERT INTO events_fts(rowid, title, body)
                    VALUES (new.rowid, new.title, new.body);
                END;

                -- Which file each event touched. Feeds file-keyed injection.
                CREATE TABLE IF NOT EXISTS event_files (
                    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    path     TEXT NOT NULL,
                    project  TEXT NOT NULL,
                    PRIMARY KEY (event_id, path)
                );
                CREATE INDEX IF NOT EXISTS event_files_lookup
                    ON event_files(project, path);

                -- Circuit-breaker state per summarizer CLI. Derived in the
                -- sense that losing it only costs one wasted retry.
                CREATE TABLE IF NOT EXISTS summarizer_health (
                    cli            TEXT PRIMARY KEY,
                    failures       INTEGER NOT NULL DEFAULT 0,
                    cooldown_until TEXT,
                    last_error     TEXT,
                    last_failed_at TEXT
                );

                -- How a session was invoked. Cached because classifying it
                -- costs a process spawn, and a hook may not spend that on
                -- every single event.
                -- Where the host CLI keeps this session's transcript. A
                -- POINTER, never content: consolidation reads it, summarizes,
                -- and persists only the summary.
                CREATE TABLE IF NOT EXISTS session_transcript (
                    session TEXT PRIMARY KEY,
                    path    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_invocation (
                    session    TEXT PRIMARY KEY,
                    invocation TEXT NOT NULL
                );

                -- How many sessions a project has consolidated since its
                -- durable knowledge pages were last synthesized. The trigger
                -- for semantic memory, kept as a watermark so nothing needs a
                -- scheduler.
                CREATE TABLE IF NOT EXISTS knowledge_state (
                    project        TEXT PRIMARY KEY,
                    last_synth_at  TEXT,
                    sessions_since INTEGER NOT NULL DEFAULT 0
                );

                -- The concrete things a session was about: files, services,
                -- tables, commands. A second retrieval stream beside FTS5,
                -- matched lexically - two mentions are the same entity when
                -- they are the same string, and nothing cleverer.
                CREATE TABLE IF NOT EXISTS entities (
                    name    TEXT NOT NULL,
                    session TEXT NOT NULL,
                    project TEXT NOT NULL,
                    PRIMARY KEY (name, session)
                );
                CREATE INDEX IF NOT EXISTS entities_by_project ON entities(project, name);

                -- Which pointers a session has already been shown, so nothing
                -- is injected twice and the byte budget can be enforced.
                CREATE TABLE IF NOT EXISTS injected (
                    session  TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    PRIMARY KEY (session, event_id)
                );
                CREATE TABLE IF NOT EXISTS injected_bytes (
                    session TEXT PRIMARY KEY,
                    bytes   INTEGER NOT NULL DEFAULT 0
                );
                -- What recall handed to a session. Distinct from `injected`,
                -- which is what we pushed: this is what the agent asked for.
                CREATE TABLE IF NOT EXISTS recalled (
                    session  TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    PRIMARY KEY (session, event_id)
                );

                CREATE TABLE IF NOT EXISTS injected_files (
                    session TEXT NOT NULL,
                    path    TEXT NOT NULL,
                    PRIMARY KEY (session, path)
                );

                -- What consolidation has already done for a session, so a
                -- debounced trigger and a timer backstop do not redo work.
                CREATE TABLE IF NOT EXISTS session_state (
                    session       TEXT PRIMARY KEY,
                    project       TEXT NOT NULL,
                    last_run_at   TEXT,
                    last_event_id TEXT,
                    last_tier     TEXT
                );
                ",
            )
            .context("apply schema")?;
        self.add_missing_columns()
    }

    /// Add columns introduced after a database was first created.
    ///
    /// `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a new
    /// column never reaches a database someone has been using — every insert
    /// then fails against a schema the code no longer matches. The index is
    /// rebuildable, but silently breaking capture until someone runs
    /// `brain reindex` is not an acceptable upgrade path.
    fn add_missing_columns(&self) -> Result<()> {
        for (table, column, definition) in
            [
                ("events", "topic", "TEXT"),
                ("events", "invocation", "TEXT"),
                ("events", "forgotten", "INTEGER NOT NULL DEFAULT 0"),
                ("events", "corrected_by", "TEXT"),
                ("events", "read_count", "INTEGER NOT NULL DEFAULT 0"),
                ("events", "confidence", "INTEGER NOT NULL DEFAULT 0"),
                ("summarizer_health", "last_failed_at", "TEXT"),
            ]
        {
            if self.has_column(table, column)? {
                continue;
            }
            self.conn
                .execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {definition};"))
                .with_context(|| format!("add {table}.{column}"))?;
        }
        Ok(())
    }

    fn has_column(&self, table: &str, column: &str) -> Result<bool> {
        let mut stmt = self
            .conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .context("read table info")?;
        let mut rows = stmt.query([]).context("run table info")?;
        while let Some(row) = rows.next().context("read column row")? {
            if row.get::<_, String>(1).context("read column name")? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Index one event.
    ///
    /// Idempotent by event id: replaying a log line, or a hook that fired
    /// twice, must not produce a duplicate row. The log stays append-only;
    /// this is derived state, so last-write-wins is correct here.
    ///
    /// # Errors
    /// Returns an error when the insert fails.
    pub fn index(&self, event: &Event) -> Result<()> {
        let files = serde_json::to_string(&event.files).unwrap_or_else(|_| "[]".to_string());
        self.conn
            .execute(
                "INSERT INTO events
                    (id, ts, workspace, project, session, cli, hook, kind, title, body,
                     files, topic, invocation, consolidated)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                 ON CONFLICT(id) DO UPDATE SET
                     title = excluded.title,
                     body = excluded.body,
                     files = excluded.files,
                     topic = excluded.topic,
                     invocation = excluded.invocation,
                     consolidated = excluded.consolidated",
                params![
                    event.id,
                    event.ts,
                    event.workspace.to_string(),
                    event.project.to_string(),
                    event.session.to_string(),
                    event.source.cli,
                    event.source.hook,
                    event.kind.as_str(),
                    event.title,
                    event.body,
                    files,
                    event.topic,
                    event.extra.get("invocation").and_then(serde_json::Value::as_str),
                    i32::from(event.consolidated),
                ],
            )
            .context("index event")?;

        // A tombstone or a correction is about ANOTHER event. Both are
        // ordinary appended lines - nothing is deleted or edited in the log -
        // and what they change is the derived row they point at.
        match event.kind {
            EventKind::Tombstone => {
                for target in &event.links {
                    self.conn
                        .execute(
                            "UPDATE events SET forgotten = 1 WHERE id = ?1",
                            params![target],
                        )
                        .context("apply tombstone")?;
                }
            }
            EventKind::Note if event.source.hook == "correct" => {
                for target in &event.links {
                    self.conn
                        .execute(
                            "UPDATE events SET title = ?2, body = ?3, corrected_by = ?4
                             WHERE id = ?1",
                            params![target, event.title, event.body, event.id],
                        )
                        .context("apply correction")?;
                }
            }
            _ => {}
        }

        for path in &event.files {
            self.conn
                .execute(
                    "INSERT OR IGNORE INTO event_files (event_id, path, project)
                     VALUES (?1, ?2, ?3)",
                    params![event.id, path, event.project.to_string()],
                )
                .context("index event file")?;
        }
        Ok(())
    }

    /// Full-text search within one project, most relevant first.
    ///
    /// # Errors
    /// Returns an error when the query cannot be executed. A malformed FTS5
    /// query (an unbalanced quote typed by an agent) is reported as an error
    /// rather than silently returning nothing.
    pub fn search(&self, project: &str, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT e.id, e.ts, e.cli, e.kind, e.title,
                        snippet(events_fts, 1, '[', ']', ' … ', 24)
                 FROM events_fts
                 JOIN events e ON e.rowid = events_fts.rowid
                 WHERE events_fts MATCH ?1 AND e.project = ?2 AND e.forgotten = 0
                       AND e.kind != 'tombstone'
                 -- Relevance decides the order, but an entry a human called
                 -- stale should not sit at the top of it. Flagging has to
                 -- change what the user SEES, or it is a counter nobody can
                 -- observe.
                 ORDER BY CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END, rank
                 LIMIT ?3",
            )
            .context("prepare search")?;

        let mut read = |query: &str| -> rusqlite::Result<Vec<Hit>> {
            stmt.query_map(params![query, project, limit as i64], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: row.get(5)?,
                })
            })?
            .collect()
        };

        // FTS5 has its own query grammar, and the things people naturally
        // search for break it: `src/billing.rs` is a syntax error near `/`,
        // as is anything with an unbalanced quote. Falling back to the same
        // text as a quoted phrase turns a failed search into a literal one,
        // which is what someone typing a path meant anyway.
        match read(query) {
            Ok(hits) => Ok(hits),
            Err(_) => {
                let phrase = format!("\"{}\"", query.replace('"', " "));
                read(&phrase).context("read search results")
            }
        }
    }

    /// Fetch full events by id, in the order requested.
    ///
    /// # Errors
    /// Returns an error when a lookup fails.
    pub fn get(&self, ids: &[String]) -> Result<Vec<Event>> {
        let mut out = Vec::with_capacity(ids.len());
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, ts, workspace, project, session, cli, hook, kind, title, body,
                        files, topic, consolidated
                 FROM events WHERE id = ?1",
            )
            .context("prepare get")?;
        for id in ids {
            let found = stmt
                .query_row(params![id], |row| {
                    let files: String = row.get(10)?;
                    let kind: String = row.get(7)?;
                    Ok(Event {
                        v: crate::event::SCHEMA_VERSION,
                        id: row.get(0)?,
                        ts: row.get(1)?,
                        workspace: parse_uuid(&row.get::<_, String>(2)?),
                        project: parse_uuid(&row.get::<_, String>(3)?),
                        session: parse_uuid(&row.get::<_, String>(4)?),
                        source: Source { cli: row.get(5)?, hook: row.get(6)? },
                        kind: parse_kind(&kind),
                        title: row.get(8)?,
                        body: row.get(9)?,
                        files: serde_json::from_str(&files).unwrap_or_default(),
                        links: Vec::new(),
                        topic: row.get(11)?,
                        consolidated: row.get::<_, i32>(12)? != 0,
                        extra: serde_json::Map::new(),
                    })
                })
                .optional()
                .context("get event")?;
            if let Some(event) = found {
                out.push(event);
            }
        }
        Ok(out)
    }

    /// Most recent events in a project, newest first.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn recent(&self, project: &str, limit: usize) -> Result<Vec<Hit>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, ts, cli, kind, title
                 FROM events WHERE project = ?1 AND forgotten = 0 AND kind != 'tombstone'
                 ORDER BY id DESC LIMIT ?2",
            )
            .context("prepare recent")?;
        let rows = stmt
            .query_map(params![project, limit as i64], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: String::new(),
                })
            })
            .context("run recent")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read recent results")
    }

    /// Number of indexed events.
    ///
    /// # Errors
    /// Returns an error when the count query fails.
    pub fn count(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM events", [], |row| row.get(0))
            .context("count events")
    }

    /// Per-CLI event counts, for `brain doctor`.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn counts_by_cli(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT cli, COUNT(*) FROM events GROUP BY cli ORDER BY 2 DESC")
            .context("prepare cli counts")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .context("run cli counts")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read cli counts")
    }

    /// Forget what a session has been shown, and give it its budget back.
    ///
    /// Called when the agent's context is wiped. The session id survives a
    /// compaction but the context does not, so everything we track against
    /// that id has to start over — otherwise the de-duplication guard becomes
    /// the cause of the amnesia it exists to prevent.
    ///
    /// # Errors
    /// Returns an error when the writes fail.
    pub fn reset_injection_state(&self, session: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM injected WHERE session = ?1", params![session])
            .context("reset injected ids")?;
        self.conn
            .execute("DELETE FROM injected_files WHERE session = ?1", params![session])
            .context("reset injected files")?;
        self.conn
            .execute("DELETE FROM injected_bytes WHERE session = ?1", params![session])
            .context("reset injected bytes")?;
        Ok(())
    }

    /// Remember where a session's transcript lives.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_transcript_path(&self, session: &str, path: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO session_transcript (session, path) VALUES (?1, ?2)
                 ON CONFLICT(session) DO UPDATE SET path = excluded.path",
                params![session, path],
            )
            .context("record transcript path")?;
        Ok(())
    }

    /// Where a session's transcript lives, if the CLI told us.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn transcript_path(&self, session: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT path FROM session_transcript WHERE session = ?1",
                params![session],
                |row| row.get(0),
            )
            .optional()
            .context("read transcript path")
    }

    /// Which CLI most recently worked in this project.
    ///
    /// The MCP server is spawned by a host CLI and told nothing about which
    /// one - and the session id it invents for itself matches no hook's, so
    /// asking about its own session always answered "nobody". The project's
    /// most recent capture is the closest true answer available, and it is
    /// only used to decide which cheap tier to borrow first.
    pub fn project_cli(&self, project: &str) -> Result<Option<String>> {
        let cli = self.conn.query_row(
            "SELECT cli FROM events WHERE project = ?1 ORDER BY id DESC LIMIT 1",
            [project],
            |row| row.get::<_, String>(0),
        );
        match cli {
            Ok(cli) => Ok(Some(cli)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    /// How this session was invoked, if we have already worked it out.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn session_invocation(&self, session: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT invocation FROM session_invocation WHERE session = ?1",
                params![session],
                |row| row.get(0),
            )
            .optional()
            .context("read session invocation")
    }

    /// Remember how a session was invoked.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_session_invocation(&self, session: &str, invocation: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO session_invocation (session, invocation) VALUES (?1, ?2)",
                params![session, invocation],
            )
            .context("record session invocation")?;
        Ok(())
    }

    /// Remember that recall surfaced these ids to a session.
    ///
    /// Feeds the rule that a model may only withdraw memory it has actually
    /// been shown.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_recalled<'a>(
        &self,
        session: &str,
        ids: impl Iterator<Item = &'a str>,
    ) -> Result<()> {
        for id in ids {
            let inserted = self
                .conn
                .execute(
                    "INSERT OR IGNORE INTO recalled (session, event_id) VALUES (?1, ?2)",
                    params![session, id],
                )
                .context("record recalled id")?;
            // Count a session's first read only. Re-reading the same entry in
            // one conversation says nothing extra about its worth.
            if inserted > 0 {
                self.conn
                    .execute(
                        "UPDATE events SET read_count = read_count + 1 WHERE id = ?1",
                        params![id],
                    )
                    .context("count read")?;
            }
        }
        Ok(())
    }

    /// Did recall surface this id to this session?
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn was_recalled(&self, session: &str, id: &str) -> Result<bool> {
        let found: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM recalled WHERE session = ?1 AND event_id = ?2",
                params![session, id],
                |row| row.get(0),
            )
            .optional()
            .context("check recalled")?;
        Ok(found.is_some())
    }

    /// Note that a project consolidated another session.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn note_session_consolidated(&self, project: &str) -> Result<i64> {
        self.conn
            .execute(
                "INSERT INTO knowledge_state (project, sessions_since) VALUES (?1, 1)
                 ON CONFLICT(project) DO UPDATE SET
                     sessions_since = knowledge_state.sessions_since + 1",
                params![project],
            )
            .context("note consolidated session")?;
        self.conn
            .query_row(
                "SELECT sessions_since FROM knowledge_state WHERE project = ?1",
                params![project],
                |row| row.get(0),
            )
            .context("read sessions since")
    }

    /// Reset the counter after synthesizing knowledge pages.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn note_knowledge_synthesized(&self, project: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO knowledge_state (project, last_synth_at, sessions_since)
                 VALUES (?1, ?2, 0)
                 ON CONFLICT(project) DO UPDATE SET
                     last_synth_at = excluded.last_synth_at, sessions_since = 0",
                params![project, jiff::Timestamp::now().to_string()],
            )
            .context("note synthesis")?;
        Ok(())
    }

    /// Titles of knowledge already synthesized for one project.
    ///
    /// Synthesis runs again every few sessions and will happily rediscover
    /// what it found last time; without this, one durable fact would accrete
    /// one entry per run until the primer said little else.
    pub fn knowledge_titles(&self, project: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT title FROM events WHERE project = ?1 AND kind = 'knowledge' AND forgotten = 0",
        )?;
        let rows = stmt.query_map([project], |row| row.get::<_, String>(0))?;
        Ok(rows.filter_map(std::result::Result::ok).collect())
    }

    /// Recent session summaries for a project, newest first.
    ///
    /// The raw material for durable knowledge: what each session concluded,
    /// rather than every event it produced.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn recent_summaries(&self, project: &str, limit: usize) -> Result<Vec<Event>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id FROM events
                 WHERE project = ?1 AND kind = 'session_summary' AND forgotten = 0
                 ORDER BY id DESC LIMIT ?2",
            )
            .context("prepare recent summaries")?;
        let ids = stmt
            .query_map(params![project, limit as i64], |row| row.get::<_, String>(0))
            .context("run recent summaries")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("read summary ids")?;
        self.get(&ids)
    }

    /// Record what a session was about.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_entities(&self, session: &str, project: &str, names: &[String]) -> Result<()> {
        for name in names {
            self.conn
                .execute(
                    "INSERT OR IGNORE INTO entities (name, session, project) VALUES (?1, ?2, ?3)",
                    params![name, session, project],
                )
                .context("record entity")?;
        }
        Ok(())
    }

    /// Every entity in a project, with how many sessions touched it.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn entities(&self, project: &str) -> Result<Vec<(String, i64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT name, COUNT(*) FROM entities
                 WHERE project = ?1 GROUP BY name ORDER BY 2 DESC, 1",
            )
            .context("prepare entities")?;
        let rows = stmt
            .query_map(params![project], |row| Ok((row.get(0)?, row.get(1)?)))
            .context("run entities")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read entities")
    }

    /// Sessions that touched a named entity.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn sessions_for_entity(&self, project: &str, name: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT session FROM entities WHERE project = ?1 AND name = ?2 ORDER BY session",
            )
            .context("prepare entity sessions")?;
        let rows = stmt
            .query_map(params![project, name], |row| row.get(0))
            .context("run entity sessions")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read entity sessions")
    }

    /// Entries whose session touched a named entity.
    ///
    /// The second retrieval stream: a query that names a file or a service can
    /// find the work about it even when no title happens to contain the word.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn search_by_entity(&self, project: &str, name: &str, limit: usize) -> Result<Vec<Hit>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT e.id, e.ts, e.cli, e.kind, e.title
                 FROM events e
                 JOIN entities n ON n.session = e.session AND n.project = e.project
                 WHERE e.project = ?1 AND n.name = ?2 AND e.forgotten = 0
                       AND e.kind != 'tombstone'
                 ORDER BY CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END, e.id DESC
                 LIMIT ?3",
            )
            .context("prepare entity search")?;
        let rows = stmt
            .query_map(params![project, name, limit as i64], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: String::new(),
                })
            })
            .context("run entity search")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read entity search")
    }

    /// Lower an entry's standing after a human called it stale or wrong.
    ///
    /// Deliberately not a learning system: a counter and a sort key. Nothing
    /// is destroyed, because a human calling something stale is a judgement
    /// about usefulness, not a claim that it never happened.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn lower_confidence(&self, id: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE events SET confidence = confidence - 1 WHERE id = ?1",
                params![id],
            )
            .context("lower confidence")?;
        Ok(())
    }

    /// Entries a human has flagged, for the lint page.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn flagged(&self, project: &str) -> Result<Vec<Pointer>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, ts, kind, title, topic, hook FROM events
                 WHERE project = ?1 AND confidence < 0 AND forgotten = 0
                 ORDER BY confidence, id DESC LIMIT 100",
            )
            .context("prepare flagged")?;
        let rows = stmt
            .query_map(params![project], |row| {
                Ok(Pointer {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    kind: row.get(2)?,
                    title: row.get(3)?,
                    topic: row.get(4)?,
                    has_files: false,
                    hook: row.get(5)?,
                })
            })
            .context("run flagged")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read flagged")
    }

    /// Does this event exist and is it still remembered?
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn event_exists(&self, id: &str) -> Result<bool> {
        let found: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM events WHERE id = ?1 AND forgotten = 0",
                params![id],
                |row| row.get(0),
            )
            .optional()
            .context("check event")?;
        Ok(found.is_some())
    }

    /// The project and title of an event, whether or not it is forgotten.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn event_summary(&self, id: &str) -> Result<Option<(String, String)>> {
        self.conn
            .query_row(
                "SELECT project, title FROM events WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context("read event summary")
    }

    /// How many entries have been forgotten or corrected, for `brain stats`.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn revision_counts(&self) -> Result<(i64, i64)> {
        self.conn
            .query_row(
                "SELECT
                   (SELECT COUNT(*) FROM events WHERE forgotten = 1),
                   (SELECT COUNT(*) FROM events WHERE corrected_by IS NOT NULL)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .context("read revision counts")
    }

    /// How consolidation has been done, by tier.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn consolidation_tiers(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT COALESCE(last_tier, 'unknown'), COUNT(*)
                 FROM session_state GROUP BY 1 ORDER BY 2 DESC",
            )
            .context("prepare tier counts")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .context("run tier counts")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read tier counts")
    }

    /// How much injected memory was later actually pulled.
    ///
    /// The honest answer to "is the primer worth its bytes": of the pointers
    /// pushed into sessions, how many did an agent go on to read in full.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn injection_uptake(&self) -> Result<(i64, i64)> {
        self.conn
            .query_row(
                "SELECT
                   (SELECT COUNT(*) FROM injected),
                   (SELECT COUNT(*) FROM injected i
                      WHERE EXISTS (SELECT 1 FROM recalled r WHERE r.event_id = i.event_id))",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .context("read injection uptake")
    }

    /// Total recall calls that returned something.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn recall_count(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM recalled", [], |row| row.get(0))
            .context("count recalled")
    }

    /// Is there unconsolidated work older than `max_age_secs`?
    ///
    /// The backstop's whole question. Consolidation only matters when a next
    /// session will read it, and that session fires hooks — so a hook asking
    /// "is anything stale?" covers every case a wall-clock timer would, minus
    /// the one that does not matter: a backlog on a machine where no CLI ever
    /// runs again.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn has_stale_backlog(&self, max_age_secs: i64) -> Result<bool> {
        let cutoff = (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(max_age_secs))
            .to_string();
        let found: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM events
                 WHERE consolidated = 0 AND kind = 'observation' AND ts < ?1
                 LIMIT 1",
                params![cutoff],
                |row| row.get(0),
            )
            .optional()
            .context("check stale backlog")?;
        Ok(found.is_some())
    }

    /// Sessions in a project with unconsolidated events, oldest first.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn sessions_pending(&self, project: &str) -> Result<Vec<PendingSession>> {
        let mut stmt = self
            .conn
            .prepare(
                // The CLI of the newest event, not MAX(cli), which is
                // alphabetical: for a session two CLIs touched, "codex" would
                // beat "claude-code" for no reason but its spelling, and this
                // value decides whose cheap tier gets asked to summarize.
                "SELECT e.session, COUNT(*), MAX(e.id),
                        (SELECT cli FROM events
                          WHERE session = e.session ORDER BY id DESC LIMIT 1)
                 FROM events e
                 WHERE e.project = ?1 AND e.consolidated = 0 AND e.kind = 'observation'
                 GROUP BY e.session
                 ORDER BY MAX(e.id)",
            )
            .context("prepare pending sessions")?;
        let rows = stmt
            .query_map(params![project], |row| {
                Ok(PendingSession {
                    session: row.get(0)?,
                    pending: row.get(1)?,
                    newest_event_id: row.get(2)?,
                    cli: row.get(3)?,
                })
            })
            .context("run pending sessions")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read pending sessions")
    }

    /// Unconsolidated observations for one session, oldest first.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn session_events(&self, session: &str) -> Result<Vec<Event>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id FROM events
                 WHERE session = ?1 AND consolidated = 0 AND kind = 'observation'
                 ORDER BY id",
            )
            .context("prepare session events")?;
        let ids = stmt
            .query_map(params![session], |row| row.get::<_, String>(0))
            .context("run session events")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("read session event ids")?;
        self.get(&ids)
    }

    /// Mark events consolidated.
    ///
    /// Only a successful model-backed run calls this. Rule-based output
    /// deliberately leaves events pending so a later working run can produce
    /// the better version — the ladder degrades quality, never data.
    ///
    /// # Errors
    /// Returns an error when the update fails.
    pub fn mark_consolidated(&self, ids: &[String]) -> Result<()> {
        for id in ids {
            self.conn
                .execute("UPDATE events SET consolidated = 1 WHERE id = ?1", params![id])
                .context("mark consolidated")?;
        }
        Ok(())
    }

    /// Record that consolidation ran for a session.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_session_run(
        &self,
        session: &str,
        project: &str,
        newest_event_id: &str,
        tier: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO session_state (session, project, last_run_at, last_event_id, last_tier)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(session) DO UPDATE SET
                     last_run_at = excluded.last_run_at,
                     last_event_id = excluded.last_event_id,
                     last_tier = excluded.last_tier",
                params![session, project, jiff::Timestamp::now().to_string(), newest_event_id, tier],
            )
            .context("record session run")?;
        Ok(())
    }

    /// What consolidation last did for a session, if anything.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn session_run(&self, session: &str) -> Result<Option<SessionRun>> {
        self.conn
            .query_row(
                "SELECT last_run_at, last_event_id, last_tier FROM session_state
                 WHERE session = ?1",
                params![session],
                |row| {
                    Ok(SessionRun {
                        last_run_at: row.get(0)?,
                        last_event_id: row.get(1)?,
                        last_tier: row.get(2)?,
                    })
                },
            )
            .optional()
            .context("read session state")
    }

    /// Note a summarizer failure, opening the breaker at the threshold.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_summarizer_failure(&self, cli: &str, error: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO summarizer_health (cli, failures, last_error, last_failed_at)
                 VALUES (?1, 1, ?2, ?3)
                 ON CONFLICT(cli) DO UPDATE SET
                     failures = summarizer_health.failures + 1,
                     last_error = excluded.last_error,
                     last_failed_at = excluded.last_failed_at",
                params![cli, truncate_error(error), jiff::Timestamp::now().to_string()],
            )
            .context("record summarizer failure")?;

        let failures: i64 = self
            .conn
            .query_row(
                "SELECT failures FROM summarizer_health WHERE cli = ?1",
                params![cli],
                |row| row.get(0),
            )
            .context("read failure count")?;

        if failures >= crate::summarizer::failure_threshold() {
            let until = jiff::Timestamp::now()
                + jiff::SignedDuration::from_secs(
                    i64::try_from(crate::summarizer::cooldown().as_secs()).unwrap_or(1800),
                );
            self.conn
                .execute(
                    "UPDATE summarizer_health SET cooldown_until = ?2 WHERE cli = ?1",
                    params![cli, until.to_string()],
                )
                .context("open circuit breaker")?;
        }
        Ok(())
    }

    /// Note a summarizer success, closing the breaker.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_summarizer_success(&self, cli: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO summarizer_health (cli, failures, cooldown_until, last_error, last_failed_at)
                 VALUES (?1, 0, NULL, NULL, NULL)
                 ON CONFLICT(cli) DO UPDATE SET
                     failures = 0, cooldown_until = NULL, last_error = NULL,
                     last_failed_at = NULL",
                params![cli],
            )
            .context("record summarizer success")?;
        Ok(())
    }

    /// Is this CLI in a cooldown right now?
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn summarizer_in_cooldown(&self, cli: &str) -> Result<bool> {
        let until: Option<String> = self
            .conn
            .query_row(
                "SELECT cooldown_until FROM summarizer_health WHERE cli = ?1",
                params![cli],
                |row| row.get(0),
            )
            .optional()
            .context("read cooldown")?
            .flatten();

        let Some(until) = until else { return Ok(false) };
        let Ok(until) = until.parse::<jiff::Timestamp>() else { return Ok(false) };
        Ok(until > jiff::Timestamp::now())
    }

    /// Summarizer health for `brain doctor`.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn summarizer_health(&self) -> Result<Vec<SummarizerHealth>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT cli, failures, last_error, last_failed_at
                 FROM summarizer_health ORDER BY cli",
            )
            .context("prepare summarizer health")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(SummarizerHealth {
                    cli: row.get(0)?,
                    failures: row.get(1)?,
                    last_error: row.get(2)?,
                    last_failed_at: row.get(3)?,
                })
            })
            .context("run summarizer health")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read summarizer health")
    }

    /// SQL fragment ranking memory by how much it is worth recalling.
    ///
    /// Two axes, in order. A consolidated narrative outranks a classified
    /// title, which outranks an unclassified one, which outranks a raw
    /// capture. Within that, what the thing is ABOUT decides: a decision or a
    /// discovery is what someone resuming work needs; config and test noise is
    /// what they can rediscover in seconds.
    ///
    /// This ordering is what makes a byte budget safe to enforce by simply
    /// stopping: whatever gets cut is, by construction, the least useful thing
    /// left.
    fn rank(prefix: &str) -> String {
        format!(
            // Knowledge outranks a session summary because it is what
            // survived several of them.
            "CASE {prefix}kind
                 WHEN 'knowledge' THEN 0
                 WHEN 'session_summary' THEN 1
                 WHEN 'note' THEN 2
                 WHEN 'page_update' THEN 3
                 ELSE 4
             END,
             CASE WHEN {prefix}invocation = 'headless' THEN 1 ELSE 0 END,
             -- Evidence beats heuristics: something an agent went back and
             -- read is worth more than something we merely guessed at, and a
             -- human calling an entry stale outranks both.
             -{prefix}confidence,
             CASE WHEN {prefix}read_count > 0 THEN 0 ELSE 1 END,
             CASE {prefix}topic
                 WHEN 'decision' THEN 0
                 WHEN 'discovery' THEN 1
                 WHEN 'bugfix' THEN 2
                 WHEN 'feature' THEN 3
                 WHEN 'config' THEN 4
                 WHEN 'test' THEN 5
                 ELSE 6
             END"
        )
    }

    /// Ranked pointers for the session-start primer.
    ///
    /// Ranking is kind first, recency second. A consolidated summary is worth
    /// more than a rewritten title, which is worth more than a raw capture —
    /// and within a kind, newer wins. That ordering is what makes a byte
    /// budget safe to enforce by simply stopping: whatever gets cut is, by
    /// construction, the least useful thing left.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn primer_pointers(&self, project: &str, limit: usize) -> Result<Vec<Pointer>> {
        let sql = format!(
            // The same floor every other read enforces. Injection is recall
            // too - and the costlier half, because it spends bytes in every
            // future session whether or not anyone asked. A withdrawn memory
            // that only disappears from search has not been withdrawn.
            "SELECT id, ts, kind, title, topic, files, hook FROM events
             WHERE project = ?1 AND forgotten = 0 AND kind != 'tombstone'
             ORDER BY {}, id DESC
             LIMIT ?2",
            Self::rank("")
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare primer pointers")?;
        let rows = stmt
            .query_map(params![project, limit as i64], |row| {
                Ok(Pointer {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    kind: row.get(2)?,
                    title: row.get(3)?,
                    topic: row.get(4)?,
                    has_files: row
                        .get::<_, String>(5)
                        .map(|files| files.len() > 2)
                        .unwrap_or(false),
                    hook: row.get(6)?,
                })
            })
            .context("run primer pointers")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read primer pointers")
    }

    /// Pointers for events that touched one file, best first.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn pointers_for_file(
        &self,
        project: &str,
        path: &str,
        limit: usize,
    ) -> Result<Vec<Pointer>> {
        let sql = format!(
            "SELECT e.id, e.ts, e.kind, e.title, e.topic, e.hook
             FROM event_files f
             JOIN events e ON e.id = f.event_id
             WHERE f.project = ?1 AND f.path = ?2 AND e.forgotten = 0
                   AND e.kind != 'tombstone' 
             ORDER BY {}, e.id DESC
             LIMIT ?3",
            Self::rank("e.")
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare file pointers")?;
        let rows = stmt
            .query_map(params![project, path, limit as i64], |row| {
                Ok(Pointer {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    kind: row.get(2)?,
                    title: row.get(3)?,
                    topic: row.get(4)?,
                    has_files: true,
                    hook: row.get(5)?,
                })
            })
            .context("run file pointers")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read file pointers")
    }

    /// Chronological slice of a project, for `brain_timeline`.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn timeline(&self, project: &str, since: &str, limit: usize) -> Result<Vec<Hit>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, ts, cli, kind, title FROM events
                 WHERE project = ?1 AND ts >= ?2 AND forgotten = 0 AND kind != 'tombstone' 
                 ORDER BY id
                 LIMIT ?3",
            )
            .context("prepare timeline")?;
        let rows = stmt
            .query_map(params![project, since, limit as i64], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: String::new(),
                })
            })
            .context("run timeline")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read timeline")
    }

    /// Has this session already been shown this pointer?
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn already_injected(&self, session: &str, event_id: &str) -> Result<bool> {
        let found: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM injected WHERE session = ?1 AND event_id = ?2",
                params![session, event_id],
                |row| row.get(0),
            )
            .optional()
            .context("check injected")?;
        Ok(found.is_some())
    }

    /// Has this session already been given pointers for this file?
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn file_already_injected(&self, session: &str, path: &str) -> Result<bool> {
        let found: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM injected_files WHERE session = ?1 AND path = ?2",
                params![session, path],
                |row| row.get(0),
            )
            .optional()
            .context("check injected file")?;
        Ok(found.is_some())
    }

    /// Bytes this session has already spent on automatic injection.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn session_injected_bytes(&self, session: &str) -> Result<usize> {
        let bytes: Option<i64> = self
            .conn
            .query_row(
                "SELECT bytes FROM injected_bytes WHERE session = ?1",
                params![session],
                |row| row.get(0),
            )
            .optional()
            .context("read injected bytes")?;
        Ok(usize::try_from(bytes.unwrap_or(0)).unwrap_or(0))
    }

    /// Record what an injection spent.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_injected(&self, session: &str, ids: &[String], bytes: usize) -> Result<()> {
        for id in ids {
            self.conn
                .execute(
                    "INSERT OR IGNORE INTO injected (session, event_id) VALUES (?1, ?2)",
                    params![session, id],
                )
                .context("record injected id")?;
        }
        self.conn
            .execute(
                "INSERT INTO injected_bytes (session, bytes) VALUES (?1, ?2)
                 ON CONFLICT(session) DO UPDATE SET bytes = injected_bytes.bytes + ?2",
                params![session, i64::try_from(bytes).unwrap_or(0)],
            )
            .context("record injected bytes")?;
        Ok(())
    }

    /// Mark a file as covered for this session.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_injected_file(&self, session: &str, path: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO injected_files (session, path) VALUES (?1, ?2)",
                params![session, path],
            )
            .context("record injected file")?;
        Ok(())
    }

    /// Total bytes injected per session, for budget reporting.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn injection_stats(&self) -> Result<(i64, i64, i64)> {
        self.conn
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(bytes), 0)
                 FROM injected_bytes",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .context("read injection stats")
    }

    /// How many events carry each topic, for `brain doctor`.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn topic_counts(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT topic, COUNT(*) FROM events
                 WHERE topic IS NOT NULL
                 GROUP BY topic ORDER BY 2 DESC",
            )
            .context("prepare topic counts")?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .context("run topic counts")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read topic counts")
    }

    /// Delete everything. Safe because the log rebuilds it.
    ///
    /// # Errors
    /// Returns an error when the tables cannot be cleared.
    pub fn clear(&self) -> Result<()> {
        self.conn
            // Only what a replay rebuilds is cleared. Everything else in this
            // database is local bookkeeping the log does not contain -
            // `session_state` and `summarizer_health`, the injection and
            // recall counters, the knowledge watermark, and `entities`, which
            // is written by consolidation from a model's reading and cannot
            // be recomputed by replaying events. Clearing those would not
            // rebuild them; it would delete them.
            .execute_batch("DELETE FROM event_files; DELETE FROM events;")
            .context("clear index")?;
        Ok(())
    }
}

/// A memory pointer: everything an injection may carry, and nothing more.
#[derive(Debug, Clone)]
pub struct Pointer {
    pub id: String,
    pub ts: String,
    pub kind: String,
    pub title: String,
    /// What it is about, when something classified it.
    pub topic: Option<String>,
    /// Lifecycle hook that produced it. The primer's floor needs it: what a
    /// human typed is signal even with no file attached, while a tool call
    /// that touched nothing usually is not.
    pub hook: String,
    /// Whether the underlying event touched any file. Part of the primer's
    /// usefulness floor: a capture with no files and no classification is
    /// almost always a bare command nobody will recall.
    pub has_files: bool,
}

/// Health of one summarizer rung.
#[derive(Debug, Clone)]
pub struct SummarizerHealth {
    pub cli: String,
    pub failures: i64,
    pub last_error: Option<String>,
    /// When the most recent failure happened. `None` for rows written before
    /// this was recorded.
    pub last_failed_at: Option<String>,
}

/// One session with work waiting.
#[derive(Debug, Clone)]
pub struct PendingSession {
    pub session: String,
    pub pending: i64,
    pub newest_event_id: String,
    pub cli: String,
}

/// What consolidation last did for a session.
#[derive(Debug, Clone)]
pub struct SessionRun {
    pub last_run_at: Option<String>,
    pub last_event_id: Option<String>,
    pub last_tier: Option<String>,
}

/// Keep a stored error readable; the full text is in `brain.log`.
fn truncate_error(error: &str) -> String {
    crate::sanitize::truncate(error, 500)
}

fn parse_uuid(raw: &str) -> uuid::Uuid {
    uuid::Uuid::try_parse(raw).unwrap_or(uuid::Uuid::nil())
}

fn parse_kind(raw: &str) -> EventKind {
    match raw {
        "session_summary" => EventKind::SessionSummary,
        "page_update" => EventKind::PageUpdate,
        "note" => EventKind::Note,
        "knowledge" => EventKind::Knowledge,
        "tombstone" => EventKind::Tombstone,
        _ => EventKind::Observation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_event_kind_survives_the_round_trip_through_storage() {
        // parse_kind's catch-all silently turned a new kind into an
        // observation, so brain_get reported knowledge as a raw capture. A
        // list that has to be extended for the test to compile is what stops
        // the next kind from doing the same.
        for kind in [
            EventKind::Observation,
            EventKind::SessionSummary,
            EventKind::PageUpdate,
            EventKind::Note,
            EventKind::Knowledge,
            EventKind::Tombstone,
        ] {
            assert_eq!(parse_kind(kind.as_str()), kind, "`{}` did not round-trip", kind.as_str());
        }
    }
    use uuid::Uuid;

    fn event(title: &str, body: &str, project: Uuid) -> Event {
        let mut event = Event::new(
            Uuid::nil(),
            project,
            Uuid::nil(),
            Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
            EventKind::Observation,
            title.into(),
            body.into(),
        );
        event.files = vec!["src/main.rs".to_string()];
        event
    }

    #[test]
    fn fts5_is_available_and_searchable() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        store.index(&event("Fixed the auth middleware", "token expiry used <", project)).unwrap();
        let hits = store.search(&project.to_string(), "auth", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].title.contains("auth"));
    }

    #[test]
    fn a_query_that_breaks_fts_syntax_still_searches() {
        // The things people actually type: a path, a snippet with a stray
        // quote. FTS5 rejects both outright, and a search that errors is
        // indistinguishable from memory that is missing.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        store
            .index(&event("Edit: src/billing.rs", "totals at period end", project))
            .unwrap();

        for query in ["src/billing.rs", "period \"end", "a - b"] {
            let hits = store
                .search(&project.to_string(), query, 10)
                .unwrap_or_else(|error| panic!("query {query:?} errored: {error}"));
            let _ = hits;
        }
        assert_eq!(
            store.search(&project.to_string(), "src/billing.rs", 10).unwrap().len(),
            1,
            "a path query should find the work on that path"
        );
    }

    #[test]
    fn search_is_scoped_to_one_project() {
        let store = Store::open_memory().unwrap();
        let mine = Uuid::new_v4();
        let theirs = Uuid::new_v4();
        store.index(&event("shared word here", "body", mine)).unwrap();
        store.index(&event("shared word here", "body", theirs)).unwrap();
        assert_eq!(store.search(&mine.to_string(), "shared", 10).unwrap().len(), 1);
    }

    #[test]
    fn indexing_the_same_event_twice_is_idempotent() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let event = event("once", "body", project);
        store.index(&event).unwrap();
        store.index(&event).unwrap();
        assert_eq!(store.count().unwrap(), 1);
        assert_eq!(store.search(&project.to_string(), "once", 10).unwrap().len(), 1);
    }

    #[test]
    fn get_returns_full_bodies() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let event = event("title", "the full body text", project);
        store.index(&event).unwrap();
        let fetched = store.get(std::slice::from_ref(&event.id)).unwrap();
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].body, "the full body text");
        assert_eq!(fetched[0].files, vec!["src/main.rs".to_string()]);
    }

    #[test]
    fn a_database_from_before_a_column_existed_is_upgraded_in_place() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        store.index(&event("before", "body", project)).unwrap();

        // Rewind to the older schema, then re-run migration the way opening an
        // existing database does.
        store.conn.execute_batch("ALTER TABLE events DROP COLUMN topic;").unwrap();
        assert!(!store.has_column("events", "topic").unwrap());
        store.migrate().unwrap();
        assert!(store.has_column("events", "topic").unwrap());

        // Capture still works, and the pre-existing row survived.
        store.index(&event("after", "body", project)).unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn migration_is_idempotent() {
        let store = Store::open_memory().unwrap();
        store.migrate().unwrap();
        store.migrate().unwrap();
        assert!(store.has_column("events", "topic").unwrap());
    }

    #[test]
    fn clear_empties_the_index_and_its_fts_mirror() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        store.index(&event("gone soon", "body", project)).unwrap();
        store.clear().unwrap();
        assert_eq!(store.count().unwrap(), 0);
        assert!(store.search(&project.to_string(), "gone", 10).unwrap().is_empty());
    }
}
