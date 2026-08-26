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

/// How much wider than the caller's limit a search reads before spreading
/// results across sessions. Four deep enough that a busy session cannot fill
/// the pool by itself, shallow enough that the query stays one index scan.
const SEARCH_POOL_FACTOR: usize = 4;

/// How many top hits seed the graph-neighbour expansion. Three, because a
/// neighbour of the best hits is context and a neighbour of the tenth is
/// noise wearing its badge.
const NEIGHBOUR_SEEDS: usize = 3;

/// Query tokens shorter than this never reach the entity LIKE scan; two
/// letters inside every path separator is a match for the whole table.
const ENTITY_TOKEN_MIN: usize = 3;

/// Shortest query the trigram index can answer, fixed by the tokenizer:
/// it stores three-character runs and has nothing smaller to look up.
const TRIGRAM_MIN_CHARS: usize = 3;

/// Does this text use a script that is written without spaces between words?
///
/// The question `unicode61` gets wrong. Thai, Lao, Khmer, Myanmar, Tibetan,
/// Han, kana and Hangul all run words together, so a word-boundary tokenizer
/// cuts them at whatever mark it happens to consider punctuation. Latin,
/// Cyrillic, Greek, Arabic and Hebrew all separate words with spaces and are
/// served correctly by the tokenizer already — including them here would
/// only add substring noise to a ranking that is already right.
fn writes_without_spaces(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{0E00}'..='\u{0EFF}'   // Thai, Lao
            | '\u{0F00}'..='\u{0FFF}' // Tibetan
            | '\u{1000}'..='\u{109F}' // Myanmar
            | '\u{1780}'..='\u{17FF}' // Khmer
            | '\u{3040}'..='\u{30FF}' // Hiragana, Katakana
            | '\u{3400}'..='\u{4DBF}' // CJK extension A
            | '\u{4E00}'..='\u{9FFF}' // CJK unified
            | '\u{AC00}'..='\u{D7AF}' // Hangul syllables
            | '\u{F900}'..='\u{FAFF}' // CJK compatibility
        )
    })
}

/// Tokens per query that the entity scan will consider. Enough for any
/// question a person types; a pasted stack trace stops being a query here.
const ENTITY_TOKENS_MAX: usize = 8;

/// Events that represent one session in the entity stream. Two: the most
/// canonical page and one runner-up. The stream nominates sessions, not
/// events, and a busy session must not fill the pool through it.
const ENTITY_EVENTS_PER_SESSION: usize = 2;

/// How close a memory has to be before it counts as an answer.
///
/// A floor here is what lets "nothing is close enough" be an outcome rather
/// than "here is the corpus, sorted".
///
/// Set at the median cosine of UNRELATED pairs, measured on a real event
/// store rather than on invented sentences: memories from two different
/// sessions score below it half the time, memories from the same session
/// score well above it. Each model has its own scale for this and the numbers
/// do not transfer — the English model this replaced put unrelated pairs at
/// 0.101 and same-session pairs at 0.238, so its floor was 0.10; this one
/// puts them at 0.229 and 0.362. A floor carried over unchanged would have
/// admitted the whole corpus.
///
/// Deliberately low. This ranking never stands alone — RRF fuses it with the
/// keyword list, so its job is recall, and a threshold tuned for precision
/// would throw away the loose association that was the entire reason for
/// adding it.
const NEAREST_FLOOR: f32 = 0.23;

/// How much of the recall stack a caller wants.
///
/// Not a tuning knob — a safety boundary. Semantic ranking answers "closest to
/// this", which on any query has an answer, and a caller that DELETES what it
/// is handed must never be given a ranking that always returns something.
/// `forget --entity` is that caller, and its own preview promises the reach is
/// lexical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recall {
    /// Keyword and meaning, fused. What a person or an agent asking a question
    /// wants.
    Fused,
    /// Words only, exactly as typed. What anything destructive gets.
    Lexical,
}

/// A subject that more than one session has touched.
#[derive(Debug, serde::Serialize)]
pub struct Subject {
    pub name: String,
    /// How many distinct sessions named it. Recurrence is the signal: once is
    /// an incident, repeatedly is what the project is about.
    pub sessions: i64,
}

/// What a project looks like from outside any one question.
#[derive(Debug, serde::Serialize)]
pub struct Outline {
    pub sessions: i64,
    pub observations: i64,
    /// Durable knowledge titles - what survived several sessions.
    pub knowledge: Vec<String>,
    pub summaries: i64,
    pub subjects: Vec<Subject>,
}

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
    /// Which session produced it.
    ///
    /// This was hidden from callers on the grounds that a session uuid means
    /// nothing to an agent. That held while a list was read top to bottom.
    /// It stopped holding once one agent could read another's work: agents run
    /// sessions in parallel, so a newest-first list interleaves several of
    /// them, and the only thing that says which line belongs with which is
    /// this. It is opaque, and it does not need to be anything else - grouping
    /// asks for equality, not for meaning.
    pub session: String,
}

/// The derived index.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// How long before a consolidation claim is treated as abandoned.
    ///
    /// Long enough that a slow model call is not mistaken for a crash, short
    /// enough that a crash does not cost a session its summary for an hour.
    const CLAIM_STALE_SECS: i64 = 300;

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

                -- The same titles again, indexed by three-character run
                -- instead of by word.
                --
                -- `unicode61` finds word boundaries at spaces, and Thai,
                -- Khmer, Lao and the CJK scripts do not write any - so a
                -- sentence in them becomes fragments cut at whatever tone or
                -- vowel mark happened to fall inside it, and a word plainly
                -- present in the text is not findable. Measured on this
                -- corpus before this table existed: `ภาษาหลัก` returned
                -- nothing while three events contained it.
                --
                -- Titles only, and that is a measured choice rather than a
                -- cautious one: over this event store the title index cost
                -- nothing on disk (it fit in pages already allocated) while
                -- adding bodies cost 81 MB, a 65% larger database, to widen
                -- one stream of five.
                -- One-time migrations that are not a column and cannot be
                -- asked about. Filling an index added after the rows it
                -- covers is the first: the index cannot be asked whether it
                -- is empty, so the fact that it was filled is recorded here.
                CREATE TABLE IF NOT EXISTS schema_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS events_tri USING fts5(
                    title,
                    content='events',
                    content_rowid='rowid',
                    tokenize='trigram'
                );

                CREATE TRIGGER IF NOT EXISTS events_tri_ai AFTER INSERT ON events BEGIN
                    INSERT INTO events_tri(rowid, title) VALUES (new.rowid, new.title);
                END;
                CREATE TRIGGER IF NOT EXISTS events_tri_ad AFTER DELETE ON events BEGIN
                    INSERT INTO events_tri(events_tri, rowid, title)
                    VALUES ('delete', old.rowid, old.title);
                END;
                CREATE TRIGGER IF NOT EXISTS events_tri_au AFTER UPDATE ON events BEGIN
                    INSERT INTO events_tri(events_tri, rowid, title)
                    VALUES ('delete', old.rowid, old.title);
                    INSERT INTO events_tri(rowid, title) VALUES (new.rowid, new.title);
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

                -- What we last wrote to a page, so an edit made by hand in
                -- the vault can be told apart from our own output and read
                -- back into the log instead of being overwritten.
                CREATE TABLE IF NOT EXISTS page_state (
                    path    TEXT PRIMARY KEY,
                    hash    TEXT NOT NULL,
                    session TEXT NOT NULL
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
                -- `active` separates the two jobs this table does: the de-dup
                -- guard reads only active rows, while the uptake measurement
                -- reads all of them. A compaction deactivates instead of
                -- deleting - erasing the rows also erased the record of every
                -- pointer the session had pulled, and stats then reported a
                -- primer nobody used on a brain where the pulls had happened.
                CREATE TABLE IF NOT EXISTS injected (
                    session  TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    active   INTEGER NOT NULL DEFAULT 1,
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
                -- debounced trigger and the catch-up backstop do not redo work.
                -- One semantic vector per event. Separate from `events`
                -- because it is derived, regenerable, and four orders of
                -- magnitude larger than the row it belongs to; keeping it out
                -- means every query that does NOT rank semantically still
                -- reads narrow rows.
                CREATE TABLE IF NOT EXISTS event_vec (
                    event_id TEXT PRIMARY KEY,
                    vec      BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_state (
                    session       TEXT PRIMARY KEY,
                    project       TEXT NOT NULL,
                    last_run_at   TEXT,
                    last_event_id TEXT,
                    last_tier     TEXT,
                    -- Who is consolidating this session right now. Taken by
                    -- `claim_session`, cleared when that run finishes.
                    claimed_at    TEXT
                );
                ",
            )
            .context("apply schema")?;
        self.add_missing_columns()?;
        self.backfill_trigram()
    }

    /// Fill a full-text index that was added after the events it covers.
    ///
    /// `CREATE VIRTUAL TABLE IF NOT EXISTS` leaves an existing database with
    /// an empty index and triggers that only ever see new rows, so every
    /// memory captured before the upgrade would be invisible to the stream
    /// that was added to find it — silently, because an empty index answers
    /// every query with nothing rather than with an error.
    ///
    /// The index cannot be asked whether it holds anything. `COUNT(*)` on an
    /// external-content FTS5 table is answered by the CONTENT table, so the
    /// obvious check reads every row of `events` out of an index containing
    /// nothing and concludes it is full. That is how this shipped the first
    /// time, and it failed silently on a real event store: twenty thousand
    /// rows reported, two rows of actual index, every Thai query answered
    /// with nothing.
    ///
    /// So the fact is recorded instead of inferred. Idempotent through the
    /// marker, which is also what makes a rebuild forceable — drop the row
    /// and reopen.
    fn backfill_trigram(&self) -> Result<()> {
        let built: bool = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_state WHERE key = 'events_tri_built')",
                [],
                |row| row.get(0),
            )
            .context("check the substring index")?;
        if built {
            return Ok(());
        }
        self.conn
            .execute_batch(
                "INSERT INTO events_tri(events_tri) VALUES('rebuild');
                 INSERT OR REPLACE INTO schema_state (key, value)
                 VALUES ('events_tri_built', '1');",
            )
            .context("build the substring index")?;
        Ok(())
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
                ("session_state", "claimed_at", "TEXT"),
                ("injected", "active", "INTEGER NOT NULL DEFAULT 1"),
                ("injected", "in_flight", "INTEGER NOT NULL DEFAULT 0"),
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
    /// Search, optionally narrowed to one topic.
    ///
    /// Relevance ranking answers "what mentions this"; a scope answers "what
    /// did we DECIDE about this", which is a different question and the one
    /// worth asking when memory is large. The topic is the taxonomy
    /// consolidation already assigns, so this costs a WHERE clause rather
    /// than a new index.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn search(
        &self,
        project: &str,
        query: &str,
        topic: Option<&str>,
        limit: usize,
        recall: Recall,
    ) -> Result<Vec<Hit>> {
        let mut stmt = self
            .conn
            .prepare(
                // Two stages, because relevance alone lets one session own
                // the page. The window takes each session's best few first,
                // so the pool handed to `spread_across_sessions` contains
                // the quiet sessions at all - reading a flat top-N never
                // reaches them when one session holds most of the project.
                //
                // The demotion is inside the window too: an entry a human
                // called stale must not be the hit that represents its
                // session. Flagging has to change what the user SEES, or it
                // is a counter nobody can observe.
                "WITH matched AS (
                     SELECT e.id, e.ts, e.cli, e.kind, e.title, e.session,
                            snippet(events_fts, 1, '[', ']', ' … ', 24) AS snip,
                            CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END AS demoted,
                            rank AS relevance
                     FROM events_fts
                     JOIN events e ON e.rowid = events_fts.rowid
                     WHERE events_fts MATCH ?1 AND e.project = ?2 AND e.forgotten = 0
                           AND e.kind != 'tombstone'
                                 AND e.hook != 'correct'
                           AND (?5 IS NULL OR e.topic = ?5)
                 )
                 SELECT id, ts, cli, kind, title, snip, session FROM (
                     SELECT *, ROW_NUMBER() OVER (
                         PARTITION BY session ORDER BY demoted, relevance
                     ) AS per_session FROM matched
                 )
                 WHERE per_session <= ?3
                 ORDER BY demoted, relevance
                 LIMIT ?4",
            )
            .context("prepare search")?;

        // Read a pool wider than asked for, capped per session: spreading
        // results is only possible if the quiet sessions' hits were fetched
        // at all, and a flat pool never reaches them.
        let pool = limit.saturating_mul(SEARCH_POOL_FACTOR).max(limit);
        let mut read = |query: &str| -> rusqlite::Result<Vec<Hit>> {
            stmt.query_map(params![query, project, limit as i64, pool as i64, topic], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: row.get(5)?,
                    session: row.get(6)?,
                })
            })?
            .collect()
        };

        // FTS5 has its own query grammar, and the things people naturally
        // search for break it: `src/billing.rs` is a syntax error near `/`,
        // as is anything with an unbalanced quote. Falling back to the same
        // text as a quoted phrase turns a failed search into a literal one,
        // which is what someone typing a path meant anyway.
        let keyword = match read(query) {
            Ok(hits) => hits,
            Err(_) => {
                let phrase = format!("\"{}\"", query.replace('"', " "));
                read(&phrase).context("read search results")?
            }
        };
        drop(stmt);

        if recall == Recall::Lexical {
            return Ok(spread_across_sessions(keyword, limit));
        }

        // Meaning, alongside words. Failure here degrades to the keyword
        // results rather than failing the search: a model that will not load
        // is a worse search, not a broken one.
        let semantic = crate::embed::encode(query)
            .and_then(|vector| self.nearest(project, &vector, topic, pool))
            .unwrap_or_default();
        let semantic_ids: Vec<String> = semantic.into_iter().map(|(id, _)| id).collect();
        let semantic = self.hits_by_id(&semantic_ids)?;

        // Two more rankings that need no model at all, which is the point:
        // they are what keeps recall wide when every model is unreachable.
        // Entities are what a session DECLARED it was about, so they find
        // work whose words never matched; neighbours are what else touched
        // the same things, so a hit pulls in its context.
        let entity = self.entity_matches(project, query, topic, ENTITY_EVENTS_PER_SESSION, pool)?;
        // Only for the scripts the keyword tokenizer cannot cut into words.
        // An English query is already served correctly by `keyword`, and
        // substring matching would only add `author` to a search for `auth`.
        let substring = if writes_without_spaces(query) {
            self.substring_matches(project, query, topic, pool)?
        } else {
            Vec::new()
        };
        let mut seeds: Vec<String> = Vec::new();
        for hit in keyword.iter().chain(semantic.iter()).chain(substring.iter()) {
            if seeds.len() >= NEIGHBOUR_SEEDS {
                break;
            }
            if !seeds.contains(&hit.id) {
                seeds.push(hit.id.clone());
            }
        }
        let graph = self.neighbours_of(project, &seeds, topic, pool)?;

        Ok(spread_across_sessions(
            self.fuse(&[keyword, semantic, entity, substring, graph], pool)?,
            limit,
        ))
    }

    /// Combine several rankings into one.
    ///
    /// Reciprocal rank fusion, which needs only each list's ORDER — and that
    /// is the point. An FTS5 `rank`, a cosine similarity, an entity count and
    /// a shared-neighbour count are not on the same scale and never will be;
    /// any attempt to weight one against another directly is a constant
    /// someone tuned once against one corpus. Position is the only thing all
    /// the lists agree on the meaning of.
    ///
    /// An entry several rankings found outranks one that only appears in one,
    /// which is exactly the behaviour wanted: the keyword hit that is also
    /// about the right thing goes first. The lists are equal-weight on
    /// purpose — a per-stream weight is a knob nobody can justify a value
    /// for.
    fn fuse(&self, lists: &[Vec<Hit>], limit: usize) -> Result<Vec<Hit>> {
        // The conventional damping constant. Large enough that the top of
        // any one list does not dominate outright, so agreement between
        // rankings can still outweigh a single strong opinion.
        const K: f32 = 60.0;

        let mut score: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();
        for list in lists {
            for (rank, hit) in list.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let contribution = 1.0 / (K + rank as f32 + 1.0);
                *score.entry(hit.id.as_str()).or_default() += contribution;
            }
        }

        // Fusion keeps only each list's ORDER, which silently discards the one
        // thing both lists encoded outside their order: that a human flagged an
        // entry stale. Both rankings put demoted entries last on purpose -
        // "flagging has to change what the user SEES" - so the flag is carried
        // across the fusion rather than re-derived from a position it no longer
        // occupies.
        let demoted = self.demoted_among(score.keys().copied())?;
        let mut ranked: Vec<(&str, f32)> = score.into_iter().collect();
        ranked.sort_by(|a, b| {
            demoted
                .contains(a.0)
                .cmp(&demoted.contains(b.0))
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| b.0.cmp(a.0))
        });
        ranked.truncate(limit);

        // The first list that saw an id supplies its Hit. Keyword goes first
        // in every caller, so an entry the words found keeps its FTS snippet
        // - the marked-up excerpt - over another stream's plain body prefix.
        let mut known: std::collections::HashMap<&str, &Hit> = std::collections::HashMap::new();
        for list in lists {
            for hit in list {
                known.entry(hit.id.as_str()).or_insert(hit);
            }
        }

        Ok(ranked
            .into_iter()
            .filter_map(|(id, _)| known.get(id).map(|hit| (*hit).clone()))
            .collect())
    }

    /// Which of these ids a human has flagged stale.
    fn demoted_among<'a>(
        &self,
        ids: impl Iterator<Item = &'a str>,
    ) -> Result<std::collections::HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT 1 FROM events WHERE id = ?1 AND confidence < 0")
            .context("prepare demoted")?;
        let mut found = std::collections::HashSet::new();
        for id in ids {
            if stmt.exists(params![id]).unwrap_or(false) {
                found.insert(id.to_string());
            }
        }
        Ok(found)
    }

    /// What sits beside one memory.
    ///
    /// Two memories are neighbours when their sessions named the same entity —
    /// the same file, symbol, or subject. That is a different question from
    /// "what matches these words", and it is the one an agent has when it is
    /// already holding a memory: not "find me X" but "what else touched this".
    ///
    /// Ordered by how many entities the two share, because one shared file is
    /// a coincidence and four is a subject. The event itself is excluded — a
    /// memory is not related to itself, and returning it wastes the budget the
    /// caller is spending to look outward.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn related(&self, project: &str, id: &str, limit: usize) -> Result<Vec<Hit>> {
        let mut stmt = self
            .conn
            .prepare(
                "WITH subject AS (
                     SELECT n.name FROM entities n
                     JOIN events e ON e.session = n.session AND e.project = n.project
                     WHERE e.id = ?1 AND n.project = ?2
                 )
                 SELECT e.id, e.ts, e.cli, e.kind, e.title,
                        substr(COALESCE(e.body, ''), 1, 160), e.session,
                        COUNT(DISTINCT n.name) AS shared
                 FROM events e
                 JOIN entities n ON n.session = e.session AND n.project = e.project
                 WHERE n.name IN (SELECT name FROM subject)
                       AND e.project = ?2 AND e.id != ?1
                       AND e.forgotten = 0 AND e.kind != 'tombstone'
                       AND e.hook != 'correct'
                 GROUP BY e.id
                 ORDER BY shared DESC,
                          CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END,
                          CASE e.kind
                              WHEN 'knowledge' THEN 0
                              WHEN 'session_summary' THEN 1
                              ELSE 2
                          END,
                          e.id DESC
                 LIMIT ?3",
            )
            .context("prepare related")?;
        let rows = stmt
            .query_map(params![id, project, limit as i64], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: row.get(5)?,
                    session: row.get(6)?,
                })
            })
            .context("run related")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read related")
    }

    /// Memories whose titles contain the query as a substring.
    ///
    /// The stream that answers for scripts `unicode61` cannot cut into words.
    /// It is substring matching, not word matching, which is why it is gated
    /// on the query's script rather than run for everything: for English it
    /// would rank `author` against `auth`, and English already has both word
    /// boundaries and a stemmer.
    ///
    /// The query goes in as one quoted phrase. FTS5's own grammar would read
    /// a stray quote or a slash as syntax, and someone typing a sentence in
    /// Thai means the sentence.
    fn substring_matches(
        &self,
        project: &str,
        query: &str,
        topic: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Hit>> {
        // The tokenizer indexes three-character runs, so nothing shorter can
        // be looked up at all.
        if query.chars().count() < TRIGRAM_MIN_CHARS {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT e.id, e.ts, e.cli, e.kind, e.title,
                        substr(COALESCE(e.body, ''), 1, 160), e.session
                 FROM events_tri
                 JOIN events e ON e.rowid = events_tri.rowid
                 WHERE events_tri MATCH ?1 AND e.project = ?2 AND e.forgotten = 0
                       AND e.kind != 'tombstone' AND e.hook != 'correct'
                       AND (?4 IS NULL OR e.topic = ?4)
                 ORDER BY CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END, rank
                 LIMIT ?3",
            )
            .context("prepare substring matches")?;
        let phrase = format!("\"{}\"", query.replace('"', " "));
        let rows = stmt
            .query_map(params![phrase, project, limit as i64, topic], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: row.get(5)?,
                    session: row.get(6)?,
                })
            })
            .context("run substring matches")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read substring matches")
    }

    /// Sessions whose DECLARED subjects match the query's words.
    ///
    /// Entities are what consolidation said a session was about - files,
    /// symbols, subjects - so this stream finds work whose own text never
    /// contains the query. It needs no model, which is why it exists: it is
    /// one of the rankings that keeps zero-LLM recall wide.
    ///
    /// A match nominates a SESSION, not an event, because entities are
    /// recorded per session. Handing back every event in a matching session
    /// would let one busy session flood the pool, so each session is
    /// represented by its most canonical few - knowledge first, then the
    /// summary, then raw captures - the same authority order `related` uses.
    fn entity_matches(
        &self,
        project: &str,
        query: &str,
        topic: Option<&str>,
        per_session: usize,
        pool: usize,
    ) -> Result<Vec<Hit>> {
        // The same normalization entity names went through at write time;
        // matching raw query text against normalized names would miss on
        // nothing more than a capital letter.
        let tokens: Vec<String> = query
            .split_whitespace()
            .map(crate::consolidate::normalize_entity)
            .filter(|token| token.len() >= ENTITY_TOKEN_MIN)
            .map(|token| token.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"))
            .take(ENTITY_TOKENS_MAX)
            .collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let likes = tokens
            .iter()
            .enumerate()
            .map(|(index, _)| format!("n.name LIKE ?{} ESCAPE '\\'", index + 5))
            .collect::<Vec<_>>()
            .join(" OR ");
        let sql = format!(
            "WITH matched AS (
                 SELECT n.session, COUNT(DISTINCT n.name) AS matched
                 FROM entities n
                 WHERE n.project = ?1 AND ({likes})
                 GROUP BY n.session
             ),
             candidates AS (
                 SELECT e.id, e.ts, e.cli, e.kind, e.title,
                        substr(COALESCE(e.body, ''), 1, 160) AS snip, e.session,
                        m.matched,
                        CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END AS demoted,
                        CASE e.kind
                            WHEN 'knowledge' THEN 0
                            WHEN 'session_summary' THEN 1
                            ELSE 2
                        END AS authority,
                        ROW_NUMBER() OVER (
                            PARTITION BY e.session
                            ORDER BY CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END,
                                     CASE e.kind
                                         WHEN 'knowledge' THEN 0
                                         WHEN 'session_summary' THEN 1
                                         ELSE 2
                                     END,
                                     e.id DESC
                        ) AS per_session
                 FROM events e
                 JOIN matched m ON m.session = e.session
                 WHERE e.project = ?1 AND e.forgotten = 0 AND e.kind != 'tombstone'
                       AND e.hook != 'correct'
                       AND (?2 IS NULL OR e.topic = ?2)
             )
             SELECT id, ts, cli, kind, title, snip, session FROM candidates
             WHERE per_session <= ?3
             ORDER BY demoted, matched DESC, authority, id DESC
             LIMIT ?4"
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare entity matches")?;
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![
            Box::new(project.to_string()),
            Box::new(topic.map(str::to_string)),
            Box::new(per_session as i64),
            Box::new(pool as i64),
        ];
        for token in &tokens {
            params.push(Box::new(format!("%{token}%")));
        }
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params.iter().map(AsRef::as_ref)), |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: row.get(5)?,
                    session: row.get(6)?,
                })
            })
            .context("run entity matches")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read entity matches")
    }

    /// What sits beside the hits another ranking already found.
    ///
    /// `related`, widened to several seeds and folded into search: the top
    /// hits' sessions declared entities, and whatever else touched those
    /// entities is context the query's words never asked for. Needs no
    /// model - the other zero-LLM ranking.
    ///
    /// The shared-name count is taken once per SESSION, before events are
    /// touched at all. Entities are recorded per session, so every event in a
    /// session shares the same set - counting them per event asks the same
    /// question once for each event and pays for the answer every time. The
    /// join that made that possible multiplied a session's events by its
    /// shared names, so one long session was enough to turn a search into a
    /// million-row group-by: measured at 25s on a 21k-event brain, against
    /// 1.2s for this shape, with byte-identical results.
    fn neighbours_of(
        &self,
        project: &str,
        seeds: &[String],
        topic: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Hit>> {
        if seeds.is_empty() {
            return Ok(Vec::new());
        }
        let holes = |offset: usize| {
            (0..seeds.len())
                .map(|index| format!("?{}", index + offset))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let sql = format!(
            "WITH subject AS (
                 SELECT DISTINCT n.name FROM entities n
                 JOIN events e ON e.session = n.session AND e.project = n.project
                 WHERE e.id IN ({seed_holes}) AND n.project = ?1
             ),
             shared AS (
                 SELECT n.session, COUNT(DISTINCT n.name) AS shared
                 FROM entities n
                 WHERE n.project = ?1 AND n.name IN (SELECT name FROM subject)
                 GROUP BY n.session
             )
             SELECT e.id, e.ts, e.cli, e.kind, e.title,
                    substr(COALESCE(e.body, ''), 1, 160), e.session,
                    s.shared
             FROM events e
             JOIN shared s ON s.session = e.session
             WHERE e.project = ?1 AND e.id NOT IN ({seed_holes})
                   AND e.forgotten = 0 AND e.kind != 'tombstone'
                   AND e.hook != 'correct'
                   AND (?2 IS NULL OR e.topic = ?2)
             ORDER BY s.shared DESC,
                      CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END,
                      CASE e.kind
                          WHEN 'knowledge' THEN 0
                          WHEN 'session_summary' THEN 1
                          ELSE 2
                      END,
                      e.id DESC
             LIMIT ?3",
            seed_holes = holes(4),
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare neighbours")?;
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![
            Box::new(project.to_string()),
            Box::new(topic.map(str::to_string)),
            Box::new(limit as i64),
        ];
        for seed in seeds {
            params.push(Box::new(seed.clone()));
        }
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params.iter().map(AsRef::as_ref)), |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: row.get(5)?,
                    session: row.get(6)?,
                })
            })
            .context("run neighbours")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read neighbours")
    }

    /// What this project is, before anyone knows what to ask about it.
    ///
    /// An agent opening a session can search only for what it already suspects.
    /// This is the other half: the durable knowledge, the subjects that recur,
    /// and enough shape to form a question with. Counts come from the same
    /// filtered view search uses, so an outline never describes memory that
    /// recall would refuse to return.
    ///
    /// # Errors
    /// Returns an error when a query fails.
    pub fn outline(&self, project: &str, limit: usize) -> Result<Outline> {
        let live = "forgotten = 0 AND kind != 'tombstone' AND hook != 'correct'";
        let count = |extra: &str| -> Result<i64> {
            self.conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM events WHERE project = ?1 AND {live} {extra}"),
                    params![project],
                    |row| row.get(0),
                )
                .context("count outline")
        };
        let sessions: i64 = self
            .conn
            .query_row(
                &format!(
                    "SELECT COUNT(DISTINCT session) FROM events WHERE project = ?1 AND {live}"
                ),
                params![project],
                |row| row.get(0),
            )
            .context("count sessions")?;

        let mut stmt = self
            .conn
            .prepare(
                "SELECT n.name, COUNT(DISTINCT n.session) AS sessions
                 FROM entities n
                 WHERE n.project = ?1
                 GROUP BY n.name
                 HAVING sessions > 1
                 ORDER BY sessions DESC, n.name
                 LIMIT ?2",
            )
            .context("prepare outline subjects")?;
        let subjects = stmt
            .query_map(params![project, limit as i64], |row| {
                Ok(Subject { name: row.get(0)?, sessions: row.get(1)? })
            })
            .context("run outline subjects")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("read outline subjects")?;

        Ok(Outline {
            sessions,
            observations: count("")?,
            knowledge: self.knowledge_titles(project)?,
            summaries: count("AND kind = 'session_summary'")?,
            subjects,
        })
    }

    /// Hits for ids the keyword pass never saw.
    fn hits_by_id(&self, ids: &[String]) -> Result<Vec<Hit>> {
        let mut out = Vec::with_capacity(ids.len());
        let mut stmt = self
            .conn
            .prepare(
                // The same recall floor, repeated rather than assumed. Today
                // every id reaching here came through `nearest`, which already
                // enforces it - but that is a property of one caller, not of
                // this function, and a withdrawal that depends on the call
                // graph staying the shape it is today is not a withdrawal.
                "SELECT id, ts, cli, kind, title, substr(COALESCE(body, ''), 1, 160), session
                 FROM events
                 WHERE id = ?1 AND forgotten = 0 AND kind != 'tombstone'
                       AND hook != 'correct'",
            )
            .context("prepare hits by id")?;
        for id in ids {
            if let Ok(hit) = stmt.query_row(params![id], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: row.get(5)?,
                    session: row.get(6)?,
                })
            }) {
                out.push(hit);
            }
        }
        Ok(out)
    }

    /// Store one event's semantic vector.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn set_vectors(&self, rows: &[(String, Vec<u8>)]) -> Result<()> {
        // One transaction, not one per row. A two-thousand-row backlog as
        // two thousand autocommits is two thousand write-lock acquisitions,
        // and one `SQLITE_BUSY` past the timeout used to take the whole
        // consolidation run down with it.
        let transaction = self.conn.unchecked_transaction().context("begin vectors")?;
        {
            let mut stmt = transaction
                .prepare(
                    "INSERT INTO event_vec (event_id, vec) VALUES (?1, ?2)
                     ON CONFLICT(event_id) DO UPDATE SET vec = excluded.vec",
                )
                .context("prepare store vector")?;
            for (id, vector) in rows {
                // A vector of all zeros means nothing tokenized - text made
                // entirely of words this vocabulary has never seen. Storing it
                // would put a row in the ranking whose score against every
                // query is exactly 0.0, which outranks every genuinely
                // negative cosine. It is absence, so it is stored as absence.
                if vector.iter().all(|byte| *byte == 0) {
                    continue;
                }
                stmt.execute(params![id, vector]).context("store vector")?;
            }
        }
        transaction.commit().context("commit vectors")?;
        Ok(())
    }

    /// Events that still have no vector, newest first, with the text to encode.
    ///
    /// Newest first because a backlog is worked through in batches and the
    /// recent end is what anyone is about to search for.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn events_missing_vectors(&self, project: &str, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare(
                // Scoped to the project being consolidated. Globally ordered,
                // a busy project spends every run's budget on its own newest
                // rows and a quiet one never gets embedded at all.
                //
                // A vector of the wrong width counts as absent, which is how
                // a model change migrates: `similarity` scores mismatched
                // widths at 0.0, so a stale row is not a worse answer but a
                // memory that has left semantic search entirely. Listing it
                // here lets the ordinary backlog re-embed it a bounded slice
                // at a time, and `doctor`'s percentage reports the progress -
                // no migration step, and nothing to run by hand.
                "SELECT e.id, e.title || ' ' || COALESCE(e.body, '')
                 FROM events e
                 LEFT JOIN event_vec v ON v.event_id = e.id
                 WHERE (v.event_id IS NULL OR length(v.vec) != ?3)
                       AND e.kind != 'tombstone' AND e.project = ?1
                 ORDER BY e.id DESC
                 LIMIT ?2",
            )
            .context("prepare missing vectors")?;
        let rows = stmt
            .query_map(params![project, limit as i64, crate::embed::DIMS as i64], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .context("run missing vectors")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read missing vectors")
    }

    /// How many events have a vector, and how many are still waiting.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn vector_coverage(&self) -> Result<(i64, i64)> {
        // Width, not presence. A vector left behind by an older model still
        // has a row and still scores 0.0 against every query, so counting it
        // would report a full index over an empty search - and hide the
        // backlog that is quietly putting it right.
        let embedded: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM event_vec WHERE length(vec) = ?1",
                params![crate::embed::DIMS as i64],
                |row| row.get(0),
            )
            .context("count vectors")?;
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM events WHERE kind != 'tombstone'", [], |row| {
                row.get(0)
            })
            .context("count events")?;
        Ok((embedded, total))
    }

    /// Rank a project's events by meaning rather than by words.
    ///
    /// Brute force on purpose. At 512 bytes a vector, a project with a hundred
    /// thousand events is 51 MB of sequential read and a few million integer
    /// multiplies — comfortably inside the time a search is allowed to take,
    /// and it costs no index to maintain, no extension to load, and no second
    /// database to keep consistent with this one. An approximate index earns
    /// its complexity somewhere past this scale, not at it.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn nearest(
        &self,
        project: &str,
        query: &[u8],
        topic: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        if query.iter().all(|byte| *byte == 0) {
            // Nothing tokenized: an empty query, or text made entirely of
            // words this vocabulary has never seen. A zero vector is not a
            // question, and answering it with the corpus is worse than
            // answering it with nothing.
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare(
                // The same recall floor every other read enforces. A memory
                // withdrawn from search has to be withdrawn from this one too,
                // or the withdrawal was cosmetic.
                "SELECT v.event_id, v.vec, e.confidence
                 FROM event_vec v
                 JOIN events e ON e.id = v.event_id
                 WHERE e.project = ?1 AND e.forgotten = 0 AND e.kind != 'tombstone'
                       AND e.hook != 'correct'
                       AND (?2 IS NULL OR e.topic = ?2)",
            )
            .context("prepare nearest")?;
        let mut scored: Vec<(String, f32, bool)> = stmt
            .query_map(params![project, topic], |row| {
                let id: String = row.get(0)?;
                let vec: Vec<u8> = row.get(1)?;
                let confidence: i64 = row.get(2)?;
                let score = crate::embed::similarity(query, &vec);
                Ok((id, score, confidence < 0))
            })
            .context("run nearest")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("read nearest")?;

        // A cosine ranking always has a best answer, which is not the same as
        // having a relevant one. Without a floor every query returns the whole
        // project in some order - including the empty query, whose vector is
        // all zeros and scores 0.0 against everything, a flat tie that still
        // sorts. "Nothing is close enough" has to be an answer this can give.
        scored.retain(|(_, score, _)| *score >= NEAREST_FLOOR);
        // Demotion is applied AFTER the floor, not as part of the score: a
        // human flagging an entry stale should push it down the list, never
        // push it off the end of a query it genuinely matches.
        scored.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.1.total_cmp(&a.1)));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(id, score, _)| (id, score)).collect())
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
    /// One brain holds every CLI's work, so `cli` answers the question a
    /// shared brain exists to answer: what did the OTHER agent just do. It is
    /// the whole reason the column is stored per event rather than per brain -
    /// without a way to ask by it, cross-CLI handoff is a property of the data
    /// that nothing can read.
    ///
    /// `kind` is what makes that answer readable. Agents run in parallel: on
    /// the brain this was written against, seven codex sessions opened within
    /// two seconds of each other, so a flat newest-first list interleaves
    /// several pieces of unrelated work line by line and none of them can be
    /// followed. `session_summary` collapses that to one line per session -
    /// the granularity "what has it been doing" actually asks for - and
    /// `observation` is the other half: what a session is doing right now,
    /// before consolidation has written its summary.
    ///
    /// `session` closes the loop the other two open. Naming a session is how a
    /// list of sessions becomes one piece of work read whole: search or the
    /// summary list says which one, this reads it. Every hit already carries
    /// the id, so nothing has to be remembered between the two calls.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn recent(
        &self,
        project: &str,
        cli: Option<&str>,
        kind: Option<&str>,
        session: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Hit>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, ts, cli, kind, title, session
                 FROM events WHERE project = ?1 AND forgotten = 0 AND kind != 'tombstone'
                       AND hook != 'correct'
                       AND (?3 IS NULL OR cli = ?3)
                       AND (?4 IS NULL OR kind = ?4)
                       AND (?5 IS NULL OR session = ?5)
                 ORDER BY id DESC LIMIT ?2",
            )
            .context("prepare recent")?;
        let rows = stmt
            .query_map(params![project, limit as i64, cli, kind, session], |row| {
                Ok(Hit {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    cli: row.get(2)?,
                    kind: row.get(3)?,
                    title: row.get(4)?,
                    snippet: String::new(),
                    session: row.get(5)?,
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
        // Deactivated, not deleted. These rows are also the uptake
        // measurement's denominator - and their matches in `recalled` its
        // numerator - so deleting them here silently un-counted every pull a
        // session made before it compacted.
        self.conn
            .execute("UPDATE injected SET active = 0 WHERE session = ?1", params![session])
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

    /// Remember what we wrote to a page.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_page(&self, path: &str, hash: &str, session: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO page_state (path, hash, session) VALUES (?1, ?2, ?3)
                 ON CONFLICT(path) DO UPDATE SET hash = excluded.hash, session = excluded.session",
                params![path, hash, session],
            )
            .context("record page state")?;
        Ok(())
    }

    /// Every page whose file no longer matches what we wrote, with the
    /// session it belongs to.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn pages_edited_by_hand(&self) -> Result<Vec<(String, String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, hash, session FROM page_state")
            .context("prepare page state")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })
            .context("run page state")?;
        Ok(rows.filter_map(std::result::Result::ok).collect())
    }

    /// The consolidated summary for one session, if it has one.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn summary_for_session(&self, session: &str) -> Result<Option<(String, String)>> {
        let found = self.conn.query_row(
            "SELECT id, body FROM events
             WHERE session = ?1 AND kind = 'session_summary' AND forgotten = 0
             ORDER BY id DESC LIMIT 1",
            [session],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        );
        match found {
            Ok(pair) => Ok(Some(pair)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    /// A session's summary text, but only when a HUMAN corrected it.
    ///
    /// The summary row also holds whatever the last summarizer produced, so
    /// reading it unconditionally would freeze the first summary forever -
    /// a rule-based run could never be replaced by a model's better one.
    /// The join is what distinguishes "someone edited this in the vault"
    /// from "this is simply the current text".
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn human_corrected_summary(&self, session: &str) -> Result<Option<String>> {
        let found = self.conn.query_row(
            "SELECT s.body FROM events s
             JOIN events c ON c.id = s.corrected_by
             WHERE s.session = ?1 AND s.kind = 'session_summary' AND s.forgotten = 0
                   AND c.cli = 'human'
             ORDER BY s.id DESC LIMIT 1",
            [session],
            |row| row.get::<_, String>(0),
        );
        match found {
            Ok(body) => Ok(Some(body)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
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
                "SELECT e.id, e.ts, e.cli, e.kind, e.title, e.session
                 FROM events e
                 JOIN entities n ON n.session = e.session AND n.project = e.project
                 WHERE e.project = ?1 AND n.name = ?2 AND e.forgotten = 0
                       AND e.kind != 'tombstone'
                             AND e.hook != 'correct'
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
                    session: row.get(5)?,
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

    /// The same measurement, for still-unsummarized work alone.
    ///
    /// Separate because the two failures are different. A summary nobody
    /// pulls means the primer is describing the wrong things. A half-finished
    /// task nobody pulls means the reserve holding it is either too small to
    /// say anything useful or too large to be believed - and the number of
    /// lines it reserves cannot be argued about without this.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn in_flight_uptake(&self) -> Result<(i64, i64)> {
        self.conn
            .query_row(
                "SELECT
                   (SELECT COUNT(*) FROM injected WHERE in_flight = 1),
                   (SELECT COUNT(*) FROM injected i
                      WHERE i.in_flight = 1
                        AND EXISTS (SELECT 1 FROM recalled r WHERE r.event_id = i.event_id))",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .context("read in-flight uptake")
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

    /// Take exclusive rights to consolidate one session.
    ///
    /// Returns whether we got them. This is the whole of our mutual exclusion:
    /// two consolidations overlapping on one session is ordinary - a session
    /// ending starts a run for itself while another session opening starts the
    /// catch-up for everything pending - and without this both summarize the
    /// same backlog, which is a second model call the user pays for and two
    /// copies of one narrative in memory.
    ///
    /// One statement, because it has to be atomic and SQLite is already the one
    /// thing every process here agrees on. The comparable systems all reach for
    /// a single writer instead - a resident server or worker that owns the
    /// database - which this project does not have and will not add. A lock
    /// file works too, and did, but it can only report that SOMEONE holds the
    /// lock, never whether our own work is still outstanding; closing that gap
    /// took a second check and a third piece of state to make the second check
    /// safe. A claim answers the actual question in one round trip.
    ///
    /// A claim older than [`Self::CLAIM_STALE_SECS`] is taken over: a crashed
    /// run must not wedge a session's memory forever.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn claim_session(&self, session: &str, project: &str) -> Result<bool> {
        let now = jiff::Timestamp::now();
        let stale = (now - jiff::SignedDuration::from_secs(Self::CLAIM_STALE_SECS)).to_string();
        let changed = self
            .conn
            .execute(
                "INSERT INTO session_state (session, project, claimed_at)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(session) DO UPDATE SET claimed_at = ?3
                 WHERE session_state.claimed_at IS NULL
                    OR session_state.claimed_at < ?4",
                params![session, project, now.to_string(), stale],
            )
            .context("claim session")?;
        Ok(changed == 1)
    }

    /// Give the claim back, so the next run does not wait out the stale window.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn release_session(&self, session: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE session_state SET claimed_at = NULL WHERE session = ?1",
                params![session],
            )
            .context("release session")?;
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
        self.ranked_pointers(project, None, limit)
    }

    /// The same ranking, restricted to one kind.
    ///
    /// What makes a quota possible. Ranked together, knowledge takes every
    /// line the primer has as soon as there is enough of it - measured at 124
    /// entries: 21 knowledge, 5 in flight, no session summaries at all. It
    /// outranks a summary by kind and it never stops accumulating, so the
    /// question is not whether it crowds everything else out but when.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn pointers_of_kind(
        &self,
        project: &str,
        kind: &str,
        limit: usize,
    ) -> Result<Vec<Pointer>> {
        self.ranked_pointers(project, Some(kind), limit)
    }

    fn ranked_pointers(
        &self,
        project: &str,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Pointer>> {
        let sql = format!(
            // The same floor every other read enforces. Injection is recall
            // too - and the costlier half, because it spends bytes in every
            // future session whether or not anyone asked. A withdrawn memory
            // that only disappears from search has not been withdrawn.
            "SELECT id, ts, kind, title, topic, files, hook FROM events
             WHERE project = ?1 AND forgotten = 0 AND kind != 'tombstone'
                   AND hook != 'correct'
                   AND (?3 IS NULL OR kind = ?3)
             ORDER BY {}, id DESC
             LIMIT ?2",
            Self::rank("")
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare primer pointers")?;
        let rows = stmt
            .query_map(params![project, limit as i64, kind], |row| {
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

    /// The newest captures nothing has summarized yet.
    ///
    /// Deliberately not part of [`Self::primer_pointers`]'s ranking, which
    /// puts kind before recency and is right to: a consolidated summary
    /// really is worth more than a raw capture, nearly always. The exception
    /// is the one this answers - a session killed mid-task leaves captures
    /// that no summary covers, and they lose to every summary ever written.
    ///
    /// Worse than merely missing: the newest captures are often ABOUT
    /// something already summarized. The summary says the subject was
    /// handled, and the capture saying it is half-finished is the part left
    /// out - so the next session reads "done" and redoes the rest.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn unconsolidated_pointers(&self, project: &str, limit: usize) -> Result<Vec<Pointer>> {
        let mut stmt = self
            .conn
            .prepare(
                // The same test `inject::worth_injecting` applies, pushed into
                // the query so the limit counts lines that will survive it.
                // Taking the newest five rows instead spends the reserve on
                // whatever ran last - a session_start, a stop - and the
                // caller drops all five, leaving the seats empty and the
                // in-flight work still invisible. Which is what shipped
                // first, and what a real event store showed within minutes.
                "SELECT id, ts, kind, title, topic, files, hook FROM events
                 WHERE project = ?1 AND consolidated = 0 AND forgotten = 0
                       AND kind != 'tombstone' AND hook != 'correct'
                       AND (topic IS NOT NULL
                            OR kind != 'observation'
                            OR files != '[]'
                            OR hook = 'user_prompt_submit'
                            OR (hook = 'stop' AND title != 'Turn finished'))
                 ORDER BY id DESC
                 LIMIT ?2",
            )
            .context("prepare unconsolidated pointers")?;
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
            .context("run unconsolidated pointers")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read unconsolidated pointers")
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
                         AND e.hook != 'correct'
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
                "SELECT id, ts, cli, kind, title, session FROM events
                 WHERE project = ?1 AND ts >= ?2 AND forgotten = 0 AND kind != 'tombstone'
                       AND hook != 'correct'
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
                    session: row.get(5)?,
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
                "SELECT 1 FROM injected WHERE session = ?1 AND event_id = ?2 AND active = 1",
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
    pub fn record_injected(
        &self,
        session: &str,
        ids: &[String],
        in_flight: usize,
        bytes: usize,
    ) -> Result<()> {
        for (at, id) in ids.iter().enumerate() {
            self.conn
                .execute(
                    // Re-arms a row a compaction deactivated: after a reset the
                    // guard reports the pointer as unseen, this pushes it
                    // again, and OR IGNORE would leave `active` at 0 - so the
                    // same pointer would then be re-pushed at every
                    // opportunity for the rest of the session.
                    // `in_flight` is sticky: a pointer first shown as
                    // unsummarized work stays counted as that, even if a
                    // later injection of the same id comes from the ranked
                    // list once a summary exists. The question the column
                    // answers is what the agent was handed, not what the
                    // event became.
                    "INSERT INTO injected (session, event_id, in_flight) VALUES (?1, ?2, ?3)
                     ON CONFLICT(session, event_id) DO UPDATE SET
                         active = 1,
                         in_flight = MAX(injected.in_flight, ?3)",
                    params![session, id, i64::from(at < in_flight)],
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

    /// When the worst-offending session was last active.
    ///
    /// `injected_bytes` has no timestamp of its own - a session's spend is
    /// permanent once recorded, with nothing to age it out - so a bug fixed
    /// today stays reported as an active failure forever unless the reader
    /// can tell how old the worst number is. The session's own most recent
    /// captured event is the closest true answer available, the same idiom
    /// `project_cli` already uses to answer "which CLI, really" without a
    /// dedicated column of its own.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn worst_injection_at(&self) -> Result<Option<String>> {
        let ts = self.conn.query_row(
            "SELECT e.ts FROM events e
             WHERE e.session = (
                 SELECT session FROM injected_bytes ORDER BY bytes DESC LIMIT 1
             )
             ORDER BY e.id DESC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        );
        match ts {
            Ok(ts) => Ok(Some(ts)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
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

/// Interleave hits so no single session can own the whole result.
///
/// Relevance alone is not enough when one session is much louder than the
/// rest: measured on a real machine, every query returned ten of ten hits
/// from the session that happened to be running - which held 97% of the
/// project's events - and memory from thirteen earlier sessions was
/// unreachable through search. Worse, those hits were things the agent could
/// already see in its own context, so pulling them bought nothing.
///
/// A permutation, never a filter: sessions are visited in the order their
/// best hit appeared, one hit each per round, so the single most relevant
/// result stays first and nothing is dropped. A search that matched only one
/// session returns exactly what it always did.
fn spread_across_sessions(hits: Vec<Hit>, limit: usize) -> Vec<Hit> {
    let mut sessions: Vec<(String, std::collections::VecDeque<Hit>)> = Vec::new();
    for hit in hits {
        match sessions.iter_mut().find(|(session, _)| *session == hit.session) {
            Some((_, queue)) => queue.push_back(hit),
            None => sessions.push((hit.session.clone(), [hit].into())),
        }
    }

    let mut out = Vec::with_capacity(limit);
    while out.len() < limit {
        let before = out.len();
        for (_, queue) in &mut sessions {
            if out.len() >= limit {
                break;
            }
            if let Some(hit) = queue.pop_front() {
                out.push(hit);
            }
        }
        // Every session is drained; nothing left to interleave.
        if out.len() == before {
            break;
        }
    }
    out
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

    fn hit(id: &str, session: &str) -> Hit {
        Hit {
            id: id.to_string(),
            ts: "2026-08-24T00:00:00Z".to_string(),
            cli: "claude-code".to_string(),
            kind: "observation".to_string(),
            title: id.to_string(),
            snippet: String::new(),
            session: session.to_string(),
        }
    }

    #[test]
    fn spreading_results_is_a_permutation_of_the_most_relevant_ones() {
        // A loud session (twelve hits) and two quiet ones (one each), in
        // relevance order as FTS returned them.
        let mut hits: Vec<Hit> = (0..12).map(|i| hit(&format!("loud{i}"), "loud")).collect();
        hits.push(hit("quiet-a", "a"));
        hits.push(hit("quiet-b", "b"));

        let out = spread_across_sessions(hits, 6);
        assert_eq!(out.len(), 6, "the caller asked for six and must get six");
        assert_eq!(out[0].id, "loud0", "the single most relevant hit still leads");

        let sessions: Vec<&str> = out.iter().map(|hit| hit.session.as_str()).collect();
        assert!(sessions.contains(&"a") && sessions.contains(&"b"), "quiet sessions unreachable: {sessions:?}");
        let loud = sessions.iter().filter(|session| **session == "loud").count();
        assert!(loud < 6, "one session still owns the whole page: {sessions:?}");
    }

    #[test]
    fn a_single_session_search_is_left_exactly_as_it_was() {
        // Nothing to interleave means nothing may change - spreading is a
        // permutation, never a filter, so a project with one session sees
        // precisely the ranking FTS produced.
        let hits: Vec<Hit> = (0..5).map(|i| hit(&format!("only{i}"), "one")).collect();
        let out = spread_across_sessions(hits, 10);
        assert_eq!(
            out.iter().map(|hit| hit.id.clone()).collect::<Vec<_>>(),
            vec!["only0", "only1", "only2", "only3", "only4"],
            "a single-session result was reordered or truncated"
        );
    }

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
        let hits = store.search(&project.to_string(), "auth", None, 10, Recall::Fused).unwrap();
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
                .search(&project.to_string(), query, None, 10, Recall::Fused)
                .unwrap_or_else(|error| panic!("query {query:?} errored: {error}"));
            let _ = hits;
        }
        assert_eq!(
            store.search(&project.to_string(), "src/billing.rs", None, 10, Recall::Fused).unwrap().len(),
            1,
            "a path query should find the work on that path"
        );
    }

    fn event_in_session(title: &str, body: &str, project: Uuid, session: Uuid) -> Event {
        let mut event = Event::new(
            Uuid::nil(),
            project,
            session,
            Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
            EventKind::Observation,
            title.into(),
            body.into(),
        );
        event.files = vec!["src/main.rs".to_string()];
        event
    }

    #[test]
    fn an_index_added_after_the_events_it_covers_gets_filled() {
        // The failure this exists to catch, found on a real event store and
        // not by any test: `COUNT(*)` on an external-content FTS5 table is
        // answered by the CONTENT table, not by the index, so an emptiness
        // check written that way reads 20,120 rows out of a completely empty
        // index and skips the rebuild. Nothing errors. Every query in the
        // affected script just returns nothing, forever.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        store.index(&event("Asked: ปรับปรุงประสิทธิภาพการค้นหา", "body", project)).unwrap();
        assert_eq!(
            store.search(&project.to_string(), "ประสิทธิภาพ", None, 10, Recall::Fused).unwrap().len(),
            1
        );

        // An index that exists but holds nothing - exactly what an upgrade
        // leaves behind.
        store.conn.execute_batch("INSERT INTO events_tri(events_tri) VALUES('delete-all');").unwrap();
        store.conn.execute_batch("DELETE FROM schema_state WHERE key = 'events_tri_built';").unwrap();
        assert_eq!(
            store.search(&project.to_string(), "ประสิทธิภาพ", None, 10, Recall::Fused).unwrap().len(),
            0,
            "the emptied index still answered - this test is not testing what it claims"
        );

        store.migrate().unwrap();
        assert_eq!(
            store.search(&project.to_string(), "ประสิทธิภาพ", None, 10, Recall::Fused).unwrap().len(),
            1,
            "opening the store did not fill an index that was empty"
        );
    }

    #[test]
    fn a_word_inside_a_thai_sentence_is_findable() {
        // Thai writes without spaces, and `unicode61` splits on tone and
        // vowel marks rather than on word boundaries - so a sentence becomes
        // fragments like `กระบบค` and `นหาให`, and searching for a word that
        // is plainly in the text returns nothing. Whether recall works then
        // depends on where the marks happened to fall, which is a lottery,
        // not an index.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        store
            .index(&event("Asked: แก้บั๊กระบบค้นหาให้รองรับภาษาไทย", "body", project))
            .unwrap();

        for query in ["ระบบ", "ค้นหา", "ภาษาไทย"] {
            let hits = store.search(&project.to_string(), query, None, 10, Recall::Fused).unwrap();
            assert_eq!(hits.len(), 1, "{query:?} did not find the sentence containing it");
        }
    }

    #[test]
    fn an_english_query_is_not_touched_by_the_substring_stream() {
        // Trigram matching is substring matching: it would rank `author` for
        // `auth`. English already has word boundaries and a stemmer, so the
        // stream stays out of the way unless the query is in a script that
        // needs it.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        store.index(&event("Wrote the authentication guide", "body", project)).unwrap();
        store.index(&event("Chose an author for the page", "body", project)).unwrap();

        let hits = store.search(&project.to_string(), "authentication", None, 10, Recall::Fused).unwrap();
        assert_eq!(hits[0].title, "Wrote the authentication guide", "{hits:?}");
    }

    #[test]
    fn coverage_counts_what_can_answer_a_query_not_what_has_a_row() {
        // The number `doctor` prints is the only thing telling anyone a model
        // change is still being absorbed. Counting rows rather than usable
        // rows reports 100% while every one of them scores 0.0 against every
        // query - a full index and an empty search, agreeing with each other.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let stale = event("Chose SQLite over Postgres", "nothing resident", project);
        let fresh = event("Wrote the installation guide", "one page", project);
        store.index(&stale).unwrap();
        store.index(&fresh).unwrap();
        store.set_vectors(&[(stale.id.clone(), vec![7u8; crate::embed::DIMS / 2])]).unwrap();
        store.set_vectors(&[(fresh.id.clone(), vec![7u8; crate::embed::DIMS])]).unwrap();

        let (embedded, total) = store.vector_coverage().unwrap();
        assert_eq!(total, 2);
        assert_eq!(embedded, 1, "a vector of the wrong width was counted as coverage");
    }

    #[test]
    fn a_vector_from_an_older_model_is_treated_as_missing() {
        // Changing the embedding model changes the width of a vector, and
        // `similarity` scores mismatched widths at 0.0 - so a stale row is
        // not a worse answer, it is a memory that has silently left semantic
        // search. The backlog has to see it as absent, or an upgrade would
        // quietly hollow out the index of every brain that already existed.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let stale = event("Chose SQLite over Postgres", "nothing resident", project);
        store.index(&stale).unwrap();
        store
            .set_vectors(&[(stale.id.clone(), vec![7u8; crate::embed::DIMS / 2])])
            .unwrap();

        let pending = store.events_missing_vectors(&project.to_string(), 10).unwrap();
        assert!(
            pending.iter().any(|(id, _)| *id == stale.id),
            "a vector of the wrong width was counted as present: {pending:?}"
        );
    }

    #[test]
    fn an_entity_match_finds_work_the_words_never_mention() {
        // Consolidation recorded "src/billing.rs" as one of the session's
        // entities, but no event text ever says "billing". Words alone cannot
        // find this session; the declared entity is the only trail.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let session = Uuid::new_v4();
        store
            .index(&event_in_session("Quarterly totals drift", "rounding at period end", project, session))
            .unwrap();
        store
            .record_entities(&session.to_string(), &project.to_string(), &["src/billing.rs".to_string()])
            .unwrap();

        let hits =
            store.search(&project.to_string(), "src/billing.rs", None, 10, Recall::Fused).unwrap();
        assert_eq!(hits.len(), 1, "the entity stream should surface the session's work");
        assert!(hits[0].title.contains("Quarterly"));
    }

    #[test]
    fn a_graph_neighbour_of_a_keyword_hit_joins_the_results() {
        // Two sessions touched the same entity. The query matches only the
        // first session's words; the second is its neighbour through the
        // shared entity, and that is how it earns a place in the results.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let seed = Uuid::new_v4();
        let neighbour = Uuid::new_v4();
        store
            .index(&event_in_session("Fixed the vermilion middleware", "expiry check", project, seed))
            .unwrap();
        store
            .index(&event_in_session("Connection pool tuning", "idle timeout raised", project, neighbour))
            .unwrap();
        for session in [seed, neighbour] {
            store
                .record_entities(&session.to_string(), &project.to_string(), &["src/gate.rs".to_string()])
                .unwrap();
        }

        let hits = store.search(&project.to_string(), "vermilion", None, 10, Recall::Fused).unwrap();
        let titles: Vec<&str> = hits.iter().map(|hit| hit.title.as_str()).collect();
        assert!(titles.iter().any(|t| t.contains("vermilion")), "keyword hit lost: {titles:?}");
        assert!(
            titles.iter().any(|t| t.contains("Connection pool")),
            "the neighbour through the shared entity never surfaced: {titles:?}"
        );
    }

    #[test]
    fn a_forgotten_event_never_returns_through_the_entity_stream() {
        // brain_forget's contract: forgotten means recall stops returning it,
        // through EVERY stream. The entity trail must not resurrect it.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let session = Uuid::new_v4();
        let target = event_in_session("Quarterly totals drift", "rounding at period end", project, session);
        let target_id = target.id.clone();
        store.index(&target).unwrap();
        store
            .record_entities(&session.to_string(), &project.to_string(), &["src/billing.rs".to_string()])
            .unwrap();

        let mut tombstone = event_in_session("forgotten", "", project, session);
        tombstone.kind = EventKind::Tombstone;
        tombstone.links = vec![target_id];
        store.index(&tombstone).unwrap();

        let hits =
            store.search(&project.to_string(), "src/billing.rs", None, 10, Recall::Fused).unwrap();
        assert!(hits.is_empty(), "a forgotten event resurfaced through its entity: {hits:?}");
    }

    #[test]
    fn a_demoted_entry_sorts_last_after_multi_stream_fusion() {
        // Both events reach the results through the entity stream; the one a
        // human flagged stale must not represent the search.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let stale = Uuid::new_v4();
        let fresh = Uuid::new_v4();
        let stale_event = event_in_session("Old approach", "superseded", project, stale);
        let stale_id = stale_event.id.clone();
        store.index(&stale_event).unwrap();
        store.index(&event_in_session("Current approach", "in force", project, fresh)).unwrap();
        for session in [stale, fresh] {
            store
                .record_entities(&session.to_string(), &project.to_string(), &["src/billing.rs".to_string()])
                .unwrap();
        }
        store.lower_confidence(&stale_id).unwrap();

        let hits =
            store.search(&project.to_string(), "src/billing.rs", None, 10, Recall::Fused).unwrap();
        assert_eq!(hits.len(), 2, "both sessions should still be found: {hits:?}");
        assert_eq!(hits.last().unwrap().title, "Old approach", "the stale entry led the results");
    }

    #[test]
    fn a_lexical_recall_stays_words_only() {
        // Recall::Lexical is an explicit "words only" request; no other
        // stream may add to it.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let session = Uuid::new_v4();
        store
            .index(&event_in_session("Quarterly totals drift", "rounding at period end", project, session))
            .unwrap();
        store
            .record_entities(&session.to_string(), &project.to_string(), &["src/billing.rs".to_string()])
            .unwrap();
        let hits =
            store.search(&project.to_string(), "src/billing.rs", None, 10, Recall::Lexical).unwrap();
        assert!(hits.is_empty(), "lexical recall used a non-lexical stream: {hits:?}");
    }

    #[test]
    fn search_is_scoped_to_one_project() {
        let store = Store::open_memory().unwrap();
        let mine = Uuid::new_v4();
        let theirs = Uuid::new_v4();
        store.index(&event("shared word here", "body", mine)).unwrap();
        store.index(&event("shared word here", "body", theirs)).unwrap();
        assert_eq!(store.search(&mine.to_string(), "shared", None, 10, Recall::Fused).unwrap().len(), 1);
    }

    #[test]
    fn indexing_the_same_event_twice_is_idempotent() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let event = event("once", "body", project);
        store.index(&event).unwrap();
        store.index(&event).unwrap();
        assert_eq!(store.count().unwrap(), 1);
        assert_eq!(store.search(&project.to_string(), "once", None, 10, Recall::Fused).unwrap().len(), 1);
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
        assert!(store.search(&project.to_string(), "gone", None, 10, Recall::Fused).unwrap().is_empty());
    }

    #[test]
    fn work_in_flight_is_counted_apart_from_what_a_summary_said() {
        // Two pointers land in one session: one was still-unsummarized work,
        // one came from the ranked list. The agent pulls only the second.
        // Overall uptake reads 1 of 2 and says nothing useful; split, it says
        // the reserve was ignored - which is the number that decides how many
        // lines the reserve should hold.
        let store = Store::open_memory().unwrap();
        store
            .record_injected("s1", &["01FLIGHT".into(), "01SUMMARY".into()], 1, 40)
            .unwrap();
        store.record_recalled("s1", ["01SUMMARY"].into_iter()).unwrap();

        assert_eq!(store.injection_uptake().unwrap(), (2, 1));
        assert_eq!(
            store.in_flight_uptake().unwrap(),
            (1, 0),
            "the in-flight pointer was not counted apart"
        );

        // Pulled later, it moves - and re-injecting it from the ranked list
        // must not quietly reclassify what the agent was originally handed.
        store.record_recalled("s1", ["01FLIGHT"].into_iter()).unwrap();
        store.record_injected("s1", &["01FLIGHT".into()], 0, 20).unwrap();
        assert_eq!(store.in_flight_uptake().unwrap(), (1, 1));
    }

    #[test]
    fn a_compaction_reset_does_not_erase_the_uptake_history() {
        let store = Store::open_memory().unwrap();
        let session = "sess-1";

        // A pointer was pushed, and the agent went on to read it in full.
        store.record_injected(session, &["01AAA".to_string()], 0, 100).unwrap();
        store.record_recalled(session, std::iter::once("01AAA")).unwrap();
        assert_eq!(store.injection_uptake().unwrap(), (1, 1));

        // The context is wiped by a compaction. The de-dup guard must forget,
        // but the measurement must not: the pull already happened.
        store.reset_injection_state(session).unwrap();
        assert!(!store.already_injected(session, "01AAA").unwrap());
        assert_eq!(store.injection_uptake().unwrap(), (1, 1));

        // Re-injecting the same pointer after the reset re-arms the guard
        // without double-counting the push.
        store.record_injected(session, &["01AAA".to_string()], 0, 100).unwrap();
        assert!(store.already_injected(session, "01AAA").unwrap());
        assert_eq!(store.injection_uptake().unwrap(), (1, 1));
    }
}
