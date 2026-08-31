//! The versioned observation log — the source of truth.
//!
//! Everything else in this program is derived state that `brain reindex` can
//! rebuild from these files. The schema is a contract: readers must ignore
//! unknown fields so a newer writer never breaks an older
//! reader, and `v` only bumps on a genuinely breaking change.
//!
//! Layout: `wiki/<workspace>/<project>/events/YYYY-MM.jsonl`, one JSON object
//! per line, append-only, fsynced per event.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ulid::Ulid;
use uuid::Uuid;

/// Current schema version. Bump only on a breaking change.
pub const SCHEMA_VERSION: u32 = 1;

/// What an event represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// A captured lifecycle observation.
    Observation,
    /// A consolidated summary of one session.
    SessionSummary,
    /// A wiki page was written or rewritten.
    PageUpdate,
    /// An explicit note from the user or agent.
    Note,
    /// Something that stayed true across sessions, synthesized from their
    /// summaries. Session summaries are episodic - what happened, once. This
    /// is the semantic half, and it outlives the sessions it came from.
    Knowledge,
    /// A deletion. Append-only holds: removal is an event, and compaction
    /// honours it. Required for the future sync contract.
    Tombstone,
    /// A retirement: the named events keep their identity - id, timestamp,
    /// title, topic, files, links - and lose their bodies from the index.
    /// The log keeps everything; replaying reproduces the drop.
    Retire,
}

impl EventKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observation => "observation",
            Self::SessionSummary => "session_summary",
            Self::PageUpdate => "page_update",
            Self::Note => "note",
            Self::Knowledge => "knowledge",
            Self::Tombstone => "tombstone",
            Self::Retire => "retire",
        }
    }
}

/// What a remembered thing is *about*.
///
/// Separate axis from [`EventKind`], which says what an event *is* (a capture,
/// a summary, a note). This says what it is about, and it is what makes an
/// index scannable: a reader skimming forty one-liners can find the decisions
/// without reading the config noise.
///
/// Deliberately short. Six kinds a cheap model can tell apart beats a dozen it
/// guesses at, and an unknown value is dropped rather than forced into the
/// nearest bucket.
pub const TOPICS: &[&str] = &["decision", "bugfix", "feature", "discovery", "config", "test"];

/// Map a model-supplied topic onto the taxonomy, or `None`.
///
/// Lenient by the same rule as the rest of answer parsing: models improvise
/// (`fix`, `bug`, `feat`), and a recognizable near-miss is worth keeping. What
/// is NOT recognizable is dropped — an unknown topic must never be coerced
/// into a wrong one, because a mislabeled decision is worse than an unlabeled
/// one.
#[must_use]
pub fn normalize_topic(raw: &str) -> Option<&'static str> {
    let cleaned = raw.trim().to_ascii_lowercase();
    let cleaned = cleaned.trim_matches(|c: char| !c.is_ascii_alphabetic());
    match cleaned {
        "decision" | "decide" | "choice" | "rationale" => Some("decision"),
        "bugfix" | "bug" | "fix" | "bugfixes" | "defect" => Some("bugfix"),
        "feature" | "feat" | "implementation" | "enhancement" => Some("feature"),
        "discovery" | "finding" | "gotcha" | "insight" | "learning" => Some("discovery"),
        "config" | "configuration" | "setup" | "chore" | "infra" => Some("config"),
        "test" | "tests" | "testing" => Some("test"),
        _ => None,
    }
}

/// Which CLI and which lifecycle hook produced an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// CLI identifier, e.g. `claude-code`.
    pub cli: String,
    /// Lifecycle hook in snake_case, e.g. `post_tool_use`.
    pub hook: String,
}

/// Is this hook the moment a human typed something at the agent?
///
/// Every CLI spells it differently and two do not have it at all:
/// `UserPromptSubmit` (Claude Code, Codex), `beforeSubmitPrompt` (Cursor),
/// `BeforeAgent` (Gemini) - normalized spellings here. Antigravity and
/// OpenCode expose no prompt event, so nothing to match. Matching only one
/// spelling quietly made every human-typed-is-signal rule Claude-only.
#[must_use]
pub fn is_user_prompt(hook: &str) -> bool {
    matches!(hook, "user_prompt_submit" | "before_submit_prompt" | "before_agent")
}

/// Does this text carry an explicit order to remember?
///
/// "จำไว้ว่าให้ใช้ rtk เสมอ" is the strongest signal any prompt can carry,
/// and it costs no model call to spot. Deliberately conservative, in both
/// languages: a phrase must read as an instruction ABOUT remembering or a
/// standing order, not merely contain "always" - "why does it always
/// crash" is a question, not a rule. A false negative costs nothing
/// (consolidation still reads every prompt); a false positive floats one
/// event it should not.
#[must_use]
pub fn carries_memory_intent(text: &str) -> bool {
    let lowered = text.to_lowercase();
    const PHRASES: &[&str] = &[
        "remember that",
        "remember this",
        "don't forget",
        "from now on",
        "always use",
        "never use",
        "make it a rule",
        "\u{e08}\u{e33}\u{e44}\u{e27}\u{e49}",           // จำไว้
        "\u{e2d}\u{e22}\u{e48}\u{e32}\u{e25}\u{e37}\u{e21}", // อย่าลืม
        "\u{e2b}\u{e49}\u{e32}\u{e21}\u{e43}\u{e0a}\u{e49}", // ห้ามใช้
        "\u{e15}\u{e49}\u{e2d}\u{e07}\u{e43}\u{e0a}\u{e49}", // ต้องใช้
        "\u{e17}\u{e38}\u{e01}\u{e04}\u{e23}\u{e31}\u{e49}\u{e07}\u{e17}\u{e35}\u{e48}", // ทุกครั้งที่
        "\u{e15}\u{e48}\u{e2d}\u{e44}\u{e1b}\u{e19}\u{e35}\u{e49}\u{e43}\u{e2b}\u{e49}", // ต่อไปนี้ให้
    ];
    PHRASES.iter().any(|phrase| lowered.contains(phrase))
}

/// One line of the log.
///
/// `#[serde(default)]` on the optional fields plus `flatten`ed `extra` is what
/// makes the format forward-compatible: a v1 reader handed a v2 line keeps the
/// fields it knows and carries the rest through untouched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Schema version.
    pub v: u32,
    /// ULID: time-ordered, globally unique, safe to merge across machines.
    pub id: String,
    /// RFC 3339 UTC timestamp.
    pub ts: String,
    pub workspace: Uuid,
    pub project: Uuid,
    pub session: Uuid,
    pub source: Source,
    pub kind: EventKind,
    /// One line. Rule-based at capture; consolidation may rewrite it later
    /// through a `page_update`, never by mutating this line.
    pub title: String,
    /// Sanitized content. The sanitizer runs before this struct is built.
    #[serde(default)]
    pub body: String,
    /// Repo-relative paths this event touched.
    #[serde(default)]
    pub files: Vec<String>,
    /// Related event or page ids.
    #[serde(default)]
    pub links: Vec<String>,
    /// What this is about, from [`TOPICS`]. Absent when nothing classified it.
    ///
    /// Additive: a reader that predates this field ignores it, and one that
    /// postdates a line without it sees `None`. The schema version does not
    /// move for a new optional field — only for a change in what an existing
    /// field means.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    /// Which store appended this line. Stamped at append time, absent on
    /// events written before the field existed. Provenance for a future
    /// multi-store merge: which replica wrote what cannot be reconstructed
    /// after the fact, so it is recorded now, needed or not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Local processing state — deliberately NOT part of the sync contract.
    #[serde(default)]
    pub consolidated: bool,
    /// Unknown fields from a newer writer, preserved verbatim on read.
    #[serde(flatten, default)]
    pub extra: serde_json::Map<String, Value>,
}

impl Event {
    /// Build a new event with a fresh ULID and the current time.
    #[must_use]
    pub fn new(
        workspace: Uuid,
        project: Uuid,
        session: Uuid,
        source: Source,
        kind: EventKind,
        title: String,
        body: String,
    ) -> Self {
        Self {
            v: SCHEMA_VERSION,
            id: Ulid::new().to_string(),
            ts: jiff::Timestamp::now().to_string(),
            workspace,
            project,
            session,
            source,
            kind,
            title,
            body,
            files: Vec::new(),
            links: Vec::new(),
            topic: None,
            origin: None,
            consolidated: false,
            extra: serde_json::Map::new(),
        }
    }

    /// Is this event ABOUT another event - a tombstone, a retirement, a
    /// correction or a flag? Replay applies these after everything they
    /// could target, because cross-machine clocks make id order unreliable
    /// between writers.
    #[must_use]
    pub fn is_revision(&self) -> bool {
        matches!(self.kind, EventKind::Tombstone | EventKind::Retire)
            || (self.kind == EventKind::Note
                && matches!(self.source.hook.as_str(), "correct" | "feedback" | "supersede"))
    }

    /// Month bucket this event belongs to, from its own timestamp.
    #[must_use]
    pub fn month(&self) -> String {
        // ULID and RFC 3339 both sort lexicographically; the first 7 chars of
        // an RFC 3339 timestamp are exactly `YYYY-MM`.
        self.ts.get(..7).unwrap_or("unknown").to_string()
    }
}

/// Append-only event log rooted at one project directory.
pub struct EventLog {
    events_dir: PathBuf,
}

impl EventLog {
    /// Open (creating if needed) the log under a project directory.
    ///
    /// # Errors
    /// Returns an error when the events directory cannot be created.
    pub fn open(project_dir: &Path) -> Result<Self> {
        let events_dir = project_dir.join("events");
        fs::create_dir_all(&events_dir)
            .with_context(|| format!("create event log directory {}", events_dir.display()))?;
        Ok(Self { events_dir })
    }

    /// Path of the monthly file an event belongs in.
    #[must_use]
    pub fn file_for(&self, month: &str) -> PathBuf {
        self.events_dir.join(format!("{month}.jsonl"))
    }

    /// Append one event and fsync it.
    ///
    /// A single `write` of one line opened with `O_APPEND` is atomic against
    /// concurrent appenders for the sizes we write, which is what lets several
    /// CLIs capture into one project at once without a lock. The fsync is the
    /// durability half: a crash immediately after this call must not lose the
    /// event, because nothing else stores it.
    ///
    /// # Errors
    /// Returns an error when the file cannot be opened, written, or synced.
    pub fn append(&self, event: &Event) -> Result<()> {
        let path = self.file_for(&event.month());
        // Stamp which store wrote this line, unless the caller already knows
        // better (an import replaying another machine's events must keep
        // their origin, not claim it).
        let mut event = event.clone();
        if event.origin.is_none() {
            event.origin = crate::ids::origin();
        }
        let mut line = serde_json::to_string(&event).context("serialize event")?;
        line.push('\n');

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open event log {}", path.display()))?;
        file.write_all(line.as_bytes())
            .with_context(|| format!("append to event log {}", path.display()))?;
        file.sync_data()
            .with_context(|| format!("fsync event log {}", path.display()))?;
        Ok(())
    }

    /// Every monthly log file, oldest first.
    ///
    /// # Errors
    /// Returns an error when the events directory cannot be read.
    pub fn files(&self) -> Result<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = fs::read_dir(&self.events_dir)
            .with_context(|| format!("read {}", self.events_dir.display()))?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        files.sort();
        Ok(files)
    }

    /// Read every event in order.
    ///
    /// A malformed line is skipped rather than aborting the read: one corrupt
    /// line from an interrupted write must not make the rest of the log
    /// unreadable. The count of skipped lines is returned alongside.
    ///
    /// # Errors
    /// Returns an error when a log file cannot be read.
    pub fn read_all(&self) -> Result<(Vec<Event>, usize)> {
        let mut events = Vec::new();
        let mut skipped = 0usize;
        for path in self.files()? {
            let text = fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Event>(line) {
                    Ok(event) => events.push(event),
                    Err(_) => skipped += 1,
                }
            }
        }
        // ULID order is causal order on one machine, and only there. A
        // clock running ahead on another machine can hand a correction an
        // id SMALLER than the event it corrects; replayed in pure id order
        // that correction runs before its target exists, its UPDATE lands
        // on no row, and the revision silently evaporates on the very
        // rebuild that was supposed to reproduce it. So: everything a
        // revision could target replays first, revisions after, id order
        // within each half (two corrections of one event stay last-wins).
        // Complete in two passes because revisions never target revisions -
        // correction and feedback notes are excluded from every surface, so
        // nothing can cite one.
        events.sort_by(|a, b| {
            (a.is_revision(), &a.id).cmp(&(b.is_revision(), &b.id))
        });
        Ok((events, skipped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_correction_from_a_fast_clock_still_lands_on_its_target() {
        // Another machine's clock running ahead gives a correction an id
        // SMALLER than the event it corrects. Pure id order would replay
        // it first, into nothing.
        let dir = std::env::temp_dir().join(format!("brain-skew-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log = EventLog::open(&dir).unwrap();

        let mut target = sample();
        target.id = "01TESTTARGET000000000000ZZ".to_string();
        let mut correction = sample();
        correction.id = "01TESTCORRECTION00000000AA".to_string();
        correction.kind = EventKind::Note;
        correction.source.hook = "correct".to_string();
        correction.links = vec![target.id.clone()];

        log.append(&correction).unwrap();
        log.append(&target).unwrap();
        let (events, skipped) = log.read_all().unwrap();
        assert_eq!(skipped, 0);
        assert_eq!(
            events.iter().map(|event| event.id.as_str()).collect::<Vec<_>>(),
            vec![target.id.as_str(), correction.id.as_str()],
            "a revision must replay after anything it could target"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_order_to_remember_is_heard_and_a_question_is_not() {
        for yes in [
            "Remember that we deploy on Fridays only",
            "from now on, always use rtk grep",
            "\u{e08}\u{e33}\u{e44}\u{e27}\u{e49}\u{e27}\u{e48}\u{e32}\u{e43}\u{e2b}\u{e49}\u{e43}\u{e0a}\u{e49} rtk", // จำไว้ว่าให้ใช้ rtk
            "\u{e2d}\u{e22}\u{e48}\u{e32}\u{e25}\u{e37}\u{e21} run lint", // อย่าลืม run lint
        ] {
            assert!(carries_memory_intent(yes), "missed an order: {yes}");
        }
        for no in [
            "why does it always crash on startup",
            "I never understood this part of the code",
            "fix the login bug",
        ] {
            assert!(!carries_memory_intent(no), "a question read as an order: {no}");
        }
    }

    #[test]
    fn a_supersession_is_a_revision_for_replay_purposes() {
        // It rewrites another event, so a fast clock elsewhere must not be
        // able to replay it into nothing.
        let mut note = sample();
        note.kind = EventKind::Note;
        note.source.hook = "supersede".to_string();
        assert!(note.is_revision());
    }

    fn sample() -> Event {
        Event::new(
            Uuid::nil(),
            Uuid::nil(),
            Uuid::nil(),
            Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
            EventKind::Observation,
            "Edited src/main.rs".into(),
            "body".into(),
        )
    }

    #[test]
    fn append_then_read_round_trips() {
        let dir = std::env::temp_dir().join(format!("brain-test-{}", Ulid::new()));
        let log = EventLog::open(&dir).unwrap();
        let event = sample();
        log.append(&event).unwrap();
        let (events, skipped) = log.read_all().unwrap();
        assert_eq!(skipped, 0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, event.id);
        assert_eq!(events[0].v, SCHEMA_VERSION);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_line_is_skipped_not_fatal() {
        let dir = std::env::temp_dir().join(format!("brain-test-{}", Ulid::new()));
        let log = EventLog::open(&dir).unwrap();
        log.append(&sample()).unwrap();
        let path = log.file_for(&sample().month());
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{ this is not json\n").unwrap();
        let (events, skipped) = log.read_all().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(skipped, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn topics_normalize_from_what_models_actually_write() {
        assert_eq!(normalize_topic("decision"), Some("decision"));
        assert_eq!(normalize_topic("  Fix "), Some("bugfix"));
        assert_eq!(normalize_topic("FEAT"), Some("feature"));
        assert_eq!(normalize_topic("\"gotcha\""), Some("discovery"));
        assert_eq!(normalize_topic("chore"), Some("config"));
    }

    #[test]
    fn an_unrecognized_topic_is_dropped_never_guessed() {
        // A mislabeled decision is worse than an unlabeled one.
        assert_eq!(normalize_topic("refactor"), None);
        assert_eq!(normalize_topic(""), None);
        assert_eq!(normalize_topic("misc"), None);
    }

    #[test]
    fn every_normalized_topic_is_in_the_taxonomy() {
        for candidate in ["decision", "fix", "feat", "gotcha", "chore", "tests"] {
            let topic = normalize_topic(candidate).unwrap();
            assert!(TOPICS.contains(&topic), "{topic} is not in TOPICS");
        }
    }

    #[test]
    fn a_line_without_a_topic_stays_byte_identical() {
        // The field is additive: it must not appear at all when unset, or every
        // existing log line would look changed to a differ.
        let event = sample();
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("topic"), "absent topic must not be serialized");
    }

    #[test]
    fn unknown_fields_survive_a_round_trip() {
        let line = r#"{"v":2,"id":"01J","ts":"2026-08-22T00:00:00Z","workspace":"00000000-0000-0000-0000-000000000000","project":"00000000-0000-0000-0000-000000000000","session":"00000000-0000-0000-0000-000000000000","source":{"cli":"codex","hook":"stop"},"kind":"observation","title":"t","future_field":"kept"}"#;
        let event: Event = serde_json::from_str(line).unwrap();
        assert_eq!(event.v, 2);
        assert_eq!(event.extra.get("future_field").unwrap(), "kept");
        let back = serde_json::to_string(&event).unwrap();
        assert!(back.contains("future_field"));
    }
}
