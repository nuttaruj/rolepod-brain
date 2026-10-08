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
/// Bytes the write-ahead log is cut back to after a checkpoint.
const WAL_SIZE_LIMIT: i64 = 64 * 1024 * 1024;

/// Hooks whose events carry a tool call or a lifecycle marker, not prose.
/// Claude Code, Gemini and OpenCode spell the same hooks differently, and
/// each stored spelling is listed (`normalize_hook` output).
const PAYLOAD_HOOKS: &str = "'post_tool_use', 'pre_tool_use', 'session_start', 'session_end', \
     'pre_compact', 'after_tool', 'pre_compress', 'tool_execute_after'";

/// SQL for "this event is prose the embedder keeps", over the `events` table
/// aliased `alias`. A delegate's footsteps are skipped; its `subagent_stop`
/// report is prose and stays, matching `consolidate::is_delegate_footstep`.
fn embeddable(alias: &str) -> String {
    format!(
        "{alias}.hook NOT IN ({PAYLOAD_HOOKS}) AND ({alias}.agent IS NULL OR {alias}.hook = 'subagent_stop')"
    )
}

/// A `WITH` clause naming each project in `events` once, as
/// `projects(project)`, found by skipping through an index from one name to
/// the next rather than by reading rows. A statement that joins it to
/// `events e ON e.project = projects.project AND e.consolidated = 0` reads
/// only pending rows, through `events_unconsolidated`: a few reads per
/// project on a settled store, where any other path reads every row.
const PROJECTS_CTE: &str = "WITH RECURSIVE projects(project) AS (
     SELECT MIN(project) FROM events
     UNION ALL
     SELECT (SELECT MIN(project) FROM events WHERE project > projects.project)
     FROM projects WHERE projects.project IS NOT NULL)";

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

/// What each stream's opinion is worth, in the order `search` fuses them:
/// keyword, semantic, entity, substring, graph, keyword_relaxed.
///
/// Not equal, because they do not answer the same question. Keyword,
/// semantic and substring rank EVENTS against the query. Entity and graph
/// rank SESSIONS - entities are recorded per session, so both nominate every
/// event of any session that touched a matching name, and a session that
/// touched a thousand names is nominated by nearly every query.
///
/// Equal weight let that arithmetic win. RRF pays 1/(K+rank), so at K=60 an
/// entry sitting tenth in three lists (3 x 1/70 = 0.043) beats one sitting
/// FIRST in a single list (1/61 = 0.016) - and the streams that agree most
/// readily are the two that nominate whole sessions. Measured on a 22k-event
/// brain: entries a reranker judged correct sat at a median rank of 4 in
/// their best stream and rank 10 after fusion; 20 of 28 were pushed DOWN by
/// being fused. One entry took #1 for 6 of 17 queries.
///
/// At half weight, those two streams still widen recall - fewer correct
/// entries fall out of the pool than before, not more - without deciding the
/// order. The same 17 queries then returned 17 different first results, and
/// the worst repeat was 1.
///
/// `keyword_relaxed` stays at 1.0: like keyword it ranks EVENTS, and it
/// exists to let in partial matches the AND query cannot, so halving it
/// would bury the very entries it adds. It is a superset of keyword, so an
/// entry both find collects two close contributions; if that proves too
/// strong, drop ids already in `keyword` from it rather than lowering this.
const STREAM_WEIGHTS: [f32; 6] = [1.0, 1.0, 0.5, 1.0, 0.5, 1.0];
/// The same order, named - what `Trace` calls each stream.
const STREAM_NAMES: [&str; 6] =
    ["keyword", "semantic", "entity", "substring", "graph", "keyword_relaxed"];
/// Each stream's position in `STREAM_NAMES` and in the `lists` array
/// `search_traced` fuses; `fuse` records which streams found an id as one bit
/// per position. A test pins these to the names.
const KEYWORD: usize = 0;
const SEMANTIC: usize = 1;
const ENTITY: usize = 2;
const SUBSTRING: usize = 3;
const GRAPH: usize = 4;
const RELAXED: usize = 5;

/// The `seen_in` bit of one stream.
const fn bit(stream: usize) -> u8 {
    1 << stream
}

/// English words that say what kind of question this is, not what it is
/// about. Memmy's 22 plus the interrogatives and auxiliaries a natural
/// question is mostly made of.
const RELAXED_STOP_WORDS: [&str; 34] = [
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "with", "you", "your", "how", "why", "what", "which", "does", "did", "do", "get",
    "was", "were", "about", "not",
];

/// Wall clock a prompt-time lookup may take before it is interrupted.
const PROMPT_POINTER_DEADLINE: std::time::Duration = std::time::Duration::from_millis(150);

/// Wall clock `related` may take before it is interrupted. It answers an
/// agent that is waiting, and the old join shape ran over 100 s on the
/// 115k-event project; the session-grouped shape does far less work, so this
/// caps a bad plan rather than a normal run.
const RELATED_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Shared names are counted once per session, before events are touched:
/// entities are recorded per session, so every event in it shares the same
/// set, and a per-event join multiplies a session's events by its names
/// (see `neighbours_of`). Unlike there, the seed's own session stays in - it
/// is a neighbour here - and only the seed event itself is left out.
///
/// Like `neighbours_sql`, the events are ranked from the columns
/// `events_recall` holds and their rows are read by id once they are down to
/// the limit: fetching every event of the 45 sharing sessions took 3.4 s cold
/// on the 115k-event project, counting them in the index 0.12 s. The seed is
/// looked up by primary key, with `project` only a check.
const RELATED_SQL: &str = "WITH seed_sessions AS (
             SELECT DISTINCT session FROM events
             WHERE id = ?1 AND +project = ?2
         ),
         subject AS (
             SELECT DISTINCT n.name FROM entities n
             WHERE n.session IN (SELECT session FROM seed_sessions)
                   AND n.project = ?2
         ),
         shared AS (
             SELECT n.session, COUNT(DISTINCT n.name) AS shared
             FROM entities n
             WHERE n.project = ?2 AND n.name IN (SELECT name FROM subject)
             GROUP BY n.session
         )
         SELECT h.id, h.ts, h.cli, h.kind, h.title,
                substr(COALESCE(h.body, ''), 1, 160), h.session,
                c.shared
         FROM (SELECT e.id, s.shared,
                      CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END AS demoted,
                      CASE e.kind
                          WHEN 'knowledge' THEN 0
                          WHEN 'session_summary' THEN 1
                          WHEN 'source' THEN 1
                          ELSE 2
                      END AS authority
               FROM events e
               JOIN shared s ON s.session = e.session
               WHERE e.project = ?2 AND e.id != ?1
                     AND e.forgotten = 0 AND e.kind NOT IN ('tombstone', 'retire')
                     AND e.hook NOT IN ('correct', 'feedback', 'supersede')
               ORDER BY s.shared DESC, demoted, authority, e.id DESC
               LIMIT ?3) c
         JOIN events h ON h.id = c.id
         ORDER BY c.shared DESC, c.demoted, c.authority, c.id DESC";

/// The statement behind [`Store::nearest`], with the same recall floor every
/// other read enforces. A memory withdrawn from search has to be withdrawn
/// from this one too, or the withdrawal was cosmetic.
const NEAREST_SQL: &str = "SELECT v.event_id, v.vec, e.confidence
         FROM event_vec v
         JOIN events e ON e.id = v.event_id
         WHERE e.project = ?1 AND e.forgotten = 0 AND e.kind NOT IN ('tombstone', 'retire')
               AND e.hook NOT IN ('correct', 'feedback', 'supersede')
               AND (?2 IS NULL OR e.topic = ?2)";

/// The keyword stream of [`Store::search_traced`].
///
/// Two stages, because relevance alone lets one session own the page. The
/// window takes each session's best few first, so the pool handed to
/// `spread_across_sessions` contains the quiet sessions at all - reading a
/// flat top-N never reaches them when one session holds most of the project.
///
/// The demotion is inside the window too: an entry a human called stale must
/// not be the hit that represents its session. Flagging has to change what
/// the user SEES, or it is a counter nobody can observe.
///
/// The window ranks by rowid, and title, snippet and the rest are read only
/// for the rows that make the pool. `snippet()` opens each match's body, and a
/// query whose terms are common matches thousands of rows: 4,450 of them took
/// 1.5 s cold on the 115k-event project. `pos` carries the pool's own order to
/// the final read, numbered by an explicit `ORDER BY` so it does not lean on
/// the order a subquery happens to emit.
const KEYWORD_SQL: &str = "WITH matched AS (
         SELECT e.rowid AS rid, e.session,
                CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END AS demoted,
                rank AS relevance
         FROM events_fts
         JOIN events e ON e.rowid = events_fts.rowid
         WHERE events_fts MATCH ?1 AND e.project = ?2 AND e.forgotten = 0
               AND e.kind NOT IN ('tombstone', 'retire')
               AND e.hook NOT IN ('correct', 'feedback', 'supersede')
               AND (?5 IS NULL OR e.topic = ?5)
     ),
     pool AS (
         SELECT rid, ROW_NUMBER() OVER (ORDER BY demoted, relevance) AS pos FROM (
             SELECT rid, demoted, relevance FROM (
                 SELECT *, ROW_NUMBER() OVER (
                     PARTITION BY session ORDER BY demoted, relevance
                 ) AS per_session FROM matched
             )
             WHERE per_session <= ?3
             ORDER BY demoted, relevance
             LIMIT ?4
         )
     )
     SELECT e.id, e.ts, e.cli, e.kind, e.title,
            snippet(events_fts, 1, '[', ']', ' … ', 24), e.session
     FROM events_fts
     JOIN events e ON e.rowid = events_fts.rowid
     JOIN pool ON pool.rid = events_fts.rowid
     WHERE events_fts MATCH ?1
     ORDER BY pool.pos";

/// How much of a prompt becomes the query.
const PROMPT_QUERY_CHARS: usize = 400;

/// Terms a relaxed query keeps. Its groups are always three terms wide: the
/// nested loops in `compile_fts` and the `3` in its guard and trace note
/// are that width.
const RELAXED_TERMS: usize = 5;

/// The first read of a prompt-time lookup, with its phrase fallback.
///
/// A query FTS5 cannot parse is read as a phrase instead, as `search_traced`
/// does. An interrupt is not a parse error: it propagates, so a lookup that
/// ran out of time never goes on to run more reads past its deadline.
fn raw_or_phrase(
    first: rusqlite::Result<Vec<Pointer>>,
    phrase: impl FnOnce() -> rusqlite::Result<Vec<Pointer>>,
) -> Result<Vec<Pointer>> {
    match first {
        Ok(found) => Ok(found),
        Err(rusqlite::Error::SqliteFailure(failure, _))
            if failure.code == rusqlite::ErrorCode::OperationInterrupted =>
        {
            Err(anyhow::anyhow!("prompt lookup interrupted at its deadline"))
        }
        Err(_) => phrase().context("read prompt pointers"),
    }
}

/// A prompt as `compile_fts` can relax it: a prompt's quotes, `*` and
/// capitalised `OR`/`AND`/`NOT`/`NEAR` are prose (quoted error text, markdown
/// bold), not FTS5 syntax, so they must not switch the relaxed read off.
fn relaxable(prompt: &str) -> String {
    prompt
        .replace(['"', '*'], " ")
        .split_whitespace()
        .map(|word| {
            if matches!(word, "OR" | "AND" | "NOT") || word.starts_with("NEAR") {
                word.to_lowercase()
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A natural-language question as an FTS5 query that needs only SOME of its
/// words: an OR of every three-term AND group over its (at most five)
/// longest terms.
///
/// The raw query is an implicit AND, so a question with one word the answer
/// never used finds nothing. `how does retry backoff jitter get capped` should
/// still reach the note about retry, backoff and jitter. Terms are quoted, so
/// a `/` or `.` inside a path stays one phrase. Thai goes through untouched:
/// the index's own tokenizer cuts it, exactly as it does for the raw query.
///
/// `None` when the query already speaks FTS5 (a quote, `*`, `OR`/`AND`/`NOT`/
/// `NEAR`) - the person meant it - or when fewer than three terms remain,
/// where relaxing would be the raw query again.
fn compile_fts(query: &str) -> Option<String> {
    if query.contains('"') || query.contains('*') {
        return None;
    }
    if query
        .split_whitespace()
        .any(|word| matches!(word, "OR" | "AND" | "NOT") || word.starts_with("NEAR"))
    {
        return None;
    }
    let edge = |c: char| c.is_ascii_punctuation() || matches!(c, '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}');
    let mut terms: Vec<String> = Vec::new();
    for word in query.split_whitespace() {
        let word = word.trim_matches(edge);
        let pieces: Vec<&str> = if word.contains('/') || word.contains('.') {
            vec![word]
        } else {
            word.split(|c: char| edge(c) && c != '_' && c != '-').collect()
        };
        for piece in pieces {
            let piece = piece.trim_matches(|c: char| c == '-' || c == '_');
            if piece.chars().count() < 3 || RELAXED_STOP_WORDS.contains(&piece.to_lowercase().as_str()) {
                continue;
            }
            if !terms.iter().any(|seen| seen.eq_ignore_ascii_case(piece)) {
                terms.push(piece.to_string());
            }
        }
    }
    if terms.len() > RELAXED_TERMS {
        // Longest first, earlier wins a tie; then back to the order asked.
        let mut order: Vec<usize> = (0..terms.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(terms[i].chars().count()));
        order.truncate(RELAXED_TERMS);
        order.sort_unstable();
        terms = order.into_iter().map(|i| terms[i].clone()).collect();
    }
    if terms.len() < 3 {
        return None;
    }
    let mut groups: Vec<String> = Vec::new();
    for a in 0..terms.len() {
        for b in a + 1..terms.len() {
            for c in b + 1..terms.len() {
                groups.push(format!("(\"{}\" \"{}\" \"{}\")", terms[a], terms[b], terms[c]));
            }
        }
    }
    Some(groups.join(" OR "))
}

/// Length past which an entry starts paying for being long, in bytes of
/// title plus body. The corpus median, measured: 212 on a 22k-event brain.
const LENGTH_PIVOT: f32 = 200.0;

/// How steeply an over-long entry is discounted.
///
/// A long entry is not a worse answer for being long - it is a worse answer
/// because of what length does to every stream that ranks it. An embedding
/// of a thousand words is the centroid of everything it covers, which sits
/// near any query and answers none of them; BM25 sees more terms and scores
/// more matches. Both effects reward breadth over aboutness.
///
/// Measured on the same brain: entries a reranker judged correct had a
/// median length of 82 bytes, while the entries occupying the top five that
/// it did NOT pick ran to 815. The distilled `knowledge` note that answers
/// the question was losing to the session summary that mentions it in
/// passing.
///
/// `score / (1 + 0.6 * ln(len / 200))`, floored so nothing shorter than the
/// pivot is touched. Logarithmic because the difference between 200 and 2000
/// bytes matters and the difference between 20k and 22k does not. Mean rank
/// of a reranker's picks fell from 11.67 to 7.40 with no entry lost from the
/// pool; 0.3 and 1.0 both land near 7.9, so the exact value is not load
/// bearing.
const LENGTH_PENALTY: f32 = 0.6;

/// Events that represent one session in the entity stream. Two: the most
/// canonical page and one runner-up. The stream nominates sessions, not
/// events, and a busy session must not fill the pool through it.
const ENTITY_EVENTS_PER_SESSION: usize = 2;

/// Sessions still in flight that lead a list of session summaries.
///
/// A session that ended without a summary - the CLI hit its limit, the
/// machine slept, the backstop has not reached it - is invisible to a list of
/// summaries, and the summary list is what the tool description tells an
/// agent to ask for. A real store showed the cost: a codex session with 187
/// captures ended, a Claude session opened 22 seconds later, and "what was
/// codex doing" was answered from the previous session's summary because
/// nothing newer had one. The same in-flight rule the primer uses leads the
/// list instead; a few is enough, because agents run a few sessions at once.
const RECENT_IN_FLIGHT: usize = 3;

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
/// One recorded rerank, as `brain stats` reads it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankRun {
    pub engine: String,
    pub reason: String,
    pub ms: u64,
    pub cold: bool,
}

/// Seconds in a day: the window `brain stats` and `brain doctor` read.
pub const DAY_SECS: i64 = 24 * 3600;

/// One model call made for consolidation, synthesis or ingest. `outcome` is
/// `ok`, `unusable`, `unparseable` (JSON of the wrong shape), `timeout` or
/// `spawn_error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummarizerCall {
    pub session: String,
    pub purpose: String,
    pub cli: String,
    pub model: String,
    pub prompt_bytes: u64,
    pub answer_bytes: u64,
    pub ms: u64,
    pub outcome: String,
}

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

/// A cosine at or above this is a match on meaning alone; between
/// [`NEAREST_FLOOR`] and this it is a loose association that needs company or
/// a place in the semantic top [`SEMANTIC_TOP_KEEP`].
/// 0.36 is the median cosine of the live store's real hits.
const SEMANTIC_MATCH: f32 = 0.36;

/// How many of the semantic stream's own best guesses the floor keeps even
/// below [`SEMANTIC_MATCH`] (they still need [`NEAREST_FLOOR`]). A short query
/// such as "authentication" scores a true meaning-match under 0.36 against
/// the static multilingual model, and finding it is what semantic search is
/// for; the deep tail of the same list is the noise. Ranks are 0-based here.
const SEMANTIC_TOP_KEEP: usize = 3;

/// Why the relevance floor kept an id off the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dropped {
    /// Only the semantic stream found it, below [`SEMANTIC_MATCH`] and outside
    /// its top [`SEMANTIC_TOP_KEEP`].
    LooseSemantic,
    Entity,
    Graph,
    EntityAndGraph,
}

impl std::fmt::Display for Dropped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LooseSemantic => write!(f, "semantic only, below {SEMANTIC_MATCH}"),
            Self::Entity => f.write_str("entity only"),
            Self::Graph => f.write_str("graph only"),
            Self::EntityAndGraph => f.write_str("entity and graph only"),
        }
    }
}

impl serde::Serialize for Dropped {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// The relevance floor for one id: `None` keeps it, `Some(why)` drops it.
/// `seen_in` has a bit per fusion list (see [`bit`]); `cosine` is its
/// semantic score when it had one; `semantic_rank` is its 0-based place in the
/// semantic stream, kept while within [`SEMANTIC_TOP_KEEP`] and at or above
/// [`NEAREST_FLOOR`].
fn floor_reason(seen_in: u8, cosine: Option<f32>, semantic_rank: Option<usize>) -> Option<Dropped> {
    const WORDS: u8 = bit(KEYWORD) | bit(SUBSTRING) | bit(RELAXED);
    const COMPANY: u8 = bit(ENTITY) | bit(GRAPH);
    if seen_in & WORDS != 0 || cosine.is_some_and(|c| c >= SEMANTIC_MATCH) {
        return None;
    }
    // Two loose semantic hits in sessions that share an entity keep each
    // other: company is not independent evidence, a trade-off the rule accepts.
    // `nearest` sorts stale-flagged entries last, so with fewer than 3
    // non-demoted candidates above the floor one can survive as a top guess;
    // it still sorts last on the page.
    if seen_in & bit(SEMANTIC) != 0 {
        let top = semantic_rank.is_some_and(|rank| rank < SEMANTIC_TOP_KEEP)
            && cosine.is_some_and(|c| c >= NEAREST_FLOOR);
        return if top || seen_in & COMPANY != 0 { None } else { Some(Dropped::LooseSemantic) };
    }
    Some(match seen_in & COMPANY {
        x if x == bit(ENTITY) => Dropped::Entity,
        x if x == bit(GRAPH) => Dropped::Graph,
        _ => Dropped::EntityAndGraph,
    })
}

/// Why a search returned what it did, one stream at a time.
///
/// A fused order is the sum of five opinions, and the order alone cannot say
/// which opinion put an entry where it is, or why an entry someone expected
/// is missing. Tuning the fusion on a 22k-event brain meant reconstructing
/// each stream's list by hand to answer that; this is the reconstruction,
/// kept. `brain search --explain` prints it.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Trace {
    /// In fusion order: keyword, semantic, entity, substring, graph,
    /// keyword_relaxed.
    pub streams: Vec<StreamTrace>,
    /// The fusion arithmetic for every id any stream nominated.
    pub fused: std::collections::HashMap<String, Fused>,
}

/// One stream's opinion.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StreamTrace {
    pub name: &'static str,
    /// Its share in the fusion — see `STREAM_WEIGHTS`.
    pub weight: f32,
    /// What it nominated, in its own order, with the stream's own score when
    /// that score means something outside the order (cosine, for meaning).
    pub ranked: Vec<(String, Option<f32>)>,
    /// Why the list is empty or shorter than the query deserved, when the
    /// stream knows: a model that would not load, a query its vocabulary
    /// cannot read, a fallback taken, a stream skipped on purpose.
    pub note: Option<String>,
}

/// How one id's fused score was arrived at.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Fused {
    /// The weighted reciprocal ranks, summed.
    pub raw: f32,
    /// Title plus body, in bytes.
    pub length: i64,
    /// What `raw` was divided by for being long; 1.0 at or under the pivot.
    pub discount: f32,
    /// `raw / discount`: the number the order came from.
    pub score: f32,
    /// A human flagged it stale, so it sorts last whatever the score.
    pub demoted: bool,
    /// Why the relevance floor kept this id off the page, when it did.
    pub dropped: Option<Dropped>,
}

impl Default for Fused {
    /// No discount until a length says otherwise: an id whose `events` row
    /// is gone must keep `raw`, not divide it by zero.
    fn default() -> Self {
        Self { raw: 0.0, length: 0, discount: 1.0, score: 0.0, demoted: false, dropped: None }
    }
}

#[cfg(test)]
impl Trace {
    /// Where a stream ranked an id, counted from one; `None` when it did not
    /// nominate it. The tests' question; `brain search --explain` walks the
    /// streams once instead.
    pub fn rank_in(&self, stream: &str, id: &str) -> Option<usize> {
        self.streams
            .iter()
            .find(|s| s.name == stream)?
            .ranked
            .iter()
            .position(|(candidate, _)| candidate == id)
            .map(|index| index + 1)
    }
}

/// One bounded slice of the retention pass; see [`Store::retention_step`].
#[derive(Debug, PartialEq, Eq)]
pub struct RetentionStep {
    /// The events whose bodies the rule drops, oldest rowid first.
    pub ids: Vec<String>,
    /// The rowid the next slice starts after.
    pub next: i64,
    /// Nothing is left past `next`: the pass has seen the whole table.
    pub end: bool,
}

/// The derived index.
pub struct Store {
    conn: Connection,
}

/// Which ledger a spilled id belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ledger {
    /// `recalled`, offered by a search or listing.
    Recalled,
    /// `recalled`, opened in full.
    Opened,
    /// `injected`, a pointer a hook pushed.
    Injected,
    /// `injected_files`; the "id" is the file's path.
    File,
}

/// One line of `surfaced.jsonl`: ids a session was shown while the ledger
/// could not be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfacedLine {
    pub session: String,
    pub kind: Ledger,
    pub ids: Vec<String>,
}

/// How an entry reached a session: offered by a search, or opened on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// A search or listing returned it. Says nothing about whether it helped.
    Offered,
    /// The agent asked for its full body, having seen the title. The closest
    /// thing to a relevance judgement this project can observe.
    Opened,
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
        Self::open_waiting(path, std::time::Duration::from_millis(u64::from(BUSY_TIMEOUT_MS)))
    }

    /// [`Store::open`], waiting at most `wait` on a lock another holds - the
    /// migration included, which is why it cannot be set after the open.
    ///
    /// # Errors
    /// Returns an error when the database cannot be opened or migrated.
    pub fn open_waiting(path: &Path, wait: std::time::Duration) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("open database {}", path.display()))?;
        conn.busy_timeout(wait).context("set busy timeout")?;
        // `journal_mode` is persistent, but setting it every open costs
        // nothing and keeps a hand-copied database correct.
        conn.pragma_update(None, "journal_mode", "WAL").context("enable WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL").context("set synchronous")?;
        // A checkpoint that cannot truncate (a reader held it) must not leave a
        // huge log behind: the file shrinks back to this after the next one.
        conn.pragma_update(None, "journal_size_limit", WAL_SIZE_LIMIT).context("set journal_size_limit")?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// One number out of any query, for tests in other modules.
    #[cfg(test)]
    pub fn raw_count(&self, sql: &str) -> i64 {
        self.conn.query_row(sql, [], |row| row.get(0)).expect("raw count")
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

    /// Run raw SQL on the index, so a test can take a table away and prove
    /// that a code path never reaches it.
    ///
    /// # Errors
    /// Returns an error when the statement fails.
    #[cfg(test)]
    pub fn execute_batch_for_test(&self, sql: &str) -> Result<()> {
        self.conn.execute_batch(sql).context("test sql")
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
                    -- How often a primer or file pointer pushed this line.
                    -- Paired with read_count: offered many sessions and never
                    -- pulled means the line spends budget and buys nothing.
                    injected_count INTEGER NOT NULL DEFAULT 0,
                    -- Lowered when a human says an entry is stale or wrong.
                    -- Nothing is destroyed; it just stops crowding the primer.
                    confidence   INTEGER NOT NULL DEFAULT 0,
                    consolidated INTEGER NOT NULL DEFAULT 0,
                    -- Published by a teammate rather than derived here.
                    -- Searchable and injectable like any other lesson, and
                    -- excluded from everything that REWRITES knowledge:
                    -- nobody edits another person's entry.
                    team         INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS events_project_ts ON events(project, ts);
                -- None by kind here: a store gets `events_session_id`,
                -- `events_kind_proj` and `events_recall` from
                -- `build_primer_indexes`, in a consolidate run. Creating them
                -- here would build them in whichever hook opens an existing
                -- store first. `events_session` stays beside them: an older
                -- binary's open recreates it, cold, under the write lock, when
                -- it is missing.
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
                -- The update triggers (`events_au`, `events_tri_au`) are made
                -- by `scope_fts_triggers`, once the schema is in place.

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
                    -- 0: a search offered it. 1: the agent asked to read it
                    -- in full. See Store::record_recalled.
                    opened   INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (session, event_id)
                );
                -- Both are keyed session-first, so the replay's count by
                -- event id scanned the table once per event. Side tables of
                -- tens of thousands of rows: the build is a few hundred ms cold, once.
                CREATE INDEX IF NOT EXISTS injected_by_event ON injected(event_id);
                CREATE INDEX IF NOT EXISTS recalled_by_event ON recalled(event_id);

                CREATE TABLE IF NOT EXISTS injected_files (
                    session TEXT NOT NULL,
                    path    TEXT NOT NULL,
                    PRIMARY KEY (session, path)
                );

                -- What each rerank did: which engine answered, how long it
                -- took, and why the faster engine did not. Local counters for
                -- `brain stats`; whether a machine without a local model
                -- should wait on a CLI at all cannot be decided without them.
                CREATE TABLE IF NOT EXISTS rerank_runs (
                    ts     TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    ms     INTEGER NOT NULL,
                    cold   INTEGER NOT NULL DEFAULT 0
                );

                -- Every model call consolidation, synthesis and ingest made,
                -- failed ones too: what the summarizer costs per session is a
                -- number nobody could read before. Local bookkeeping the log
                -- cannot rebuild, so `clear()` leaves it alone. Advisory
                -- reranks are in `rerank_runs`, not here.
                CREATE TABLE IF NOT EXISTS summarizer_calls (
                    ts           TEXT NOT NULL,
                    session      TEXT NOT NULL,
                    purpose      TEXT NOT NULL,
                    cli          TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    prompt_bytes INTEGER NOT NULL,
                    answer_bytes INTEGER NOT NULL,
                    ms           INTEGER NOT NULL,
                    outcome      TEXT NOT NULL
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
                    claimed_at    TEXT,
                    -- Consecutive runs that failed or fell to the rule-based
                    -- floor while a model was reachable. Parked at
                    -- `PARK_AFTER`; a success resets it.
                    attempts        INTEGER NOT NULL DEFAULT 0,
                    last_error      TEXT,
                    last_attempt_at TEXT
                );

                -- A consolidation ask, written BEFORE the run lock is tried, so
                -- an ask that finds the lock held is not lost: the holder reads
                -- the rows still unconsumed once it has let go of the lock.
                -- `session` NULL is not a scope; `all_projects` and `cwd` are.
                CREATE TABLE IF NOT EXISTS consolidation_requests (
                    id           INTEGER PRIMARY KEY,
                    session      TEXT,
                    all_projects INTEGER NOT NULL,
                    force        INTEGER NOT NULL,
                    cwd          TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    consumed_at  TEXT
                );

                -- One row per `brain consolidate` invocation, written by the
                -- command, so a run that yielded or failed is still on record.
                CREATE TABLE IF NOT EXISTS consolidation_runs (
                    started     TEXT NOT NULL,
                    ended       TEXT NOT NULL,
                    mode        TEXT NOT NULL,
                    yielded     INTEGER NOT NULL,
                    sessions    INTEGER NOT NULL,
                    events      INTEGER NOT NULL,
                    failed      INTEGER NOT NULL,
                    rule_based  INTEGER NOT NULL,
                    error       TEXT
                );
                ",
            )
            .context("apply schema")?;
        self.add_missing_columns()?;
        self.scope_fts_triggers()?;
        self.backfill_trigram()?;
        self.reconcile_embedding_model()
    }

    /// Make the two update triggers fire only when the text they index changes.
    ///
    /// Unscoped, they ran on every `UPDATE`, so flipping `consolidated`,
    /// `injected_count` or `read_count` deleted and re-inserted the row in
    /// both indexes: 1.6 ms a row, for text that had not changed.
    ///
    /// Any future column fed to an index must join its `OF` list, or an edit
    /// to it leaves the index answering from the old text.
    ///
    /// Triggers are DDL and take milliseconds, so this runs at open. The
    /// marker keeps an existing store from being dropped and recreated on
    /// every open; a fresh store has nothing to drop and gets the scoped ones
    /// directly.
    fn scope_fts_triggers(&self) -> Result<()> {
        let done: bool = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_state WHERE key = 'fts_triggers_scoped')",
                [],
                |row| row.get(0),
            )
            .context("check the trigger scope")?;
        if done {
            return Ok(());
        }
        self.conn
            .execute_batch(
                "BEGIN;
                 DROP TRIGGER IF EXISTS events_au;
                 DROP TRIGGER IF EXISTS events_tri_au;
                 CREATE TRIGGER events_au AFTER UPDATE OF title, body ON events BEGIN
                     INSERT INTO events_fts(events_fts, rowid, title, body)
                     VALUES ('delete', old.rowid, old.title, old.body);
                     INSERT INTO events_fts(rowid, title, body)
                     VALUES (new.rowid, new.title, new.body);
                 END;
                 CREATE TRIGGER events_tri_au AFTER UPDATE OF title ON events BEGIN
                     INSERT INTO events_tri(events_tri, rowid, title)
                     VALUES ('delete', old.rowid, old.title);
                     INSERT INTO events_tri(rowid, title) VALUES (new.rowid, new.title);
                 END;
                 INSERT OR REPLACE INTO schema_state (key, value)
                 VALUES ('fts_triggers_scoped', '1');
                 COMMIT;",
            )
            .context("scope the text index triggers")?;
        Ok(())
    }

    /// Drop, one bounded batch, the vectors of events that are no longer
    /// embedded. Returns how many it deleted; 0 means the migration is done.
    ///
    /// Tool-call payloads and a delegate's footsteps are JSON, not prose; they
    /// crowded the semantic stream and cost three quarters of the index.
    /// Touches `event_vec` only, never the log or `events`.
    ///
    /// Never run from `open`: a store this size cannot be scanned and written
    /// inside a hook's budget. The backlog pass calls it instead, until a
    /// batch finds nothing and records `payload_vectors_dropped`. Once that
    /// marker is set this is a single indexed read.
    ///
    /// # Errors
    /// Returns an error when a query fails.
    pub fn drop_payload_vectors(&self, batch: usize) -> Result<usize> {
        let done: bool = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_state WHERE key = 'payload_vectors_dropped')",
                [],
                |row| row.get(0),
            )
            .context("check payload vectors")?;
        if done {
            return Ok(0);
        }
        let sql = format!(
            "DELETE FROM event_vec WHERE event_id IN (
                 SELECT v.event_id FROM event_vec v JOIN events e ON e.id = v.event_id
                 WHERE NOT ({}) LIMIT ?1
             )",
            embeddable("e")
        );
        let deleted = self.conn.execute(&sql, params![batch as i64]).context("drop payload vectors")?;
        if deleted == 0 {
            self.conn
                .execute(
                    "INSERT OR IGNORE INTO schema_state (key, value) VALUES ('payload_vectors_dropped', '1')",
                    [],
                )
                .context("mark payload vectors dropped")?;
        }
        Ok(deleted)
    }

    /// Empty the vector index when a different model wrote it.
    ///
    /// A vector is only meaningful against the table that produced it, and
    /// the bytes do not say which table that was. Width caught the one
    /// change so far because the widths differed; a swap between two models
    /// of the same width would leave every old vector in place, each scoring
    /// noise against every new query, with `doctor` reporting a full index.
    /// So the model's signature is recorded beside the vectors, and a
    /// mismatch drops them all - which is not a migration, it is the same
    /// backlog `events_missing_vectors` already works through a slice at a
    /// time.
    ///
    /// No record at all means the model this build carries: every store
    /// without one was last embedded after the width change that introduced
    /// the current model, so assuming otherwise would re-embed a hundred
    /// thousand correct vectors on the upgrade that adds the record.
    ///
    /// This runs once, at open. A process that outlives the swap - an MCP
    /// server from before the upgrade, a consolidation run in flight - still
    /// carries the old model after a newer binary has re-stamped the index,
    /// so [`Self::set_vectors`] and [`Self::nearest`] each check the record
    /// again and refuse to touch an index that is no longer theirs.
    fn reconcile_embedding_model(&self) -> Result<()> {
        let current = crate::embed::signature();
        let recorded = self.recorded_embedding_model()?;
        if recorded.as_deref() == Some(current.as_str()) {
            return Ok(());
        }
        if recorded.is_some() {
            self.conn
                .execute_batch("DELETE FROM event_vec;")
                .context("drop vectors from a different model")?;
        }
        self.conn
            .execute(
                "INSERT OR REPLACE INTO schema_state (key, value) VALUES ('embedding_model', ?1)",
                params![current],
            )
            .context("record the embedding model")?;
        Ok(())
    }

    fn recorded_embedding_model(&self) -> Result<Option<String>> {
        self.conn
            .query_row("SELECT value FROM schema_state WHERE key = 'embedding_model'", [], |row| row.get(0))
            .optional()
            .context("read the embedding model record")
    }

    /// Fail unless the index still belongs to the model this process carries.
    ///
    /// The record is checked at open; this re-checks it on every vector read
    /// and write, for the process that was already running when a newer
    /// build re-stamped the index. Its vectors would score noise against
    /// everything the new model wrote, and nothing in the bytes would say so.
    fn vectors_are_this_models(&self) -> Result<()> {
        let current = crate::embed::signature();
        match self.recorded_embedding_model()? {
            Some(recorded) if recorded != current => anyhow::bail!(
                "the vector index now belongs to {recorded}; this process carries {current} \
                 and has to be restarted before it can use meaning"
            ),
            _ => Ok(()),
        }
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
                ("events", "injected_count", "INTEGER NOT NULL DEFAULT 0"),
                ("events", "confidence", "INTEGER NOT NULL DEFAULT 0"),
                ("events", "team", "INTEGER NOT NULL DEFAULT 0"),
                ("events", "agent", "TEXT"),
                ("events", "clamped", "INTEGER NOT NULL DEFAULT 0"),
                ("summarizer_health", "last_failed_at", "TEXT"),
                ("session_state", "claimed_at", "TEXT"),
                ("session_state", "attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("session_state", "last_error", "TEXT"),
                ("session_state", "last_attempt_at", "TEXT"),
                ("injected", "active", "INTEGER NOT NULL DEFAULT 1"),
                ("injected", "in_flight", "INTEGER NOT NULL DEFAULT 0"),
                ("recalled", "opened", "INTEGER NOT NULL DEFAULT 0"),
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

    /// Index one event published by a teammate.
    ///
    /// The same row as any other, flagged so the rewriting paths skip it.
    /// Searchable, recallable and injectable exactly like a lesson derived
    /// here, because a lesson's worth does not depend on who wrote it.
    ///
    /// # Errors
    /// Returns an error when the row cannot be written.
    pub fn index_team_event(&self, event: &Event) -> Result<()> {
        self.index(event)?;
        self.conn
            .execute("UPDATE events SET team = 1 WHERE id = ?1", params![event.id])
            .context("flag team event")?;
        Ok(())
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
        self.index_event(event, true)
    }

    /// Index an event a hook has just minted.
    ///
    /// The same as `index`, minus the two count lookups a replay needs: a
    /// fresh id has no `recalled` or `injected` row to restore from, and the
    /// lookups were 76% of a hook's SQL. Anything that can see an id twice -
    /// a reindex, a catch-up, a team merge - calls `index`.
    ///
    /// # Errors
    /// Returns an error when the insert fails.
    pub fn index_captured(&self, event: &Event) -> Result<()> {
        self.index_event(event, false)
    }

    fn index_event(&self, event: &Event, restore_counts: bool) -> Result<()> {
        let subject = self.subject_files_for(event)?;
        let files = serde_json::to_string(&subject).unwrap_or_else(|_| "[]".to_string());
        // A prompt that ORDERS remembering starts one rung up. Derived from
        // the event's own text, so a replay reproduces it; set only at
        // insert, so feedback demotes survive re-indexing.
        let intent = event.kind == EventKind::Observation
            && crate::event::is_user_prompt(&event.source.hook)
            && (crate::event::carries_memory_intent(&event.title)
                || crate::event::carries_memory_intent(&event.body));
        let (body, clamped) = clamp_for_index(event);
        self.conn
            .execute(
                "INSERT INTO events
                    (id, ts, workspace, project, session, cli, hook, kind, title, body,
                     files, topic, invocation, confidence, consolidated, agent, clamped)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
                 ON CONFLICT(id) DO UPDATE SET
                     title = excluded.title,
                     body = excluded.body,
                     clamped = excluded.clamped,
                     files = excluded.files,
                     topic = excluded.topic,
                     invocation = excluded.invocation,
                     consolidated = excluded.consolidated,
                     agent = excluded.agent",
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
                    body,
                    files,
                    event.topic,
                    event.extra.get("invocation").and_then(serde_json::Value::as_str),
                    i32::from(intent),
                    i32::from(event.consolidated),
                    event.agent.as_deref(),
                    i32::from(clamped),
                ],
            )
            .context("index event")?;

        // A tombstone or a correction is about ANOTHER event. Both are
        // ordinary appended lines - nothing is deleted or edited in the log -
        // and what they change is the derived row they point at.
        match event.kind {
            EventKind::Retire => {
                // Body only: title, topic, files and links stay, so the
                // pointer remains findable and the provenance intact. The
                // external-content FTS sees the UPDATE through its triggers
                // and drops the body text itself; the trigram index holds
                // titles only and never carried it.
                for target in &event.links {
                    self.conn
                        .execute(
                            "UPDATE events SET body = '', clamped = 0 WHERE id = ?1",
                            params![target],
                        )
                        .context("apply retirement")?;
                }
            }
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
            // A human saying "this is stale" is a judgement the log has to
            // carry, for the same reason a correction does. It did not: the
            // MCP handler lowered the column and appended nothing, so a
            // rebuild raised every flagged entry back to the top of its
            // ranking and emptied the review page - the one place flagging
            // leads anywhere. Unlike a read count there was no second table to
            // recover it from; on this machine two rebuilds in a day left zero
            // flags behind and no way to tell what had been flagged.
            EventKind::Note if event.source.hook == "feedback" => {
                for target in &event.links {
                    self.conn
                        .execute(
                            "UPDATE events SET confidence = confidence - 1 WHERE id = ?1",
                            params![target],
                        )
                        .context("apply feedback")?;
                }
            }
            // A supersession is the synthesis pass landing newer wording on
            // a fact it re-derived from NEW summaries. The rewrite is the
            // same as a correction's; the confidence bump is not - being
            // derived again is recurrence evidence, the thing this table
            // ranks on, while a user's fix says the entry was WRONG and
            // earns nothing.
            //
            // Unless a human already rewrote the page. A correction is the
            // one signal that the model's derivation was WRONG, and a later
            // session re-deriving the same wrong fact from the same kind of
            // evidence is exactly the case the correction exists for - so
            // the human's wording stands, and only the recurrence counts.
            // Measured shape: the fold picks "newest wording" by id, and a
            // freshly derived page is always newer than the fix it undoes.
            EventKind::Note if event.source.hook == "supersede" => {
                for target in &event.links {
                    self.conn
                        .execute(
                            "UPDATE events SET confidence = confidence + 1 WHERE id = ?1",
                            params![target],
                        )
                        .context("count supersession")?;
                    if self.human_corrected(std::slice::from_ref(target))?.is_empty() {
                        self.conn
                            .execute(
                                "UPDATE events SET title = ?2, body = ?3, corrected_by = ?4
                                 WHERE id = ?1",
                                params![target, event.title, event.body, event.id],
                            )
                            .context("apply supersession")?;
                    }
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

        // A hook-minted id has no recalled or injected rows to restore.
        if restore_counts {
            self.restore_counts(&event.id)?;
        }

        // Consolidation progress has to survive a rebuild, and it very nearly
        // did not: `mark_consolidated` writes only to this table, so a
        // `reindex` replayed the log, found every observation flagged unread,
        // and re-summarised the entire history - 141 model calls in ten
        // minutes on the store this was found on, each one work already done.
        //
        // The log does carry the answer. A summary names the events it was
        // drawn from, and the log replays in order, so those events are
        // already here when it arrives. Deriving the flag from the summary
        // rather than storing it separately is what makes the index disposable
        // in fact and not only in the README.
        if event.kind == EventKind::SessionSummary && !event.links.is_empty() {
            // Only a model-backed summary finishes its events. The rule-based
            // floor writes a page too, and leaves them pending on purpose so a
            // working model can redo them later - the one property this whole
            // ladder is built to keep. Restoring progress without checking
            // which tier wrote the page consumed those events on a rebuild,
            // which an end-to-end test caught before it shipped.
            //
            // A summary written before the tier was recorded has none. Those
            // are treated as done: the alternative is re-summarising every
            // session a store has ever had, which is the failure this whole
            // change exists to stop, and it was measured at 141 model calls in
            // ten minutes. The cost is that a legacy rule-based page keeps its
            // rule-based text.
            let model_backed = event
                .extra
                .get("tier")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|tier| tier != "rule-based");
            if model_backed {
                self.mark_consolidated(&event.links)?;
            }
        }

        for path in &subject {
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

    /// Put back the two counters that sit on the row but are kept in tables
    /// `clear()` spares.
    ///
    /// What an agent went back and read outlives the index that recorded it:
    /// `read_count` sits on the row, which a replay deletes and re-inserts at
    /// its default. So every rebuild quietly flattened the one signal `rank()`
    /// has that is evidence rather than heuristic, and nothing said a word:
    /// measured after two rebuilds in one day, 0 of 28,854 events had a read
    /// to their name while `recalled` still held 529 rows.
    ///
    /// Counted per session, matching how it is incremented: re-reading an
    /// entry in one conversation says nothing extra about its worth.
    ///
    /// The other counter has the same disease: how often a pointer was
    /// offered lives on the row too, and a replay resets it while `injected`
    /// keeps the truth. Without this, every rebuild hands each stale pointer
    /// a fresh five-session decay budget.
    fn restore_counts(&self, event_id: &str) -> Result<()> {
        if let Ok(reads) = self.conn.query_row(
            "SELECT COUNT(*) FROM recalled WHERE event_id = ?1",
            params![event_id],
            |row| row.get::<_, i64>(0),
        ) {
            if reads > 0 {
                self.conn
                    .execute(
                        "UPDATE events SET read_count = ?2 WHERE id = ?1",
                        params![event_id, reads],
                    )
                    .context("restore read count")?;
            }
        }
        if let Ok(times) = self.conn.query_row(
            "SELECT COUNT(*) FROM injected WHERE event_id = ?1",
            params![event_id],
            |row| row.get::<_, i64>(0),
        ) {
            if times > 0 {
                self.conn
                    .execute(
                        "UPDATE events SET injected_count = ?2 WHERE id = ?1",
                        params![event_id, times],
                    )
                    .context("restore injected count")?;
            }
        }
        Ok(())
    }

    /// The files this event should be findable by.
    ///
    /// Normally the event's own list. Summaries and knowledge pages written
    /// before they carried one are the exception: they name the events they
    /// were drawn from, so the list can be rebuilt from those rather than lost.
    ///
    /// That is what makes `brain reindex` a backfill rather than a copy. The
    /// log is replayed in order, so a summary's observations are already
    /// indexed when it arrives, and a knowledge page's summaries have already
    /// been through this same derivation - two hops, both of them upstream.
    ///
    /// Only these two kinds, and only when the list is empty: an event that
    /// carries files carries them because something meant it to, and a
    /// derivation that overruled that would be inventing history rather than
    /// completing it.
    fn subject_files_for(&self, event: &Event) -> Result<Vec<String>> {
        if !event.files.is_empty()
            || !matches!(event.kind, EventKind::SessionSummary | EventKind::Knowledge)
            || event.links.is_empty()
        {
            return Ok(event.files.clone());
        }
        let slots = std::iter::repeat_n("?", event.links.len()).collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT path, COUNT(*) AS touches FROM event_files
             WHERE event_id IN ({slots})
             GROUP BY path
             ORDER BY touches DESC, path
             LIMIT {}",
            crate::consolidate::SUBJECT_FILES_MAX
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare subject files")?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(event.links.iter()), |row| {
                row.get::<_, String>(0)
            })
            .context("run subject files")?;
        Ok(rows.filter_map(std::result::Result::ok).collect())
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
        self.search_traced(project, query, topic, limit, recall, false).map(|(hits, _)| hits)
    }

    /// [`Self::search`], with the working shown.
    ///
    /// With `floor` off the hits are exactly what `search` returns; the
    /// [`Trace`] beside them is every stream's own list and the fusion
    /// arithmetic, so a surprising order can be read rather than guessed at.
    ///
    /// `floor` keeps off the page what only a loose association nominated:
    /// an id survives when a word stream (keyword, keyword_relaxed,
    /// substring) found it, when its cosine reaches [`SEMANTIC_MATCH`], or
    /// when it clears [`NEAREST_FLOOR`] and entity or graph agrees or it is
    /// among the semantic stream's top [`SEMANTIC_TOP_KEEP`]. Callers
    /// that rerank pass `false`: the reranker is the judge of a wide pool.
    /// The trace's `fused` names each id the floor dropped and why.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn search_traced(
        &self,
        project: &str,
        query: &str,
        topic: Option<&str>,
        limit: usize,
        recall: Recall,
        floor: bool,
    ) -> Result<(Vec<Hit>, Trace)> {
        let mut stmt = self.conn.prepare(KEYWORD_SQL).context("prepare search")?;

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
        let (keyword, keyword_note) = match read(query) {
            Ok(hits) => (hits, None),
            Err(_) => {
                let phrase = format!("\"{}\"", query.replace('"', " "));
                let hits = read(&phrase).context("read search results")?;
                (hits, Some("FTS5 rejected the query's syntax; searched it as a quoted phrase".to_string()))
            }
        };
        // The same query with only some of its words required. Fused, not
        // substituted: the raw AND stays the first opinion, this one only
        // lets a partial match into the pool.
        let (relaxed, relaxed_note) = if recall == Recall::Lexical {
            (Vec::new(), None)
        } else {
            match compile_fts(query) {
                None => (
                    Vec::new(),
                    Some("skipped: the query is FTS syntax or has fewer than three terms".to_string()),
                ),
                Some(fts) => match read(&fts) {
                    Ok(hits) => (hits, Some(format!("any 3 of the longest terms: {fts}"))),
                    Err(error) => (Vec::new(), Some(format!("did not run: {error}"))),
                },
            }
        };
        drop(stmt);

        if recall == Recall::Lexical {
            let trace = Trace {
                streams: vec![StreamTrace {
                    name: STREAM_NAMES[0],
                    weight: STREAM_WEIGHTS[0],
                    ranked: keyword.iter().map(|hit| (hit.id.clone(), None)).collect(),
                    note: keyword_note,
                }],
                fused: std::collections::HashMap::new(),
            };
            return Ok((spread_across_sessions(keyword, limit), trace));
        }

        // Meaning, alongside words. Failure here degrades to the keyword
        // results rather than failing the search: a model that will not load
        // is a worse search, not a broken one. The trace keeps the reason,
        // because "semantic found nothing" and "semantic did not run" call
        // for different fixes and look identical in the results.
        let (semantic_scored, semantic_note) = match crate::embed::encode(query) {
            Err(error) => (Vec::new(), Some(format!("did not run: {error}"))),
            Ok(vector) if vector.iter().all(|byte| *byte == 0) => (
                Vec::new(),
                Some("did not run: no word of the query is in the model's vocabulary".to_string()),
            ),
            Ok(vector) => match self.nearest(project, &vector, topic, pool) {
                Ok(scored) => (scored, None),
                Err(error) => (Vec::new(), Some(format!("did not run: {error}"))),
            },
        };
        let semantic_ids: Vec<String> = semantic_scored.iter().map(|(id, _)| id.clone()).collect();
        let semantic = self.hits_by_id(&semantic_ids)?;
        // The recall floor is the one number a reader of this list needs
        // beside it: a cosine just above it is a loose association, not a
        // match, and the floor is what "found nothing" was measured against.
        let semantic_note = semantic_note.or_else(|| Some(format!("cosine floor {NEAREST_FLOOR}")));

        // Two more rankings that need no model at all, which is the point:
        // they are what keeps recall wide when every model is unreachable -
        // wide for a reranker; a floored page lets them in only as company
        // for a semantic hit (see `floor_reason`).
        // Entities are what a session DECLARED it was about, so they find
        // work whose words never matched; neighbours are what else touched
        // the same things, so a hit pulls in its context.
        let entity = self.entity_matches(project, query, topic, ENTITY_EVENTS_PER_SESSION, pool)?;
        // Only for the scripts the keyword tokenizer cannot cut into words.
        // An English query is already served correctly by `keyword`, and
        // substring matching would only add `author` to a search for `auth`.
        let (substring, substring_note) = if writes_without_spaces(query) {
            (self.substring_matches(project, query, topic, pool)?, None)
        } else {
            (
                Vec::new(),
                Some("skipped: the query has word boundaries, so the keyword stream already covers it".to_string()),
            )
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
        let graph_note =
            Some(format!("neighbours of {} seed(s) from keyword, semantic and substring", seeds.len()));

        // One array feeds both the fusion and the trace, so a stream's name,
        // weight and list cannot drift apart by position. The order is
        // KEYWORD, SEMANTIC, ENTITY, SUBSTRING, GRAPH, RELAXED.
        let lists = [keyword, semantic, entity, substring, graph, relaxed];
        let notes = [keyword_note, semantic_note, None, substring_note, graph_note, relaxed_note];
        let cosines: std::collections::HashMap<String, f32> = semantic_scored.iter().cloned().collect();
        let streams = STREAM_NAMES
            .into_iter()
            .zip(STREAM_WEIGHTS)
            .zip(&lists)
            .zip(notes)
            .map(|(((name, weight), list), note)| StreamTrace {
                name,
                weight,
                ranked: list
                    .iter()
                    .map(|hit| {
                        let own = (name == STREAM_NAMES[SEMANTIC]).then(|| cosines.get(&hit.id).copied()).flatten();
                        (hit.id.clone(), own)
                    })
                    .collect(),
                note,
            })
            .collect();
        let (hits, fused) = self.fuse(&lists, pool, floor.then_some(&cosines))?;
        Ok((spread_across_sessions(hits, limit), Trace { streams, fused }))
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
    /// about the right thing goes first. Not equal-weight: see
    /// `STREAM_WEIGHTS` for the two streams held at half and the
    /// measurement that put them there.
    fn fuse(
        &self,
        lists: &[Vec<Hit>],
        limit: usize,
        floor: Option<&std::collections::HashMap<String, f32>>,
    ) -> Result<(Vec<Hit>, std::collections::HashMap<String, Fused>)> {
        // The conventional damping constant. Large enough that the top of
        // any one list does not dominate outright, so agreement between
        // rankings can still outweigh a single strong opinion.
        const K: f32 = 60.0;

        let mut fused: std::collections::HashMap<&str, Fused> = std::collections::HashMap::new();
        // Which streams nominated each id, as a bit per list position.
        let mut seen_in: std::collections::HashMap<&str, u8> = std::collections::HashMap::new();
        // Each id's 0-based rank in the semantic list (first occurrence).
        let mut semantic_rank: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (index, list) in lists.iter().enumerate() {
            let weight = STREAM_WEIGHTS.get(index).copied().unwrap_or(1.0);
            for (rank, hit) in list.iter().enumerate() {
                if index == SEMANTIC {
                    semantic_rank.entry(hit.id.as_str()).or_insert(rank);
                }
                #[allow(clippy::cast_precision_loss)]
                let contribution = weight / (K + rank as f32 + 1.0);
                fused.entry(hit.id.as_str()).or_default().raw += contribution;
                *seen_in.entry(hit.id.as_str()).or_default() |= bit(index);
            }
        }

        // Fusion keeps only each list's ORDER, which silently discards the one
        // thing both lists encoded outside their order: that a human flagged an
        // entry stale. Both rankings put demoted entries last on purpose -
        // "flagging has to change what the user SEES" - so the flag is carried
        // across the fusion rather than re-derived from a position it no longer
        // occupies.
        for (id, length) in self.lengths_of(fused.keys().copied())? {
            if let Some(entry) = fused.get_mut(id.as_str()) {
                #[allow(clippy::cast_precision_loss)]
                let over = (length as f32 / LENGTH_PIVOT).ln().max(0.0);
                entry.length = length;
                entry.discount = 1.0 + LENGTH_PENALTY * over;
            }
        }
        let demoted = self.demoted_among(fused.keys().copied())?;
        for (id, entry) in &mut fused {
            entry.score = entry.raw / entry.discount;
            entry.demoted = demoted.contains(*id);
            if let Some(cosines) = floor {
                entry.dropped = floor_reason(
                    seen_in.get(id).copied().unwrap_or(0),
                    cosines.get(*id).copied(),
                    semantic_rank.get(id).copied(),
                );
            }
        }

        let mut ranked: Vec<(&str, &Fused)> =
            fused.iter().filter(|(_, entry)| entry.dropped.is_none()).map(|(id, entry)| (*id, entry)).collect();
        ranked.sort_by(|a, b| {
            a.1.demoted
                .cmp(&b.1.demoted)
                .then_with(|| b.1.score.total_cmp(&a.1.score))
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

        let hits = ranked
            .into_iter()
            .filter_map(|(id, _)| known.get(id).map(|hit| (*hit).clone()))
            .collect();
        let fused = fused.into_iter().map(|(id, entry)| (id.to_string(), entry)).collect();
        Ok((hits, fused))
    }

    /// Which of these ids a human has flagged stale.
    /// Title-plus-body byte length of each id, in one statement.
    fn lengths_of<'a>(
        &self,
        ids: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<(String, i64)>> {
        let ids: Vec<String> = ids.map(str::to_string).collect();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let holes = (0..ids.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, length(COALESCE(title, '')) + length(COALESCE(body, '')) \
             FROM events WHERE id IN ({holes})"
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare lengths")?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .context("run lengths")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read lengths")
    }

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
    /// Returns an error when the query fails or is interrupted at
    /// [`RELATED_DEADLINE`].
    pub fn related(&self, project: &str, id: &str, limit: usize) -> Result<Vec<Hit>> {
        self.within(RELATED_DEADLINE, || self.related_rows(project, id, limit))
    }

    fn related_rows(&self, project: &str, id: &str, limit: usize) -> Result<Vec<Hit>> {
        let mut stmt = self.conn.prepare(RELATED_SQL).context("prepare related")?;
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
                       AND e.kind NOT IN ('tombstone', 'retire') AND e.hook NOT IN ('correct', 'feedback', 'supersede')
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

    /// The statement behind [`Self::entity_matches`] for `tokens` name patterns.
    ///
    /// The window ranks sessions' events from columns `events_recall` holds,
    /// so the recall floor never fetches a row; title and body are read by id
    /// for the few that make the limit. Carrying them through the window
    /// fetched every event of every matching session: 2.4 s cold on the
    /// 115k-event project.
    fn entity_matches_sql(tokens: usize) -> String {
        let likes = (0..tokens)
            .map(|index| format!("n.name LIKE ?{} ESCAPE '\\'", index + 5))
            .collect::<Vec<_>>()
            .join(" OR ");
        format!(
            "WITH matched AS (
                 SELECT n.session, COUNT(DISTINCT n.name) AS matched
                 FROM entities n
                 WHERE n.project = ?1 AND ({likes})
                 GROUP BY n.session
             ),
             candidates AS (
                 SELECT e.id, e.session, m.matched,
                        CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END AS demoted,
                        CASE e.kind
                            WHEN 'knowledge' THEN 0
                            WHEN 'session_summary' THEN 1
                            WHEN 'source' THEN 1
                            ELSE 2
                        END AS authority,
                        ROW_NUMBER() OVER (
                            PARTITION BY e.session
                            ORDER BY CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END,
                                     CASE e.kind
                                         WHEN 'knowledge' THEN 0
                                         WHEN 'session_summary' THEN 1
                                         WHEN 'source' THEN 1
                                         ELSE 2
                                     END,
                                     e.id DESC
                        ) AS per_session
                 FROM events e
                 JOIN matched m ON m.session = e.session
                 WHERE e.project = ?1 AND e.forgotten = 0 AND e.kind NOT IN ('tombstone', 'retire')
                       AND e.hook NOT IN ('correct', 'feedback', 'supersede')
                       AND (?2 IS NULL OR e.topic = ?2)
             )
             SELECT h.id, h.ts, h.cli, h.kind, h.title,
                    substr(COALESCE(h.body, ''), 1, 160), h.session
             FROM (SELECT id, demoted, matched, authority FROM candidates
                   WHERE per_session <= ?3
                   ORDER BY demoted, matched DESC, authority, id DESC
                   LIMIT ?4) c
             JOIN events h ON h.id = c.id
             ORDER BY c.demoted, c.matched DESC, c.authority, c.id DESC"
        )
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

        let sql = Self::entity_matches_sql(tokens.len());
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

    /// The statement behind [`Self::neighbours_of`] for `seeds` seed ids.
    ///
    /// The seeds are looked up by primary key: `project` is only a check, and
    /// left indexable it let the planner walk the whole project through
    /// `events_recall` to find three ids. The neighbours are ranked from the
    /// columns that index holds and read by id once they are down to `limit`,
    /// like [`Self::entity_matches_sql`]: fetching every event of every
    /// sharing session took 3.2 s cold on the 115k-event project.
    fn neighbours_sql(seeds: usize) -> String {
        let holes = |offset: usize| {
            (0..seeds)
                .map(|index| format!("?{}", index + offset))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "WITH seed_sessions AS (
                 SELECT DISTINCT session FROM events
                 WHERE id IN ({seed_holes}) AND +project = ?1
             ),
             subject AS (
                 SELECT DISTINCT n.name FROM entities n
                 WHERE n.session IN (SELECT session FROM seed_sessions)
                       AND n.project = ?1
             ),
             shared AS (
                 SELECT n.session, COUNT(DISTINCT n.name) AS shared
                 FROM entities n
                 WHERE n.project = ?1 AND n.name IN (SELECT name FROM subject)
                       AND n.session NOT IN (SELECT session FROM seed_sessions)
                 GROUP BY n.session
             )
             SELECT h.id, h.ts, h.cli, h.kind, h.title,
                    substr(COALESCE(h.body, ''), 1, 160), h.session,
                    c.shared
             FROM (SELECT e.id, s.shared,
                          CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END AS demoted,
                          CASE e.kind
                              WHEN 'knowledge' THEN 0
                              WHEN 'session_summary' THEN 1
                              WHEN 'source' THEN 1
                              ELSE 2
                          END AS authority
                   FROM events e
                   JOIN shared s ON s.session = e.session
                   WHERE e.project = ?1
                         AND e.forgotten = 0 AND e.kind NOT IN ('tombstone', 'retire')
                         AND e.hook NOT IN ('correct', 'feedback', 'supersede')
                         AND (?2 IS NULL OR e.topic = ?2)
                   ORDER BY s.shared DESC, demoted, authority, e.id DESC
                   LIMIT ?3) c
             JOIN events h ON h.id = c.id
             ORDER BY c.shared DESC, c.demoted, c.authority, c.id DESC",
            seed_holes = holes(4),
        )
    }

    /// What sits beside the hits another ranking already found.
    ///
    /// `related`, widened to several seeds and folded into search: the top
    /// hits' sessions declared entities, and whatever else touched those
    /// entities is context the query's words never asked for. Needs no
    /// model - the other zero-LLM ranking.
    ///
    /// A seed's own session is excluded, not merely its own event. Entities
    /// are recorded per session, so a seed drags in every name its session
    /// ever declared - which means that session shares 100% of `subject` with
    /// itself and wins the stream outright, no matter how the count is
    /// normalised. Measured on a 22k-event brain: one session's summary took
    /// #1 in 21 of 23 queries, one of them nonsense words absent from the
    /// database. The stream is for what the query's words could NOT reach; a
    /// seed's own session was already reached, and returning the rest of it
    /// is the same opinion counted again, once per event it happens to hold.
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
        let sql = Self::neighbours_sql(seeds.len());
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
        let live = "forgotten = 0 AND kind NOT IN ('tombstone', 'retire') AND hook NOT IN ('correct', 'feedback', 'supersede')";
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
            knowledge: self.knowledge_entries(project)?.into_iter().map(|(_, title)| title).collect(),
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
                 WHERE id = ?1 AND forgotten = 0 AND kind NOT IN ('tombstone', 'retire')
                       AND hook NOT IN ('correct', 'feedback', 'supersede')",
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
        self.vectors_are_this_models()?;
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
        let sql = format!(
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
                       AND e.kind NOT IN ('tombstone', 'retire') AND e.project = ?1
                       AND {}
                 ORDER BY e.id DESC
                 LIMIT ?2",
            embeddable("e")
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare missing vectors")?;
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
                &format!(
                    "SELECT COUNT(*) FROM event_vec v JOIN events e ON e.id = v.event_id
                     WHERE length(v.vec) = ?1 AND e.kind NOT IN ('tombstone', 'retire') AND {}",
                    embeddable("e")
                ),
                params![crate::embed::DIMS as i64],
                |row| row.get(0),
            )
            .context("count vectors")?;
        let total: i64 = self
            .conn
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM events e WHERE e.kind NOT IN ('tombstone', 'retire') AND {}",
                    embeddable("e")
                ),
                [],
                |row| row.get(0),
            )
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
        self.vectors_are_this_models()?;
        let mut stmt = self.conn.prepare(NEAREST_SQL).context("prepare nearest")?;
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
        //
        // Equal scores are common (a dozen "Session started" summaries embed
        // alike) and the oldest comes first. Without a tie-break they kept the
        // order the rows were read in, which is the order of whichever index
        // the planner walked.
        scored.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.1.total_cmp(&a.1)).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(id, score, _)| (id, score)).collect())
    }

    /// Drop the index's copy of these rows' bodies and say the log holds them.
    ///
    /// The row stays: title, topic, files and links keep it findable. Unlike a
    /// retirement this writes nothing to the log, so it is index state only -
    /// a reindex restores the body, and `clamped = 1` is what sends `brain_get`
    /// back to the log for it while the body is gone. Rows already empty are
    /// left alone. Returns how many rows changed; the caller owns the
    /// transaction, so a chunk of ids is one commit.
    ///
    /// # Errors
    /// Returns an error when an update fails.
    pub fn drop_index_bodies(&self, ids: &[String]) -> Result<usize> {
        let mut stmt = self
            .conn
            .prepare_cached("UPDATE events SET body = '', clamped = 1 WHERE id = ?1 AND body != ''")
            .context("prepare body drop")?;
        let mut changed = 0;
        for id in ids {
            changed += stmt.execute(params![id]).context("drop index body")?;
        }
        Ok(changed)
    }

    /// Bytes the index holds in these rows' bodies.
    fn body_bytes(&self, ids: &[String]) -> Result<i64> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT COALESCE(SUM(length(CAST(body AS BLOB))), 0) FROM events WHERE id = ?1")
            .context("prepare body size")?;
        let mut total = 0i64;
        for id in ids {
            let bytes: i64 = stmt.query_row(params![id], |row| row.get(0)).context("measure body")?;
            total = total.saturating_add(bytes);
        }
        Ok(total)
    }

    /// Is the log the only place this row's whole body lives?
    ///
    /// True when the body was cut down on the way into the index, or dropped
    /// from it later. The log still holds the whole line; `brain_get` reads it
    /// from there. A retirement clears the flag: it withdrew the body on
    /// purpose, and the log must not hand it back.
    /// Separate from [`Store::get`] on purpose: consolidation reads through
    /// `get` and wants the bounded body.
    ///
    /// # Errors
    /// Returns an error when the lookup fails.
    pub fn is_clamped(&self, id: &str) -> Result<bool> {
        let clamped: Option<i32> = self
            .conn
            .query_row("SELECT clamped FROM events WHERE id = ?1", params![id], |row| row.get(0))
            .optional()
            .context("read clamped flag")?;
        Ok(clamped.unwrap_or(0) != 0)
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
                        files, topic, consolidated, agent
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
                        agent: row.get(13)?,
                        origin: None,
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
        let mut stmt = self.conn.prepare(Self::recent_sql()).context("prepare recent")?;
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
        let listed: Vec<Hit> =
            rows.collect::<rusqlite::Result<Vec<_>>>().context("read recent results")?;
        drop(stmt);

        // Only the summary list needs this. A list of everything already
        // leads with the newest captures themselves, and one session's work
        // is what `session` asks for whole. The row is keyed by the newest
        // capture, the way the primer's line is, so reading that session
        // counts as pulling it.
        if session.is_some() || kind != Some("session_summary") {
            return Ok(listed);
        }
        let mut hits: Vec<Hit> = self
            .unconsolidated_sessions(project, RECENT_IN_FLIGHT)?
            .into_iter()
            .filter(|work| cli.is_none_or(|cli| work.cli == cli))
            .map(|work| Hit {
                id: work.newest_id,
                ts: work.newest_ts,
                title: format!(
                    "{} session, {} capture(s) not yet summarized - brain_recent(kind: \"raw\", session: \"{}\") reads them",
                    work.cli, work.captures, work.session
                ),
                cli: work.cli,
                kind: "observation".to_string(),
                snippet: String::new(),
                session: work.session,
            })
            .collect();
        hits.extend(listed);
        hits.truncate(limit);
        Ok(hits)
    }

    /// The statement behind [`Self::recent`]. The floor, the kind and the
    /// session are columns of `events_recall`, so the inner select applies
    /// them without fetching a row; the outer one reads the listed ids
    /// newest first, fetching rows (and testing `cli`, which the index does
    /// not hold) only until `limit` are found; a rare `cli` still costs a row
    /// fetch per skipped id. Before the build the inner
    /// select reads the project through `events_unconsolidated`, as the
    /// single select did.
    fn recent_sql() -> &'static str {
        "SELECT id, ts, cli, kind, title, session FROM events
         WHERE id IN (SELECT id FROM events
                      WHERE project = ?1 AND forgotten = 0 AND kind NOT IN ('tombstone', 'retire')
                            AND hook NOT IN ('correct', 'feedback', 'supersede')
                            AND (?4 IS NULL OR kind = ?4)
                            AND (?5 IS NULL OR session = ?5))
               AND (?3 IS NULL OR cli = ?3)
         ORDER BY id DESC LIMIT ?2"
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

    /// Which of these ids a human last rewrote.
    ///
    /// The join tells a `brain correct` (hook `correct`) or a vault edit
    /// (cli `human`) apart from the machine's own `supersede`, which also
    /// lands in `corrected_by`. Both machine passes that rewrite knowledge -
    /// the write-time supersede and the duplicate fold - ask this before
    /// touching a page: a human's wording is never overwritten and never
    /// withdrawn by a model re-deriving the claim it fixed.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn human_corrected(&self, ids: &[String]) -> Result<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let holes = (0..ids.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT t.id FROM events t
             JOIN events c ON c.id = t.corrected_by
             WHERE t.id IN ({holes}) AND (c.hook = 'correct' OR c.cli = 'human')"
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare human_corrected")?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(ids.iter()), |row| row.get::<_, String>(0))
            .context("run human_corrected")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read human_corrected")
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

    /// What an existing index already knows about how a session was invoked,
    /// read without opening it for writes.
    ///
    /// `open` migrates, and a migration under a held write lock waits out the
    /// whole busy timeout - longer than a hook is allowed to take. A capture
    /// needs this answer before it appends to the log, so it asks here, waits
    /// a moment at most, and treats every failure as "not known".
    #[must_use]
    pub fn peek_session_invocation(path: &Path, session: &str) -> Option<String> {
        let conn =
            Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
        conn.busy_timeout(std::time::Duration::from_millis(200)).ok()?;
        conn.query_row(
            "SELECT invocation FROM session_invocation WHERE session = ?1",
            params![session],
            |row| row.get(0),
        )
        .optional()
        .ok()
        .flatten()
    }

    /// Remember how a session was invoked.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    /// Put a settled session back in front of the summarizer.
    ///
    /// Quiet and headless sessions are settled by marking their events done
    /// without writing anything, which is also what stops the backstop from
    /// asking about them forever. `--force --session` is the way to say "I
    /// want the summary after all": the events go back to pending and the
    /// run record goes, so the next pass treats the session as never seen.
    /// A session that was summarized by a model is left alone - forcing it
    /// would write the same narrative twice.
    ///
    /// # Errors
    /// Returns an error when a statement fails.
    pub fn reopen_settled_session(&self, session: &str) -> Result<bool> {
        let settled: Option<String> = self
            .conn
            .query_row(
                "SELECT last_tier FROM session_state
                 WHERE session = ?1 AND last_tier IN ('quiet', 'headless')",
                params![session],
                |row| row.get(0),
            )
            .optional()
            .context("read settled session")?;
        if settled.is_none() {
            return Ok(false);
        }
        self.conn
            .execute(
                "UPDATE events SET consolidated = 0
                 WHERE session = ?1 AND kind = 'observation' AND forgotten = 0",
                params![session],
            )
            .context("reopen settled events")?;
        self.conn
            .execute("DELETE FROM session_state WHERE session = ?1", params![session])
            .context("forget settled run")?;
        Ok(true)
    }

    /// Was this session classified as a one-shot run when it opened?
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn session_is_headless(&self, session: &str) -> Result<bool> {
        Ok(self
            .session_invocation(session)?
            .is_some_and(|raw| crate::invocation::parse(&raw).is_headless()))
    }

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
        self.record_recall(session, ids, Reach::Offered)
    }

    /// The same, for ids the agent asked to read in full.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_opened<'a>(
        &self,
        session: &str,
        ids: impl Iterator<Item = &'a str>,
    ) -> Result<()> {
        self.record_recall(session, ids, Reach::Opened)
    }

    /// Write one recall, remembering whether it was merely offered by a
    /// search or actually opened.
    ///
    /// The distinction is the only honest relevance signal this project can
    /// collect. A search returns thirty entries and says nothing about which
    /// of them answered anything; an agent calling `brain_get` on one of
    /// them has judged it worth the tokens. Ranking work so far has had to
    /// score itself against a model's opinion of a list of titles, which is
    /// a proxy that shares the ranking's own blind spots. This is the
    /// beginning of a record that does not.
    ///
    /// `opened` only ever goes up: a search that offers an entry already
    /// opened must not erase that it was opened.
    fn record_recall<'a>(
        &self,
        session: &str,
        ids: impl Iterator<Item = &'a str>,
        how: Reach,
    ) -> Result<()> {
        for id in ids {
            let seen: Option<i64> = self
                .conn
                .query_row(
                    "SELECT opened FROM recalled WHERE session = ?1 AND event_id = ?2",
                    params![session, id],
                    |row| row.get(0),
                )
                .optional()
                .context("read recalled row")?;
            match seen {
                None => {
                    self.conn
                        .execute(
                            "INSERT INTO recalled (session, event_id, opened) VALUES (?1, ?2, ?3)",
                            params![session, id, i64::from(how == Reach::Opened)],
                        )
                        .context("record recalled id")?;
                    // Count a session's first read only. Re-reading the same
                    // entry in one conversation says nothing extra about its
                    // worth.
                    self.conn
                        .execute(
                            "UPDATE events SET read_count = read_count + 1 WHERE id = ?1",
                            params![id],
                        )
                        .context("count read")?;
                }
                Some(0) if how == Reach::Opened => {
                    // Offered earlier in this session, opened now. The upgrade
                    // is the signal; the read was already counted.
                    self.conn
                        .execute(
                            "UPDATE recalled SET opened = 1 WHERE session = ?1 AND event_id = ?2",
                            params![session, id],
                        )
                        .context("mark opened")?;
                }
                Some(_) => {}
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
    pub fn knowledge_entries(&self, project: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            // `team = 0`: this list is what folding, superseding and
            // dedup work from, and all three REWRITE what they find. A
            // teammate's entry is read here and rewritten nowhere - the
            // one mechanical guarantee behind "nobody edits another
            // person's memory".
            "SELECT id, title FROM events
             WHERE project = ?1 AND kind = 'knowledge' AND forgotten = 0 AND team = 0",
        )?;
        let rows = stmt
            .query_map([project], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
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
                // Documents read in sit beside session summaries: both are
                // one episode's account, and a claim they agree on recurs.
                "SELECT id FROM events
                 WHERE project = ?1 AND kind IN ('session_summary', 'source') AND forgotten = 0
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

    /// Corrections and flags a person made, newest first.
    ///
    /// The one reader of `hook IN ('correct','feedback')` events: every other
    /// query filters them out because they mutate their target and then have
    /// nothing left to say. Synthesis reads them because a correction the
    /// user had to make twice IS the durable fact - a standing rule this
    /// project keeps violating.
    ///
    /// Machine rewordings are excluded twice over: `supersede_knowledge`
    /// now writes its own `hook = 'supersede'` (not matched here), and the
    /// `corrected_by` guard below still catches the pre-rename shape in old
    /// logs - a rule distilled from our own rewordings would be a rule
    /// about bookkeeping.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn recent_corrections(&self, project: &str, limit: usize) -> Result<Vec<Event>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id FROM events
                 WHERE project = ?1 AND kind = 'note'
                   AND hook IN ('correct','feedback') AND forgotten = 0
                   AND NOT EXISTS (SELECT 1 FROM events t
                                   WHERE t.corrected_by = events.id
                                     AND t.kind = 'knowledge')
                 ORDER BY id DESC LIMIT ?2",
            )
            .context("prepare recent corrections")?;
        let ids = stmt
            .query_map(params![project, limit as i64], |row| row.get::<_, String>(0))
            .context("run recent corrections")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("read correction ids")?;
        self.get(&ids)
    }

    /// Consolidated observation bodies older than `cutoff` that were never
    /// surfaced: not injected, not recalled, no read to their name. Usage
    /// beats age - anything anyone ever saw survives - the storage study's
    /// rule, checked against the tables that survive a rebuild rather than
    /// the counter a rebuild used to flatten.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn retirable(&self, project: &str, cutoff: &str) -> Result<(Vec<String>, i64)> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, LENGTH(body) FROM events
                 WHERE project = ?1 AND kind = 'observation' AND consolidated = 1
                   AND forgotten = 0 AND body != '' AND read_count = 0
                   AND ts < ?2
                   AND id NOT IN (SELECT event_id FROM injected)
                   AND id NOT IN (SELECT event_id FROM recalled)
                 ORDER BY id",
            )
            .context("prepare retirable")?;
        let rows = stmt
            .query_map(params![project, cutoff], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("run retirable")?;
        let mut ids = Vec::new();
        let mut bytes = 0i64;
        for row in rows {
            let (id, len) = row.context("read retirable row")?;
            ids.push(id);
            bytes += len;
        }
        Ok((ids, bytes))
    }

    /// Who the retention pass drops, as one `WHERE` clause over `events`;
    /// `?1` is the cutoff.
    ///
    /// The same rule as [`Self::retirable`] and every project at once, written
    /// with `NOT EXISTS` so a NULL in a usage table cannot empty the answer,
    /// and with the timestamp first so a recent row is rejected before its
    /// body is read. `+kind` keeps `events_kind_proj` out of the plan: the pass
    /// walks rowids, and an index on kind would send it through every
    /// observation's row in id order instead.
    ///
    /// A prose row also needs its vector first. The embed backlog encodes
    /// `title || ' ' || body`, so a body dropped before its vector exists
    /// leaves a vector made from the title alone, and nothing ever redoes it.
    /// A row the embedder skips has no vector to wait for. The cost is that
    /// retention trails the backlog, and a store without the model keeps its
    /// bodies.
    fn retention_rule() -> String {
        format!(
            "ts < ?1 AND +kind = 'observation' AND consolidated = 1
             AND forgotten = 0 AND read_count = 0 AND body != ''
             AND NOT EXISTS (SELECT 1 FROM injected i WHERE i.event_id = events.id)
             AND NOT EXISTS (SELECT 1 FROM recalled r WHERE r.event_id = events.id)
             AND (NOT ({}) OR EXISTS (
                 SELECT 1 FROM event_vec v WHERE v.event_id = events.id AND length(v.vec) = {}))",
            embeddable("events"),
            crate::embed::DIMS
        )
    }

    fn retention_step_sql() -> String {
        format!(
            "SELECT rowid, id FROM events
             WHERE rowid > ?2 AND rowid <= ?3 AND {}
             ORDER BY rowid LIMIT ?4",
            Self::retention_rule()
        )
    }

    /// Read the next bounded slice of the retention pass: up to `limit` rows
    /// the rule drops among the `span` rowids after `after`.
    ///
    /// The window is what bounds the read. Scanning the table for the next
    /// match would read all of it when few rows match, 7 to 9 s cold on a
    /// 1.8 GB store, and nothing can interrupt a statement; a window of rowids
    /// is a sequential read of a known size, and the caller checks its clock
    /// between windows.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn retention_step(&self, cutoff: &str, after: i64, span: i64, limit: usize) -> Result<RetentionStep> {
        let mut stmt = self.conn.prepare_cached(&Self::retention_step_sql()).context("prepare retention step")?;
        let rows = stmt
            .query_map(params![cutoff, after, after.saturating_add(span), limit as i64], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .context("run retention step")?;
        let mut ids = Vec::new();
        let mut last = after;
        for row in rows {
            let (rowid, id) = row.context("read retention row")?;
            last = rowid;
            ids.push(id);
        }
        // A full slice stopped at its last match; a short one saw its whole window.
        let next = if ids.len() == limit { last } else { after.saturating_add(span) };
        let newest: Option<i64> = self
            .conn
            .query_row("SELECT MAX(rowid) FROM events", [], |row| row.get(0))
            .context("read the newest rowid")?;
        Ok(RetentionStep { ids, next, end: newest.is_none_or(|newest| next >= newest) })
    }

    /// How many of the `span` rowids after `after` the retention rule would
    /// drop now.
    ///
    /// Bounded like a step: counting the rest of the table reads all of it,
    /// 5 to 9 s cold on 1.9 GB, and `brain doctor` asks. So the answer is a
    /// floor for the table past the window, exact inside it.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn retention_pending(&self, cutoff: &str, after: i64, span: i64) -> Result<i64> {
        self.conn
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM events WHERE rowid > ?2 AND rowid <= ?3 AND {}",
                    Self::retention_rule()
                ),
                params![cutoff, after, after.saturating_add(span)],
                |row| row.get(0),
            )
            .context("count retirable bodies")
    }

    /// One value out of `schema_state`, the table of small facts the index
    /// keeps about itself.
    ///
    /// # Errors
    /// Returns an error when the read fails.
    pub fn state(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_row("SELECT value FROM schema_state WHERE key = ?1", params![key], |row| row.get(0))
            .optional()
            .with_context(|| format!("read {key}"))
    }

    /// Write one `schema_state` value, replacing what was there.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn set_state(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .execute("INSERT OR REPLACE INTO schema_state (key, value) VALUES (?1, ?2)", params![key, value])
            .with_context(|| format!("write {key}"))?;
        Ok(())
    }

    /// Remove one `schema_state` value; absent is fine.
    ///
    /// # Errors
    /// Returns an error when the delete fails.
    pub fn clear_state(&self, key: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM schema_state WHERE key = ?1", params![key])
            .with_context(|| format!("clear {key}"))?;
        Ok(())
    }

    /// Change how long this connection waits on a lock another holds.
    ///
    /// # Errors
    /// Returns an error when the timeout cannot be set.
    pub fn set_busy_timeout(&self, wait: std::time::Duration) -> Result<()> {
        self.conn.busy_timeout(wait).context("set busy timeout")
    }

    /// What the compact window needs to decide, and what it last did, in one read.
    ///
    /// # Errors
    /// Returns an error when a read fails.
    pub fn compact_state(&self) -> Result<CompactState> {
        let number = |key: &str| -> Result<u64> {
            Ok(self.state(key)?.and_then(|value| value.parse().ok()).unwrap_or(0))
        };
        let (pages, free_pages): (i64, i64) = self.page_use()?;
        let page_size: i64 =
            self.conn.pragma_query_value(None, "page_size", |row| row.get(0)).context("read page_size")?;
        Ok(CompactState {
            done_at: self.state("compact_done_at")?,
            tried_at: self.state("compact_tried_at")?.and_then(|value| value.parse().ok()),
            skip: self.state("compact_skip")?,
            retention_pending: self.state("retention_cursor")?.is_some(),
            retention_dropped: number("retention_dropped")?,
            retention_done_at: self.retention_done_at()?,
            retention_dropped_bytes: number("retention_dropped_bytes")?,
            compact_dropped_bytes: number("compact_dropped_bytes")?,
            file_bytes: u64::try_from(pages).unwrap_or(0).saturating_mul(u64::try_from(page_size).unwrap_or(0)),
            free_bytes: u64::try_from(free_pages).unwrap_or(0).saturating_mul(u64::try_from(page_size).unwrap_or(0)),
            before_bytes: None,
            after_bytes: None,
        })
    }

    /// Write down what a compact attempt did: `done_at` (which also settles the
    /// dropped-bytes counter and clears the last skip), `tried_at`, `skip`.
    /// Only the fields that are set are written.
    ///
    /// # Errors
    /// Returns an error when a write fails.
    pub fn record_compact(&self, state: &CompactState) -> Result<()> {
        let transaction = self.conn.unchecked_transaction().context("begin compact record")?;
        if let Some(tried) = state.tried_at {
            self.set_state("compact_tried_at", &tried.to_string())?;
        }
        if let Some(skip) = &state.skip {
            self.set_state("compact_skip", skip)?;
        }
        if let Some(done) = &state.done_at {
            self.set_state("compact_done_at", done)?;
            self.set_state("compact_dropped_bytes", &state.compact_dropped_bytes.to_string())?;
            if state.skip.is_none() {
                self.clear_state("compact_skip")?;
            }
        }
        if let Some(bytes) = state.before_bytes {
            self.set_state("compact_bytes_before", &bytes.to_string())?;
        }
        if let Some(bytes) = state.after_bytes {
            self.set_state("compact_bytes_after", &bytes.to_string())?;
        }
        transaction.commit().context("commit compact record")
    }

    /// The rowid the retention pass resumes after; 0 starts a pass.
    ///
    /// # Errors
    /// Returns an error when the read fails.
    pub fn retention_cursor(&self) -> Result<i64> {
        Ok(self.state("retention_cursor")?.and_then(|value| value.parse().ok()).unwrap_or(0))
    }

    /// When the last full retention pass finished, as written by
    /// [`Self::finish_retention_pass`].
    ///
    /// # Errors
    /// Returns an error when the read fails.
    pub fn retention_done_at(&self) -> Result<Option<String>> {
        self.state("retention_done_at")
    }

    /// Bodies the retention pass has dropped from this index so far.
    ///
    /// # Errors
    /// Returns an error when the read fails.
    pub fn retention_dropped(&self) -> Result<i64> {
        Ok(self.state("retention_dropped")?.and_then(|value| value.parse().ok()).unwrap_or(0))
    }

    /// Drop one slice's bodies and move the cursor past them, in one
    /// transaction: a crash repeats the slice, which changes nothing the
    /// second time. Returns how many bodies it emptied.
    ///
    /// The rule is not asked again here: a row surfaced between the step's
    /// read and this commit loses its index body too. The window is
    /// milliseconds, and `brain_get` still returns the whole body from the log.
    ///
    /// # Errors
    /// Returns an error when a write fails; the transaction rolls back.
    pub fn commit_retention_step(&self, ids: &[String], cursor: i64) -> Result<usize> {
        let transaction = self.conn.unchecked_transaction().context("begin retention step")?;
        // Measured in this transaction, before the drop, in bytes: `length()`
        // of a text value counts characters.
        let bytes = self.body_bytes(ids)?;
        let dropped = self.drop_index_bodies(ids)?;
        self.conn
            .execute(
                "INSERT INTO schema_state (key, value) VALUES ('retention_cursor', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![cursor.to_string()],
            )
            .context("move the retention cursor")?;
        self.conn
            .execute(
                "INSERT INTO schema_state (key, value) VALUES ('retention_dropped', ?1)
                 ON CONFLICT(key) DO UPDATE
                 SET value = CAST(value AS INTEGER) + CAST(excluded.value AS INTEGER)",
                params![dropped.to_string()],
            )
            .context("count the dropped bodies")?;
        self.conn
            .execute(
                "INSERT INTO schema_state (key, value) VALUES ('retention_dropped_bytes', ?1)
                 ON CONFLICT(key) DO UPDATE
                 SET value = CAST(value AS INTEGER) + CAST(excluded.value AS INTEGER)",
                params![bytes.to_string()],
            )
            .context("count the dropped bytes")?;
        transaction.commit().context("commit retention step")?;
        Ok(dropped)
    }

    /// Record that a pass reached the end of the table: the next one waits for
    /// a day to pass, and starts from the first row.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn finish_retention_pass(&self, now: &str) -> Result<()> {
        let transaction = self.conn.unchecked_transaction().context("begin retention finish")?;
        self.conn
            .execute("DELETE FROM schema_state WHERE key = 'retention_cursor'", [])
            .context("clear the retention cursor")?;
        self.conn
            .execute(
                "INSERT INTO schema_state (key, value) VALUES ('retention_done_at', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![now],
            )
            .context("record the retention pass")?;
        transaction.commit().context("commit retention finish")
    }

    /// Give the pages a retention step freed back to the files: merge a bounded
    /// amount of the text index, then checkpoint the WAL.
    ///
    /// Emptying a body writes delete markers into the text index, and only a
    /// merge folds them away. `merge` with a page count does at most that much
    /// work, so it fits the same bound as a step. The checkpoint is PASSIVE
    /// and never waits for a reader; when it caught up completely a TRUNCATE
    /// follows with no busy timeout, so a hook holding the database sends it
    /// away instead of waiting on it. Neither `optimize` nor VACUUM belongs
    /// here: both rewrite more than a run may spend.
    ///
    /// # Errors
    /// Returns an error when the merge or the checkpoint fails.
    pub fn settle_after_retention(&self, merge_pages: i64) -> Result<()> {
        self.conn
            .execute("INSERT INTO events_fts(events_fts, rank) VALUES ('merge', ?1)", params![merge_pages])
            .context("merge the text index")?;
        let (busy, log, done): (i64, i64, i64) = self
            .conn
            .query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .context("checkpoint the WAL")?;
        if busy == 0 && log > 0 && log == done {
            self.conn.busy_timeout(std::time::Duration::ZERO).context("drop the busy timeout")?;
            let truncated = self
                .conn
                .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get::<_, i64>(0))
                .context("truncate the WAL");
            self.conn
                .busy_timeout(std::time::Duration::from_millis(u64::from(BUSY_TIMEOUT_MS)))
                .context("restore the busy timeout")?;
            truncated?;
        }
        Ok(())
    }

    /// Pages in the database file and how many of them hold nothing.
    ///
    /// Dropped bodies leave their pages on the free list, where SQLite reuses
    /// them but never hands them back to the filesystem.
    ///
    /// # Errors
    /// Returns an error when a pragma cannot be read.
    pub fn page_use(&self) -> Result<(i64, i64)> {
        let pages = self.conn.pragma_query_value(None, "page_count", |row| row.get(0)).context("read page_count")?;
        let free =
            self.conn.pragma_query_value(None, "freelist_count", |row| row.get(0)).context("read freelist_count")?;
        Ok((pages, free))
    }

    /// Fold both text indexes into one segment each, so the delete markers a
    /// dropped body left behind stop taking room. Rewrites the whole index:
    /// `brain compact` only, never a hook or a consolidation.
    ///
    /// # Errors
    /// Returns an error when an index cannot be optimized.
    pub fn optimize_text_indexes(&self) -> Result<()> {
        for table in ["events_fts", "events_tri"] {
            self.conn
                .execute(&format!("INSERT INTO {table}({table}) VALUES ('optimize')"), [])
                .with_context(|| format!("optimize {table}"))?;
        }
        Ok(())
    }

    /// Rewrite the database file without its free pages. Needs the whole file
    /// to itself and room for a second copy; `brain compact` checks both.
    ///
    /// # Errors
    /// Returns an error when the rewrite fails, e.g. a hook held the database
    /// past the busy timeout. The file is unchanged then.
    pub fn vacuum(&self) -> Result<()> {
        self.conn.execute_batch("VACUUM").context("vacuum")
    }

    /// Empty the write-ahead log into the database file and cut it to nothing.
    /// False when a reader kept it from finishing; the next checkpoint will.
    ///
    /// # Errors
    /// Returns an error when the checkpoint itself fails.
    pub fn truncate_wal(&self) -> Result<bool> {
        let busy: i64 = self
            .conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))
            .context("truncate the WAL")?;
        Ok(busy == 0)
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
                       AND e.kind NOT IN ('tombstone', 'retire')
                             AND e.hook NOT IN ('correct', 'feedback', 'supersede')
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

    /// Entries a human has flagged, for the lint page.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn flagged(&self, project: &str) -> Result<Vec<Pointer>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, ts, kind, title, topic FROM events
                 WHERE project = ?1 AND confidence < 0 AND forgotten = 0
                 ORDER BY confidence, id DESC LIMIT 100",
            )
            .context("prepare flagged")?;
        let rows = stmt
            .query_map(params![project], Self::pointer_row)
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
    /// Write down what one rerank did. See `rerank::Outcome`.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_rerank(&self, engine: &str, reason: &str, ms: u64, cold: bool) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO rerank_runs (ts, engine, reason, ms, cold) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    jiff::Timestamp::now().to_string(),
                    engine,
                    reason,
                    i64::try_from(ms).unwrap_or(i64::MAX),
                    i32::from(cold)
                ],
            )
            .context("record rerank")?;
        Ok(())
    }

    /// What reranking has cost, per engine: every run's latency, warm and
    /// cold apart, and the reasons the engine above did not answer.
    ///
    /// Percentiles are left to the caller: the rows are few (one per
    /// reranked search) and a median over them is one sort.
    ///
    /// # Errors
    /// Returns an error when the read fails.
    pub fn rerank_runs(&self) -> Result<Vec<RerankRun>> {
        let mut stmt = self
            .conn
            .prepare("SELECT engine, reason, ms, cold FROM rerank_runs ORDER BY rowid")
            .context("prepare rerank runs")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(RerankRun {
                    engine: row.get(0)?,
                    reason: row.get(1)?,
                    ms: row.get::<_, i64>(2)?.max(0).unsigned_abs(),
                    cold: row.get::<_, i32>(3)? != 0,
                })
            })
            .context("read rerank runs")?
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("read rerank runs")?;
        Ok(rows)
    }

    /// Write down one summarizer model call, whatever came of it.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_summarizer_call(&self, call: &SummarizerCall) -> Result<()> {
        let size = |n: u64| i64::try_from(n).unwrap_or(i64::MAX);
        self.conn
            .execute(
                "INSERT INTO summarizer_calls
                   (ts, session, purpose, cli, model, prompt_bytes, answer_bytes, ms, outcome)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    jiff::Timestamp::now().to_string(),
                    call.session,
                    call.purpose,
                    call.cli,
                    call.model,
                    size(call.prompt_bytes),
                    size(call.answer_bytes),
                    size(call.ms),
                    call.outcome
                ],
            )
            .context("record summarizer call")?;
        Ok(())
    }

    /// Summarizer calls made in the last `secs` seconds, oldest first.
    ///
    /// # Errors
    /// Returns an error when the read fails.
    pub fn summarizer_calls_since(&self, secs: i64) -> Result<Vec<SummarizerCall>> {
        let cutoff = (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(secs)).to_string();
        let mut stmt = self
            .conn
            .prepare(
                "SELECT session, purpose, cli, model, prompt_bytes, answer_bytes, ms, outcome
                 FROM summarizer_calls WHERE ts >= ?1 ORDER BY rowid",
            )
            .context("prepare summarizer calls")?;
        let size = |row: &rusqlite::Row<'_>, i: usize| -> rusqlite::Result<u64> {
            Ok(row.get::<_, i64>(i)?.max(0).unsigned_abs())
        };
        let rows = stmt
            .query_map(params![cutoff], |row| {
                Ok(SummarizerCall {
                    session: row.get(0)?,
                    purpose: row.get(1)?,
                    cli: row.get(2)?,
                    model: row.get(3)?,
                    prompt_bytes: size(row, 4)?,
                    answer_bytes: size(row, 5)?,
                    ms: size(row, 6)?,
                    outcome: row.get(7)?,
                })
            })
            .context("read summarizer calls")?
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("read summarizer calls")?;
        Ok(rows)
    }

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

    /// Of the entries recall has offered, how many were opened.
    ///
    /// Returns `(offered, opened)`. This is the project's only unproxied
    /// relevance signal: a search returns thirty entries and asserts nothing
    /// about which of them answered anything, while an agent that asks for a
    /// body has decided, for its own reasons, that a title was worth the
    /// tokens. Ranking changes have so far been scored against a model's
    /// opinion of a list of titles - a proxy that shares the ranking's blind
    /// spots. Given enough sessions, this does not.
    ///
    /// Counting begins when a brain is upgraded, not at first capture:
    /// everything recalled before then reads as offered-and-never-opened,
    /// because that is all the old rows can say.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn recall_precision(&self) -> Result<(i64, i64)> {
        self.conn
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(opened), 0) FROM recalled",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .context("read recall precision")
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
            .query_row(&Self::stale_backlog_sql(), params![cutoff], |row| row.get(0))
            .optional()
            .context("check stale backlog")?;
        Ok(found.is_some())
    }

    /// The statement behind [`Self::has_stale_backlog`]: list each project
    /// once by skipping through the index, then read only that project's
    /// pending rows through `events_unconsolidated`.
    ///
    /// The plain form, one `SELECT` over `events`, scanned every row: 4 s on
    /// a clean store of 321k rows, inside a hook with a 5 s budget. It was
    /// fast only while an old pending row happened to match early in the
    /// scan. `+e.ts` keeps the planner from walking `events_project_ts` over
    /// every old row of a project instead.
    fn stale_backlog_sql() -> String {
        // A parked session is not work a run can finish, so it must
        // not summon one: that is the loop this guard exists to stop.
        format!(
            "{PROJECTS_CTE}
             SELECT 1 FROM projects
             JOIN events e ON e.project = projects.project AND e.consolidated = 0
             WHERE +e.kind = 'observation' AND +e.ts < ?1
               AND e.session NOT IN ({})
             LIMIT 1",
            Self::parked_sessions_sql()
        )
    }

    /// Take the idle-sweep window: true when the last sweep check is older
    /// than `debounce_secs` before `now`, and the window is spent from here on.
    ///
    /// A plain read answers most stops; only a stop that finds the window open
    /// writes. That write is one conditional upsert, so two stops racing take
    /// it once. The caller spends the window whatever it finds next: a check
    /// that found nothing must not be repeated every turn.
    ///
    /// # Errors
    /// Returns an error when the read or the write fails.
    pub fn claim_idle_sweep(&self, now: jiff::Timestamp, debounce_secs: i64) -> Result<bool> {
        let cutoff = (now - jiff::SignedDuration::from_secs(debounce_secs)).to_string();
        let last: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM schema_state WHERE key = 'last_idle_sweep_at'",
                [],
                |row| row.get(0),
            )
            .optional()
            .context("read the idle sweep window")?;
        if last.is_some_and(|at| at >= cutoff) {
            return Ok(false);
        }
        self.take_idle_sweep(&now.to_string(), &cutoff)
    }

    /// The write half of [`Self::claim_idle_sweep`]: succeeds only while the
    /// stored window is still older than `cutoff`. A stop that read the window
    /// open before another one wrote it loses here, so one window spawns once.
    fn take_idle_sweep(&self, now: &str, cutoff: &str) -> Result<bool> {
        let taken = self
            .conn
            .execute(
                "INSERT INTO schema_state (key, value) VALUES ('last_idle_sweep_at', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value
                 WHERE schema_state.value < ?2",
                params![now, cutoff],
            )
            .context("claim the idle sweep")?;
        Ok(taken > 0)
    }

    /// The sessions set aside: out of attempts, and nothing new to try on.
    ///
    /// A session that has failed [`Self::PARK_AFTER`] times stays parked only
    /// while it has no pending observation newer than its last failed attempt;
    /// an event that arrived afterwards is new work, and gets one more try
    /// (the count stays, so one more failure parks it again).
    fn parked_sessions_sql() -> String {
        format!(
            "SELECT ss.session FROM session_state ss
             WHERE ss.attempts >= {}
               AND NOT EXISTS (
                   SELECT 1 FROM events n
                   WHERE n.session = ss.session AND n.consolidated = 0
                     AND n.kind = 'observation' AND n.ts > ss.last_attempt_at)",
            Self::PARK_AFTER
        )
    }

    /// Timestamp of the oldest observation still waiting for a summary, not
    /// counting a parked session's: what `brain doctor` ages the backlog by.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn oldest_pending_ts(&self) -> Result<Option<String>> {
        self.conn
            .query_row(&Self::oldest_pending_sql(), [], |row| row.get(0))
            .context("read oldest pending event")
    }

    /// The statement behind [`Self::oldest_pending_ts`]; the unary plus is
    /// for the reason given at [`Self::sessions_pending_sql`].
    fn oldest_pending_sql() -> String {
        format!(
            "SELECT MIN(ts) FROM events
             WHERE consolidated = 0 AND +kind = 'observation'
               AND session NOT IN ({})",
            Self::parked_sessions_sql()
        )
    }

    /// Write a consolidation ask down. Done before the run lock is tried.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn add_consolidation_request(
        &self,
        session: Option<&str>,
        all_projects: bool,
        force: bool,
        cwd: &str,
    ) -> Result<i64> {
        self.conn
            .execute(
                "INSERT INTO consolidation_requests
                     (session, all_projects, force, cwd, requested_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![session, all_projects, force, cwd, jiff::Timestamp::now().to_string()],
            )
            .context("record consolidation request")?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Mark an ask taken. `false` when it already was: somebody else holds it.
    ///
    /// One statement, so two takers cannot both win.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn consume_consolidation_request(&self, id: i64) -> Result<bool> {
        let changed = self
            .conn
            .execute(
                "UPDATE consolidation_requests SET consumed_at = ?2
                 WHERE id = ?1 AND consumed_at IS NULL",
                params![id, jiff::Timestamp::now().to_string()],
            )
            .context("consume consolidation request")?;
        if changed != 1 {
            return Ok(false);
        }
        // Identical open asks are one piece of work: this run serves them all,
        // so they must not each cost a drain round and a full pass. They stay
        // behind this one if it is handed back, since it stands for them. An
        // `--all` ask covers every known project from any directory, so its
        // cwd does not make it different work: 51 open `--all` asks from six
        // directories once cost six rounds.
        self.conn
            .execute(
                "UPDATE consolidation_requests SET consumed_at = ?2
                 WHERE consumed_at IS NULL AND id != ?1
                   AND session IS (SELECT session FROM consolidation_requests WHERE id = ?1)
                   AND all_projects = (SELECT all_projects FROM consolidation_requests WHERE id = ?1)
                   AND force = (SELECT force FROM consolidation_requests WHERE id = ?1)
                   AND (all_projects = 1
                        OR cwd = (SELECT cwd FROM consolidation_requests WHERE id = ?1))",
                params![id, jiff::Timestamp::now().to_string()],
            )
            .context("consume identical consolidation requests")?;
        Ok(true)
    }

    /// Delete the open asks that have no work left: a non-force ask for a
    /// session with nothing pending. Returns how many it deleted.
    ///
    /// Such an ask is written after its events are indexed, so a later run
    /// settled them, or the session was settled when it was asked for. Left
    /// open, each one costs a holder a drain round: 572 of 629 open asks were
    /// this kind on one machine. A force ask stays, because it reopens a
    /// settled session, and so does an `--all` ask, which names no session.
    /// A session that gets new events is asked for again at its next
    /// boundary, or by the backstop.
    ///
    /// # Errors
    /// Returns an error when the delete fails.
    pub fn drop_stale_asks(&self) -> Result<usize> {
        self.conn
            .execute(&Self::drop_stale_asks_sql(), [])
            .context("drop stale consolidation requests")
    }

    /// The statement behind [`Self::drop_stale_asks`]: the sessions with
    /// pending work, read by project through `events_unconsolidated`, are
    /// what an ask must name to stay. Looked up per ask through
    /// `events_session`, it read each asked session's events cold: 1.1 s on
    /// one store, under the write lock a hook waits on; this form took
    /// 15 ms. `CROSS JOIN` keeps the projects first: with a plain join
    /// inside the `DELETE`, SQLite built an automatic index on `consolidated`
    /// instead, a scan of the whole table (6.0 s). `+e.kind` keeps any index
    /// on kind out of the plan.
    fn drop_stale_asks_sql() -> String {
        format!(
            "DELETE FROM consolidation_requests
             WHERE consumed_at IS NULL AND session IS NOT NULL AND force = 0
               AND session NOT IN (
                   {PROJECTS_CTE}
                   SELECT e.session FROM projects
                   CROSS JOIN events e ON e.project = projects.project AND e.consolidated = 0
                   WHERE +e.kind = 'observation')"
        )
    }

    /// Whether a session has an observation no run has consolidated yet. One
    /// read through the `events_session` index. A parked session counts: its
    /// events are set aside, not settled, so its ask stays as it did before.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn session_has_pending(&self, session: &str) -> Result<bool> {
        self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM events
                     WHERE session = ?1 AND consolidated = 0 AND kind = 'observation')",
                params![session],
                |row| row.get(0),
            )
            .context("read pending session events")
    }

    /// Hand an ask back, for a run that took it and could not finish it.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn release_consolidation_request(&self, id: i64) -> Result<()> {
        self.conn
            .execute("UPDATE consolidation_requests SET consumed_at = NULL WHERE id = ?1", params![id])
            .context("release consolidation request")?;
        Ok(())
    }

    /// The oldest ask nobody has taken. Never compared against a run's start:
    /// whether it is still open is `consumed_at`, and only that.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn next_consolidation_request(&self) -> Result<Option<ConsolidationRequest>> {
        self.conn
            .query_row(
                "SELECT id, session, all_projects, force, cwd FROM consolidation_requests
                 WHERE consumed_at IS NULL ORDER BY id LIMIT 1",
                [],
                |row| {
                    Ok(ConsolidationRequest {
                        id: row.get(0)?,
                        session: row.get(1)?,
                        all_projects: row.get(2)?,
                        force: row.get(3)?,
                        cwd: row.get(4)?,
                    })
                },
            )
            .optional()
            .context("read consolidation request")
    }

    /// Drop taken asks, and ledger rows, older than `days`.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn purge_consumed_requests(&self, days: i64) -> Result<()> {
        let cutoff =
            (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(days * DAY_SECS)).to_string();
        self.conn
            .execute(
                "DELETE FROM consolidation_requests
                 WHERE consumed_at IS NOT NULL AND consumed_at < ?1",
                params![cutoff],
            )
            .context("purge consolidation requests")?;
        // The runs ledger goes on the same clock: one row per spawned
        // invocation, hook-spawned yields included, would otherwise grow forever.
        self.conn
            .execute("DELETE FROM consolidation_runs WHERE started < ?1", params![cutoff])
            .context("purge consolidation runs")?;
        Ok(())
    }

    /// Record one `brain consolidate` invocation.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_consolidation_run(&self, run: &ConsolidationRun) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO consolidation_runs
                     (started, ended, mode, yielded, sessions, events, failed, rule_based, error)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    run.started,
                    run.ended,
                    run.mode,
                    run.yielded,
                    run.sessions,
                    run.events,
                    run.failed,
                    run.rule_based,
                    run.error.as_deref().map(truncate_error),
                ],
            )
            .context("record consolidation run")?;
        Ok(())
    }

    /// Invocations that began within the last `secs`, oldest first.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn consolidation_runs_since(&self, secs: i64) -> Result<Vec<ConsolidationRun>> {
        let cutoff = (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(secs)).to_string();
        let mut stmt = self
            .conn
            .prepare(
                "SELECT started, ended, mode, yielded, sessions, events, failed, rule_based, error
                 FROM consolidation_runs WHERE started >= ?1 ORDER BY rowid",
            )
            .context("prepare consolidation runs")?;
        let rows = stmt
            .query_map(params![cutoff], |row| {
                Ok(ConsolidationRun {
                    started: row.get(0)?,
                    ended: row.get(1)?,
                    mode: row.get(2)?,
                    yielded: row.get(3)?,
                    sessions: row.get(4)?,
                    events: row.get(5)?,
                    failed: row.get(6)?,
                    rule_based: row.get(7)?,
                    error: row.get(8)?,
                })
            })
            .context("read consolidation runs")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("collect consolidation runs")
    }

    /// Sessions in a project with unconsolidated events, oldest first.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn sessions_pending(&self, project: &str) -> Result<Vec<PendingSession>> {
        let mut stmt = self.conn.prepare(&Self::sessions_pending_sql()).context("prepare pending sessions")?;
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

    /// The statement behind [`Self::unconsolidated_sessions`]. The unary plus
    /// on `kind` keeps the planner off `events_kind_proj`, which would read
    /// every observation of the project to find the few pending ones; the one
    /// on `session` keeps it off `events_recall`, whose order serves the
    /// `GROUP BY` and which then fetched every row of the project.
    fn unconsolidated_sessions_sql() -> &'static str {
        "SELECT session, MIN(cli), COUNT(*), MAX(id), MAX(ts) FROM events
         WHERE project = ?1 AND consolidated = 0 AND forgotten = 0
               AND +kind = 'observation'
               AND hook NOT IN ('correct', 'feedback', 'supersede')
               AND (topic IS NOT NULL
                    OR files != '[]'
                    OR hook IN ('user_prompt_submit', 'before_submit_prompt', 'before_agent')
                    OR (hook = 'stop' AND title != 'Turn finished'))
               -- A one-shot run is not work someone will come back
               -- to; naming it as unfinished sends the next session
               -- to read a reviewer's transcript as the latest work.
               AND session NOT IN (SELECT session FROM session_invocation
                                    WHERE invocation = 'headless')
         GROUP BY +session
         ORDER BY MAX(id) DESC
         LIMIT ?2"
    }

    /// The statement behind [`Self::sessions_pending`]. The unary plus on
    /// `kind` keeps the planner off `events_kind_proj`, which would read every
    /// observation of the project to find the few pending ones; the one on
    /// `session` keeps it off `events_recall`, as at
    /// [`Self::unconsolidated_sessions_sql`].
    fn sessions_pending_sql() -> String {
        format!(
            // The CLI of the newest event, not MAX(cli), which is
            // alphabetical: for a session two CLIs touched, "codex" would
            // beat "claude-code" for no reason but its spelling, and this
            // value decides whose cheap tier gets asked to summarize.
            "SELECT e.session, COUNT(*), MAX(e.id),
                    (SELECT cli FROM events
                      WHERE session = e.session ORDER BY id DESC LIMIT 1)
             FROM events e
             WHERE e.project = ?1 AND e.consolidated = 0 AND +e.kind = 'observation'
               AND e.session NOT IN ({})
             GROUP BY +e.session
             ORDER BY MAX(e.id)",
            Self::parked_sessions_sql()
        )
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
        let transaction = self.conn.unchecked_transaction().context("begin mark consolidated")?;
        for id in ids {
            transaction
                .execute("UPDATE events SET consolidated = 1 WHERE id = ?1", params![id])
                .context("mark consolidated")?;
        }
        transaction.commit().context("commit mark consolidated")?;
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
                     last_tier = excluded.last_tier,
                     attempts = CASE WHEN excluded.last_tier = 'rule-based' THEN attempts ELSE 0 END,
                     last_error = CASE WHEN excluded.last_tier = 'rule-based' THEN last_error ELSE NULL END",
                params![session, project, jiff::Timestamp::now().to_string(), newest_event_id, tier],
            )
            .context("record session run")?;
        Ok(())
    }

    /// A session stops being retried after this many failed attempts in a row.
    pub const PARK_AFTER: i64 = 3;

    /// Count one failed attempt on a session and keep why.
    ///
    /// Not a run: `last_run_at` and the watermark stay as they were, so a
    /// failure never makes a later run think the work was done.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_session_failure(&self, session: &str, project: &str, error: &str) -> Result<()> {
        let now = jiff::Timestamp::now().to_string();
        self.conn
            .execute(
                "INSERT INTO session_state (session, project, attempts, last_error, last_attempt_at)
                 VALUES (?1, ?2, 1, ?3, ?4)
                 ON CONFLICT(session) DO UPDATE SET
                     attempts = attempts + 1,
                     last_error = excluded.last_error,
                     last_attempt_at = excluded.last_attempt_at",
                params![session, project, truncate_error(error), now],
            )
            .context("record session failure")?;
        Ok(())
    }

    /// Give a parked session its attempts back; `--session X --force` is the
    /// one way a person says "try it again".
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn reset_session_attempts(&self, session: &str) -> Result<()> {
        self.conn
            .execute("UPDATE session_state SET attempts = 0 WHERE session = ?1", params![session])
            .context("reset session attempts")?;
        Ok(())
    }

    /// Parked sessions whose last failed attempt is older than `cutoff`: the
    /// ones a daily retry may give their attempts back.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn parked_before(&self, cutoff: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT session FROM session_state
                 WHERE attempts >= ?1 AND last_attempt_at < ?2 ORDER BY session",
            )
            .context("prepare parked sessions")?;
        let rows = stmt
            .query_map(params![Self::PARK_AFTER, cutoff], |row| row.get(0))
            .context("run parked sessions")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read parked sessions")
    }

    /// Sessions that have failed at least `min_attempts` times in a row, worst
    /// first, as (session, attempts, last_error).
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn failing_sessions(&self, min_attempts: i64) -> Result<Vec<(String, i64, String)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT session, attempts, COALESCE(last_error, '') FROM session_state
                 WHERE attempts >= ?1 ORDER BY attempts DESC, session",
            )
            .context("prepare failing sessions")?;
        let rows = stmt
            .query_map(params![min_attempts], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .context("run failing sessions")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read failing sessions")
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
    /// Offers a pointer may spend unread before it stops outranking fresh
    /// material. Five sessions of being pushed and never once pulled is the
    /// budget talking: the line buys nothing where it is.
    const INJECTIONS_BEFORE_DECAY: i64 = 5;

    fn rank(prefix: &str) -> String {
        format!(
            // Knowledge outranks a session summary because it is what
            // survived several of them.
            // {decay} = INJECTIONS_BEFORE_DECAY.
            "CASE {prefix}kind
                 WHEN 'knowledge' THEN 0
                 WHEN 'session_summary' THEN 1
                 WHEN 'source' THEN 1
                 WHEN 'note' THEN 2
                 WHEN 'page_update' THEN 3
                 ELSE 4
             END,
             -- A standing rule first among lessons: it was distilled from
             -- corrections a person made more than once, which is the
             -- strongest evidence this table holds. Only knowledge rows
             -- carry hook = 'rule', so nothing else moves.
             CASE WHEN {prefix}hook = 'rule' THEN 0 ELSE 1 END,
             CASE WHEN {prefix}invocation = 'headless' THEN 1 ELSE 0 END,
             -- Evidence beats heuristics: something an agent went back and
             -- read is worth more than something we merely guessed at, and a
             -- human calling an entry stale outranks both.
             -{prefix}confidence,
             CASE WHEN {prefix}read_count > 0 THEN 0 ELSE 1 END,
             -- Decay on the push side: offered this many sessions and never
             -- once pulled, a pointer stops crowding out fresh lines.
             -- Knowledge is exempt - its worth is the line itself (83% of it
             -- ever surfaced), not a body nobody needs to pull.
             CASE WHEN {prefix}kind != 'knowledge'
                       AND {prefix}injected_count >= {decay}
                       AND {prefix}read_count = 0
                  THEN 1 ELSE 0 END,
             CASE {prefix}topic
                 WHEN 'decision' THEN 0
                 WHEN 'discovery' THEN 1
                 WHEN 'bugfix' THEN 2
                 WHEN 'feature' THEN 3
                 WHEN 'config' THEN 4
                 WHEN 'test' THEN 5
                 ELSE 6
             END",
            decay = Self::INJECTIONS_BEFORE_DECAY
        )
    }

    /// Ranked pointers for the session-start primer's remainder: every kind
    /// that earns a line, so no observation is read only to be dropped.
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
        self.ranked_pointers(project, Kinds::Pushed, limit)
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
        self.ranked_pointers(project, Kinds::One(kind), limit)
    }

    /// The ranked read behind the primer's layers and its remainder.
    ///
    /// The primer asks for one kind per layer and for the pushed set.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub(crate) fn ranked_pointers(&self, project: &str, kinds: Kinds<'_>, limit: usize) -> Result<Vec<Pointer>> {
        let mut stmt = self.conn.prepare(&Self::ranked_pointers_sql(&kinds)).context("prepare primer pointers")?;
        let rows = match kinds {
            Kinds::One(kind) => stmt.query_map(params![project, limit as i64, kind], Self::pointer_row),
            _ => stmt.query_map(params![project, limit as i64], Self::pointer_row),
        }
        .context("run primer pointers")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read primer pointers")
    }

    /// The statement behind [`Self::ranked_pointers`].
    ///
    /// The kind is a plain `kind = ?3` or a fixed `IN` list, never
    /// `(?3 IS NULL OR kind = ?3)`: that form let no index on `kind` serve
    /// the read, so each of the primer's three queries walked every row of
    /// the project (115k of them on the largest) and sorted them, 4.8 to
    /// 6.0 s cold apiece. The same fixed list is what the primer keeps
    /// anyway; observations never earn a line of their own.
    fn ranked_pointers_sql(kinds: &Kinds<'_>) -> String {
        let kind = match kinds {
            #[cfg(test)]
            Kinds::Any => "",
            Kinds::One(_) => "AND kind = ?3",
            Kinds::Pushed => "AND kind IN ('knowledge', 'session_summary', 'source', 'note', 'page_update')",
        };
        format!(
            // The same floor every other read enforces. Injection is recall
            // too - and the costlier half, because it spends bytes in every
            // future session whether or not anyone asked. A withdrawn memory
            // that only disappears from search has not been withdrawn.
            "SELECT id, ts, kind, title, topic FROM events
             WHERE project = ?1 AND forgotten = 0 AND kind NOT IN ('tombstone', 'retire')
                   AND hook NOT IN ('correct', 'feedback', 'supersede')
                   {kind}
             ORDER BY {}, id DESC
             LIMIT ?2",
            Self::rank("")
        )
    }

    /// Sessions whose captures nothing has summarized yet, newest first.
    ///
    /// One row per session, not one per capture. The primer used to push the
    /// newest unsummarized captures themselves, four or five lines at a time,
    /// and a real store showed what that bought: of 476 such lines, two were
    /// ever opened. Forty percent were the reader's own captures echoed back
    /// after a compaction, and half the rest were a finished session the
    /// backstop had not reached - not work killed mid-task at all.
    ///
    /// What the next session needs is not the captures, it is to know that a
    /// session is still unsummarized and how to read it. One line carries
    /// that; `brain_recent` with the session id carries the rest, on demand.
    ///
    /// A session counts only through captures that mean something - a
    /// prompt, an answer, a file touched, a classification - so a session
    /// that only opened and ran a few commands is not "in flight". The
    /// newest such capture's id names the line, which is what dedup and
    /// uptake are keyed by: a resume shows the line again only when new work
    /// arrived, and the line counts as pulled when that session is read.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn unconsolidated_sessions(&self, project: &str, limit: usize) -> Result<Vec<InFlight>> {
        let mut stmt = self
            .conn
            .prepare(Self::unconsolidated_sessions_sql())
            .context("prepare unconsolidated sessions")?;
        let rows = stmt
            .query_map(params![project, limit as i64], |row| {
                Ok(InFlight {
                    session: row.get(0)?,
                    cli: row.get(1)?,
                    captures: row.get(2)?,
                    newest_id: row.get(3)?,
                    newest_ts: row.get(4)?,
                })
            })
            .context("run unconsolidated sessions")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read unconsolidated sessions")
    }

    /// The newest summary another session wrote, with the CLI that wrote it.
    ///
    /// The hand-off line's source: not a ranking, just "what did the last
    /// person here finish". A one-shot (headless) run is skipped for the
    /// reason `rank` demotes it, and nothing older than `max_age` counts - a
    /// hand-off from last week is history, not a hand-off. The CLI rides
    /// beside the pointer rather than in it so no other read pays for it.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn newest_summary(
        &self,
        project: &str,
        exclude_session: &str,
        max_age: std::time::Duration,
    ) -> Result<Option<(Pointer, String)>> {
        let age = jiff::SignedDuration::try_from(max_age).context("hand-off age out of range")?;
        let cutoff = jiff::Timestamp::now()
            .checked_sub(age)
            .context("hand-off cutoff out of range")?
            .to_string();
        self.conn
            .query_row(
                Self::newest_summary_sql(),
                params![project, exclude_session, cutoff],
                |row| Ok((Self::pointer_row(row)?, row.get::<_, String>(5)?)),
            )
            .optional()
            .context("read newest summary")
    }

    /// The statement behind [`Self::newest_summary`]. With `events_kind_proj`
    /// built, `ORDER BY id DESC LIMIT 1` walks that index from the newest end;
    /// without it, it sorts every summary of the project.
    fn newest_summary_sql() -> &'static str {
        "SELECT id, ts, kind, title, topic, cli FROM events
         WHERE project = ?1 AND kind = 'session_summary' AND forgotten = 0
               AND session != ?2
               AND (invocation IS NULL OR invocation != 'headless')
               -- A summary event never carries `invocation`; the run
               -- is recorded per session, as `unconsolidated_sessions`
               -- reads it.
               AND session NOT IN (SELECT session FROM session_invocation
                                    WHERE invocation = 'headless')
               -- The unary plus keeps the planner off `events_project_ts`, whose
               -- range on `ts` looks cheaper than it is and then sorts.
               AND +ts >= ?3
         ORDER BY id DESC LIMIT 1"
    }

    /// The CLI a session was captured through, from its first capture.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn session_cli(&self, session: &str) -> Result<Option<String>> {
        self.conn
            .query_row(Self::session_cli_sql(), params![session], |row| row.get(0))
            .optional()
            .context("read session cli")
    }

    /// The statement behind [`Self::session_cli`]. `events_session_id` makes
    /// it one probe; through `events_session` it read every row of the
    /// session to find the smallest id, 1.4 s cold for a 22k-event session.
    fn session_cli_sql() -> &'static str {
        "SELECT cli FROM events WHERE session = ?1 ORDER BY id LIMIT 1"
    }

    /// Build the indexes the primer's and search's reads need, once.
    ///
    /// `events(kind, project, id)` serves the per-kind layers and the hand-off
    /// line; `events(session, id)` serves `session_cli`. It covers
    /// `events(session)` for every lookup by session, but that index stays: a
    /// 0.64.0 binary on the same store would recreate it on its next open,
    /// 4.7 s cold under the write lock, and its SessionStart took 7.4 to 8.6 s.
    /// `events_recall` holds every column of the recall floor, so the reads
    /// that apply it to a project's events (`nearest`, `entity_matches`,
    /// `neighbours_of`, `related`) filter from the index instead of fetching
    /// each row; it is 44 MB on a 327k-event store. The builds cost several
    /// seconds cold on a store of millions of rows, and the write lock for as
    /// long, so this runs only in a consolidate run, under the run lock, never
    /// in `open` or a hook. A hook that cannot index meanwhile leaves its
    /// event to the log catch-up. Until it has run the reads stay correct,
    /// just slower: nothing names these indexes.
    ///
    /// Returns whether it built anything.
    ///
    /// # Errors
    /// Returns an error when a statement fails; the transaction rolls back and
    /// the next run tries again.
    pub fn build_primer_indexes(&self) -> Result<bool> {
        let built: bool = self
            .conn
            .query_row("SELECT EXISTS(SELECT 1 FROM schema_state WHERE key = 'primer_indexes_built')", [], |row| {
                row.get(0)
            })
            .context("check the primer indexes")?;
        if built {
            return Ok(false);
        }
        let transaction = self.conn.unchecked_transaction().context("begin primer indexes")?;
        transaction
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS events_kind_proj ON events(kind, project, id);
                 CREATE INDEX IF NOT EXISTS events_session_id ON events(session, id);
                 CREATE INDEX IF NOT EXISTS events_recall
                     ON events(project, session, forgotten, kind, hook, topic, confidence, id);
                 INSERT OR REPLACE INTO schema_state (key, value) VALUES ('primer_indexes_built', '1');",
            )
            .context("build primer indexes")?;
        transaction.commit().context("commit primer indexes")?;
        Ok(true)
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
            "SELECT e.id, e.ts, e.kind, e.title, e.topic
             FROM event_files f
             JOIN events e ON e.id = f.event_id
             WHERE f.project = ?1 AND f.path = ?2 AND e.forgotten = 0
                   AND e.kind NOT IN ('tombstone', 'retire')
                         AND e.hook NOT IN ('correct', 'feedback', 'supersede')
                   AND NOT (e.kind = 'observation'
                            AND (e.title LIKE 'Read:%' OR e.title LIKE 'Grep:%' OR e.title LIKE 'Glob:%'))
             ORDER BY {}, e.id DESC
             LIMIT ?3",
            Self::rank("e.")
        );
        let mut stmt = self.conn.prepare(&sql).context("prepare file pointers")?;
        let rows = stmt
            .query_map(params![project, path, limit as i64], Self::pointer_row)
            .context("run file pointers")?;
        rows.collect::<rusqlite::Result<Vec<_>>>().context("read file pointers")
    }

    /// Up to `limit` durable entries (knowledge, summaries, notes) that a task
    /// prompt's words reach, for the prompt-time push.
    ///
    /// Lexical only: the raw FTS query first, then the relaxed one
    /// ([`compile_fts`]), the kind filter in SQL. No vector, no entity, no
    /// model - this runs on every task prompt, inside the hook's budget, so it
    /// is held to [`PROMPT_POINTER_DEADLINE`] of wall clock. A query still
    /// running at the deadline is interrupted and the call is an error: the
    /// caller injects nothing rather than something late.
    ///
    /// # Errors
    /// Returns an error when the query fails or is interrupted at the deadline.
    pub fn prompt_pointers(&self, project: &str, query: &str, limit: usize) -> Result<Vec<Pointer>> {
        self.prompt_pointers_within(project, query, limit, PROMPT_POINTER_DEADLINE)
    }

    fn prompt_pointers_within(
        &self,
        project: &str,
        query: &str,
        limit: usize,
        deadline: std::time::Duration,
    ) -> Result<Vec<Pointer>> {
        self.within(deadline, || self.prompt_pointer_rows(project, query, limit))
    }

    /// Run `work` against this connection, interrupting it at `deadline`.
    ///
    /// The interrupt surfaces as an error from `work`, never as a partial
    /// answer. When the timer thread cannot start there is no guard, so
    /// `work` does not run and the call is an error.
    fn within<T>(&self, deadline: std::time::Duration, work: impl FnOnce() -> Result<T>) -> Result<T> {
        use std::sync::mpsc::{channel, RecvTimeoutError};
        let handle = self.conn.get_interrupt_handle();
        let (finished, waiting) = channel::<()>();
        // Cancelable: dropping `finished` wakes the timer at once, so a query
        // that completes never leaves a thread sleeping out the deadline, and
        // never gets interrupted after the fact.
        let timer = std::thread::Builder::new()
            .spawn(move || {
                if waiting.recv_timeout(deadline) == Err(RecvTimeoutError::Timeout) {
                    handle.interrupt();
                }
            })
            .context("start the lookup deadline timer")?;
        let result = work();
        drop(finished);
        let _ = timer.join();
        result
    }

    fn prompt_pointer_rows(&self, project: &str, query: &str, limit: usize) -> Result<Vec<Pointer>> {
        // A prompt can be a page long; its opening says what the task is, and
        // a bounded query is a bounded cost.
        let query: String = query.chars().take(PROMPT_QUERY_CHARS).collect();
        let mut stmt = self
            .conn
            .prepare(
                // Confidence below zero is a flagged entry: never pushed.
                "SELECT e.id, e.ts, e.kind, e.title, e.topic
                 FROM events_fts
                 JOIN events e ON e.rowid = events_fts.rowid
                 WHERE events_fts MATCH ?1 AND e.project = ?2 AND e.forgotten = 0
                       AND e.kind IN ('knowledge', 'session_summary', 'note')
                       AND e.hook NOT IN ('correct', 'feedback', 'supersede')
                       AND COALESCE(e.confidence, 0) >= 0
                 ORDER BY rank
                 LIMIT ?3",
            )
            .context("prepare prompt pointers")?;
        let mut read = |fts: &str| -> rusqlite::Result<Vec<Pointer>> {
            stmt.query_map(params![fts, project, limit as i64], Self::pointer_row)?.collect()
        };
        let mut pointers = raw_or_phrase(read(&query), || read(&format!("\"{}\"", query.replace('"', " "))))?;
        if let Some(fts) = compile_fts(&relaxable(&query)) {
            for pointer in read(&fts).context("read relaxed prompt pointers")? {
                if !pointers.iter().any(|seen| seen.id == pointer.id) {
                    pointers.push(pointer);
                }
            }
        }
        pointers.truncate(limit);
        Ok(pointers)
    }

    /// The `id, ts, kind, title, topic` columns as a [`Pointer`].
    fn pointer_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Pointer> {
        Ok(Pointer {
            id: row.get(0)?,
            ts: row.get(1)?,
            kind: row.get(2)?,
            title: row.get(3)?,
            topic: row.get(4)?,
        })
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
                 WHERE project = ?1 AND ts >= ?2 AND forgotten = 0 AND kind NOT IN ('tombstone', 'retire')
                       AND hook NOT IN ('correct', 'feedback', 'supersede')
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

    /// Record what an injection spent, if the session can still afford it.
    ///
    /// Returns `false`, recording nothing, when `bytes` would take the
    /// session past `cap`. Every caller checked the budget before building
    /// its text, but hooks run as separate processes and a burst of tool
    /// calls arrives at once: each read the same remaining budget, each spent
    /// it, and the session ended over its ceiling. The check that binds is
    /// this one statement, which SQLite runs atomically.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn record_injected(
        &self,
        session: &str,
        ids: &[String],
        in_flight: usize,
        bytes: usize,
        cap: usize,
    ) -> Result<bool> {
        let reserved = self
            .conn
            .execute(
                "INSERT INTO injected_bytes (session, bytes) SELECT ?1, ?2 WHERE ?2 <= ?3
                 ON CONFLICT(session) DO UPDATE SET bytes = injected_bytes.bytes + ?2
                 WHERE injected_bytes.bytes + ?2 <= ?3",
                params![
                    session,
                    i64::try_from(bytes).unwrap_or(i64::MAX),
                    i64::try_from(cap).unwrap_or(i64::MAX)
                ],
            )
            .context("record injected bytes")?;
        if reserved == 0 {
            return Ok(false);
        }
        for (at, id) in ids.iter().enumerate() {
            self.note_injected_id(session, id, at < in_flight)?;
        }
        Ok(true)
    }

    /// One pointer a session was shown: counted once per session, matching
    /// read_count - re-pushing the same pointer inside one conversation says
    /// nothing new about how often it gets offered.
    fn note_injected_id(&self, session: &str, id: &str, in_flight: bool) -> Result<()> {
        let seen: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM injected WHERE session = ?1 AND event_id = ?2",
                params![session, id],
                |row| row.get(0),
            )
            .optional()
            .context("read injected row")?;
        if seen.is_none() {
            self.conn
                .execute(
                    "UPDATE events SET injected_count = injected_count + 1 WHERE id = ?1",
                    params![id],
                )
                .context("count injection")?;
        }
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
                params![session, id, i64::from(in_flight)],
            )
            .context("record injected id")?;
        Ok(())
    }

    /// Fold what the hot paths spilled to `surfaced.jsonl` into the ledger.
    /// Idempotent - a row already there is left alone - so a fold that died
    /// before its file was removed can simply run again. Returns the ids seen.
    ///
    /// # Errors
    /// Returns an error when the writes fail; nothing is then applied.
    pub fn fold_surfaced(&self, lines: impl Iterator<Item = SurfacedLine>) -> Result<usize> {
        let transaction = self.conn.unchecked_transaction().context("begin fold surfaced")?;
        let mut seen = 0;
        for line in lines {
            for id in &line.ids {
                match line.kind {
                    Ledger::Recalled => {
                        self.record_recall(&line.session, std::iter::once(id.as_str()), Reach::Offered)?;
                    }
                    Ledger::Opened => {
                        self.record_recall(&line.session, std::iter::once(id.as_str()), Reach::Opened)?;
                    }
                    Ledger::Injected => self.note_injected_id(&line.session, id, false)?,
                    Ledger::File => self.record_injected_file(&line.session, id)?,
                }
                seen += 1;
            }
        }
        transaction.commit().context("commit fold surfaced")?;
        Ok(seen)
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

    /// Whether the index holds this event id. One primary-key probe.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn has_event(&self, id: &str) -> Result<bool> {
        self.conn
            .query_row("SELECT EXISTS(SELECT 1 FROM events WHERE id = ?1)", params![id], |row| row.get(0))
            .context("probe event id")
    }

    /// How far the log-tail catch-up has read one log file, under a key naming
    /// the file. The value is `<end>:<last line start>:<fingerprint>`; the
    /// catch-up's own doc says what each part is for.
    ///
    /// # Errors
    /// Returns an error when the query fails.
    pub fn log_watermark(&self, key: &str) -> Result<Option<String>> {
        self.state(key).context("read log watermark")
    }

    /// Record where the catch-up stopped in one log file.
    ///
    /// # Errors
    /// Returns an error when the write fails.
    pub fn set_log_watermark(&self, key: &str, value: &str) -> Result<()> {
        self.set_state(key, value).context("write log watermark")
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
            // be recomputed by replaying events - and the `summarizer_calls`
            // and `rerank_runs` ledgers, which record spend that already
            // happened. Clearing those would not rebuild them; it would
            // delete them.
            // The log-tail watermarks go too: they say what the index holds,
            // so after a clear that fails part-way they must not claim a log
            // is fully read. So do the retention marks: the replay restores
            // every body, and a cursor or a spent day left behind would hide
            // them from the next pass. The compact marks go with them: they
            // measure against the retention counters, and a counter that
            // restarts at zero beside a larger one would read as negative.
            .execute_batch(
                "DELETE FROM event_files; DELETE FROM events;
                 DELETE FROM schema_state WHERE key LIKE 'log_tail:%' OR key LIKE 'retention\\_%' ESCAPE '\\'
                    OR key LIKE 'compact\\_%' ESCAPE '\\';",
            )
            .context("clear index")?;
        Ok(())
    }

    /// Mark settled again what a replay reopened: each quiet or headless
    /// session's observations up to the event its verdict covered. Returns how
    /// many events it marked.
    ///
    /// Those two verdicts are written only to `session_state`, never to the
    /// log, and [`Self::clear`] keeps that table. A replay sets every flag to
    /// the log's, which is unconsolidated, and `should_wait` then finds the
    /// session's newest event already covered by its last run and passes over
    /// it on every run after: pending for good, and old enough to keep the
    /// backstop asking. A model run is not affected, because its summary's
    /// links set its flags back as the log is read.
    ///
    /// This restores the verdict exactly, with no model and no page. Events
    /// newer than the verdict stay pending: they are work.
    ///
    /// # Errors
    /// Returns an error when the update fails.
    pub fn resettle_replayed(&self) -> Result<usize> {
        self.conn
            .execute(&Self::resettle_replayed_sql(), [])
            .context("resettle replayed sessions")
    }

    /// The statement behind [`Self::resettle_replayed`]: pending rows, read
    /// by project through `events_unconsolidated`, then kept when their
    /// session's verdict is quiet or headless. Read by session through
    /// `events_session`, it walked every event of the 3,539 quiet and
    /// headless sessions on one store cold: 3.3 s, under the write lock a
    /// hook waits on for 5 s; this form took 90 ms. `CROSS JOIN` keeps the
    /// projects first, and `+e.kind` keeps any index on kind out of the plan.
    fn resettle_replayed_sql() -> String {
        format!(
            "UPDATE events SET consolidated = 1 WHERE id IN (
                 {PROJECTS_CTE}
                 SELECT e.id FROM projects
                 CROSS JOIN events e ON e.project = projects.project AND e.consolidated = 0
                 JOIN session_state ss ON ss.session = e.session
                 WHERE ss.last_tier IN ('quiet', 'headless')
                   AND +e.kind = 'observation' AND e.id <= ss.last_event_id)"
        )
    }

    /// [`Self::resettle_replayed`], once per store, for a store a `reindex`
    /// stranded before that command settled its own replay. Returns 0 once
    /// `replayed_settled` is recorded, which makes every later call a single
    /// indexed read.
    ///
    /// Never run from `open`: the update holds the write lock that hooks
    /// wait on. The run that holds the consolidation lock calls it instead.
    ///
    /// # Errors
    /// Returns an error when a query fails.
    pub fn resettle_replayed_once(&self) -> Result<usize> {
        let done: bool = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_state WHERE key = 'replayed_settled')",
                [],
                |row| row.get(0),
            )
            .context("check replayed sessions")?;
        if done {
            return Ok(0);
        }
        let settled = self.resettle_replayed()?;
        self.conn
            .execute("INSERT OR IGNORE INTO schema_state (key, value) VALUES ('replayed_settled', '1')", [])
            .context("mark replayed sessions settled")?;
        Ok(settled)
    }
}

/// A memory pointer: everything an injection may carry, and nothing more.
/// A session with captures nothing has summarized yet - see
/// [`Store::unconsolidated_sessions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InFlight {
    pub session: String,
    pub cli: String,
    /// Captures that count: prompts, answers, file touches, classified work.
    pub captures: i64,
    /// The newest of them; the id the primer line is keyed by.
    pub newest_id: String,
    pub newest_ts: String,
}

/// Which kinds a ranked pointer read covers.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Kinds<'a> {
    /// Every kind; only the ranking tests ask for it.
    #[cfg(test)]
    Any,
    One(&'a str),
    /// The kinds that earn a primer line of their own: everything but
    /// observations (and the bookkeeping kinds no read shows).
    Pushed,
}

#[derive(Debug, Clone)]
pub struct Pointer {
    pub id: String,
    pub ts: String,
    pub kind: String,
    pub title: String,
    /// What it is about, when something classified it.
    pub topic: Option<String>,
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

/// What the index knows about its own compactions, with the page figures that
/// say how much a rewrite would give back. Read by [`Store::compact_state`];
/// [`Store::record_compact`] writes back the fields that are set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactState {
    /// RFC 3339 time of the last finished compact.
    pub done_at: Option<String>,
    /// Unix seconds of the last attempt that got as far as the disk guard.
    pub tried_at: Option<i64>,
    /// `<reason>@<RFC 3339>` of the last attempt that stood aside or failed.
    pub skip: Option<String>,
    /// A retention pass is part-way: its dropped counters are still moving.
    pub retention_pending: bool,
    pub retention_dropped: u64,
    pub retention_done_at: Option<String>,
    /// Bytes of body text retention has dropped, and how much of that a compact
    /// has already settled.
    pub retention_dropped_bytes: u64,
    pub compact_dropped_bytes: u64,
    /// The database file's size by page count, and the part on the free list.
    pub file_bytes: u64,
    pub free_bytes: u64,
    /// Sizes around a finished compact, for [`Store::record_compact`].
    pub before_bytes: Option<u64>,
    pub after_bytes: Option<u64>,
}

/// An ask to consolidate, as written before the run lock was tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationRequest {
    pub id: i64,
    pub session: Option<String>,
    pub all_projects: bool,
    pub force: bool,
    /// The asker's directory, from which its project is resolved.
    pub cwd: String,
}

/// One `brain consolidate` invocation, for the runs ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationRun {
    pub started: String,
    pub ended: String,
    pub mode: String,
    pub yielded: bool,
    pub sessions: i64,
    pub events: i64,
    pub failed: i64,
    pub rule_based: i64,
    pub error: Option<String>,
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

/// Largest observation body the index keeps. The log keeps the whole line.
const INDEX_BODY_MAX: usize = 4096;

/// Leaf ceilings tried in turn when the body is a JSON object.
const INDEX_LEAF_CEILINGS: [usize; 3] = [2048, 1024, 512];

/// The body as the index stores it, and whether it was cut.
///
/// Only an observation is cut: a note, summary or lesson is written to be
/// read whole. A JSON body (a tool call) shrinks leaf by leaf, so it stays
/// parseable and the result keeps its tail; anything else, or a body whose
/// structure alone is too big, is cut head and tail. Deterministic, so a
/// reindex reproduces the same row.
fn clamp_for_index(event: &Event) -> (String, bool) {
    if event.kind != EventKind::Observation || event.body.len() <= INDEX_BODY_MAX {
        return (event.body.clone(), false);
    }
    if let Ok(value @ serde_json::Value::Object(_)) = serde_json::from_str(&event.body) {
        for ceiling in INDEX_LEAF_CEILINGS {
            let mut shrunk = value.clone();
            clamp_leaves(&mut shrunk, ceiling);
            if let Ok(text) = serde_json::to_string(&shrunk) {
                if text.len() <= INDEX_BODY_MAX {
                    return (text, true);
                }
            }
        }
    }
    (crate::sanitize::truncate_head_tail(&event.body, INDEX_BODY_MAX), true)
}

fn clamp_leaves(value: &mut serde_json::Value, max: usize) {
    match value {
        serde_json::Value::String(text) => *text = crate::sanitize::clamp_leaf(text, max),
        serde_json::Value::Array(items) => items.iter_mut().for_each(|v| clamp_leaves(v, max)),
        serde_json::Value::Object(map) => map.values_mut().for_each(|v| clamp_leaves(v, max)),
        _ => {}
    }
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
        "retire" => EventKind::Retire,
        "source" => EventKind::Source,
        _ => EventKind::Observation,
    }
}

#[cfg(test)]
mod tests {
    /// A statement that cannot finish by itself.
    const ENDLESS: &str =
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c";

    #[test]
    fn a_lookup_still_running_at_its_deadline_is_an_error() {
        let store = Store::open_memory().unwrap();
        let ran = store.within(std::time::Duration::from_millis(1), || {
            let n: i64 = store.conn.query_row(ENDLESS, [], |row| row.get(0))?;
            Ok(n)
        });
        assert!(ran.is_err(), "an endless statement came back: {ran:?}");
    }

    #[test]
    fn an_interrupt_in_the_first_read_is_not_retried_as_a_phrase() {
        let store = Store::open_memory().unwrap();
        let interrupted = store
            .within(std::time::Duration::from_millis(1), || {
                store.conn.query_row(ENDLESS, [], |row| row.get::<_, i64>(0)).map_err(Into::into)
            })
            .unwrap_err();
        let error = interrupted.downcast::<rusqlite::Error>().expect("the SQLite error survives");
        let mut phrase_ran = false;
        let found = raw_or_phrase(Err(error), || {
            phrase_ran = true;
            Ok(Vec::new())
        });
        assert!(found.is_err(), "an interrupted lookup returned pointers");
        assert!(!phrase_ran, "the phrase read ran after the deadline");

        // A real parse error still falls back.
        let syntax = store
            .conn
            .query_row("SELECT * FROM events_fts WHERE events_fts MATCH 'a\"'", [], |row| row.get::<_, i64>(0))
            .unwrap_err();
        let found = raw_or_phrase(Err(syntax), || {
            phrase_ran = true;
            Ok(Vec::new())
        });
        assert!(found.is_ok() && phrase_ran, "a parse error must fall back to the phrase");
    }

    #[test]
    fn quoted_error_text_and_markdown_bold_still_get_the_relaxed_read() {
        let store = Store::open_memory().unwrap();
        let project = uuid::Uuid::from_u128(9);
        let mut entry = Event::new(
            uuid::Uuid::nil(),
            project,
            uuid::Uuid::nil(),
            Source { cli: "claude-code".into(), hook: "consolidate".into() },
            EventKind::Knowledge,
            "Retry backoff jitter is capped at thirty seconds".into(),
            String::new(),
        );
        entry.consolidated = true;
        store.index(&entry).unwrap();
        for prompt in [
            "error: \"retry\" backoff **jitter** capped",
            "retry AND backoff OR jitter NOT capped extra",
        ] {
            let found = store.prompt_pointers(&project.to_string(), prompt, 3).unwrap();
            assert_eq!(found.len(), 1, "{prompt:?} lost the relaxed read");
        }
    }

    /// Two stops can both read the window open; only the first write may win.
    /// The second handle arrives at the write after the first has claimed.
    #[test]
    fn a_stop_that_read_the_window_open_loses_to_the_one_that_wrote_first() {
        let dir = std::env::temp_dir().join(format!("brain-sweep-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("brain.db");
        let (a, b) = (Store::open(&db).unwrap(), Store::open(&db).unwrap());
        let now = jiff::Timestamp::now();
        let cutoff = (now - jiff::SignedDuration::from_secs(900)).to_string();
        assert!(a.claim_idle_sweep(now, 900).unwrap(), "the first stop claims the window");
        // `b` read before `a` wrote, so its read said open; its write must fail.
        assert!(!b.take_idle_sweep(&now.to_string(), &cutoff).unwrap(), "the window was claimed twice");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A rule-based run is the fall the drain counts as a failed attempt, right
    /// after `record_session_run`; if that run reset the counter the session
    /// would sit at 1 forever and never be parked.
    #[test]
    fn a_rule_based_run_does_not_reset_attempts_but_a_model_run_does() {
        let store = Store::open_memory().unwrap();
        for round in 1..=3 {
            store.record_session_run("s1", "p1", "01A", "rule-based").unwrap();
            store.record_session_failure("s1", "p1", "fell to rule-based").unwrap();
            assert_eq!(store.failing_sessions(1).unwrap()[0].1, round);
        }
        store.record_session_run("s1", "p1", "01A", "claude-code").unwrap();
        assert!(store.failing_sessions(1).unwrap().is_empty());
    }

    /// Parked means "no new work": an event that arrives after the last failed
    /// attempt gives the session another try, in all three readers.
    #[test]
    fn a_parked_session_is_pending_again_once_a_newer_event_arrives() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let session = Uuid::new_v4();
        let mut old = event("Edit: a.rs", "{}", project);
        old.session = session;
        old.ts = (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(3 * 3600)).to_string();
        store.index(&old).unwrap();
        for _ in 0..Store::PARK_AFTER {
            store.record_session_failure(&session.to_string(), &project.to_string(), "boom").unwrap();
        }
        let key = project.to_string();
        assert!(store.sessions_pending(&key).unwrap().is_empty(), "parked and nothing new");
        assert!(!store.has_stale_backlog(0).unwrap());
        assert_eq!(store.oldest_pending_ts().unwrap(), None);

        let mut fresh = event("Edit: b.rs", "{}", project);
        fresh.session = session;
        fresh.ts = (jiff::Timestamp::now() + jiff::SignedDuration::from_secs(60)).to_string();
        store.index(&fresh).unwrap();
        assert_eq!(store.sessions_pending(&key).unwrap().len(), 1, "a newer event must reopen it");
        assert!(store.has_stale_backlog(0).unwrap());
        assert_eq!(store.oldest_pending_ts().unwrap(), Some(old.ts.clone()));

        // One more failure parks it again: the count kept going.
        store.record_session_failure(&session.to_string(), &key, "boom").unwrap();
        assert_eq!(store.failing_sessions(1).unwrap()[0].1, Store::PARK_AFTER + 1);
    }

    /// The backstop runs inside every `session_start` hook. On a clean store
    /// a plan that scans `events` reads every row before it can say no, so the
    /// question must reach pending rows through their own index.
    #[test]
    fn the_backstop_question_reads_only_pending_rows() {
        let store = Store::open_memory().unwrap();
        let (mut a, mut b) = (Uuid::new_v4(), Uuid::new_v4());
        if a.to_string() > b.to_string() {
            std::mem::swap(&mut a, &mut b);
        }
        let old = (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(2 * 3600)).to_string();
        let mut done = Vec::new();
        for project in [a, b] {
            for i in 0..50 {
                let mut settled = event(&format!("Edit: {i}.rs"), "{}", project);
                settled.ts.clone_from(&old);
                store.index(&settled).unwrap();
                done.push(settled.id);
            }
        }
        store.mark_consolidated(&done).unwrap();
        assert!(!store.has_stale_backlog(900).unwrap(), "every row is settled");

        // The one pending row sits in the project that sorts last.
        let mut pending = event("Edit: late.rs", "{}", b);
        pending.ts = old;
        store.index(&pending).unwrap();
        assert!(store.has_stale_backlog(900).unwrap(), "the pending row in B was missed");
        store.mark_consolidated(std::slice::from_ref(&pending.id)).unwrap();
        assert!(!store.has_stale_backlog(900).unwrap());

        // The answers above hold on a full scan too; the plan is what guards the cost.
        let plan: Vec<String> = store
            .conn
            .prepare(&format!("EXPLAIN QUERY PLAN {}", Store::stale_backlog_sql()))
            .unwrap()
            .query_map(params!["now"], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert!(
            plan.iter().any(|row| row.starts_with("SEARCH e ") && row.contains("events_unconsolidated")),
            "pending rows are not read through events_unconsolidated: {plan:#?}"
        );
        assert!(
            !plan.iter().any(|row| matches!(
                row.strip_prefix("SCAN ").and_then(|rest| rest.split_whitespace().next()),
                Some("events" | "e")
            )),
            "the backstop scans the events table: {plan:#?}"
        );
    }

    /// A rebuild replays every line with its flag as the log has it: cleared.
    /// A model run gets its flag back from its summary's links; a quiet or
    /// headless verdict lives only in `session_state`, which the rebuild
    /// keeps, so those sessions came back pending and every later run passed
    /// over them as already done.
    #[test]
    fn a_rebuild_keeps_quiet_and_headless_sessions_settled() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let key = project.to_string();
        let (s1, s2, s3, s4) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        store.record_session_invocation(&s2.to_string(), "headless").unwrap();
        // Ids in minting order, one millisecond apart, so "newer" is not left
        // to the random half of a ULID.
        let start = u64::try_from(jiff::Timestamp::now().as_millisecond()).unwrap() - 60_000;
        let mut minted = 0;
        let mut mint = |session: Uuid| {
            minted += 1;
            let mut event = event_in_session("Bash: ls", "{}", project, session);
            event.id = ulid::Ulid::from_parts(start + minted, 0).to_string();
            event
        };
        let mut log = Vec::new();
        for (session, tier) in [(s1, "quiet"), (s2, "headless"), (s3, "rule-based"), (s4, "quiet")] {
            let events: Vec<Event> = (0..3).map(|_| mint(session)).collect();
            let ids: Vec<String> = events.iter().map(|event| event.id.clone()).collect();
            for event in &events {
                store.index(event).unwrap();
            }
            // A rule-based floor leaves its events pending on purpose.
            if tier != "rule-based" {
                store.mark_consolidated(&ids).unwrap();
            }
            store.record_session_run(&session.to_string(), &key, ids.last().unwrap(), tier).unwrap();
            log.extend(events);
        }
        // s4 went on working after its quiet verdict.
        let later = mint(s4);
        store.index(&later).unwrap();
        log.push(later);

        // What `reindex` does: the same lines, each unconsolidated in the log.
        store.clear().unwrap();
        for event in &log {
            assert!(!event.consolidated);
            store.index(event).unwrap();
        }
        let pending = |store: &Store| {
            let mut found: Vec<(String, i64)> =
                store.sessions_pending(&key).unwrap().into_iter().map(|p| (p.session, p.pending)).collect();
            found.sort();
            found
        };
        assert_eq!(pending(&store).len(), 4, "the replay did not reopen the settled sessions");

        assert_eq!(store.resettle_replayed().unwrap(), 9, "3 events each for s1, s2 and s4");
        let mut left = vec![(s3.to_string(), 3), (s4.to_string(), 1)];
        left.sort();
        assert_eq!(pending(&store), left, "only the floor and s4's newer event are work");
    }

    #[test]
    fn a_request_stays_open_until_taken_and_is_taken_once() {
        let store = Store::open_memory().unwrap();
        let first = store.add_consolidation_request(Some("s"), false, true, "/w").unwrap();
        let second = store.add_consolidation_request(None, true, false, "/v").unwrap();
        let next = store.next_consolidation_request().unwrap().unwrap();
        assert_eq!(
            next,
            ConsolidationRequest {
                id: first,
                session: Some("s".into()),
                all_projects: false,
                force: true,
                cwd: "/w".into()
            }
        );
        assert!(store.consume_consolidation_request(first).unwrap());
        assert!(!store.consume_consolidation_request(first).unwrap(), "taken twice");
        assert_eq!(store.next_consolidation_request().unwrap().unwrap().id, second);
        store.release_consolidation_request(first).unwrap();
        assert_eq!(store.next_consolidation_request().unwrap().unwrap().id, first);
        // Only a taken row older than a week goes.
        store.consume_consolidation_request(first).unwrap();
        let old = (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(8 * DAY_SECS)).to_string();
        store
            .conn
            .execute("UPDATE consolidation_requests SET consumed_at = ?1 WHERE id = ?2", params![old, first])
            .unwrap();
        store.purge_consumed_requests(7).unwrap();
        assert_eq!(store.next_consolidation_request().unwrap().unwrap().id, second);
        let rows: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM consolidation_requests", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 1);
    }

    /// Identical open asks are one piece of work: taking one takes them all, and
    /// a different ask is left alone.
    #[test]
    fn taking_an_ask_takes_its_identical_twins_only() {
        let store = Store::open_memory().unwrap();
        let a = store.add_consolidation_request(None, true, false, "/w").unwrap();
        let twin = store.add_consolidation_request(None, true, false, "/w").unwrap();
        let other = store.add_consolidation_request(None, true, true, "/w").unwrap();
        let sess_a = store.add_consolidation_request(Some("s"), false, true, "/w").unwrap();
        let sess_b = store.add_consolidation_request(Some("s"), false, true, "/w").unwrap();
        assert!(store.consume_consolidation_request(a).unwrap());
        assert!(!store.consume_consolidation_request(twin).unwrap(), "twin still open");
        assert_eq!(store.next_consolidation_request().unwrap().unwrap().id, other);
        assert!(store.consume_consolidation_request(other).unwrap());
        assert!(store.consume_consolidation_request(sess_a).unwrap());
        assert!(!store.consume_consolidation_request(sess_b).unwrap(), "session twin still open");
        assert!(store.next_consolidation_request().unwrap().is_none());
    }

    /// An `--all` pass covers every known project wherever it was asked from,
    /// so the directory does not make two `--all` asks different work. Before,
    /// each directory cost a drain round and a full pass of its own.
    #[test]
    fn an_all_projects_ask_takes_its_twins_from_any_directory() {
        let store = Store::open_memory().unwrap();
        let here = store.add_consolidation_request(None, true, false, "/a").unwrap();
        let there = store.add_consolidation_request(None, true, false, "/b").unwrap();
        let forced = store.add_consolidation_request(None, true, true, "/a").unwrap();
        assert!(store.consume_consolidation_request(here).unwrap());
        assert!(!store.consume_consolidation_request(there).unwrap(), "the /b twin still open");
        assert_eq!(store.next_consolidation_request().unwrap().unwrap().id, forced, "force is other work");
    }

    /// A non-force ask for a session with nothing pending has no work left:
    /// the run that would serve it only repeats a pass. A force ask reopens
    /// settled sessions and an `--all` ask names none, so both stay.
    #[test]
    fn a_session_ask_with_nothing_pending_is_dropped() {
        let store = Store::open_memory().unwrap();
        let (done, busy) = (Uuid::new_v4(), Uuid::new_v4());
        store.index(&event_in_session("Edit: a.rs", "{}", Uuid::new_v4(), busy)).unwrap();
        let stale = store.add_consolidation_request(Some(&done.to_string()), false, false, "/w").unwrap();
        let pending = store.add_consolidation_request(Some(&busy.to_string()), false, false, "/w").unwrap();
        let forced = store.add_consolidation_request(Some(&done.to_string()), false, true, "/w").unwrap();
        let all = store.add_consolidation_request(None, true, false, "/w").unwrap();
        assert!(!store.session_has_pending(&done.to_string()).unwrap());
        assert!(store.session_has_pending(&busy.to_string()).unwrap());

        assert_eq!(store.drop_stale_asks().unwrap(), 1);
        let open: Vec<i64> = store
            .conn
            .prepare("SELECT id FROM consolidation_requests WHERE consumed_at IS NULL ORDER BY id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(open, [pending, forced, all], "only the stale ask {stale} goes");
    }

    /// The heal runs in every run that holds the lock, inside the write lock
    /// a hook's write waits on for at most 5 s. Reached by session, its two
    /// statements read every quiet session's events cold: 3.3 s and 1.1 s on
    /// one store. Pending rows have their own index. Which rows change is
    /// pinned by the existing tests of each function; the plan guards the cost.
    #[test]
    fn the_heal_reads_only_pending_rows() {
        let store = Store::open_memory().unwrap();
        for (name, sql) in [
            ("resettle_replayed", Store::resettle_replayed_sql()),
            ("drop_stale_asks", Store::drop_stale_asks_sql()),
        ] {
            let plan: Vec<String> = store
                .conn
                .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
                .unwrap()
                .query_map([], |row| row.get::<_, String>(3))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert!(
                plan.iter()
                    .any(|row| row == "SEARCH e USING INDEX events_unconsolidated (project=? AND consolidated=?)"),
                "{name} does not read pending rows through events_unconsolidated: {plan:#?}"
            );
            assert!(
                !plan.iter().any(|row| row.contains("events_session")
                    || row.contains("AUTOMATIC")
                    || matches!(
                        row.strip_prefix("SCAN ").and_then(|rest| rest.split_whitespace().next()),
                        Some("events" | "e")
                    )),
                "{name} walks events by session or scans them: {plan:#?}"
            );
        }
    }

    /// The plan, one line per step, of a statement taking `args` text values.
    fn plan_of(store: &Store, sql: &str, args: usize) -> Vec<String> {
        store
            .conn
            .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
            .unwrap()
            .query_map(rusqlite::params_from_iter(std::iter::repeat_n("x", args)), |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    /// The primer reads the project's rows by kind, through `events_kind_proj`.
    /// Through the wrong index each of its three layers read every row of the
    /// project and sorted them: 4.8 to 6.0 s cold on the 115k-event project,
    /// where the plan below read 135, 32 and 297 ms. The plans can flip, so
    /// they are pinned, and so is the one statement an added kind index once
    /// turned into a scan of every observation.
    #[test]
    fn the_primer_reads_go_through_the_kind_index() {
        let store = Store::open_memory().unwrap();
        assert!(store.build_primer_indexes().unwrap());
        for (name, sql, args) in [
            ("one kind", Store::ranked_pointers_sql(&Kinds::One("knowledge")), 3),
            ("pushed kinds", Store::ranked_pointers_sql(&Kinds::Pushed), 2),
            ("newest summary", Store::newest_summary_sql().to_string(), 3),
        ] {
            let plan = plan_of(&store, &sql, args);
            assert!(
                plan.iter().any(|row| row.starts_with("SEARCH events USING INDEX events_kind_proj (kind=? AND project=?")),
                "{name} does not read through events_kind_proj: {plan:#?}"
            );
            assert!(
                !plan.iter().any(|row| row.contains("events_unconsolidated")
                    || row.contains("events_project_ts")
                    || row.contains("AUTOMATIC")
                    || row.starts_with("SCAN events")),
                "{name} reads the project by another index or scans: {plan:#?}"
            );
        }

        let plan = plan_of(&store, &Store::stale_backlog_sql(), 1);
        assert!(
            plan.iter().any(|row| row.starts_with("SEARCH e ") && row.contains("events_unconsolidated")),
            "the stale backlog left events_unconsolidated: {plan:#?}"
        );

        // Each looks for the few pending rows among millions of observations.
        // With the kind index in reach the planner walked every observation of
        // the project (or of the store) instead; `+kind` keeps it off.
        for (name, sql, args) in [
            ("stale backlog", Store::stale_backlog_sql(), 1),
            ("sessions pending", Store::sessions_pending_sql(), 1),
            ("oldest pending", Store::oldest_pending_sql(), 0),
            ("unconsolidated sessions", Store::unconsolidated_sessions_sql().to_string(), 2),
        ] {
            let plan = plan_of(&store, &sql, args);
            assert!(
                !plan.iter().any(|row| row.contains("events_kind_proj")
                    || matches!(
                        row.strip_prefix("SCAN ").and_then(|rest| rest.split_whitespace().next()),
                        Some("events" | "e")
                    )),
                "{name} reads observations by kind or scans events: {plan:#?}"
            );
            // `events_recall` is ordered by session, which `GROUP BY session`
            // wants, so the planner read the whole project through it and
            // fetched every row to find the few pending ones: 3.2 s cold on
            // the 115k-event project, inside `brain_recent`. The oldest
            // pending event is the one read that names no project.
            assert!(
                name == "oldest pending"
                    || (plan.iter().any(|row| row.contains("events_unconsolidated"))
                        && !plan.iter().any(|row| row.contains("events_recall"))),
                "{name} does not read the pending rows through events_unconsolidated: {plan:#?}"
            );
        }
    }

    #[test]
    fn the_session_cli_is_one_probe_of_the_session_index() {
        let store = Store::open_memory().unwrap();
        assert!(store.build_primer_indexes().unwrap());
        // With `events_session` kept beside it, as the build leaves a store.
        let kept: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE name = 'events_session'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(kept, 1, "events_session is not beside events_session_id");
        let plan = plan_of(&store, Store::session_cli_sql(), 1);
        assert!(
            plan.iter().any(|row| row.starts_with("SEARCH events USING INDEX events_session_id (session=?")),
            "session_cli does not probe events_session_id: {plan:#?}"
        );
        assert!(
            !plan.iter().any(|row| row.contains("TEMP B-TREE")),
            "session_cli sorts the session's rows: {plan:#?}"
        );
    }

    /// The recall floor for `nearest` and `entity_matches` filters a project's
    /// events on `forgotten`, `kind`, `hook`, `topic` and `confidence`. Through
    /// `events_unconsolidated` that fetched every event row of the project to
    /// test them: 3.4 s and 2.4 s cold on the 115k-event project, against
    /// ~0.12 s counting the same rows in an index that holds the columns.
    /// Only the final few hits may fetch a row, by primary key.
    #[test]
    fn the_recall_floor_reads_a_covering_index() {
        let store = Store::open_memory().unwrap();
        assert!(store.build_primer_indexes().unwrap());
        for (name, sql, args) in [
            ("nearest", NEAREST_SQL.to_string(), 2),
            ("entity matches", Store::entity_matches_sql(3), 7),
            ("neighbours", Store::neighbours_sql(3), 6),
            ("related", RELATED_SQL.to_string(), 3),
        ] {
            let plan = plan_of(&store, &sql, args);
            assert!(
                plan.iter().any(|row| row.starts_with("SEARCH e USING COVERING INDEX events_recall (project=?")),
                "{name} does not filter through events_recall: {plan:#?}"
            );
            assert!(
                !plan.iter().any(|row| (row.starts_with("SEARCH e ") && !row.contains("COVERING INDEX events_recall"))
                    || row.starts_with("SCAN e")
                    || row.contains("events_unconsolidated")
                    || row.contains("AUTOMATIC COVERING INDEX (project")),
                "{name} fetches event rows to apply the floor: {plan:#?}"
            );
        }
        for (name, sql, args) in [
            ("entity matches", Store::entity_matches_sql(3), 7),
            ("neighbours", Store::neighbours_sql(3), 6),
            ("related", RELATED_SQL.to_string(), 3),
        ] {
            let plan = plan_of(&store, &sql, args);
            assert!(
                plan.iter().any(|row| row == "SEARCH h USING INDEX sqlite_autoindex_events_1 (id=?)"),
                "{name} does not fetch its final rows by id: {plan:#?}"
            );
        }

        // A covering index that can answer `id IN (...) AND project = ?` made
        // the planner walk the project to find three seeds.
        for (name, sql, args) in [("neighbours", Store::neighbours_sql(3), 6), ("related", RELATED_SQL.to_string(), 3)] {
            let plan = plan_of(&store, &sql, args);
            assert!(
                plan.iter().any(|row| row == "SEARCH events USING INDEX sqlite_autoindex_events_1 (id=?)"),
                "{name} does not look its seeds up by id: {plan:#?}"
            );
            assert!(
                !plan.iter().any(|row| row.starts_with("SEARCH events ") && row.contains("events_recall")),
                "{name} walks the project to find its seeds: {plan:#?}"
            );
        }
    }

    /// `brain_recent` lists the newest rows of a project, so it must stop at
    /// `limit`. With the kind, hook and session filters as plain row tests it
    /// fetched every row of the project and sorted them: 3.35 to 3.58 s cold on
    /// the 115k-event project. The floor and the kind and session filters
    /// are columns of `events_recall`, so they are applied there, and the
    /// rows are fetched by id in `id DESC` order, which stops at the limit.
    #[test]
    fn recent_stops_at_the_limit_instead_of_sorting_the_project() {
        let store = Store::open_memory().unwrap();
        assert!(store.build_primer_indexes().unwrap());
        let plan = plan_of(&store, Store::recent_sql(), 5);
        assert!(
            plan.iter().any(|row| row.starts_with("SEARCH events USING COVERING INDEX events_recall (project=?")),
            "recent does not filter through events_recall: {plan:#?}"
        );
        assert!(
            plan.iter().any(|row| row == "SEARCH events USING INDEX sqlite_autoindex_events_1 (id=?)"),
            "recent does not fetch its rows by id: {plan:#?}"
        );
        assert!(
            !plan.iter().any(|row| row.contains("TEMP B-TREE")
                || row.contains("events_unconsolidated")
                || row.starts_with("SCAN events")),
            "recent sorts or walks the project's rows: {plan:#?}"
        );
    }

    /// The same answers before and after the build, for each filter the tool
    /// takes, with `k` cutting inside a run of equal filters.
    #[test]
    fn recent_answers_the_same_before_and_after_the_build() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let other = Uuid::new_v4();
        let session = Uuid::new_v4();
        let mut ids = Vec::new();
        for (n, (kind, cli, hook)) in [
            (EventKind::Observation, "claude-code", "post_tool_use"),
            (EventKind::SessionSummary, "codex", "consolidate"),
            (EventKind::Observation, "codex", "post_tool_use"),
            (EventKind::Knowledge, "claude-code", "correct"),
            (EventKind::SessionSummary, "claude-code", "consolidate"),
            (EventKind::Tombstone, "claude-code", "forget"),
            (EventKind::Observation, "claude-code", "post_tool_use"),
        ]
        .into_iter()
        .enumerate()
        {
            let mut entry = event(&format!("entry {n}"), "body", project);
            entry.kind = kind;
            entry.source.cli = cli.to_string();
            entry.source.hook = hook.to_string();
            if n % 2 == 0 {
                entry.session = session;
            }
            entry.id = format!("01RECENT{n:019}");
            store.index(&entry).unwrap();
            ids.push(entry.id);
        }
        let mut elsewhere = event("not this project", "body", other);
        elsewhere.id = format!("01RECENT{:019}", 9);
        store.index(&elsewhere).unwrap();
        store.conn.execute("UPDATE events SET forgotten = 1 WHERE id = ?1", params![ids[6]]).unwrap();

        let session = session.to_string();
        let asks = [
            (None, None, None, 10),
            (None, None, None, 2),
            (None, Some("session_summary"), None, 10),
            (None, Some("observation"), None, 1),
            (Some("codex"), None, None, 10),
            (Some("claude-code"), Some("session_summary"), None, 10),
            (None, None, Some(session.as_str()), 10),
        ];
        let read = |store: &Store| -> Vec<Vec<String>> {
            asks.iter()
                .map(|(cli, kind, of_session, limit)| {
                    store
                        .recent(&project.to_string(), *cli, *kind, *of_session, *limit)
                        .unwrap()
                        .into_iter()
                        .map(|hit| hit.id)
                        .collect()
                })
                .collect()
        };
        let before = read(&store);
        assert_eq!(before[0], [ids[4].clone(), ids[2].clone(), ids[1].clone(), ids[0].clone()], "the floor: no tombstone, correction or forgotten row");
        assert_eq!(before[1], [ids[4].clone(), ids[2].clone()], "k cuts the newest two");
        assert_eq!(before[3], [ids[2].clone()]);
        assert_eq!(before[4], [ids[2].clone(), ids[1].clone()], "cli");
        assert_eq!(before[6], [ids[4].clone(), ids[2].clone(), ids[0].clone()], "session");
        assert!(before[2].ends_with(&[ids[4].clone(), ids[1].clone()]), "kind: {:?}", before[2]);
        assert!(before[5].ends_with(&[ids[4].clone()]), "cli and kind: {:?}", before[5]);
        assert!(store.build_primer_indexes().unwrap());
        assert_eq!(read(&store), before, "the build changed an answer");
    }

    /// Entries that score alike come back oldest first, whichever way the
    /// index they are read through is ordered. `events_recall` walks a project
    /// by session, where `events_unconsolidated` walked it in insertion order,
    /// and the three "Session started" summaries of one project swapped places
    /// in a query about resuming a session: ties kept the order they were read.
    #[test]
    fn nearest_orders_equal_scores_by_age_not_by_index() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let vector = vec![7u8; crate::embed::DIMS];
        // Oldest first by id, in sessions that sort the other way round.
        let mut ids = Vec::new();
        for session in [0xff_u128, 0xcc, 0xaa, 0xdd] {
            let session = Uuid::from_u128(session);
            // A ULID is only ordered across milliseconds.
            std::thread::sleep(std::time::Duration::from_millis(2));
            let entry = event_in_session("Session started. No work performed.", "nothing", project, session);
            store.index(&entry).unwrap();
            store.set_vectors(&[(entry.id.clone(), vector.clone())]).unwrap();
            ids.push(entry.id);
        }
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "the fixture's ids are not in age order: {ids:?}");
        for built in [false, true] {
            if built {
                assert!(store.build_primer_indexes().unwrap());
            }
            let found: Vec<String> =
                store.nearest(&project.to_string(), &vector, None, 10).unwrap().into_iter().map(|(id, _)| id).collect();
            assert_eq!(found, ids, "built={built}");
        }
    }

    /// `entity_matches` as it was before the floor was applied to columns only:
    /// every candidate row carried its title and a body snippet through the
    /// window function.
    fn old_entity_matches_sql(tokens: usize) -> String {
        let likes = (0..tokens)
            .map(|index| format!("n.name LIKE ?{} ESCAPE '\\'", index + 5))
            .collect::<Vec<_>>()
            .join(" OR ");
        format!(
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
                            WHEN 'source' THEN 1
                            ELSE 2
                        END AS authority,
                        ROW_NUMBER() OVER (
                            PARTITION BY e.session
                            ORDER BY CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END,
                                     CASE e.kind
                                         WHEN 'knowledge' THEN 0
                                         WHEN 'session_summary' THEN 1
                                         WHEN 'source' THEN 1
                                         ELSE 2
                                     END,
                                     e.id DESC
                        ) AS per_session
                 FROM events e
                 JOIN matched m ON m.session = e.session
                 WHERE e.project = ?1 AND e.forgotten = 0 AND e.kind NOT IN ('tombstone', 'retire')
                       AND e.hook NOT IN ('correct', 'feedback', 'supersede')
                       AND (?2 IS NULL OR e.topic = ?2)
             )
             SELECT id, ts, cli, kind, title, snip, session FROM candidates
             WHERE per_session <= ?3
             ORDER BY demoted, matched DESC, authority, id DESC
             LIMIT ?4"
        )
    }

    /// Three sessions that share one, three and two of the names `src/alpha.rs`,
    /// `src/beta.rs` and `src/gamma.rs`, each holding every kind of event the
    /// recall floor and the ranking tell apart: knowledge, a summary, captures,
    /// a forgotten event, a demoted one, a correction, a tombstone and a topic.
    /// Returns the store, the project and the first capture of the first session.
    fn recall_floor_fixture() -> (Store, Uuid, String) {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let key = project.to_string();
        let names: Vec<String> = ["src/alpha.rs", "src/beta.rs", "src/gamma.rs"].map(String::from).to_vec();
        let mut seed = String::new();
        for (n, shared) in [3usize, 1, 2].into_iter().enumerate() {
            let session = Uuid::new_v4();
            let mut events = Vec::new();
            for k in 0..3 {
                events.push(event_in_session(&format!("Capture {n}.{k}"), "a capture", project, session));
            }
            if n == 0 {
                seed = events[0].id.clone();
            }
            let mut lesson = event_in_session(&format!("Lesson {n}"), "kept", project, session);
            lesson.kind = EventKind::Knowledge;
            let mut summary = event_in_session(&format!("Summary {n}"), "what happened", project, session);
            summary.kind = EventKind::SessionSummary;
            let mut demoted = event_in_session(&format!("Demoted {n}"), "flagged stale", project, session);
            demoted.kind = EventKind::Knowledge;
            let forgotten = event_in_session(&format!("Forgotten {n}"), "withdrawn", project, session);
            let mut correction = event_in_session(&format!("Correction {n}"), "fixes one", project, session);
            correction.source.hook = "correct".to_string();
            let mut gone = event_in_session(&format!("Tombstone {n}"), "", project, session);
            gone.kind = EventKind::Tombstone;
            let topical = event_in_session(&format!("Topical {n}"), "on a topic", project, session);
            events.extend([lesson, summary, demoted.clone(), forgotten.clone(), correction, gone, topical.clone()]);
            for event in &events {
                store.index(event).unwrap();
            }
            store.conn.execute("UPDATE events SET confidence = -1 WHERE id = ?1", params![demoted.id]).unwrap();
            store.conn.execute("UPDATE events SET forgotten = 1 WHERE id = ?1", params![forgotten.id]).unwrap();
            store.conn.execute("UPDATE events SET topic = 'billing' WHERE id = ?1", params![topical.id]).unwrap();
            store.record_entities(&session.to_string(), &key, &names[..shared]).unwrap();
        }
        (store, project, seed)
    }

    /// Reading the floor from the index must not change which events pass it
    /// or in what order.
    #[test]
    fn entity_matches_returns_what_the_row_fetching_query_returned() {
        let (store, project, _) = recall_floor_fixture();
        let key = project.to_string();
        let old = |topic: Option<&str>, per_session: usize, pool: usize| -> Vec<String> {
            let mut stmt = store.conn.prepare(&old_entity_matches_sql(2)).unwrap();
            let rows = stmt
                .query_map(params![key, topic, per_session as i64, pool as i64, "%src/alpha%", "%src/beta%"], |row| {
                    row.get::<_, String>(0)
                })
                .unwrap();
            rows.collect::<rusqlite::Result<Vec<_>>>().unwrap()
        };
        for built in [false, true] {
            if built {
                assert!(store.build_primer_indexes().unwrap());
            }
            for (topic, per_session, pool) in [(None, 2, 20), (None, 1, 20), (None, 4, 5), (Some("billing"), 2, 20)] {
                let new: Vec<String> = store
                    .entity_matches(&key, "src/alpha src/beta", topic, per_session, pool)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.id)
                    .collect();
                let expected = old(topic, per_session, pool);
                assert!(!expected.is_empty(), "the fixture matched nothing for {topic:?}");
                assert_eq!(new, expected, "built={built} topic={topic:?} per_session={per_session} pool={pool}");
            }
        }
    }

    /// `neighbours_of` as it was before its rows were fetched last.
    const OLD_NEIGHBOURS: &str = "WITH seed_sessions AS (
             SELECT DISTINCT session FROM events WHERE id = ?4 AND project = ?1
         ),
         subject AS (
             SELECT DISTINCT n.name FROM entities n
             WHERE n.session IN (SELECT session FROM seed_sessions) AND n.project = ?1
         ),
         shared AS (
             SELECT n.session, COUNT(DISTINCT n.name) AS shared
             FROM entities n
             WHERE n.project = ?1 AND n.name IN (SELECT name FROM subject)
                   AND n.session NOT IN (SELECT session FROM seed_sessions)
             GROUP BY n.session
         )
         SELECT e.id FROM events e
         JOIN shared s ON s.session = e.session
         WHERE e.project = ?1
               AND e.forgotten = 0 AND e.kind NOT IN ('tombstone', 'retire')
               AND e.hook NOT IN ('correct', 'feedback', 'supersede')
               AND (?2 IS NULL OR e.topic = ?2)
         ORDER BY s.shared DESC,
                  CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END,
                  CASE e.kind
                      WHEN 'knowledge' THEN 0
                      WHEN 'session_summary' THEN 1
                      WHEN 'source' THEN 1
                      ELSE 2
                  END,
                  e.id DESC
         LIMIT ?3";

    #[test]
    fn neighbours_of_returns_what_the_row_fetching_query_returned() {
        let (store, project, seed) = recall_floor_fixture();
        let key = project.to_string();
        for built in [false, true] {
            if built {
                assert!(store.build_primer_indexes().unwrap());
            }
            for (topic, limit) in [(None, 50), (None, 3), (Some("billing"), 50)] {
                let new: Vec<String> = store
                    .neighbours_of(&key, std::slice::from_ref(&seed), topic, limit)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.id)
                    .collect();
                let mut stmt = store.conn.prepare(OLD_NEIGHBOURS).unwrap();
                let old: Vec<String> = stmt
                    .query_map(params![key, topic, limit as i64, seed], |row| row.get(0))
                    .unwrap()
                    .collect::<rusqlite::Result<_>>()
                    .unwrap();
                assert!(!old.is_empty(), "the fixture matched nothing for {topic:?}");
                assert_eq!(new, old, "built={built} topic={topic:?} limit={limit}");
            }
        }
    }

    /// The keyword stream as it was before its snippet was computed for the
    /// pool only: every match carried its title and snippet through the window.
    const OLD_KEYWORD: &str = "WITH matched AS (
             SELECT e.id, e.ts, e.cli, e.kind, e.title, e.session,
                    snippet(events_fts, 1, '[', ']', ' … ', 24) AS snip,
                    CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END AS demoted,
                    rank AS relevance
             FROM events_fts
             JOIN events e ON e.rowid = events_fts.rowid
             WHERE events_fts MATCH ?1 AND e.project = ?2 AND e.forgotten = 0
                   AND e.kind NOT IN ('tombstone', 'retire')
                   AND e.hook NOT IN ('correct', 'feedback', 'supersede')
                   AND (?5 IS NULL OR e.topic = ?5)
         )
         SELECT id, ts, cli, kind, title, snip, session FROM (
             SELECT *, ROW_NUMBER() OVER (
                 PARTITION BY session ORDER BY demoted, relevance
             ) AS per_session FROM matched
         )
         WHERE per_session <= ?3
         ORDER BY demoted, relevance
         LIMIT ?4";

    #[test]
    fn the_keyword_stream_returns_what_the_snippet_per_match_query_returned() {
        let (store, project, _) = recall_floor_fixture();
        let key = project.to_string();
        let run = |sql: &str, query: &str, topic: Option<&str>, per_session: usize, pool: usize| -> Vec<(String, String)> {
            let mut stmt = store.conn.prepare(sql).unwrap();
            stmt.query_map(params![query, key, per_session as i64, pool as i64, topic], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(5)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
        };
        let query = "capture OR kept OR happened OR flagged OR stale OR topic OR withdrawn OR fixes";
        for (topic, per_session, pool) in [(None, 2, 20), (None, 1, 20), (None, 4, 5), (Some("billing"), 2, 20)] {
            let expected = run(OLD_KEYWORD, query, topic, per_session, pool);
            assert!(expected.len() > 2, "the fixture matched too little for {topic:?}: {expected:?}");
            assert_eq!(
                run(KEYWORD_SQL, query, topic, per_session, pool),
                expected,
                "topic={topic:?} per_session={per_session} pool={pool}"
            );
        }
    }

    /// The build is once-only and changes no answer; before it the reads work.
    #[test]
    fn the_primer_indexes_build_once_and_change_no_answer() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let key = project.to_string();
        let session = Uuid::new_v4();
        let mut summary = event("a summary", "body", project);
        summary.kind = EventKind::SessionSummary;
        summary.session = session;
        let mut lesson = event("a lesson", "body", project);
        lesson.kind = EventKind::Knowledge;
        let capture = event("a capture", "body", project);
        for each in [&summary, &lesson, &capture] {
            store.index(each).unwrap();
        }
        let read = |store: &Store| {
            (
                store.ranked_pointers(&key, Kinds::Pushed, 10).unwrap().into_iter().map(|p| p.id).collect::<Vec<_>>(),
                store.pointers_of_kind(&key, "knowledge", 10).unwrap().into_iter().map(|p| p.id).collect::<Vec<_>>(),
                store.session_cli(&session.to_string()).unwrap(),
                store.newest_summary(&key, "other", std::time::Duration::from_secs(3600)).unwrap().map(|(p, cli)| (p.id, cli)),
            )
        };
        let index_names = |store: &Store| -> Vec<String> {
            store
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'events' ORDER BY name")
                .unwrap()
                .query_map([], |row| row.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        assert!(!index_names(&store).iter().any(|name| name.starts_with("events_kind")), "open built the kind index");

        let before = read(&store);
        assert_eq!(before.0.len(), 2, "the pushed set is the summary and the lesson, not the capture");
        assert!(!before.0.contains(&capture.id), "an observation is in the pushed set");
        assert_eq!(before.2.as_deref(), Some("claude-code"));

        assert!(store.build_primer_indexes().unwrap());
        assert!(!store.build_primer_indexes().unwrap(), "the marker did not hold");
        assert_eq!(read(&store), before, "the build changed an answer");
        let names = index_names(&store);
        assert!(names.contains(&"events_kind_proj".to_string()) && names.contains(&"events_session_id".to_string()));
        assert!(names.contains(&"events_session".to_string()), "the build dropped events_session: {names:?}");

        // An open leaves the set as the build left it.
        store.migrate().unwrap();
        assert_eq!(index_names(&store), names);
    }

    #[test]
    fn the_runs_ledger_is_pruned_with_the_taken_asks() {
        let store = Store::open_memory().unwrap();
        let run = |started: String| ConsolidationRun {
            started: started.clone(),
            ended: started,
            mode: "all".into(),
            yielded: false,
            sessions: 0,
            events: 0,
            failed: 0,
            rule_based: 0,
            error: None,
        };
        let now = jiff::Timestamp::now();
        let old = (now - jiff::SignedDuration::from_secs(8 * DAY_SECS)).to_string();
        store.record_consolidation_run(&run(old)).unwrap();
        store.record_consolidation_run(&run(now.to_string())).unwrap();
        store.purge_consumed_requests(7).unwrap();
        assert_eq!(store.consolidation_runs_since(30 * DAY_SECS).unwrap().len(), 1);
    }

    #[test]
    fn summarizer_calls_since_keeps_only_the_window() {
        let store = Store::open_memory().unwrap();
        let call = |session: &str| SummarizerCall {
            session: session.into(),
            purpose: "consolidate".into(),
            cli: "codex".into(),
            model: "m".into(),
            prompt_bytes: 1,
            answer_bytes: 1,
            ms: 1,
            outcome: "ok".into(),
        };
        store.record_summarizer_call(&call("old")).unwrap();
        store.record_summarizer_call(&call("new")).unwrap();
        let three_days_ago =
            (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(3 * DAY_SECS)).to_string();
        store
            .conn
            .execute("UPDATE summarizer_calls SET ts = ?1 WHERE session = 'old'", params![three_days_ago])
            .unwrap();
        let day: Vec<String> =
            store.summarizer_calls_since(DAY_SECS).unwrap().into_iter().map(|c| c.session).collect();
        assert_eq!(day, ["new"]);
        let week: Vec<String> =
            store.summarizer_calls_since(7 * DAY_SECS).unwrap().into_iter().map(|c| c.session).collect();
        assert_eq!(week, ["old", "new"]);
    }

    /// Every rerank is written down with what answered and why the faster
    /// engine did not, and comes back in the order it happened.
    #[test]
    fn rerank_runs_are_recorded_and_read_back_in_order() {
        let store = super::Store::open_memory().unwrap();
        store.record_rerank("local", "", 2400, true).unwrap();
        store.record_rerank("local", "", 1800, false).unwrap();
        store.record_rerank("none", "cli-no-answer", 20_000, false).unwrap();
        let runs = store.rerank_runs().unwrap();
        assert_eq!(runs.len(), 3);
        assert_eq!((runs[0].engine.as_str(), runs[0].ms, runs[0].cold), ("local", 2400, true));
        assert_eq!((runs[1].engine.as_str(), runs[1].ms, runs[1].cold), ("local", 1800, false));
        assert_eq!((runs[2].engine.as_str(), runs[2].reason.as_str()), ("none", "cli-no-answer"));
    }

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

    /// An event the embedder keeps (a prompt), unlike `event`'s tool call.
    fn prose_event(title: &str, body: &str, project: Uuid) -> Event {
        let mut event = event(title, body, project);
        event.source.hook = "user_prompt_submit".into();
        event
    }

    fn tool_call_body(stdout: &str) -> String {
        serde_json::json!({
            "tool_name": "Bash",
            "tool_input": {"command": "make"},
            "tool_response": {"stdout": stdout},
        })
        .to_string()
    }

    fn stored_body(store: &Store, id: &str) -> (String, i32) {
        store
            .conn
            .query_row("SELECT body, clamped FROM events WHERE id = ?1", params![id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap()
    }

    #[test]
    fn a_big_observation_is_clamped_in_the_index_and_stays_json_with_its_tail() {
        let store = Store::open_memory().unwrap();
        let stdout = format!("{}RESULT_TAIL", "output line\n".repeat(1000));
        let big = event("Bash: make", &tool_call_body(&stdout), Uuid::new_v4());
        assert!(big.body.len() > 10 * 1024);
        store.index(&big).unwrap();

        let (body, clamped) = stored_body(&store, &big.id);
        assert!(body.len() <= 4096, "index kept {} bytes", body.len());
        assert_eq!(clamped, 1);
        assert!(store.is_clamped(&big.id).unwrap());
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("still JSON");
        assert!(parsed["tool_response"]["stdout"].as_str().unwrap().ends_with("RESULT_TAIL"));
        assert_eq!(parsed["tool_name"], "Bash");
        // The consolidation path reads through `get`: the bounded body.
        assert_eq!(store.get(std::slice::from_ref(&big.id)).unwrap()[0].body, body);
    }

    #[test]
    fn a_big_body_that_is_not_json_is_cut_head_and_tail() {
        let store = Store::open_memory().unwrap();
        let text = format!("HEAD{}TAIL", "y".repeat(10 * 1024));
        let big = event("prompt", &text, Uuid::new_v4());
        store.index(&big).unwrap();
        let (body, clamped) = stored_body(&store, &big.id);
        assert!(body.len() <= 4096 && clamped == 1);
        assert!(body.starts_with("HEAD") && body.ends_with("TAIL"));
    }

    #[test]
    fn only_an_observation_is_clamped_and_a_small_one_is_untouched() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut note = event("note", &"n".repeat(10 * 1024), project);
        note.kind = EventKind::Note;
        store.index(&note).unwrap();
        let (body, clamped) = stored_body(&store, &note.id);
        assert_eq!((body.len(), clamped), (10 * 1024, 0));

        let small = event("small", &tool_call_body("ok"), project);
        store.index(&small).unwrap();
        let (body, clamped) = stored_body(&store, &small.id);
        assert_eq!((body, clamped), (small.body.clone(), 0));
        assert!(!store.is_clamped(&small.id).unwrap());
    }

    #[test]
    fn a_reindex_clamps_the_same_way_again() {
        let store = Store::open_memory().unwrap();
        let big = event("Bash: make", &tool_call_body(&"z".repeat(10 * 1024)), Uuid::new_v4());
        store.index(&big).unwrap();
        let first = stored_body(&store, &big.id);
        store.index(&big).unwrap();
        assert_eq!(stored_body(&store, &big.id), first);
        assert_eq!(first.1, 1);
    }

    #[test]
    fn a_reindex_clamps_a_legacy_full_body_row() {
        // A pre-migration row: full body, clamped = 0. Re-indexing must clamp
        // it AND flag it, which only `clamped = excluded.clamped` does.
        let store = Store::open_memory().unwrap();
        let big = event("Bash: make", &tool_call_body(&"z".repeat(10 * 1024)), Uuid::new_v4());
        store.index(&big).unwrap();
        store
            .conn
            .execute(
                "UPDATE events SET body = ?2, clamped = 0 WHERE id = ?1",
                params![big.id, big.body],
            )
            .unwrap();
        store.index(&big).unwrap();
        let (body, clamped) = stored_body(&store, &big.id);
        assert_eq!(clamped, 1);
        assert!(body.len() <= 4096, "legacy row re-clamped, got {}", body.len());
    }

    #[test]
    fn a_retired_clamped_observation_is_no_longer_clamped() {
        // Retirement empties the body; `clamped` left at 1 would make
        // brain_get reload the whole line from the log.
        let store = Store::open_memory().unwrap();
        let mut big =
            event("Bash: make", &tool_call_body(&"z".repeat(10 * 1024)), Uuid::new_v4());
        big.consolidated = true;
        store.index(&big).unwrap();
        assert!(store.is_clamped(&big.id).unwrap());
        let mut retire = Event::new(
            Uuid::nil(),
            big.project,
            Uuid::nil(),
            Source { cli: "brain".into(), hook: "retire".into() },
            EventKind::Retire,
            "Retired 1".into(),
            String::new(),
        );
        retire.links = vec![big.id.clone()];
        store.index(&retire).unwrap();
        assert_eq!(stored_body(&store, &big.id), (String::new(), 0));
        assert!(!store.is_clamped(&big.id).unwrap());
    }

    #[test]
    fn a_dropped_index_body_points_at_the_log_until_retired_or_reindexed() {
        let store = Store::open_memory().unwrap();
        let mut note = event("Bash: make", "built fine, 3 warnings", Uuid::new_v4());
        note.consolidated = true;
        store.index(&note).unwrap();
        let ids = vec![note.id.clone()];

        assert_eq!(store.drop_index_bodies(&ids).unwrap(), 1);
        assert_eq!(stored_body(&store, &note.id), (String::new(), 1));
        assert!(store.is_clamped(&note.id).unwrap());
        assert_eq!(store.drop_index_bodies(&ids).unwrap(), 0, "an empty body is not dropped twice");

        // The log was never touched, so indexing the same line again restores it.
        store.index(&note).unwrap();
        assert_eq!(stored_body(&store, &note.id), ("built fine, 3 warnings".to_string(), 0));

        // A retirement after a drop still withdraws the body for good.
        store.drop_index_bodies(&ids).unwrap();
        let mut retire = Event::new(
            Uuid::nil(),
            note.project,
            Uuid::nil(),
            Source { cli: "brain".into(), hook: "retire".into() },
            EventKind::Retire,
            "Retired 1".into(),
            String::new(),
        );
        retire.links = ids;
        store.index(&retire).unwrap();
        assert_eq!(stored_body(&store, &note.id), (String::new(), 0));
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
    fn a_flag_only_update_does_not_touch_the_text_indexes() {
        // An FTS trigger on every UPDATE rewrote both indexes for a row
        // whose title and body had not changed: 1.6 ms a row to flip
        // `consolidated`. `total_changes` counts trigger writes too, so one
        // change means the row alone was written.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let e = event("Asked: flag only", "body", project);
        store.index(&e).unwrap();

        for set in ["consolidated = 1", "injected_count = 5", "read_count = 5"] {
            let before = store.conn.total_changes();
            store
                .conn
                .execute(&format!("UPDATE events SET {set} WHERE id = ?1"), params![e.id])
                .unwrap();
            assert_eq!(store.conn.total_changes() - before, 1, "`{set}` fired an FTS trigger");
        }
    }

    #[test]
    fn a_title_or_body_update_still_reaches_both_text_indexes() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let e = event("Asked: alpha", "bravo", project);
        store.index(&e).unwrap();
        store
            .conn
            .execute(
                "UPDATE events SET title = 'Asked: ประสิทธิภาพ', body = 'charlie' WHERE id = ?1",
                params![e.id],
            )
            .unwrap();

        let find = |q: &str| store.search(&project.to_string(), q, None, 10, Recall::Fused).unwrap().len();
        assert_eq!(find("charlie"), 1, "the new body is not in the word index");
        assert_eq!(find("ประสิทธิภาพ"), 1, "the new title is not in the substring index");
        assert_eq!(find("bravo"), 0, "the old body is still in the word index");
    }

    #[test]
    fn a_store_with_the_unscoped_triggers_is_moved_to_the_scoped_ones_once() {
        let store = Store::open_memory().unwrap();
        store
            .conn
            .execute_batch(
                "DROP TRIGGER events_au; DROP TRIGGER events_tri_au;
                 CREATE TRIGGER events_au AFTER UPDATE ON events BEGIN
                     INSERT INTO events_fts(events_fts, rowid, title, body)
                     VALUES ('delete', old.rowid, old.title, old.body);
                     INSERT INTO events_fts(rowid, title, body)
                     VALUES (new.rowid, new.title, new.body);
                 END;
                 DELETE FROM schema_state WHERE key = 'fts_triggers_scoped';",
            )
            .unwrap();
        store.migrate().unwrap();
        let sql: String = store
            .conn
            .query_row("SELECT sql FROM sqlite_master WHERE name = 'events_au'", [], |r| r.get(0))
            .unwrap();
        assert!(sql.contains("UPDATE OF title, body"), "{sql}");
        let sql: String = store
            .conn
            .query_row("SELECT sql FROM sqlite_master WHERE name = 'events_tri_au'", [], |r| r.get(0))
            .unwrap();
        assert!(sql.contains("UPDATE OF title"), "{sql}");
    }

    #[test]
    fn marking_events_consolidated_flips_every_flag() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let events: Vec<Event> = (0..3).map(|i| event(&format!("Asked: n{i}"), "b", project)).collect();
        for e in &events {
            store.index(e).unwrap();
        }
        let ids: Vec<String> = events.iter().map(|e| e.id.clone()).collect();
        store.mark_consolidated(&ids).unwrap();
        let left: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM events WHERE consolidated = 0", [], |r| r.get(0))
            .unwrap();
        assert_eq!(left, 0);
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
        let stale = prose_event("Chose SQLite over Postgres", "nothing resident", project);
        let fresh = prose_event("Wrote the installation guide", "one page", project);
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
        let stale = prose_event("Chose SQLite over Postgres", "nothing resident", project);
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
    fn a_vector_from_a_different_model_of_the_same_width_is_dropped_on_open() {
        // Width caught the last model change because the widths differed.
        // Two models at one width leave nothing in the bytes to tell them
        // apart, so the model's name is what the store checks - and a
        // mismatch has to empty the index, or every old vector scores noise
        // against every new query while doctor reports a full index.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let old = prose_event("Chose SQLite over Postgres", "nothing resident", project);
        store.index(&old).unwrap();
        store.set_vectors(&[(old.id.clone(), vec![7u8; crate::embed::DIMS])]).unwrap();
        assert_eq!(store.vector_coverage().unwrap(), (1, 1));

        // What a store looks like after a binary carrying another model of
        // the same width wrote it.
        store
            .conn
            .execute(
                "UPDATE schema_state SET value = 'potion-base-8M:256' WHERE key = 'embedding_model'",
                [],
            )
            .unwrap();
        store.migrate().unwrap();

        assert_eq!(store.vector_coverage().unwrap(), (0, 1), "the other model's vector survived reopening");
        let pending = store.events_missing_vectors(&project.to_string(), 10).unwrap();
        assert!(pending.iter().any(|(id, _)| *id == old.id), "not queued for re-embedding: {pending:?}");
        assert_eq!(recorded_embedding_model(&store), crate::embed::signature());
    }

    #[test]
    fn a_store_without_a_model_record_keeps_its_vectors() {
        // The record is new. Every store without one was embedded by the
        // model this build carries - the last change was caught by width -
        // so the upgrade that adds the record must not throw a hundred
        // thousand correct vectors away to be sure.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let fresh = prose_event("Wrote the installation guide", "one page", project);
        store.index(&fresh).unwrap();
        store.set_vectors(&[(fresh.id.clone(), vec![7u8; crate::embed::DIMS])]).unwrap();
        store.conn.execute("DELETE FROM schema_state WHERE key = 'embedding_model'", []).unwrap();

        store.migrate().unwrap();

        assert_eq!(
            store.vector_coverage().unwrap(),
            (1, 1),
            "vectors were dropped with no evidence of a model change"
        );
        assert_eq!(recorded_embedding_model(&store), crate::embed::signature());
    }

    fn event_from(hook: &str, kind: EventKind, title: &str, project: Uuid) -> Event {
        Event::new(
            Uuid::nil(),
            project,
            Uuid::nil(),
            Source { cli: "claude-code".into(), hook: hook.into() },
            kind,
            title.into(),
            "body words".into(),
        )
    }

    /// Every spelling of a hook that carries a tool call or a lifecycle marker.
    const SKIPPED_HOOKS: [&str; 8] = [
        "post_tool_use",
        "pre_tool_use",
        "session_start",
        "session_end",
        "pre_compact",
        "after_tool",
        "pre_compress",
        "tool_execute_after",
    ];

    /// Hooks whose events are prose and stay embedded.
    const KEPT_HOOKS: [&str; 6] =
        ["user_prompt_submit", "stop", "subagent_stop", "page_update", "knowledge", "note"];

    fn delegate_footstep(project: Uuid) -> Event {
        let mut e = event_from("stop", EventKind::Observation, "delegate step", project);
        e.agent = Some("worker".into());
        e
    }

    fn delegate_report(project: Uuid) -> Event {
        let mut e = event_from("subagent_stop", EventKind::Observation, "delegate report", project);
        e.agent = Some("worker".into());
        e
    }

    #[test]
    fn only_prose_is_queued_for_embedding_and_counted_in_coverage() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut skipped: Vec<Event> =
            SKIPPED_HOOKS.iter().map(|h| event_from(h, EventKind::Observation, h, project)).collect();
        skipped.push(delegate_footstep(project));
        let mut kept: Vec<Event> =
            KEPT_HOOKS.iter().map(|h| event_from(h, EventKind::Observation, h, project)).collect();
        kept.push(event_from("consolidate", EventKind::SessionSummary, "summed up", project));
        kept.push(delegate_report(project));
        for e in skipped.iter().chain(kept.iter()) {
            store.index(e).unwrap();
        }

        let pending = store.events_missing_vectors(&project.to_string(), 100).unwrap();
        let mut ids: Vec<_> = pending.iter().map(|(id, _)| id.clone()).collect();
        ids.sort();
        let mut want: Vec<_> = kept.iter().map(|e| e.id.clone()).collect();
        want.sort();
        assert_eq!(ids, want, "exactly the prose, including a delegate's subagent_stop report");

        let rows: Vec<_> = want.iter().map(|id| (id.clone(), vec![7u8; crate::embed::DIMS])).collect();
        store.set_vectors(&rows).unwrap();
        let total = kept.len() as i64;
        assert_eq!(store.vector_coverage().unwrap(), (total, total));
        assert!(store.events_missing_vectors(&project.to_string(), 100).unwrap().is_empty());

        // A vector left on a skipped event must not push coverage past 100%.
        store.set_vectors(&[(skipped[0].id.clone(), vec![7u8; crate::embed::DIMS])]).unwrap();
        assert_eq!(store.vector_coverage().unwrap(), (total, total));
    }

    #[test]
    fn payload_vectors_are_dropped_in_batches_once_and_the_log_is_left_alone() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut skipped: Vec<Event> =
            SKIPPED_HOOKS.iter().map(|h| event_from(h, EventKind::Observation, h, project)).collect();
        skipped.push(delegate_footstep(project));
        let mut kept: Vec<Event> =
            KEPT_HOOKS.iter().map(|h| event_from(h, EventKind::Observation, h, project)).collect();
        kept.push(delegate_report(project));
        let all: Vec<&Event> = skipped.iter().chain(kept.iter()).collect();
        for e in &all {
            store.index(e).unwrap();
        }
        let rows: Vec<_> = all.iter().map(|e| (e.id.clone(), vec![7u8; crate::embed::DIMS])).collect();
        store.set_vectors(&rows).unwrap();
        let count = |sql: &str| -> i64 { store.conn.query_row(sql, [], |r| r.get(0)).unwrap() };
        let marked = || count("SELECT COUNT(*) FROM schema_state WHERE key = 'payload_vectors_dropped'");

        // Opening never does it: the vectors are all still there.
        store.migrate().unwrap();
        assert_eq!(count("SELECT COUNT(*) FROM event_vec"), all.len() as i64);
        assert_eq!(marked(), 0);

        // Bounded: a batch of 4 leaves the rest, and the marker waits for an empty batch.
        assert_eq!(store.drop_payload_vectors(4).unwrap(), 4);
        assert_eq!(marked(), 0);
        let mut guard = 0;
        while store.drop_payload_vectors(4).unwrap() > 0 {
            guard += 1;
            assert!(guard < 10, "the migration did not converge");
        }
        assert_eq!(marked(), 1);
        assert_eq!(count("SELECT COUNT(*) FROM event_vec"), kept.len() as i64);
        assert_eq!(count("SELECT COUNT(*) FROM events"), all.len() as i64);

        // Once marked, a later vector for a skipped event is not touched.
        store.set_vectors(&[(skipped[0].id.clone(), vec![7u8; crate::embed::DIMS])]).unwrap();
        assert_eq!(store.drop_payload_vectors(100).unwrap(), 0);
        assert_eq!(count("SELECT COUNT(*) FROM event_vec"), kept.len() as i64 + 1, "the migration ran twice");
    }

    fn recorded_embedding_model(store: &Store) -> String {
        store
            .conn
            .query_row("SELECT value FROM schema_state WHERE key = 'embedding_model'", [], |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn a_process_that_outlived_a_model_swap_neither_writes_nor_reads_vectors() {
        // The open-time check cannot reach a process already running: an MCP
        // server from before the upgrade, a consolidation run in flight. Once
        // a newer build has re-stamped the index, that process's vectors are
        // noise to the new model and the new model's vectors are noise to
        // it, so both directions have to refuse - and say why.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let page = event("Chose SQLite over Postgres", "one file", project);
        store.index(&page).unwrap();
        store.set_vectors(&[(page.id.clone(), vec![7u8; crate::embed::DIMS])]).unwrap();

        // Another process, a newer build, re-stamps the index underneath us.
        store
            .conn
            .execute("UPDATE schema_state SET value = 'a-newer-model:256' WHERE key = 'embedding_model'", [])
            .unwrap();

        let written = store.set_vectors(&[(page.id.clone(), vec![9u8; crate::embed::DIMS])]);
        let error = written.expect_err("a stale process wrote into an index that is no longer its own");
        assert!(error.to_string().contains("a-newer-model:256"), "{error}");
        assert!(error.to_string().contains(&crate::embed::signature()), "{error}");
        let stored: Vec<u8> = store
            .conn
            .query_row("SELECT vec FROM event_vec WHERE event_id = ?1", params![page.id], |row| row.get(0))
            .unwrap();
        assert_eq!(stored, vec![7u8; crate::embed::DIMS], "the refused write still changed the row");

        let read = store.nearest(&project.to_string(), &vec![7u8; crate::embed::DIMS], None, 10);
        let error = read.expect_err("a stale process ranked vectors another model wrote");
        assert!(error.to_string().contains("restarted"), "{error}");
    }

    #[test]
    fn the_trace_shows_each_streams_rank_and_the_fusion_arithmetic() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let short = event("Chose SQLite over Postgres", "one file, no daemon", project);
        let long = event("SQLite considered at length", &"words and more words ".repeat(60), project);
        let other = event("Wrote the installation guide", "one page", project);
        store.index(&short).unwrap();
        store.index(&long).unwrap();
        store.index(&other).unwrap();

        let (hits, trace) =
            store.search_traced(&project.to_string(), "sqlite", None, 10, Recall::Fused, false).unwrap();

        let names: Vec<&str> = trace.streams.iter().map(|stream| stream.name).collect();
        assert_eq!(names, ["keyword", "semantic", "entity", "substring", "graph", "keyword_relaxed"]);
        assert_eq!(trace.rank_in("keyword", &short.id), Some(1), "{trace:?}");
        assert_eq!(trace.rank_in("keyword", &long.id), Some(2), "{trace:?}");
        assert_eq!(trace.rank_in("keyword", &other.id), None);
        // An English query has word boundaries: the substring stream says
        // why it stood aside rather than looking like it found nothing.
        assert!(
            trace.streams[3].note.as_deref().unwrap_or("").starts_with("skipped"),
            "{:?}",
            trace.streams[3].note
        );

        let fused = trace.fused.get(&short.id).expect("a keyword hit has fusion arithmetic");
        assert!(fused.raw > 0.0);
        assert!((fused.discount - 1.0).abs() < f32::EPSILON, "a short entry pays no length discount");
        assert!((fused.score - fused.raw).abs() < f32::EPSILON);
        assert!(!fused.demoted);

        let fused = trace.fused.get(&long.id).expect("the long hit has fusion arithmetic");
        assert!(fused.length > 200, "{}", fused.length);
        assert!(fused.discount > 1.0, "a long entry was not discounted: {fused:?}");
        assert!(fused.score < fused.raw);
        assert_eq!(hits[0].id, short.id, "{hits:?}");
    }

    #[test]
    fn a_natural_question_compiles_to_an_or_of_three_term_groups() {
        let fts = compile_fts("how does the summarizer ladder pick a CLI").expect("a question compiles");
        for dropped in ["how", "does", "\"the\"", "\"a\""] {
            assert!(!fts.contains(dropped), "{dropped} survived: {fts}");
        }
        // summarizer, ladder, pick, cli: four terms, C(4,3) groups.
        assert_eq!(fts.matches(" OR ").count(), 3, "{fts}");
        assert!(fts.contains("(\"summarizer\" \"ladder\" \"pick\")"), "{fts}");
    }

    #[test]
    fn a_query_that_speaks_fts_is_left_alone() {
        for raw in [
            "\"exact phrase\" here today",
            "sqlite OR postgres store",
            "alpha NOT beta gamma",
            "alpha NEAR(beta gamma) delta",
            "migrat* the store layer",
            "two words",
        ] {
            assert_eq!(compile_fts(raw), None, "{raw}");
        }
    }

    #[test]
    fn thai_terms_and_paths_survive_compilation() {
        let fts = compile_fts("ทำไม ภาษาไทย ค้นไม่เจอ").unwrap();
        for term in ["ทำไม", "ภาษาไทย", "ค้นไม่เจอ"] {
            assert!(fts.contains(&format!("\"{term}\"")), "{term} lost: {fts}");
        }
        let fts = compile_fts("why does src/billing.rs fail on retry").unwrap();
        assert!(fts.contains("\"src/billing.rs\""), "{fts}");
    }

    #[test]
    fn more_than_five_terms_keep_the_five_longest() {
        let fts = compile_fts("aaa bbbb ccccc dddddd eeeeeee ffffffff ggg").unwrap();
        for kept in ["bbbb", "ccccc", "dddddd", "eeeeeee", "ffffffff"] {
            assert!(fts.contains(kept), "{kept} dropped: {fts}");
        }
        assert!(!fts.contains("aaa") && !fts.contains("ggg"), "{fts}");
        assert_eq!(fts.matches(" OR ").count(), 9, "C(5,3) = 10 groups: {fts}");
    }

    #[test]
    fn a_note_matching_three_of_five_terms_arrives_through_keyword_relaxed() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut note = event("Retry policy", "retry backoff jitter keeps the herd apart", project);
        note.kind = EventKind::Knowledge;
        let noise = event("Wrote the installation guide", "one page", project);
        store.index(&note).unwrap();
        store.index(&noise).unwrap();

        let (hits, trace) = store
            .search_traced(
                &project.to_string(),
                "how does retry backoff jitter get capped by ceiling",
                None,
                10,
                Recall::Fused,
                false,
            )
            .unwrap();

        assert_eq!(trace.rank_in("keyword", &note.id), None, "the raw AND found it: {trace:?}");
        assert_eq!(trace.rank_in("keyword_relaxed", &note.id), Some(1), "{trace:?}");
        assert!(hits.iter().any(|hit| hit.id == note.id), "{hits:?}");
        assert!(trace.streams.last().unwrap().note.is_some());
    }

    #[test]
    fn an_fts_query_is_searched_as_written_with_the_relaxed_stream_skipped() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let note = event("Retry policy", "retry backoff jitter keeps the herd apart", project);
        store.index(&note).unwrap();

        let (_, trace) = store
            .search_traced(&project.to_string(), "retry OR backoff OR jitter", None, 10, Recall::Fused, false)
            .unwrap();

        let relaxed = trace.streams.last().unwrap();
        assert_eq!(relaxed.name, "keyword_relaxed");
        assert!(relaxed.ranked.is_empty(), "{trace:?}");
        assert!(relaxed.note.as_deref().unwrap_or("").starts_with("skipped"), "{trace:?}");
        assert_eq!(trace.rank_in("keyword", &note.id), Some(1), "{trace:?}");
    }

    #[test]
    fn the_trace_carries_a_human_demotion() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let stale = event("SQLite chosen for the cache", "revisited later", project);
        let live = event("SQLite chosen for the store", "still true", project);
        store.index(&stale).unwrap();
        store.index(&live).unwrap();
        store.conn.execute("UPDATE events SET confidence = -1 WHERE id = ?1", params![stale.id]).unwrap();

        let (hits, trace) =
            store.search_traced(&project.to_string(), "sqlite", None, 10, Recall::Fused, false).unwrap();

        assert!(trace.fused[&stale.id].demoted, "{trace:?}");
        assert!(!trace.fused[&live.id].demoted);
        assert_eq!(hits.last().unwrap().id, stale.id, "a demoted hit did not sort last: {hits:?}");
    }

    #[test]
    fn the_floor_keeps_what_words_or_a_close_cosine_found_and_drops_loose_company() {
        let tail = Some(SEMANTIC_TOP_KEEP);
        assert_eq!(floor_reason(bit(KEYWORD), None, None), None, "a keyword hit stays");
        assert_eq!(floor_reason(bit(RELAXED), None, None), None, "a relaxed keyword hit stays");
        assert_eq!(floor_reason(bit(SUBSTRING), None, None), None, "a substring hit stays");
        assert_eq!(floor_reason(bit(SEMANTIC), Some(0.40), tail), None, "a close cosine stays");
        assert_eq!(floor_reason(bit(SEMANTIC) | bit(ENTITY), Some(0.30), tail), None, "cosine above the nearest floor with entity stays");
        assert_eq!(floor_reason(bit(SEMANTIC) | bit(GRAPH), Some(0.30), tail), None, "... or with graph");
        assert_eq!(floor_reason(bit(SEMANTIC), Some(0.30), tail), Some(Dropped::LooseSemantic), "a loose tail cosine alone goes");
        assert_eq!(floor_reason(bit(SEMANTIC), Some(0.30), Some(SEMANTIC_TOP_KEEP - 1)), None, "the stream's top guesses stay");
        assert_eq!(floor_reason(bit(ENTITY), None, None), Some(Dropped::Entity));
        assert_eq!(floor_reason(bit(GRAPH), None, None), Some(Dropped::Graph));
        assert_eq!(floor_reason(bit(ENTITY) | bit(GRAPH), None, None), Some(Dropped::EntityAndGraph));
        assert_eq!(Dropped::LooseSemantic.to_string(), "semantic only, below 0.36");
    }

    #[test]
    fn the_stream_positions_match_the_stream_names() {
        for (position, name) in [
            (KEYWORD, "keyword"),
            (SEMANTIC, "semantic"),
            (ENTITY, "entity"),
            (SUBSTRING, "substring"),
            (GRAPH, "graph"),
            (RELAXED, "keyword_relaxed"),
        ] {
            assert_eq!(STREAM_NAMES[position], name);
        }
    }

    #[test]
    fn fuse_applies_the_floor_per_stream_combination() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut ids = std::collections::HashMap::new();
        for name in [
            "sem_below_floor", "sem_top", "sem_close", "sem_rank4", "sem_loose_graph", "entity_graph", "entity",
            "graph", "keyword", "relaxed", "at_boundary", "just_below",
        ] {
            let event = event(&format!("Note {name}"), "body", project);
            store.index(&event).unwrap();
            ids.insert(name, event.id);
        }
        let hit = |name: &str| store.hits_by_id(&[ids[name].clone()]).unwrap().remove(0);
        let mut lists: Vec<Vec<Hit>> = vec![Vec::new(); 6];
        lists[SEMANTIC] = [
            "sem_below_floor", "sem_top", "sem_close", "sem_rank4", "sem_loose_graph", "at_boundary", "just_below",
        ]
        .map(hit)
        .to_vec();
        lists[GRAPH] = ["sem_loose_graph", "entity_graph", "graph"].map(hit).to_vec();
        lists[ENTITY] = ["entity_graph", "entity"].map(hit).to_vec();
        lists[KEYWORD] = vec![hit("keyword")];
        lists[RELAXED] = vec![hit("relaxed")];
        let cosines: std::collections::HashMap<String, f32> = [
            ("sem_below_floor", NEAREST_FLOOR - 0.03),
            ("sem_top", 0.30),
            ("sem_close", 0.40),
            ("sem_rank4", 0.30),
            ("sem_loose_graph", 0.30),
            ("at_boundary", SEMANTIC_MATCH),
            ("just_below", SEMANTIC_MATCH - 0.001),
        ]
        .into_iter()
        .map(|(name, cosine)| (ids[name].clone(), cosine))
        .collect();

        let (hits, fused) = store.fuse(&lists, 20, Some(&cosines)).unwrap();

        let kept: std::collections::HashSet<&str> = hits.iter().map(|hit| hit.id.as_str()).collect();
        let expect = [
            ("sem_below_floor", Some(Dropped::LooseSemantic)), // index 0, but under NEAREST_FLOOR
            ("sem_top", None),                                 // index 1, cos 0.30: top-3 keeps it
            ("sem_close", None),
            ("sem_rank4", Some(Dropped::LooseSemantic)),       // index 3, cos 0.30: the tail
            ("sem_loose_graph", None),
            ("entity_graph", Some(Dropped::EntityAndGraph)),
            ("entity", Some(Dropped::Entity)),
            ("graph", Some(Dropped::Graph)),
            ("keyword", None),
            ("relaxed", None),
            ("at_boundary", None),
            ("just_below", Some(Dropped::LooseSemantic)),
        ];
        for (name, dropped) in expect {
            let id = ids[name].as_str();
            assert_eq!(fused[id].dropped, dropped, "{name}");
            assert_eq!(kept.contains(id), dropped.is_none(), "{name} on the page: {hits:?}");
        }

        let (open_hits, open) = store.fuse(&lists, 20, None).unwrap();
        assert_eq!(open_hits.len(), 12, "with the floor off nothing leaves the pool");
        assert!(open.values().all(|entry| entry.dropped.is_none()));
    }

    #[test]
    fn an_id_only_the_entity_stream_proposed_stays_off_the_floored_page() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let named = event("Investigated zebrafish assays", "the zebrafish colony", project);
        let bystander = event("Ordered lunch", "sandwiches and soup", project);
        store.index(&named).unwrap();
        store.index(&bystander).unwrap();
        store
            .record_entities(&Uuid::nil().to_string(), &project.to_string(), &["zebrafish".to_string()])
            .unwrap();

        let ask = |floor| {
            store.search_traced(&project.to_string(), "zebrafish", None, 10, Recall::Fused, floor).unwrap()
        };
        let (open, open_trace) = ask(false);
        assert!(open.iter().any(|hit| hit.id == bystander.id), "entity brings the session along: {open:?}");
        assert!(open_trace.fused.values().all(|entry| entry.dropped.is_none()), "floor off drops nothing");
        let (hits, trace) = ask(true);
        assert!(hits.iter().any(|hit| hit.id == named.id), "{hits:?}");
        assert!(!hits.iter().any(|hit| hit.id == bystander.id), "{hits:?}");
        assert_eq!(trace.fused[&bystander.id].dropped, Some(Dropped::Entity), "{trace:?}");
        assert_eq!(trace.fused[&named.id].dropped, None);
        // The floored page is a strict subset of the unfloored pool, and the
        // difference is exactly the ids the trace names as dropped.
        let floored: std::collections::HashSet<&str> = hits.iter().map(|hit| hit.id.as_str()).collect();
        let pool: std::collections::HashSet<&str> = open.iter().map(|hit| hit.id.as_str()).collect();
        let named_dropped: std::collections::HashSet<&str> =
            trace.fused.iter().filter(|(_, entry)| entry.dropped.is_some()).map(|(id, _)| id.as_str()).collect();
        assert!(floored.is_subset(&pool) && floored.len() < pool.len(), "{floored:?} vs {pool:?}");
        assert_eq!(pool.difference(&floored).copied().collect::<std::collections::HashSet<_>>(), named_dropped);
    }

    #[test]
    fn a_lexical_search_traces_only_the_keyword_stream() {
        // Lexical recall is what anything destructive gets: the trace must
        // not claim streams that never ran.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let only = event("Chose SQLite over Postgres", "one file", project);
        store.index(&only).unwrap();

        let (hits, trace) =
            store.search_traced(&project.to_string(), "sqlite", None, 10, Recall::Lexical, false).unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(trace.streams.len(), 1);
        assert_eq!(trace.streams[0].name, "keyword");
        assert_eq!(trace.rank_in("keyword", &only.id), Some(1));
        assert!(trace.fused.is_empty(), "nothing was fused, so nothing has fusion arithmetic");
    }

    #[test]
    fn a_delegates_capture_keeps_its_agent_through_the_index() {
        // The tag is what lets a summary tell a scout's footsteps from the
        // lead's work; lost between the log and the index it is no tag.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut footstep = event("Read: src/main.rs", "{}", project);
        footstep.agent = Some("rolepod:scout".to_string());
        let own = event("Edit: src/main.rs", "{}", project);
        store.index(&footstep).unwrap();
        store.index(&own).unwrap();

        let back = store.get(&[footstep.id.clone(), own.id.clone()]).unwrap();
        assert_eq!(back[0].agent.as_deref(), Some("rolepod:scout"));
        assert_eq!(back[1].agent, None);
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

    /// A seed's own session is not its neighbour.
    ///
    /// Measured on a real 22k-event brain: one session's summary took #1 for
    /// 21 of 23 queries, including one whose words appear nowhere in the
    /// database. Every seed came from that session, so `subject` absorbed all
    /// 1098 of its entity names and the session shared 100% of them with
    /// itself - unbeatable under any normalisation, because the number was
    /// never about relevance. The graph stream then filled with that
    /// session's other events, and fusion handed them the top of the search.
    ///
    /// The stream exists to reach what the query's words could not. A seed's
    /// own session is already in the results; returning the rest of it is not
    /// a second opinion, it is the same opinion counted again - and a long
    /// session gets to count it more times than anyone else.
    /// What a search offered, and what an agent actually opened.
    ///
    /// Every ranking change in this project has had to score itself against
    /// a model's opinion of a list of titles - a proxy that shares the
    /// ranking's own blind spots. An agent calling `brain_get` on an entry it
    /// saw only the title of is a judgement made for its own reasons. This
    /// records the difference so that, given enough sessions, the next
    /// ranking change can be measured against something real.
    #[test]
    fn opening_an_entry_is_recorded_apart_from_merely_being_offered() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let session = Uuid::new_v4().to_string();
        let offered = event_in_session("Offered only", "", project, Uuid::new_v4());
        let opened = event_in_session("Opened on purpose", "", project, Uuid::new_v4());
        store.index(&offered).unwrap();
        store.index(&opened).unwrap();

        // A search offers both.
        store
            .record_recalled(&session, [offered.id.as_str(), opened.id.as_str()].into_iter())
            .unwrap();
        // The agent reads one of them in full.
        store.record_opened(&session, [opened.id.as_str()].into_iter()).unwrap();

        let flag = |id: &str| -> i64 {
            store
                .conn
                .query_row(
                    "SELECT opened FROM recalled WHERE session = ?1 AND event_id = ?2",
                    params![&session, id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(flag(&offered.id), 0, "an entry only listed was marked as opened");
        assert_eq!(flag(&opened.id), 1, "opening after an offer was not recorded");

        // The offer must not erase it afterwards.
        store
            .record_recalled(&session, [opened.id.as_str()].into_iter())
            .unwrap();
        assert_eq!(flag(&opened.id), 1, "a later search offer erased that it was opened");

        // And a read is still counted once per session, not once per call.
        let reads: i64 = store
            .conn
            .query_row("SELECT read_count FROM events WHERE id = ?1", params![&opened.id], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(reads, 1, "the same entry was counted as read more than once");
    }

    /// A distilled note beats the long summary that merely mentions it.
    ///
    /// Length is not a fault in itself, but it flatters every stream that
    /// ranks: an embedding of a long entry sits near any query because it is
    /// the centroid of everything it covers, and BM25 finds more terms to
    /// match. Measured on a real brain, the entries a reranker judged
    /// correct had a median length of 82 bytes and the top-five entries it
    /// rejected ran to 815 - the answer was losing to the retrospective that
    /// mentions it in passing.
    #[test]
    fn a_short_answer_outranks_the_long_entry_that_only_mentions_it() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let session = Uuid::new_v4();

        // The answer: distilled, one subject, the word used once.
        store
            .index(&event_in_session("Notes on the month", "vermilion expiry is off by one", project, session))
            .unwrap();
        // A retrospective that covers eighty subjects and happens to say the
        // word more often than the answer does - so every stream that counts
        // terms or averages meaning prefers it.
        let sprawl = (0..80)
            .map(|n| {
                if n % 10 == 0 { format!("vermilion came up in topic {n}") } else { format!("unrelated topic {n} we also covered") }
            })
            .collect::<Vec<_>>()
            .join(" ");
        store.index(&event_in_session("Notes on the month", &sprawl, project, session)).unwrap();

        let hits = store.search(&project.to_string(), "vermilion", None, 10, Recall::Fused).unwrap();
        // Same title on both, so only length can separate them.
        assert_eq!(
            hits.first().map(|hit| hit.snippet.contains("off by one")),
            Some(true),
            "the sprawling entry outranked the answer: {:?}",
            hits.iter().map(|h| h.snippet.as_str()).collect::<Vec<_>>()
        );
        assert_eq!(hits.len(), 2, "the long entry must still be found, only ranked lower");
    }

    /// A first-place opinion outranks two vague agreements.
    ///
    /// RRF pays 1/(K+rank), so three lists nodding at rank ten used to beat
    /// one list certain at rank one. The two lists that nod most easily are
    /// entity and graph, and both nominate whole SESSIONS: a session that
    /// touched many names is nominated by nearly every query, and its events
    /// then arrive with two votes each while the entry that actually answers
    /// arrives with one.
    #[test]
    fn a_single_confident_stream_outranks_two_that_merely_agree() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let answer_session = Uuid::new_v4();
        let busy = Uuid::new_v4();

        // The entry the words point at, alone in its session.
        store
            .index(&event_in_session("Vermilion expiry check", "the answer", project, answer_session))
            .unwrap();
        // A session that touched the same file over and over, about
        // something else entirely.
        for n in 0..6 {
            store
                .index(&event_in_session(
                    &format!("Unrelated chore {n}"),
                    "another thing that touched the same file",
                    project,
                    busy,
                ))
                .unwrap();
        }
        for session in [answer_session, busy] {
            store
                .record_entities(&session.to_string(), &project.to_string(), &["src/gate.rs".to_string()])
                .unwrap();
        }

        let hits = store.search(&project.to_string(), "vermilion", None, 10, Recall::Fused).unwrap();
        assert_eq!(
            hits.first().map(|hit| hit.title.as_str()),
            Some("Vermilion expiry check"),
            "the session-nominating streams outvoted the entry the query names: {:?}",
            hits.iter().map(|h| h.title.as_str()).collect::<Vec<_>>()
        );
        // Still widening recall, not silenced: the busy session is present,
        // it just does not lead.
        assert!(
            hits.iter().any(|hit| hit.title.starts_with("Unrelated chore")),
            "halving a stream's weight must not remove what it finds: {:?}",
            hits.iter().map(|h| h.title.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_seeds_own_session_does_not_crowd_out_real_neighbours() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let busy = Uuid::new_v4();
        let neighbour = Uuid::new_v4();

        // The seed the query will actually match, plus a pile of unrelated
        // work that happened in the same long session.
        store
            .index(&event_in_session("Fixed the vermilion middleware", "expiry check", project, busy))
            .unwrap();
        for n in 0..8 {
            store
                .index(&event_in_session(
                    &format!("Unrelated chore number {n}"),
                    "nothing to do with the query",
                    project,
                    busy,
                ))
                .unwrap();
        }
        store
            .index(&event_in_session("Connection pool tuning", "idle timeout raised", project, neighbour))
            .unwrap();
        for session in [busy, neighbour] {
            store
                .record_entities(&session.to_string(), &project.to_string(), &["src/gate.rs".to_string()])
                .unwrap();
        }

        let hits = store.search(&project.to_string(), "vermilion", None, 10, Recall::Fused).unwrap();
        let titles: Vec<&str> = hits.iter().map(|hit| hit.title.as_str()).collect();
        assert!(titles.iter().any(|t| t.contains("vermilion")), "keyword hit lost: {titles:?}");
        assert!(
            titles.iter().any(|t| t.contains("Connection pool")),
            "the real neighbour never surfaced: {titles:?}"
        );
        assert!(
            !titles.iter().any(|t| t.contains("Unrelated chore")),
            "the seed's own session filled the graph stream: {titles:?}"
        );
    }

    /// `related` as it was before the shared-name count moved to the session:
    /// the join runs per event, so a session's events multiply by its names.
    const OLD_RELATED: &str = "WITH subject AS (
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
               AND e.forgotten = 0 AND e.kind NOT IN ('tombstone', 'retire')
               AND e.hook NOT IN ('correct', 'feedback', 'supersede')
         GROUP BY e.id
         ORDER BY shared DESC,
                  CASE WHEN e.confidence < 0 THEN 1 ELSE 0 END,
                  CASE e.kind
                      WHEN 'knowledge' THEN 0
                      WHEN 'session_summary' THEN 1
                      WHEN 'source' THEN 1
                      ELSE 2
                  END,
                  e.id DESC
         LIMIT ?3";

    /// A seed session with `names` entities and `chores` events, and a few
    /// sessions that share some of those names.
    fn related_fixture(chores: usize, names: usize) -> (Store, Uuid, String) {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let busy = Uuid::new_v4();
        let seed = event_in_session("The seed", "where it starts", project, busy);
        let seed_id = seed.id.clone();
        store.index(&seed).unwrap();
        for n in 0..chores {
            store.index(&event_in_session(&format!("Chore {n}"), "same session", project, busy)).unwrap();
        }
        let all: Vec<String> = (0..names).map(|n| format!("src/file{n}.rs")).collect();
        store.record_entities(&busy.to_string(), &project.to_string(), &all).unwrap();
        // The three branches of the recall floor and the ranking that a plain
        // capture never reaches: knowledge, a withdrawn event, a demoted one.
        let mixed = |label: &str, session: Uuid| {
            let mut lesson = event_in_session(&format!("Lesson {label}"), "kept", project, session);
            lesson.kind = EventKind::Knowledge;
            let forgotten = event_in_session(&format!("Forgotten {label}"), "withdrawn", project, session);
            let demoted = event_in_session(&format!("Demoted {label}"), "flagged stale", project, session);
            for event in [&lesson, &forgotten, &demoted] {
                store.index(event).unwrap();
            }
            store.conn.execute("UPDATE events SET forgotten = 1 WHERE id = ?1", params![forgotten.id]).unwrap();
            store.conn.execute("UPDATE events SET confidence = -1 WHERE id = ?1", params![demoted.id]).unwrap();
        };
        mixed("seed", busy);
        for (n, shared) in [3usize, 1, 2, 3].into_iter().enumerate() {
            let other = Uuid::new_v4();
            for k in 0..2 {
                store
                    .index(&event_in_session(&format!("Neighbour {n}.{k}"), "elsewhere", project, other))
                    .unwrap();
            }
            mixed(&n.to_string(), other);
            store.record_entities(&other.to_string(), &project.to_string(), &all[..shared]).unwrap();
        }
        (store, project, seed_id)
    }

    fn old_related(store: &Store, project: Uuid, id: &str, limit: usize) -> Vec<String> {
        let mut stmt = store.conn.prepare(OLD_RELATED).unwrap();
        let rows = stmt
            .query_map(params![id, project.to_string(), limit as i64], |row| row.get::<_, String>(0))
            .unwrap();
        rows.collect::<rusqlite::Result<Vec<_>>>().unwrap()
    }

    #[test]
    fn related_returns_what_the_per_event_join_returned() {
        let (store, project, seed) = related_fixture(4, 5);
        for built in [false, true] {
            if built {
                assert!(store.build_primer_indexes().unwrap());
            }
            let new: Vec<String> =
                store.related(&project.to_string(), &seed, 50).unwrap().into_iter().map(|hit| hit.id).collect();
            let old = old_related(&store, project, &seed, 50);
            assert!(old.len() > 8, "the fixture is too thin to compare: {}", old.len());
            assert_eq!(new, old, "built={built}");
            for limit in [3, 7] {
                let few: Vec<String> = store
                    .related(&project.to_string(), &seed, limit)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.id)
                    .collect();
                assert_eq!(few, old_related(&store, project, &seed, limit), "built={built} limit={limit}");
            }
        }
    }

    #[test]
    fn related_does_not_multiply_a_session_by_its_shared_names() {
        let (store, project, seed) = related_fixture(40, 30);
        let steps = |sql: &str| -> i32 {
            let mut stmt = store.conn.prepare(sql).unwrap();
            let mut rows = stmt.query(params![seed, project.to_string(), 50_i64]).unwrap();
            while rows.next().unwrap().is_some() {}
            drop(rows);
            stmt.get_status(rusqlite::StatementStatus::VmStep)
        };
        let (old, new) = (steps(OLD_RELATED), steps(RELATED_SQL));
        assert!(new * 3 < old, "related still scales with events x names: {new} steps against {old}");
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
    fn a_flag_survives_the_rebuild_that_forgets_everything_else() {
        // `brain_feedback` used to lower the column and append nothing, so a
        // rebuild raised every flagged entry back to the top and emptied the
        // review page - the only place flagging leads anywhere. There was no
        // second table to recover from either: on a real store two rebuilds in
        // one day left zero flags and no record of what had been flagged.
        //
        // A correction and a withdrawal both travel as events. So does this.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let target = event("Old approach", "superseded", project);
        store.index(&target).unwrap();

        let mut flag = event("Flagged: Old approach", "", project);
        flag.kind = EventKind::Note;
        flag.source.hook = "feedback".to_string();
        flag.links = vec![target.id.clone()];
        store.index(&flag).unwrap();

        let confidence = |label: &str| -> i64 {
            store
                .conn
                .query_row("SELECT confidence FROM events WHERE id = ?1", [&target.id], |r| {
                    r.get(0)
                })
                .expect(label)
        };
        assert!(confidence("before") < 0, "precondition: flagging demotes");

        store.clear().unwrap();
        store.index(&target).unwrap();
        store.index(&flag).unwrap();

        assert!(
            confidence("after") < 0,
            "the rebuild raised a flagged entry back to the top of its ranking"
        );
        assert_eq!(store.flagged(&project.to_string()).unwrap().len(), 1, "the review page emptied");

        // And the flag itself is bookkeeping, not memory: it must not come
        // back as a search result of its own, the same as a correction note.
        let hits =
            store.search(&project.to_string(), "Flagged", None, 10, Recall::Fused).unwrap();
        assert!(hits.iter().all(|h| h.id != flag.id), "a flag surfaced as a memory: {hits:?}");
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
        // Through the log, the way `brain_feedback` does it: the flag has to
        // survive a rebuild, so it travels as an event rather than a column
        // write nobody records.
        let mut flag = event_in_session("Flagged: Old approach", "", project, stale);
        flag.kind = EventKind::Note;
        flag.source.hook = "feedback".to_string();
        flag.links = vec![stale_id.clone()];
        store.index(&flag).unwrap();

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
    fn a_supersession_never_overwrites_a_human_correction() {
        // A correction says the derived wording was WRONG. A later session
        // re-deriving the same claim is the case the correction exists for,
        // so its wording must not come back; only the recurrence counts.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let page = Event::new(
            Uuid::nil(),
            project,
            Uuid::nil(),
            Source { cli: "brain".into(), hook: "gotcha".into() },
            EventKind::Knowledge,
            "release targets four platforms".into(),
            "four".into(),
        );
        store.index(&page).unwrap();
        let note = |hook: &str, title: &str, body: &str| {
            let mut note = Event::new(
                Uuid::nil(),
                project,
                Uuid::nil(),
                Source { cli: "brain".into(), hook: hook.into() },
                EventKind::Note,
                title.into(),
                body.into(),
            );
            note.links = vec![page.id.clone()];
            note
        };
        let row = || store.get(std::slice::from_ref(&page.id)).unwrap().remove(0);
        let confidence = || -> i64 {
            store
                .conn
                .query_row("SELECT confidence FROM events WHERE id = ?1", [&page.id], |r| r.get(0))
                .unwrap()
        };

        // Untouched page: the machine's newer wording lands, as before.
        store.index(&note("supersede", "release targets five platforms", "five")).unwrap();
        assert_eq!(row().title, "release targets five platforms");
        assert!(store.human_corrected(std::slice::from_ref(&page.id)).unwrap().is_empty());
        let before = confidence();

        // Corrected page: the human's wording stands through another
        // supersession, and the recurrence is still counted.
        store
            .index(&note("correct", "release targets five platforms, Intel macOS excluded", "minus one"))
            .unwrap();
        store.index(&note("supersede", "release targets five platforms", "five")).unwrap();
        let after = row();
        assert_eq!(
            after.title, "release targets five platforms, Intel macOS excluded",
            "the model undid a human fix"
        );
        assert_eq!(after.body, "minus one");
        assert_eq!(confidence(), before + 1, "recurrence on a corrected page went uncounted");
        assert_eq!(
            store.human_corrected(std::slice::from_ref(&page.id)).unwrap(),
            vec![page.id.clone()]
        );
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
    fn a_rebuild_remembers_what_an_agent_went_back_and_read() {
        // `rank()` puts an entry an agent actually opened above one merely
        // guessed at - evidence over heuristic, and the only such signal here.
        // It lives in `read_count` on the row, which a replay deletes and
        // re-inserts at its default, while the `recalled` table it came from
        // is deliberately spared by `clear`. So every rebuild flattened it in
        // silence: on a real store after two rebuilds in one day, 0 of 28,854
        // events had a read to their name and `recalled` still held 529 rows.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let event = event("read me", "body", project);
        store.index(&event).unwrap();

        store.record_recalled("session-a", std::iter::once(event.id.as_str())).unwrap();
        store.record_opened("session-b", std::iter::once(event.id.as_str())).unwrap();
        let before: i64 = store
            .conn
            .query_row("SELECT read_count FROM events WHERE id = ?1", [&event.id], |r| r.get(0))
            .unwrap();
        assert!(before > 0, "precondition: a read is counted, got {before}");

        store.clear().unwrap();
        store.index(&event).unwrap();

        let after: i64 = store
            .conn
            .query_row("SELECT read_count FROM events WHERE id = ?1", [&event.id], |r| r.get(0))
            .unwrap();
        assert_eq!(after, before, "the rebuild forgot that this entry had been read");
    }

    #[test]
    fn a_captured_event_runs_no_count_query() {
        // A hook mints the id a moment ago, so no `recalled` or `injected`
        // row can name it. Those two counts were 76% of a hook's SQL. The
        // rows are planted for an id the store has not seen - the only way to
        // tell a skipped count from a count that found nothing - and the
        // replay path, which does need them, is the test above.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let event = event("just captured", "body", project);
        store.record_recalled("session-a", std::iter::once(event.id.as_str())).unwrap();
        store.record_injected("session-a", std::slice::from_ref(&event.id), 0, 10, usize::MAX).unwrap();

        store.index_captured(&event).unwrap();

        let (reads, offers): (i64, i64) = store
            .conn
            .query_row(
                "SELECT read_count, injected_count FROM events WHERE id = ?1",
                [&event.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!((reads, offers), (0, 0), "the capture path still ran a count query");

        // Control: the planted rows are real, so the replay path counts them.
        store.index(&event).unwrap();
        let (reads, offers): (i64, i64) = store
            .conn
            .query_row(
                "SELECT read_count, injected_count FROM events WHERE id = ?1",
                [&event.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!((reads, offers), (1, 1), "the planted rows were never reachable");
    }

    #[test]
    fn the_replay_count_lookups_use_an_index_on_event_id() {
        // Both tables are keyed (session, event_id), so a lookup by event_id
        // alone scanned the whole table once per replayed event.
        let store = Store::open_memory().unwrap();
        for table in ["recalled", "injected"] {
            let plan: Vec<String> = store
                .conn
                .prepare(&format!("EXPLAIN QUERY PLAN SELECT COUNT(*) FROM {table} WHERE event_id = ?1"))
                .unwrap()
                .query_map(["x"], |r| r.get::<_, String>(3))
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            let plan = plan.join(" | ");
            assert!(plan.contains("USING COVERING INDEX") || plan.contains("USING INDEX"), "{table}: {plan}");
            assert!(!plan.contains("SCAN"), "{table} is scanned: {plan}");
        }
    }

    #[test]
    fn a_pointer_pushed_five_sessions_unread_stops_outranking_fresh_lines() {
        // The push-side twin of read_count: `injected` records every offer,
        // and five sessions of offers with zero pulls is the budget saying
        // the line buys nothing. The stale entry is created second so the
        // id DESC tie-break puts it first until decay - the flip below is
        // the decay bit and nothing else.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut fresh = event("never offered", "body", project);
        fresh.kind = EventKind::SessionSummary;
        fresh.id = "01TESTFRESH00000000000000A".to_string();
        let mut stale = event("offered forever", "body", project);
        stale.kind = EventKind::SessionSummary;
        // The larger id: wins the id DESC tie-break until decay flips it.
        stale.id = "01TESTSTALE00000000000000B".to_string();
        store.index(&fresh).unwrap();
        store.index(&stale).unwrap();

        for n in 0..4 {
            store.record_injected(&format!("s{n}"), std::slice::from_ref(&stale.id), 0, 10, usize::MAX).unwrap();
        }
        let before = store.pointers_of_kind(&project.to_string(), "session_summary", 10).unwrap();
        assert_eq!(before[0].id, stale.id, "four offers are not yet decay");

        store.record_injected("s4", std::slice::from_ref(&stale.id), 0, 10, usize::MAX).unwrap();
        let after = store.pointers_of_kind(&project.to_string(), "session_summary", 10).unwrap();
        assert_eq!(after[0].id, fresh.id, "the fifth unread offer must sink the pointer");
    }

    #[test]
    fn knowledge_is_exempt_from_push_decay() {
        // Knowledge earns its place as the line itself - 83% of it ever
        // surfaced against 2% of observations - and it is never pulled via
        // brain_get, so zero reads there is not evidence of uselessness.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut fresh = event("young lesson", "body", project);
        fresh.kind = EventKind::Knowledge;
        fresh.id = "01TESTFRESH00000000000000A".to_string();
        let mut stale = event("old lesson", "body", project);
        stale.kind = EventKind::Knowledge;
        stale.id = "01TESTSTALE00000000000000B".to_string();
        store.index(&fresh).unwrap();
        store.index(&stale).unwrap();

        for n in 0..6 {
            store.record_injected(&format!("s{n}"), std::slice::from_ref(&stale.id), 0, 10, usize::MAX).unwrap();
        }
        let pointers = store.pointers_of_kind(&project.to_string(), "knowledge", 10).unwrap();
        assert_eq!(pointers[0].id, stale.id, "a lesson must not decay for being shown");
    }

    #[test]
    fn a_standing_rule_outranks_every_other_lesson() {
        // hook = 'rule' marks knowledge distilled from corrections a person
        // made more than once - the strongest evidence in the table, so it
        // reads out first. The rule is created FIRST (older id) so only the
        // rank bit, not the id tie-break, can put it on top.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut rule = event("Never test against the real store", "body", project);
        rule.kind = EventKind::Knowledge;
        rule.source.hook = "rule".to_string();
        // The smaller id: the id DESC tie-break would bury it, so only the
        // rank bit can put it first.
        rule.id = "01TESTRULE000000000000000A".to_string();
        let mut gotcha = event("The fixture leaks between files", "body", project);
        gotcha.kind = EventKind::Knowledge;
        gotcha.source.hook = "gotcha".to_string();
        gotcha.id = "01TESTGOTCHA0000000000000B".to_string();
        store.index(&rule).unwrap();
        store.index(&gotcha).unwrap();

        let pointers = store.pointers_of_kind(&project.to_string(), "knowledge", 10).unwrap();
        assert_eq!(pointers[0].id, rule.id, "a rule must read out before other knowledge");
    }

    #[test]
    fn a_rebuild_remembers_how_often_a_pointer_was_offered() {
        // injected_count sits on the row like read_count did, and the same
        // rebuild that flattened read_count would hand every stale pointer a
        // fresh decay budget. `injected` survives `clear()`; restore from it.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let event = event("offered twice", "body", project);
        store.index(&event).unwrap();
        store.record_injected("session-a", std::slice::from_ref(&event.id), 0, 10, usize::MAX).unwrap();
        store.record_injected("session-b", std::slice::from_ref(&event.id), 0, 10, usize::MAX).unwrap();

        store.clear().unwrap();
        store.index(&event).unwrap();

        let count: i64 = store
            .conn
            .query_row("SELECT injected_count FROM events WHERE id = ?1", [&event.id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 2, "the rebuild forgot how often this pointer was offered");
    }

    #[test]
    fn retirable_spares_anything_ever_offered_or_read() {
        // Usage beats age, and "used" has two faces the counters split:
        // a read bumps read_count, but a pointer merely OFFERED by a primer
        // leaves only an `injected` row - read_count stays zero, and only
        // the NOT IN guard keeps it alive.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let make = |id: &str, title: &str| {
            let mut e = event(title, "an old body", project);
            e.id = id.to_string();
            e.consolidated = true;
            store.index(&e).unwrap();
            e
        };
        let unseen = make("01TESTUNSEEN000000000000AA", "nobody ever needed this");
        let offered = make("01TESTOFFERED00000000000BB", "offered but never read");
        let read = make("01TESTREAD00000000000000CC", "actually read");
        let raw = {
            let mut e = event("still in flight", "an old body", project);
            e.id = "01TESTRAW000000000000000DD".to_string();
            store.index(&e).unwrap();
            e
        };

        store.record_injected("s1", std::slice::from_ref(&offered.id), 0, 10, usize::MAX).unwrap();
        store.record_recalled("s1", std::iter::once(read.id.as_str())).unwrap();

        let (ids, bytes) =
            store.retirable(&project.to_string(), "9999-01-01T00:00:00Z").unwrap();
        assert_eq!(ids, vec![unseen.id.clone()], "only the never-surfaced body retires");
        assert!(bytes > 0, "the measurement must count the body it would drop");
        let _ = raw;
    }

    const LONG_AGO: &str = "2020-01-01T00:00:00.000000Z";
    const CUTOFF: &str = "2026-01-01T00:00:00.000000Z";

    /// An old, settled observation in `project`: the shape the rule drops.
    fn old_observation(store: &Store, title: &str, project: Uuid) -> Event {
        let mut old = event(title, "an old body", project);
        old.ts = LONG_AGO.to_string();
        old.consolidated = true;
        store.index(&old).unwrap();
        old
    }

    #[test]
    fn the_retention_rule_spares_whatever_anyone_saw_and_whatever_is_still_in_flight() {
        let store = Store::open_memory().unwrap();
        let (mine, yours) = (Uuid::new_v4(), Uuid::new_v4());
        let unseen = old_observation(&store, "nobody needed this", mine);
        let elsewhere = old_observation(&store, "nor this, in another project", yours);
        let injected = old_observation(&store, "offered by a primer", mine);
        let offered = old_observation(&store, "offered by a search", mine);
        let opened = old_observation(&store, "opened in full", mine);
        let counted = old_observation(&store, "read before the ledgers", mine);
        let forgotten = old_observation(&store, "withdrawn", mine);
        let mut unsettled = event("not consolidated yet", "an old body", mine);
        unsettled.ts = LONG_AGO.to_string();
        store.index(&unsettled).unwrap();
        let mut knowledge = event("a rule", "an old body", mine);
        knowledge.ts = LONG_AGO.to_string();
        knowledge.kind = EventKind::Knowledge;
        knowledge.consolidated = true;
        store.index(&knowledge).unwrap();
        let mut recent = event("this morning", "a fresh body", mine);
        recent.consolidated = true;
        store.index(&recent).unwrap();

        store.record_injected("s1", std::slice::from_ref(&injected.id), 0, 10, usize::MAX).unwrap();
        store.record_recalled("s1", std::iter::once(offered.id.as_str())).unwrap();
        store.record_opened("s1", std::iter::once(opened.id.as_str())).unwrap();
        store.conn.execute("UPDATE events SET read_count = 1 WHERE id = ?1", [&counted.id]).unwrap();
        store.conn.execute("UPDATE events SET forgotten = 1 WHERE id = ?1", [&forgotten.id]).unwrap();

        let step = store.retention_step(CUTOFF, 0, 1_000, 100).unwrap();
        let mut ids = step.ids.clone();
        ids.sort();
        let mut expected = vec![unseen.id.clone(), elsewhere.id.clone()];
        expected.sort();
        assert_eq!(ids, expected, "only an old, settled, never-surfaced observation goes, in any project");
        assert!(step.end, "the whole table fit in the window");
        assert_eq!(store.retention_pending(CUTOFF, 0, 1_000).unwrap(), 2);
        let first = store.conn.query_row("SELECT rowid FROM events WHERE id = ?1", [&unseen.id], |r| r.get(0)).unwrap();
        assert_eq!(store.retention_pending(CUTOFF, first, 1_000).unwrap(), 1, "a count from the cursor skips what is behind it");
        assert_eq!(store.retention_pending(CUTOFF, 0, first).unwrap(), 1, "a count stops at the end of its window");
    }

    #[test]
    fn the_retention_rule_keeps_a_prose_body_until_its_vector_exists() {
        // The embed backlog encodes title plus body, so a body dropped first
        // would leave a vector made from the title alone, for good.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let old_prose = |title: &str| {
            let mut old = prose_event(title, "an old body", project);
            old.ts = LONG_AGO.to_string();
            old.consolidated = true;
            store.index(&old).unwrap();
            old
        };
        let without = old_prose("no vector yet");
        let narrow = old_prose("a vector of another model");
        let embedded = old_prose("embedded");
        // A tool call is JSON the embedder skips, so there is nothing to wait for.
        let tool_call = old_observation(&store, "a tool call", project);
        let vector = |id: &str, width: usize| (id.to_string(), vec![1u8; width]);
        store
            .set_vectors(&[vector(&narrow.id, crate::embed::DIMS - 1), vector(&embedded.id, crate::embed::DIMS)])
            .unwrap();

        let step = store.retention_step(CUTOFF, 0, 1_000, 100).unwrap();
        let mut ids = step.ids.clone();
        ids.sort();
        let mut expected = vec![embedded.id.clone(), tool_call.id.clone()];
        expected.sort();
        assert_eq!(ids, expected, "a prose body went before its vector (without {}, narrow {})", without.id, narrow.id);
    }

    #[test]
    fn a_retention_step_reads_one_window_and_says_where_to_resume() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let ids: Vec<String> =
            (0..5).map(|n| old_observation(&store, &format!("old {n}"), project).id).collect();

        // A full slice stops at its last match.
        let first = store.retention_step(CUTOFF, 0, 100, 2).unwrap();
        assert_eq!((first.ids.as_slice(), first.next, first.end), (&ids[0..2], 2, false));
        let second = store.retention_step(CUTOFF, first.next, 100, 2).unwrap();
        assert_eq!((second.ids.as_slice(), second.next, second.end), (&ids[2..4], 4, false));
        // A short one saw its whole window, and the table ended inside it.
        let last = store.retention_step(CUTOFF, second.next, 100, 2).unwrap();
        assert_eq!((last.ids.as_slice(), last.next, last.end), (&ids[4..5], 104, true));

        // A window smaller than the table is the bound on one read.
        let narrow = store.retention_step(CUTOFF, 0, 3, 100).unwrap();
        assert_eq!((narrow.ids.as_slice(), narrow.next, narrow.end), (&ids[0..3], 3, false));
    }

    #[test]
    fn a_retention_step_walks_rowids_and_probes_the_usage_indexes() {
        // A scan of the table in rowid order is what bounds the read; through
        // an index on kind the same statement fetched every observation by id.
        let store = Store::open_memory().unwrap();
        assert!(store.build_primer_indexes().unwrap());
        let plan = plan_of(&store, &Store::retention_step_sql(), 4);
        assert!(
            plan.iter().any(|row| row.starts_with("SEARCH events USING INTEGER PRIMARY KEY (rowid>? AND rowid<?)")),
            "the step does not read a rowid window: {plan:#?}"
        );
        for table in ["injected", "recalled"] {
            assert!(
                plan.iter().any(|row| row.contains(&format!("COVERING INDEX {table}_by_event (event_id=?)"))),
                "{table} is not probed by event: {plan:#?}"
            );
        }
        assert!(!plan.iter().any(|row| row.contains("events_kind_proj")), "{plan:#?}");
    }

    #[test]
    fn a_committed_retention_step_empties_bodies_moves_the_cursor_and_a_clear_forgets_both() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let old = old_observation(&store, "old", project);
        let ids = vec![old.id.clone()];

        assert_eq!(store.retention_cursor().unwrap(), 0);
        assert_eq!(store.commit_retention_step(&ids, 7).unwrap(), 1);
        assert_eq!(stored_body(&store, &old.id), (String::new(), 1));
        assert_eq!((store.retention_cursor().unwrap(), store.retention_dropped().unwrap()), (7, 1));
        assert_eq!(store.commit_retention_step(&ids, 9).unwrap(), 0, "an empty body is not dropped twice");
        assert_eq!((store.retention_cursor().unwrap(), store.retention_dropped().unwrap()), (9, 1));

        store.finish_retention_pass("2026-10-08T00:00:00Z").unwrap();
        assert_eq!(store.retention_cursor().unwrap(), 0, "a finished pass starts over");
        assert_eq!(store.retention_done_at().unwrap().as_deref(), Some("2026-10-08T00:00:00Z"));

        // The replay restores every body, so what said they were dropped goes.
        store.clear().unwrap();
        assert_eq!(store.retention_done_at().unwrap(), None);
        assert_eq!(store.retention_dropped().unwrap(), 0);
    }

    #[test]
    fn settling_after_retention_merges_and_checkpoints_without_error() {
        // On a file, where there is a WAL for the checkpoint to empty.
        let dir = std::env::temp_dir().join(format!("brain-settle-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("brain.db");
        let store = Store::open(&db).unwrap();
        let old = old_observation(&store, "old", Uuid::new_v4());
        store.commit_retention_step(std::slice::from_ref(&old.id), 1).unwrap();
        store.settle_after_retention(10).unwrap();
        let wal = db.with_file_name("brain.db-wal");
        assert_eq!(std::fs::metadata(&wal).map_or(0, |wal| wal.len()), 0, "the checkpoint left the WAL full");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_rederived_fact_gains_confidence_and_a_user_fix_does_not() {
        // Being derived again from NEW summaries is recurrence evidence;
        // a user's correction says the entry was WRONG and earns nothing.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut fact = event("vitest runs file-by-file here", "body", project);
        fact.kind = EventKind::Knowledge;
        store.index(&fact).unwrap();

        let mut supersede = event("vitest must run file-by-file", "newer body", project);
        supersede.kind = EventKind::Note;
        supersede.source.hook = "supersede".to_string();
        supersede.links = vec![fact.id.clone()];
        store.index(&supersede).unwrap();

        let confidence: i64 = store
            .conn
            .query_row("SELECT confidence FROM events WHERE id = ?1", [&fact.id], |r| r.get(0))
            .unwrap();
        assert_eq!(confidence, 1, "a re-derivation must count as recurrence");

        let mut fix = event("actually it is per-directory", "fixed body", project);
        fix.kind = EventKind::Note;
        fix.source.hook = "correct".to_string();
        fix.links = vec![fact.id.clone()];
        store.index(&fix).unwrap();
        let confidence: i64 = store
            .conn
            .query_row("SELECT confidence FROM events WHERE id = ?1", [&fact.id], |r| r.get(0))
            .unwrap();
        assert_eq!(confidence, 1, "a user's fix must not be counted as recurrence");
    }

    #[test]
    fn a_supersession_note_never_reaches_search() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut fact = event("the cache is write-through", "body", project);
        fact.kind = EventKind::Knowledge;
        store.index(&fact).unwrap();
        let mut supersede = event("the zeppelin cache is write-through", "b", project);
        supersede.kind = EventKind::Note;
        supersede.source.hook = "supersede".to_string();
        supersede.links = vec![fact.id.clone()];
        store.index(&supersede).unwrap();

        let hits = store.search(&project.to_string(), "zeppelin", None, 10, Recall::Fused).unwrap();
        assert_eq!(hits.len(), 1, "only the fact itself may surface: {hits:?}");
        assert_eq!(hits[0].id, fact.id, "the bookkeeping note leaked into search");
    }

    #[test]
    fn a_prompt_that_orders_remembering_floats_above_the_noise() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let mut plain = event("Asked: what does the scheduler do", "how does it work", project);
        plain.source.hook = "user_prompt_submit".to_string();
        plain.id = "01TESTPLAIN0000000000000ZZ".to_string();
        let mut order = event(
            "Asked: remember that deploys happen on Fridays only",
            "remember that deploys happen on Fridays only",
            project,
        );
        order.source.hook = "user_prompt_submit".to_string();
        // The smaller id: only the intent boost can put it first.
        order.id = "01TESTORDER0000000000000AA".to_string();
        store.index(&plain).unwrap();
        store.index(&order).unwrap();

        let pointers = store.ranked_pointers(&project.to_string(), Kinds::Any, 10).unwrap();
        assert_eq!(
            pointers[0].id, order.id,
            "an explicit order to remember must outrank ordinary prompts"
        );

        // The same order typed into Cursor - a differently spelled prompt
        // hook - must get the same boost, or the rule is Claude-only.
        let mut cursor = event(
            "Asked: from now on, always use pnpm",
            "from now on, always use pnpm",
            project,
        );
        cursor.source.cli = "cursor".to_string();
        cursor.source.hook = "before_submit_prompt".to_string();
        cursor.id = "01TESTCURSOR000000000000AB".to_string();
        store.index(&cursor).unwrap();
        let confidence: i64 = store
            .conn
            .query_row("SELECT confidence FROM events WHERE id = ?1", [&cursor.id], |r| r.get(0))
            .unwrap();
        assert_eq!(confidence, 1, "an order typed into Cursor was not heard");
    }

    #[test]
    fn a_retirement_drops_the_body_and_keeps_the_pointer() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let target = event("Investigated the cache", "the zeta cipher payload", project);
        store.index(&target).unwrap();
        assert_eq!(
            store.search(&project.to_string(), "zeta", None, 10, Recall::Fused).unwrap().len(),
            1,
            "precondition: the body is searchable"
        );

        let mut retire = event("Retired 1 observation body", "", project);
        retire.kind = EventKind::Retire;
        retire.source.hook = "retire".to_string();
        retire.links = vec![target.id.clone()];
        store.index(&retire).unwrap();

        assert!(
            store.search(&project.to_string(), "zeta", None, 10, Recall::Fused).unwrap().is_empty(),
            "a retired body still matched"
        );
        let hits = store.search(&project.to_string(), "cache", None, 10, Recall::Fused).unwrap();
        assert_eq!(hits.len(), 1, "the title must stay findable");
        assert_eq!(hits[0].id, target.id);
        let body = store.get(std::slice::from_ref(&target.id)).unwrap().remove(0).body;
        assert!(body.is_empty(), "the body survived retirement: {body}");
        // The retire event itself is bookkeeping, not memory.
        assert!(
            store.search(&project.to_string(), "Retired", None, 10, Recall::Fused).unwrap().is_empty(),
            "the retire event leaked into search"
        );
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
    fn clear_drops_the_log_watermarks() {
        let store = Store::open_memory().unwrap();
        store.set_log_watermark("log_tail:/a.jsonl", "1:0:ab").unwrap();
        store.set_log_watermark("other", "kept").unwrap();
        store.clear().unwrap();
        assert_eq!(store.log_watermark("log_tail:/a.jsonl").unwrap(), None);
        assert_eq!(store.log_watermark("other").unwrap().as_deref(), Some("kept"));
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
            .record_injected("s1", &["01FLIGHT".into(), "01SUMMARY".into()], 1, 40, usize::MAX)
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
        store.record_injected("s1", &["01FLIGHT".into()], 0, 20, usize::MAX).unwrap();
        assert_eq!(store.in_flight_uptake().unwrap(), (1, 1));
    }

    #[test]
    fn a_compaction_reset_does_not_erase_the_uptake_history() {
        let store = Store::open_memory().unwrap();
        let session = "sess-1";

        // A pointer was pushed, and the agent went on to read it in full.
        store.record_injected(session, &["01AAA".to_string()], 0, 100, usize::MAX).unwrap();
        store.record_recalled(session, std::iter::once("01AAA")).unwrap();
        assert_eq!(store.injection_uptake().unwrap(), (1, 1));

        // The context is wiped by a compaction. The de-dup guard must forget,
        // but the measurement must not: the pull already happened.
        store.reset_injection_state(session).unwrap();
        assert!(!store.already_injected(session, "01AAA").unwrap());
        assert_eq!(store.injection_uptake().unwrap(), (1, 1));

        // Re-injecting the same pointer after the reset re-arms the guard
        // without double-counting the push.
        store.record_injected(session, &["01AAA".to_string()], 0, 100, usize::MAX).unwrap();
        assert!(store.already_injected(session, "01AAA").unwrap());
        assert_eq!(store.injection_uptake().unwrap(), (1, 1));
    }

    /// Two hooks of one session both read 192 bytes left and both built an
    /// injection to fit it. Only the first may land.
    #[test]
    fn a_racing_injection_cannot_spend_past_the_cap() {
        let store = Store::open_memory().unwrap();
        let session = "sess-1";

        assert!(store.record_injected(session, &["01AAA".to_string()], 0, 8000, 8192).unwrap());
        assert!(store.record_injected(session, &["01BBB".to_string()], 0, 192, 8192).unwrap());
        assert!(!store.record_injected(session, &["01CCC".to_string()], 0, 150, 8192).unwrap());
        assert_eq!(store.session_injected_bytes(session).unwrap(), 8192);
        // A rejected injection was never shown, so it must not read as one.
        assert!(!store.already_injected(session, "01CCC").unwrap());

        // A fresh session is bound by the same cap.
        assert!(!store.record_injected("sess-2", &["01DDD".to_string()], 0, 9000, 8192).unwrap());
        assert_eq!(store.session_injected_bytes("sess-2").unwrap(), 0);
    }

    #[test]
    fn a_session_in_flight_is_counted_by_captures_that_mean_something() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let talking = Uuid::new_v4();
        let idle = Uuid::new_v4();
        let mut n = 0;
        let mut add = |session: Uuid, hook: &str, title: &str, files: Vec<String>, consolidated: bool| {
            let mut event = Event::new(
                Uuid::nil(),
                project,
                session,
                Source { cli: "codex".into(), hook: hook.into() },
                EventKind::Observation,
                title.into(),
                String::new(),
            );
            n += 1;
            event.id = format!("01FLIGHT{n:018}");
            event.files = files;
            event.consolidated = consolidated;
            store.index(&event).unwrap();
            event.id
        };
        // Three bare commands and two prompts: only the prompts count.
        for i in 0..3 {
            add(talking, "post_tool_use", &format!("Ran: echo {i}"), vec![], false);
        }
        add(talking, "user_prompt_submit", "Asked: where were we?", vec![], false);
        let newest = add(talking, "user_prompt_submit", "Asked: finish it", vec![], false);
        // A session that only ran bare commands is not in flight.
        add(idle, "post_tool_use", "Ran: ls", vec![], false);
        // A summarized capture no longer counts, however recent.
        add(idle, "user_prompt_submit", "Asked: done already", vec![], true);

        let flight = store.unconsolidated_sessions(&project.to_string(), 5).unwrap();
        assert_eq!(flight.len(), 1, "only the talking session is in flight: {flight:?}");
        assert_eq!(flight[0].session, talking.to_string());
        assert_eq!(flight[0].cli, "codex");
        assert_eq!(flight[0].captures, 2, "bare commands must not be counted");
        assert_eq!(flight[0].newest_id, newest, "the line is keyed by the newest capture");
    }

    #[test]
    fn a_delegated_run_is_never_reported_as_work_in_flight() {
        // A review Claude Code delegated through the codex plugin: three
        // prompts, real captures, no summary yet - and none wanted. Naming it
        // as unfinished sends the next session to read a reviewer's transcript
        // as "what codex was doing".
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let review = Uuid::new_v4();
        let person = Uuid::new_v4();
        let mut n = 0;
        let mut add = |session: Uuid, title: &str| {
            let mut event = Event::new(
                Uuid::nil(),
                project,
                session,
                Source { cli: "codex".into(), hook: "user_prompt_submit".into() },
                EventKind::Observation,
                title.into(),
                String::new(),
            );
            n += 1;
            event.id = format!("01DELEGATE{n:016}");
            store.index(&event).unwrap();
        };
        add(review, "Asked: <role> adversarial review");
        add(review, "Asked: <task> the diff");
        add(review, "Asked: report");
        add(person, "Asked: fix the login redirect");
        store.record_session_invocation(&review.to_string(), "headless").unwrap();
        store.record_session_invocation(&person.to_string(), "interactive").unwrap();
        assert!(store.session_is_headless(&review.to_string()).unwrap());
        assert!(!store.session_is_headless(&person.to_string()).unwrap());

        let flight = store.unconsolidated_sessions(&project.to_string(), 5).unwrap();
        assert_eq!(flight.len(), 1, "the delegate must not be listed: {flight:?}");
        assert_eq!(flight[0].session, person.to_string());
    }

    #[test]
    fn a_session_cut_off_before_its_summary_leads_the_summary_list() {
        // The scenario: codex hit its limit mid-task, and a Claude session
        // opened seconds later asking what codex had been doing. The list of
        // summaries an agent is told to ask for must say that a newer session
        // exists and has none yet - or the previous session's summary passes
        // for the latest work.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let finished = Uuid::new_v4();
        let cut_off = Uuid::new_v4();
        let mut n = 0;
        let mut add = |session: Uuid, cli: &str, hook: &str, kind: EventKind, title: &str, files: Vec<String>, consolidated: bool| {
            let mut event = Event::new(
                Uuid::nil(),
                project,
                session,
                Source { cli: cli.into(), hook: hook.into() },
                kind,
                title.into(),
                String::new(),
            );
            n += 1;
            event.id = format!("01CUTOFF{n:018}");
            event.files = files;
            event.consolidated = consolidated;
            store.index(&event).unwrap();
            event.id
        };
        add(finished, "codex", "user_prompt_submit", EventKind::Observation, "Asked: add the form", vec![], true);
        add(finished, "codex", "consolidate", EventKind::SessionSummary, "Added the form", vec![], true);
        add(cut_off, "codex", "user_prompt_submit", EventKind::Observation, "Asked: review it", vec![], false);
        // A file touched counts; a bare command would not, and the row is
        // keyed by the newest capture that counts.
        let newest = add(cut_off, "codex", "post_tool_use", EventKind::Observation, "Edited form.ts", vec!["form.ts".into()], false);
        let project = project.to_string();

        let summaries = store.recent(&project, Some("codex"), Some("session_summary"), None, 10).unwrap();
        assert_eq!(summaries.len(), 2, "one in-flight row, one summary: {summaries:?}");
        assert_eq!(summaries[0].session, cut_off.to_string(), "the cut-off session leads");
        assert_eq!(summaries[0].id, newest, "keyed by its newest capture, like the primer's line");
        assert!(summaries[0].title.contains("2 capture(s) not yet summarized"), "{}", summaries[0].title);
        assert!(summaries[0].title.contains(&cut_off.to_string()), "the row names the session to read");
        assert_eq!(summaries[1].kind, "session_summary");

        // Another CLI's summary list does not carry codex's unfinished work.
        let other = store.recent(&project, Some("claude-code"), Some("session_summary"), None, 10).unwrap();
        assert!(other.is_empty(), "{other:?}");

        // The unfiltered list already leads with the captures themselves;
        // a synthetic row there would be the same id twice.
        let everything = store.recent(&project, None, None, None, 10).unwrap();
        assert_eq!(everything[0].id, newest);
        assert!(everything.iter().all(|hit| !hit.title.contains("not yet summarized")), "{everything:?}");

        // Reading one session whole is exactly that.
        let whole = store.recent(&project, None, None, Some(&cut_off.to_string()), 10).unwrap();
        assert_eq!(whole.len(), 2);
        assert!(whole.iter().all(|hit| hit.session == cut_off.to_string()));

        // `k` still bounds the answer.
        let one = store.recent(&project, Some("codex"), Some("session_summary"), None, 1).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].session, cut_off.to_string());
    }

    #[test]
    fn a_document_read_in_joins_the_pool_knowledge_is_distilled_from() {
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let make = |kind: crate::event::EventKind, hook: &str, title: &str, id: &str| {
            let mut event = crate::event::Event::new(
                Uuid::nil(),
                project,
                Uuid::nil(),
                crate::event::Source { cli: "brain".into(), hook: hook.into() },
                kind,
                title.into(),
                "body".into(),
            );
            event.id = id.to_string();
            event.consolidated = true;
            store.index(&event).unwrap();
        };
        make(crate::event::EventKind::SessionSummary, "consolidate", "A session", "01POOL0000000000000000001");
        make(crate::event::EventKind::Source, "ingest", "A document", "01POOL0000000000000000002");
        make(crate::event::EventKind::Note, "note", "A note", "01POOL0000000000000000003");

        let titles: Vec<String> = store
            .recent_summaries(&project.to_string(), 10)
            .unwrap()
            .into_iter()
            .map(|event| event.title)
            .collect();
        assert_eq!(titles, vec!["A document", "A session"], "a source recurs like a summary; a note does not");
    }

    #[test]
    fn a_teammates_lesson_is_never_offered_to_the_paths_that_rewrite_knowledge() {
        // `knowledge_entries` feeds folding, superseding and dedup - all
        // three REWRITE what they are handed. A teammate's entry must be
        // searchable and never appear here.
        let store = Store::open_memory().unwrap();
        let project = Uuid::new_v4();
        let make = |title: &str, id: &str| {
            let mut event = crate::event::Event::new(
                Uuid::nil(),
                project,
                Uuid::nil(),
                crate::event::Source { cli: "brain".into(), hook: "gotcha".into() },
                crate::event::EventKind::Knowledge,
                title.into(),
                "body".into(),
            );
            event.id = id.to_string();
            event.consolidated = true;
            event
        };
        store.index(&make("ours", "01TEAM0000000000000000001")).unwrap();
        store.index_team_event(&make("theirs", "01TEAM0000000000000000002")).unwrap();

        let titles: Vec<String> = store
            .knowledge_entries(&project.to_string())
            .unwrap()
            .into_iter()
            .map(|(_, title)| title)
            .collect();
        assert_eq!(titles, vec!["ours"], "a teammate's entry reached a rewriting path");

        // But it is memory like any other: recall still returns it.
        let recalled: Vec<String> = store
            .recent(&project.to_string(), None, Some("knowledge"), None, 10)
            .unwrap()
            .into_iter()
            .map(|hit| hit.title)
            .collect();
        assert!(recalled.contains(&"theirs".to_string()), "a teammate's lesson is not recallable: {recalled:?}");
    }
}
