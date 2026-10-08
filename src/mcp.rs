//! Stdio MCP server — the pull side of memory.
//!
//! Spawned per session by each CLI's MCP configuration and living exactly as
//! long as that session. This is the only path to full event bodies: automatic
//! injection carries titles and ids, and the agent calls in here when the task
//! actually needs the content.
//!
//! JSON-RPC 2.0 over newline-delimited stdin/stdout. Nothing else may ever be
//! written to stdout — a stray print corrupts the stream and the host CLI
//! reports the server as broken.

use std::io::{BufRead, Write};

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::config::Paths;
use crate::ids;
use crate::maint;
use crate::store::{Ledger, Store};

/// MCP protocol revision this server implements.
const PROTOCOL_VERSION: &str = "2025-06-18";

/// Default number of search hits when the caller does not say.
const DEFAULT_SEARCH_LIMIT: usize = 10;
/// Ceiling on hits, so one call cannot flood a context window.
const MAX_SEARCH_LIMIT: usize = 50;

/// Serve until stdin closes.
///
/// # Errors
/// Returns an error only when stdout cannot be written; malformed requests are
/// answered with JSON-RPC errors rather than terminating the session.
pub fn serve() -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let paths = Paths::resolve()?;

    // The project is fixed for the life of the session: the CLI spawned us
    // inside the checkout the user is working in.
    let cwd = std::env::current_dir().unwrap_or_default();
    let scope = ids::resolve_scope(&cwd);
    let project = scope.project_id.to_string();
    // Identifies this MCP session for the "has this id been surfaced?" check.
    // A per-process id is right: the guard is about what THIS conversation has
    // seen, and the server lives exactly as long as one.
    let mut client: Option<&'static str> = None;
    let mut session = session_key(client, std::process::id());

    for line in stdin.lock().lines() {
        let line = line.context("read MCP request")?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                write_message(&mut stdout, &error_response(Value::Null, -32700, &error.to_string()))?;
                continue;
            }
        };

        // Notifications carry no id and must not be answered at all.
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let method = request.get("method").and_then(Value::as_str).unwrap_or_default();
        let params = request.get("params").cloned().unwrap_or_else(|| json!({}));

        let response = match method {
            "initialize" => {
                (client, session) = on_initialize(&params, std::process::id());
                success(id, initialize_result())
            }
            "ping" => success(id, json!({})),
            "tools/list" => {
                // Listed once per session, so a model that finishes downloading
                // mid-session is still described as absent until the next one.
                // That is the honest reading anyway: this session's reranks
                // were quoted at the CLI price and will be charged it.
                let ready = crate::rerank::local_is_ready(
                    &paths.model_dir_for(crate::rerank::LOCAL_MODEL),
                );
                success(id, json!({ "tools": tool_definitions(ready) }))
            }
            "tools/call" => match call_tool(&paths, &project, &session, client, &params) {
                Ok(result) => success(id, result),
                // A failed tool call is reported inside the result, not as a
                // protocol error: the agent should see what went wrong and be
                // able to retry, rather than the client treating the server as
                // broken.
                Err(error) => success(
                    id,
                    json!({
                        "isError": true,
                        "content": [{"type": "text", "text": error.to_string()}],
                    }),
                ),
            },
            other => error_response(id, -32601, &format!("unknown method: {other}")),
        };

        write_message(&mut stdout, &response)?;
    }
    Ok(())
}

/// The CLI behind a MCP client, from `initialize.params.clientInfo.name`.
///
/// Only names observed on the wire belong here (probed with a logging MCP stub
/// in an isolated HOME; see the task receipt). A client that cannot be probed
/// without a model call stays out of the table and falls back to the old
/// behaviour - never a guess.
fn client_cli(params: &Value) -> Option<&'static str> {
    match params.pointer("/clientInfo/name").and_then(Value::as_str)? {
        "claude-code" => Some("claude-code"),
        _ => None,
    }
}

/// Per-session state an `initialize` request sets: the client and its session id.
fn on_initialize(params: &Value, pid: u32) -> (Option<&'static str>, String) {
    let client = client_cli(params);
    (client, session_key(client, pid))
}

/// The CLI whose cheap tier a rerank borrows: the calling client when it said
/// so, else the project's most recent CLI (looked up only then).
fn rerank_cli(
    client: Option<&str>,
    project_cli: impl FnOnce() -> Result<Option<String>>,
) -> Result<String> {
    match client {
        Some(cli) => Ok(cli.to_string()),
        None => Ok(project_cli()?.unwrap_or_default()),
    }
}

/// The id this MCP session answers to for the "has this id been surfaced?" check.
fn session_key(cli: Option<&str>, pid: u32) -> String {
    match cli {
        Some(cli) => format!("mcp-{cli}-{pid}"),
        None => format!("mcp-{pid}"),
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "rolepod-brain", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// The tool list, worded for the machine it is being listed on.
///
/// `local_rerank` is the one thing here that is not the same everywhere: a
/// build with the weights on disk answers a rerank in about 1.6s, and every
/// other build waits on a host CLI for around 12s. Quoting one number on both
/// leaves an agent either skipping a rerank it could have had for free, or
/// asking for one that costs a session's worth of patience. So the price on
/// the label is the price this machine charges.
fn tool_definitions(local_rerank: bool) -> Value {
    // Read once here rather than inside the macro: `json!` would otherwise
    // have to carry the branch, and the two strings are easier to compare
    // sitting next to each other.
    let rerank_cost = if local_rerank {
        "Runs on this machine in under two seconds - no subscription spent, \
         nothing sent anywhere - so ask for it whenever the first ordering \
         looks off."
    } else {
        "Costs about 12 seconds of waiting, 20 at most, via a host \
         CLI. Ask when the answer is worth that. The first \
         one also starts a one-off 600 MB download."
    };
    json!([
        {
            "name": "brain_search",
            "description": "Full-text search this project's memory. Returns matching \
                            observations with their ids, most relevant first (ties newest \
                            first). Use it \
                            before assuming context is lost: prior sessions in any CLI \
                            wrote here. Rank 1 is a candidate, not an answer: read the \
                            hits and take the one that fits. Empty or off-topic results \
                            earn one sharper retry - the subject named, a file path, a \
                            quoted phrase - before concluding it was never recorded. \
                            Pass an id to brain_get for the full body.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQLite FTS5 query. Bare words are ANDed; \
                                        \"quoted phrase\" matches exactly; OR and NOT work.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum hits (default 10, max 50).",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Narrow to one kind of memory: decision, \
                                        bugfix, feature, discovery, config, test. \
                                        Use it when the question is about a KIND of \
                                        thing - `topic: \"decision\"` answers \"what \
                                        did we decide\" without wading through every \
                                        mention.",
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": format!(
                            "Let a cheap model reorder results by what the \
                             question asked. {rerank_cost} Omit it \
                             for ordinary lookups; pass false to skip."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "brain_get",
            "description": "Fetch full observations by id. Ids come from brain_search \
                            results or from injected memory pointers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Event ids (ULIDs).",
                    },
                },
                "required": ["ids"],
            },
        },
        {
            "name": "brain_timeline",
            "description": "Chronological slice of this project's memory. Use it when \
                            the question is about ordering or when something changed \
                            (\"what happened after the refactor?\"), rather than about \
                            a topic.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "description": "ISO 8601 lower bound, e.g. 2026-08-01. Defaults \
                                        to the beginning of the log.",
                    },
                    "k": {"type": "integer", "description": "Maximum entries (default 10, max 50)."},
                },
            },
        },
        {
            "name": "brain_note",
            "description": "Save a durable note to this project's memory. Capture is \
                            automatic, so use this only for something worth remembering \
                            that no tool call would show: a decision and its reason, a \
                            constraint, a dead end worth not repeating.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note. One or two sentences."},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional repo-relative paths this note is about, \
                                        so it surfaces when those files are touched.",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "brain_forget",
            "description": "Withdraw a memory that is wrong or should not have been \
                            kept. Use when the user says a remembered thing is \
                            incorrect or asks you to forget it. Nothing is deleted \
                            from the log; recall stops returning it. Search for it \
                            or fetch it first: only ids this session has actually \
                            been shown can be withdrawn, and a pointer injected at \
                            session start does not count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Event id from a brain_search or brain_get in this session."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "brain_correct",
            "description": "Replace what a memory says, when the recorded version is \
                            wrong but the event itself matters. A claim can be true \
                            when written and false a week later; FINDING IT \
                            YOURSELF is the common case. If memory says something the code in front of \
                            you contradicts, correcting it is part of the work, not a \
                            favour. Write the replacement with a SHORT FIRST LINE - it \
                            becomes the title - and the detail below it. The original \
                            stays in the log; recall returns your text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Event id from a search or an injected pointer."},
                    "text": {"type": "string", "description": "What it should say instead."},
                },
                "required": ["id", "text"],
            },
        },
        {
            "name": "brain_feedback",
            "description": "Flag a memory as stale or unhelpful, without deleting \
                            it; for a wrong one use brain_correct or brain_forget.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Event id from a brain_search or brain_get in this session."},
                    "reason": {"type": "string", "description": "Optional: why, in a few words."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "brain_related",
            "description": "Given a memory id, returns other memories whose sessions \
                            touched the same files, symbols, or subjects.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Event id from a brain_search, brain_recent, or brain_get."},
                    "k": {"type": "integer", "description": "How many (default 10, max 50)."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "brain_doctor",
            "description": "Is this memory actually working? Runs the same checks \
                            as `brain doctor` at a terminal and returns them as \
                            data, one entry per check with `ok` and a `detail` \
                            line - render it however suits the conversation. \
                            Reach for it when the user asks whether brain is \
                            working, when recall looks empty or stale, or when \
                            anything about capture seems wrong. Most of what \
                            fails here fails silently, and this is the only \
                            thing that says so; a user who never opens a \
                            terminal has no other way to find out.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "brain_outline",
            "description": "What this project is, before you know what to ask. \
                            Returns durable knowledge, the subjects that recur \
                            across sessions, and how much has been captured. \
                            Call it first in an unfamiliar project: searching \
                            requires already suspecting what you are looking for.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k": {"type": "integer", "description": "How many subjects to name (default 10, max 50)."},
                },
            },
        },
        {
            "name": "brain_recent",
            "description": "Most recent observations in this project, newest first. \
                            Use it to re-orient at the start of a session or after a \
                            context compaction. Pass `cli` to see what ANOTHER \
                            agent did: this brain is shared by every CLI on the \
                            machine, so work done in codex or cursor is here even \
                            though you never saw it. Agents run several sessions \
                            at once, so pass `kind` too or the list interleaves \
                            unrelated work: `session_summary` is one line per \
                            session (what it has been doing), `raw` is what a \
                            session is doing right now, before it was summarized. \
                            To answer \"find the session where codex did X\": \
                            brain_search for X, take `session` off the hit, then \
                            call again with that `session` to read that one piece \
                            of work whole.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k": {
                        "type": "integer",
                        "description": "How many (default 10, max 50).",
                    },
                    "cli": {
                        "type": "string",
                        "description": "Only this CLI's observations — \
                                        `claude-code`, `codex`, `cursor`, \
                                        `gemini-cli`, `antigravity`, `opencode`. \
                                        Omit for every CLI.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "`session_summary` — one finished session \
                                        per line, led by any session whose captures \
                                        have not been summarized yet (a CLI that hit \
                                        its limit mid-task) with the call that reads \
                                        them. `raw` — live work not yet \
                                        summarized. Also `knowledge`, `note`, \
                                        `page_update`, `source` (a document read \
                                        in with `brain ingest`). Omit for all of them mixed, \
                                        which is only readable one session at a \
                                        time: every entry carries `session`.",
                    },
                    "session": {
                        "type": "string",
                        "description": "One session's work and nothing else. Take \
                                        the id from the `session` field of any hit \
                                        — brain_search, brain_recent, brain_related. \
                                        Combine with `kind` to read that session's \
                                        summary, or its live capture when it has \
                                        not been summarized yet.",
                    },
                },
            },
        },
    ])
}

/// Note what a tool call surfaced. Never a reason to fail the call: a write
/// the database refuses (the compact window holds it) is kept in
/// `surfaced.jsonl` and folded back by consolidation. In the window it does
/// not wait for the lock.
fn ledger<'a>(
    paths: &Paths,
    store: &Store,
    session: &str,
    how: Ledger,
    ids: impl Iterator<Item = &'a str>,
) {
    let ids: Vec<&str> = ids.collect();
    if ids.is_empty() {
        return;
    }
    if maint::active(paths) {
        let _ = store.set_busy_timeout(maint::FAIL_FAST);
    }
    let written = match how {
        Ledger::Opened => store.record_opened(session, ids.iter().copied()),
        _ => store.record_recalled(session, ids.iter().copied()),
    };
    if written.is_err() {
        let owned: Vec<String> = ids.iter().map(|id| (*id).to_string()).collect();
        maint::spill_surfaced(paths, session, how, &owned);
    }
}

/// Was this id shown to this session - by the ledger, or by a spill the
/// ledger has not taken back yet?
fn surfaced_to(paths: &Paths, store: &Store, session: &str, id: &str) -> Result<bool> {
    if maint::active(paths) {
        let _ = store.set_busy_timeout(maint::FAIL_FAST);
    }
    Ok(store.already_injected(session, id)?
        || store.was_recalled(session, id)?
        || maint::spilled(paths, session, id))
}

fn call_tool(
    paths: &Paths,
    project: &str,
    session: &str,
    client: Option<&str>,
    params: &Value,
) -> Result<Value> {
    let name = params.get("name").and_then(Value::as_str).unwrap_or_default();
    let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let store = if maint::active(paths) {
        Store::open_waiting(&paths.db(), maint::FAIL_FAST)?
    } else {
        Store::open(&paths.db())?
    };

    let payload = match name {
        "brain_search" => {
            let query = arguments
                .get("query")
                .and_then(Value::as_str)
                .filter(|q| !q.trim().is_empty())
                .context("brain_search requires a non-empty `query`")?;
            let limit = limit_from(&arguments);
            let config = crate::config::Config::load(&paths.config_file())?;
            // Per request first, config second. The config value is a standing
            // preference; the argument is this caller, on this question,
            // deciding the answer is worth waiting for. Only the caller knows
            // that, and it is not the same for two questions in a row.
            let rerank = arguments
                .get("rerank")
                .and_then(Value::as_bool)
                .unwrap_or(config.search.rerank);
            // With a reranker to sort them, it is worth pulling more than the
            // caller asked for: the entry that answers the question is often
            // just past the cut.
            // The wide pool whenever reranking is on, whichever engine ends
            // up doing it. Fetching thirty costs no more than fifteen, the
            // engine is not known until the hits are in hand, and a wider
            // candidate set reaches fusion before anything is trimmed - which
            // helps even the searches that are never reranked.
            let pool = if rerank { crate::rerank::LOCAL_POOL.max(limit) } else { limit };
            // An unknown topic would silently return nothing, which reads to
            // an agent as "no memory" rather than "wrong scope" - so a value
            // outside the taxonomy is ignored and the search runs unscoped.
            let topic = arguments
                .get("topic")
                .and_then(Value::as_str)
                .and_then(crate::event::normalize_topic);
            // The relevance floor applies only when this page is final: a
            // pool headed for the reranker stays wide for it to judge. If the
            // reranker then falls back to "none", that page goes out unfloored.
            let (mut hits, _) =
                store.search_traced(project, query, topic, pool, crate::store::Recall::Fused, !rerank)?;

            if rerank {
                // An entity lookup joins the wide pool: a query that names a
                // file or a service finds the work about it even when no title
                // contains the word. Appended rather than interleaved, so text
                // relevance still leads. Not on a floored page: an id only an
                // entity proposed is exactly what the floor keeps off it.
                let by_entity = store
                    .search_by_entity(project, &crate::consolidate::normalize_entity(query), pool)
                    .unwrap_or_default();
                let seen: std::collections::HashSet<String> =
                    hits.iter().map(|hit| hit.id.clone()).collect();
                hits.extend(by_entity.into_iter().filter(|hit| !seen.contains(&hit.id)).take(pool));
                hits.truncate(pool);

                let ladder = crate::summarizer::Ladder::new(&store, &config.summarizer);
                // Borrow the cheap tier of whichever CLI works here.
                // The client that is calling, when it said so; else the
                // project's most recent CLI as before.
                let cli = rerank_cli(client, || store.project_cli(project))?;
                let model_dir = paths.model_dir_for(crate::rerank::LOCAL_MODEL);
                let (reranked, outcome) =
                    crate::rerank::rerank(&ladder, &cli, query, &model_dir, hits);
                hits = reranked;
                // Telemetry, never a reason to fail the search.
                let _ = store.record_rerank(outcome.engine, outcome.reason, outcome.ms, outcome.cold);
            }
            hits.truncate(limit);

            ledger(paths, &store, session, Ledger::Recalled, hits.iter().map(|hit| hit.id.as_str()));
            json!({ "hits": hits, "count": hits.len() })
        }
        "brain_get" => {
            let ids: Vec<String> = arguments
                .get("ids")
                .and_then(Value::as_array)
                .context("brain_get requires an `ids` array")?
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect();
            // An id an agent is still holding - from an older primer, or its
            // own notes - must not resurrect what was withdrawn. `get` cannot
            // filter for us: consolidation reads through it and would leave
            // forgotten events pending forever.
            let mut events = store.get(&ids)?;
            events.retain(|event| store.event_exists(&event.id).unwrap_or(false));
            // The index keeps an observation bounded, or empty once retention
            // dropped it; the log has the whole line, and this is the one
            // reader that wants it. Not found (a pruned or moved log) leaves
            // what the index has.
            let clamped: Vec<&crate::event::Event> =
                events.iter().filter(|event| store.is_clamped(&event.id).unwrap_or(false)).collect();
            let mut bodies = full_bodies_from_log(paths, &clamped);
            for event in &mut events {
                if let Some(full) = bodies.remove(&event.id) {
                    event.body = full;
                }
            }
            // Not `record_recalled`: asking for a body, having seen only the
            // title, is the one moment an agent says an entry was worth the
            // tokens. Everything else in this file merely offered it.
            ledger(paths, &store, session, Ledger::Opened, events.iter().map(|event| event.id.as_str()));
            json!({ "events": events, "count": events.len() })
        }
        "brain_related" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .context("brain_related requires an `id`")?;
            let hits = store.related(project, id, limit_from(&arguments))?;
            ledger(paths, &store, session, Ledger::Recalled, hits.iter().map(|hit| hit.id.as_str()));
            json!({ "hits": hits, "count": hits.len() })
        }
        "brain_doctor" => {
            // The rendered text is for a terminal; an agent gets the checks
            // themselves, so it can say which one failed rather than quote a
            // wall of output.
            //
            // Where the memory lives rides along rather than being a tool of
            // its own. "Is it working" and "where is it kept" are one question
            // in a conversation, and a second call to answer the tail of the
            // first is a tool nobody would think to make.
            let checks = crate::doctor::run()?;
            let failing = checks.iter().filter(|check| !check.ok).count();
            let paths = Paths::resolve()?;
            json!({
                "ok": failing == 0,
                "failing": failing,
                "data_directory": paths.db().parent().map(|p| p.display().to_string()),
                "wiki": paths.wiki().display().to_string(),
                "project": project,
                "checks": checks
                    .iter()
                    .map(|check| json!({
                        "name": check.name,
                        "ok": check.ok,
                        "detail": check.detail,
                    }))
                    .collect::<Vec<_>>(),
            })
        }
        "brain_outline" => {
            let outline = store.outline(project, limit_from(&arguments))?;
            json!(outline)
        }
        "brain_recent" => {
            let cli = arguments.get("cli").and_then(Value::as_str);
            let kind = arguments.get("kind").and_then(Value::as_str).map(normalize_kind).transpose()?;
            // Not `session`: that name is taken by THIS conversation's id, and
            // the two mean opposite things - one is who is asking, the other is
            // whose work is being asked about.
            let of_session = arguments.get("session").and_then(Value::as_str);
            let hits = store.recent(project, cli, kind, of_session, limit_from(&arguments))?;
            ledger(paths, &store, session, Ledger::Recalled, hits.iter().map(|hit| hit.id.as_str()));
            json!({ "events": hits, "count": hits.len() })
        }
        "brain_forget" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .context("brain_forget requires an `id`")?;
            // A model may withdraw what it has been shown, not what it has
            // merely guessed at. Without this it could prune memory it never
            // saw, on nothing more than a plausible-looking id.
            anyhow::ensure!(
                surfaced_to(paths, &store, session, id)?,
                "id {id} has not been surfaced in this session; search for it first"
            );
            let outcome = crate::revise::forget(id)?;
            json!({ "forgot": id, "was": outcome.target_title, "recorded_as": outcome.id })
        }
        "brain_correct" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .context("brain_correct requires an `id`")?;
            let text = arguments
                .get("text")
                .and_then(Value::as_str)
                .context("brain_correct requires `text`")?;
            // The same floor forget and feedback enforce, and correct needs it
            // most: it decides what recall returns and what future sessions
            // are told, so an agent that could rewrite an id it never saw
            // could overwrite this project's memory from a guess - or from a
            // poisoned instruction it read somewhere.
            anyhow::ensure!(
                surfaced_to(paths, &store, session, id)?,
                "id {id} has not been surfaced in this session; search for it first"
            );
            let outcome = crate::revise::correct(id, text)?;
            json!({ "corrected": id, "was": outcome.target_title, "recorded_as": outcome.id })
        }
        "brain_feedback" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .context("brain_feedback requires an `id`")?;
            anyhow::ensure!(
                surfaced_to(paths, &store, session, id)?,
                "id {id} has not been surfaced in this session; search for it first"
            );
            let reason = arguments.get("reason").and_then(Value::as_str);
            crate::revise::flag(id, reason)?;
            json!({
                "flagged": id,
                "effect": "ranked lower and listed for review; nothing was deleted",
            })
        }
        "brain_timeline" => {
            let since = arguments.get("since").and_then(Value::as_str).unwrap_or("");
            let hits = store.timeline(project, since, limit_from(&arguments))?;
            json!({ "events": hits, "count": hits.len() })
        }
        "brain_note" => {
            let text = arguments
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .context("brain_note requires non-empty `text`")?;
            let files: Vec<String> = arguments
                .get("files")
                .and_then(Value::as_array)
                .map(|items| {
                    items.iter().filter_map(|item| item.as_str().map(str::to_string)).collect()
                })
                .unwrap_or_default();
            let id = write_note(paths, text, &files, client)?;
            json!({ "id": id, "saved": true })
        }
        other => anyhow::bail!("unknown tool: {other}"),
    };

    // Structured content plus a text mirror: clients that only render text
    // still show something useful.
    Ok(json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&payload)? }],
        "structuredContent": payload,
    }))
}

/// Append a hand-written note to the log and index it.
///
/// A note goes through the same sanitizer as captured text: an agent pasting a
/// config snippet into a note is exactly as likely to carry a secret as a tool
/// result is.
fn write_note(paths: &Paths, text: &str, files: &[String], client: Option<&str>) -> Result<String> {
    let scope = ids::resolve_scope(&std::env::current_dir().unwrap_or_default());
    let config = crate::config::Config::load(&paths.config_file())?;
    let sanitizer = crate::sanitize::Sanitizer::new(&config.sanitize)
        .context("compile sanitizer patterns")?;

    let body = sanitizer.scrub_body(text);
    let mut event = crate::event::Event::new(
        scope.workspace_id,
        scope.project_id,
        // A note belongs to the project, not to the session that happened to
        // write it: it must still surface when that session is long gone.
        uuid::Uuid::nil(),
        crate::event::Source { cli: client.unwrap_or("mcp").to_string(), hook: "note".to_string() },
        crate::event::EventKind::Note,
        crate::sanitize::truncate(&body, 120),
        body,
    );
    event.files = files.to_vec();
    // A note is already the durable form; there is nothing for a summarizer
    // to improve.
    event.consolidated = true;

    let log = crate::event::EventLog::open(&paths.project_dir(&scope))?;
    log.append(&event)?;
    let store = Store::open(&paths.db())?;
    store.index(&event)?;
    Ok(event.id)
}

/// The un-clamped bodies of these events, read from their monthly log lines.
///
/// One pass per project and month, however many ids it holds. The project
/// directory comes from the event's own project id, found among the known
/// projects, so an id surfaced from another project still resolves. An id the
/// log no longer has is simply absent from the result.
fn full_bodies_from_log(
    paths: &Paths,
    events: &[&crate::event::Event],
) -> std::collections::HashMap<String, String> {
    let mut by_month: std::collections::BTreeMap<(uuid::Uuid, String), std::collections::HashSet<String>> =
        std::collections::BTreeMap::new();
    for event in events {
        by_month.entry((event.project, event.month())).or_default().insert(event.id.clone());
    }
    let mut found = std::collections::HashMap::new();
    if by_month.is_empty() {
        return found;
    }
    let current = ids::resolve_scope(&std::env::current_dir().unwrap_or_default());
    let current_dir = paths.project_dir(&current);
    let mut known = None;
    for ((project, month), mut wanted) in by_month {
        let file = format!("{month}.jsonl");
        // The current scope first - the common case, no wiki walk. Only an id
        // from another project falls back to the scan, done once per call.
        if current.project_id == project {
            scan_month_file(&current_dir.join("events").join(&file), &mut wanted, &mut found);
        }
        if wanted.is_empty() {
            continue;
        }
        let projects: &Vec<(crate::ids::ProjectScope, std::path::PathBuf)> = known
            .get_or_insert_with(|| crate::consolidate::known_projects(paths).unwrap_or_default());
        // The directory already read is skipped: an id it lacked is not
        // there, and the file can be hundreds of megabytes.
        let skip = current.project_id == project;
        for (_, dir) in projects
            .iter()
            .filter(|(scope, dir)| scope.project_id == project && !(skip && *dir == current_dir))
        {
            scan_month_file(&dir.join("events").join(&file), &mut wanted, &mut found);
            if wanted.is_empty() {
                break;
            }
        }
    }
    found
}

fn scan_month_file(
    path: &std::path::Path,
    wanted: &mut std::collections::HashSet<String>,
    found: &mut std::collections::HashMap<String, String>,
) {
    if let Ok(file) = std::fs::File::open(path) {
        scan_log_for_bodies(std::io::BufReader::new(file), wanted, found);
    }
}

/// Move each wanted id's body from a month's lines into `found`, and stop at
/// the last one.
///
/// A month file reaches hundreds of megabytes, so it is read a line at a time
/// and a line is only parsed when it mentions a wanted id - a retire or a
/// correction that merely cites one still has to be told apart from the event.
/// A read error or a line that is not UTF-8 ends or skips only itself, as in
/// the event log's own reader.
fn scan_log_for_bodies(
    reader: impl std::io::BufRead,
    wanted: &mut std::collections::HashSet<String>,
    found: &mut std::collections::HashMap<String, String>,
) {
    for line in reader.split(b'\n') {
        if wanted.is_empty() {
            return;
        }
        let Ok(line) = line else { return };
        let Ok(text) = std::str::from_utf8(&line) else { continue };
        if !wanted.iter().any(|id| text.contains(id.as_str())) {
            continue;
        }
        if let Ok(logged) = serde_json::from_str::<crate::event::Event>(text) {
            if wanted.remove(&logged.id) {
                found.insert(logged.id, logged.body);
            }
        }
    }
}

/// Kinds a caller may ask for, and the one alias that has to work.
///
/// `raw` is not a kind - it is what the primer PRINTS for an untyped
/// observation, and the primer is the only place most agents ever learn this
/// vocabulary. Refusing the word they were shown, in favour of the word the
/// column happens to hold, would be a private schema detail sold as a
/// contract.
///
/// An unknown kind is an error rather than an empty list, for the same reason
/// a typo'd topic is loud in `brain search`: silence reads as "nothing
/// remembered", and an agent believes it.
fn normalize_kind(asked: &str) -> Result<&'static str> {
    match asked {
        "raw" | "observation" => Ok("observation"),
        "session_summary" => Ok("session_summary"),
        "knowledge" => Ok("knowledge"),
        "note" => Ok("note"),
        "page_update" => Ok("page_update"),
        "source" | "document" => Ok("source"),
        other => anyhow::bail!(
            "unknown kind `{other}`; known: raw (an untyped observation), \
             session_summary, knowledge, note, page_update, source (a document read in)"
        ),
    }
}

fn limit_from(arguments: &Value) -> usize {
    arguments
        .get("k")
        .and_then(Value::as_u64)
        .map_or(DEFAULT_SEARCH_LIMIT, |k| (k as usize).clamp(1, MAX_SEARCH_LIMIT))
}

fn success(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error_response(id: Value, code: i32, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn write_message(out: &mut impl Write, message: &Value) -> Result<()> {
    writeln!(out, "{message}").context("write MCP response")?;
    out.flush().context("flush MCP response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_declares_a_usable_schema() {
        // Both machines, because the list is worded per machine and a broken
        // schema on the rarer one is still a broken schema.
        for local_rerank in [false, true] {
            let tools = tool_definitions(local_rerank);
            let tools = tools.as_array().unwrap();
            assert_eq!(tools.len(), 11, "a tool was added or lost");
            for tool in tools {
                assert!(tool.get("name").and_then(Value::as_str).is_some());
                let description = tool.get("description").and_then(Value::as_str).unwrap();
                assert!(description.len() > 40, "description too thin to route on");
                assert_eq!(tool["inputSchema"]["type"], "object");
            }
        }
    }

    /// The list is loaded whole by CLIs that front-load schemas, so its size is
    /// a cost paid every session.
    #[test]
    fn the_tool_list_stays_under_its_size_ceiling() {
        for local_rerank in [false, true] {
            let size = serde_json::to_string(&tool_definitions(local_rerank)).unwrap().len();
            assert!(size <= 7_500, "tools/list is {size} B (local_rerank={local_rerank})");
        }
    }

    /// Trimming for size must not cut the nudge that makes agents correct
    /// memory unprompted, nor leave the download looking like a per-call cost.
    #[test]
    fn trimmed_descriptions_keep_their_load_bearing_clauses() {
        let tools = tool_definitions(false);
        let correct = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "brain_correct")
            .and_then(|t| t["description"].as_str())
            .unwrap()
            .to_string();
        assert!(correct.contains("FINDING IT YOURSELF"), "self-discovery nudge lost");
        let slow = tools[0]["inputSchema"]["properties"]["rerank"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(slow.contains("first one also starts"), "download reads as per-call");
    }

    /// The point of wording the list per machine is the number in it.
    ///
    /// Asserted on the substance rather than the sentence: a machine with the
    /// weights must not be quoting twelve seconds at anyone, and a machine
    /// without them must still say what the first rerank will cost.
    #[test]
    fn the_rerank_flag_quotes_this_machine_s_price() {
        let cost = |local_rerank| {
            tool_definitions(local_rerank)[0]["inputSchema"]["properties"]["rerank"]
                ["description"]
                .as_str()
                .expect("rerank description")
                .to_string()
        };

        let slow = cost(false);
        assert!(slow.contains("12 seconds"), "the CLI wait is what a bare build charges");
        assert!(slow.contains("600 MB"), "and the download is part of that price");

        let fast = cost(true);
        assert!(!fast.contains("12 seconds"), "a local reranker does not wait on a CLI");
        assert!(!fast.contains("600 MB"), "nor does it download what it already has");
        assert!(fast.contains("two seconds"), "say what it does cost, not just what it does not");

        assert_ne!(slow, fast, "one string for both machines is the bug this fixes");
    }

    #[test]
    fn limit_is_clamped_to_a_sane_range() {
        assert_eq!(limit_from(&json!({})), DEFAULT_SEARCH_LIMIT);
        assert_eq!(limit_from(&json!({"k": 0})), 1);
        assert_eq!(limit_from(&json!({"k": 5})), 5);
        assert_eq!(limit_from(&json!({"k": 9999})), MAX_SEARCH_LIMIT);
    }

    fn initialize(name: &str) -> Value {
        json!({"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": name, "version": "1"}})
    }

    #[test]
    fn a_known_client_name_maps_to_its_cli_and_an_unknown_one_to_none() {
        assert_eq!(client_cli(&initialize("claude-code")), Some("claude-code"));
        assert_eq!(client_cli(&initialize("some-new-cli")), None);
        assert_eq!(client_cli(&json!({})), None, "no clientInfo, no guess");
    }

    #[test]
    fn initialize_sets_the_client_and_rekeys_the_session() {
        assert_eq!(on_initialize(&initialize("claude-code"), 7), (Some("claude-code"), "mcp-claude-code-7".to_string()));
        assert_eq!(on_initialize(&initialize("some-new-cli"), 7), (None, "mcp-7".to_string()));
    }

    #[test]
    fn rerank_borrows_the_calling_client_else_the_project_cli() {
        let fallback = || Ok(Some("codex".to_string()));
        assert_eq!(rerank_cli(Some("claude-code"), fallback).unwrap(), "claude-code");
        assert_eq!(rerank_cli(None, fallback).unwrap(), "codex");
        assert_eq!(rerank_cli(None, || Ok(None)).unwrap(), "");
        let never = || -> Result<Option<String>> { panic!("fallback must not run when the client is known") };
        assert_eq!(rerank_cli(Some("claude-code"), never).unwrap(), "claude-code");
    }

    #[test]
    fn the_session_key_names_the_cli_when_known() {
        assert_eq!(session_key(Some("claude-code"), 7), "mcp-claude-code-7");
        assert_eq!(session_key(None, 7), "mcp-7");
    }

    /// A note carries the CLI that wrote it, and `mcp` when that is not known.
    #[test]
    fn a_note_is_attributed_to_the_cli_that_initialized_the_session() {
        let note = |client: Option<&str>| {
            let data_dir = std::env::temp_dir().join(format!("brain-mcp-note-{}", ulid::Ulid::new()));
            let paths = Paths { data_dir };
            let project = ids::resolve_scope(&std::env::current_dir().unwrap()).project_id.to_string();
            call_tool(
                &paths,
                &project,
                "s",
                client,
                &json!({"name": "brain_note", "arguments": {"text": "remember this"}}),
            )
            .unwrap();
            let cli = Store::open(&paths.db()).unwrap().project_cli(&project).unwrap();
            std::fs::remove_dir_all(&paths.data_dir).ok();
            cli
        };
        let known = client_cli(&initialize("claude-code"));
        assert_eq!(note(known).as_deref(), Some("claude-code"));
        assert_eq!(note(client_cli(&initialize("some-new-cli"))).as_deref(), Some("mcp"));
    }

    #[test]
    fn initialize_advertises_tools() {
        let result = initialize_result();
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert!(result["capabilities"].get("tools").is_some());
        assert_eq!(result["serverInfo"]["name"], "rolepod-brain");
    }

    /// `brain_get` is the one reader that gets the whole line back from the
    /// log; `Store::get` (consolidation's path) keeps the clamped body.
    #[test]
    fn brain_get_returns_the_full_body_of_a_clamped_row_from_the_log() {
        let data_dir = std::env::temp_dir().join(format!("brain-mcp-get-{}", ulid::Ulid::new()));
        let paths = Paths { data_dir };
        let scope = ids::ProjectScope {
            workspace: "default".to_string(),
            workspace_id: uuid::Uuid::new_v4(),
            project: "demo".to_string(),
            project_id: uuid::Uuid::new_v4(),
            root: std::path::PathBuf::from("/tmp/demo"),
        };
        let stdout = format!("{}END_OF_OUTPUT", "q".repeat(10 * 1024));
        let body = json!({"tool_name": "Bash", "tool_response": {"stdout": stdout}}).to_string();
        let mut event = crate::event::Event::new(
            scope.workspace_id,
            scope.project_id,
            uuid::Uuid::new_v4(),
            crate::event::Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
            crate::event::EventKind::Observation,
            "Bash: make".to_string(),
            body.clone(),
        );
        event.consolidated = true;
        crate::event::EventLog::open(&paths.project_dir(&scope)).unwrap().append(&event).unwrap();
        let store = Store::open(&paths.db()).unwrap();
        store.index(&event).unwrap();

        let got = call_tool(
            &paths,
            &scope.project_id.to_string(),
            "s",
            None,
            &json!({"name": "brain_get", "arguments": {"ids": [event.id.clone()]}}),
        )
        .unwrap();
        let text = got.to_string();
        assert!(text.contains("END_OF_OUTPUT"));
        assert!(text.len() > 10 * 1024, "the whole body came back");

        let bounded = &store.get(&[event.id.clone()]).unwrap()[0].body;
        assert!(bounded.len() <= 4096 && bounded.len() < body.len());
        std::fs::remove_dir_all(&paths.data_dir).ok();
    }

    fn big_observation_in(paths: &Paths, log_it: bool) -> (ids::ProjectScope, crate::event::Event) {
        let scope = ids::ProjectScope {
            workspace: "default".to_string(),
            workspace_id: uuid::Uuid::new_v4(),
            project: "demo".to_string(),
            project_id: uuid::Uuid::new_v4(),
            root: std::path::PathBuf::from("/tmp/demo"),
        };
        let stdout = format!("{}END_OF_OUTPUT", "q".repeat(10 * 1024));
        let body = json!({"tool_name": "Bash", "tool_response": {"stdout": stdout}}).to_string();
        let mut event = crate::event::Event::new(
            scope.workspace_id,
            scope.project_id,
            uuid::Uuid::new_v4(),
            crate::event::Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
            crate::event::EventKind::Observation,
            "Bash: make".to_string(),
            body,
        );
        event.consolidated = true;
        if log_it {
            crate::event::EventLog::open(&paths.project_dir(&scope))
                .unwrap()
                .append(&event)
                .unwrap();
        }
        Store::open(&paths.db()).unwrap().index(&event).unwrap();
        (scope, event)
    }

    #[test]
    fn brain_get_keeps_the_clamped_body_when_the_log_line_is_gone() {
        let data_dir = std::env::temp_dir().join(format!("brain-mcp-gone-{}", ulid::Ulid::new()));
        let paths = Paths { data_dir };
        let (scope, event) = big_observation_in(&paths, false);
        let got = call_tool(
            &paths,
            &scope.project_id.to_string(),
            "s",
            None,
            &json!({"name": "brain_get", "arguments": {"ids": [event.id.clone()]}}),
        )
        .unwrap();
        let text = got.to_string();
        assert!(text.contains("\"count\":1") || text.contains("\"count\": 1"));
        assert!(text.len() < 10 * 1024, "no log line: the bounded body stays");
        std::fs::remove_dir_all(&paths.data_dir).ok();
    }

    #[test]
    fn brain_get_does_not_resurrect_a_retired_observation() {
        let data_dir = std::env::temp_dir().join(format!("brain-mcp-ret-{}", ulid::Ulid::new()));
        let paths = Paths { data_dir };
        let (scope, event) = big_observation_in(&paths, true);
        let mut retire = crate::event::Event::new(
            scope.workspace_id,
            scope.project_id,
            uuid::Uuid::nil(),
            crate::event::Source { cli: "brain".into(), hook: "retire".into() },
            crate::event::EventKind::Retire,
            "Retired 1".to_string(),
            String::new(),
        );
        retire.links = vec![event.id.clone()];
        Store::open(&paths.db()).unwrap().index(&retire).unwrap();
        let got = call_tool(
            &paths,
            &scope.project_id.to_string(),
            "s",
            None,
            &json!({"name": "brain_get", "arguments": {"ids": [event.id.clone()]}}),
        )
        .unwrap();
        let text = got.to_string();
        assert!(!text.contains("END_OF_OUTPUT"), "retired body came back from the log");
        std::fs::remove_dir_all(&paths.data_dir).ok();
    }

    /// Retention drops bodies from the index and writes nothing to the log, so
    /// `brain_get` is the only way back to them.
    #[test]
    fn brain_get_returns_the_full_body_of_a_row_whose_body_the_index_dropped() {
        let data_dir = std::env::temp_dir().join(format!("brain-mcp-drop-{}", ulid::Ulid::new()));
        let paths = Paths { data_dir };
        let (scope, first) = big_observation_in(&paths, true);
        let mut second = first.clone();
        second.id = ulid::Ulid::new().to_string();
        second.body = "plain old observation, no clamp".to_string();
        crate::event::EventLog::open(&paths.project_dir(&scope)).unwrap().append(&second).unwrap();
        let store = Store::open(&paths.db()).unwrap();
        store.index(&second).unwrap();
        let ids = vec![first.id.clone(), second.id.clone()];
        assert_eq!(store.drop_index_bodies(&ids).unwrap(), 2);
        assert!(store.get(&ids).unwrap().iter().all(|event| event.body.is_empty()));

        let got = call_tool(
            &paths,
            &scope.project_id.to_string(),
            "s",
            None,
            &json!({"name": "brain_get", "arguments": {"ids": ids}}),
        )
        .unwrap();
        let events = got["structuredContent"]["events"].as_array().unwrap().clone();
        assert_eq!(events.len(), 2);
        let body_of = |id: &str| {
            events.iter().find(|event| event["id"] == id).unwrap()["body"].as_str().unwrap().to_string()
        };
        assert!(body_of(&first.id).contains("END_OF_OUTPUT") && body_of(&first.id).len() > 10 * 1024);
        assert_eq!(body_of(&second.id), second.body);
        std::fs::remove_dir_all(&paths.data_dir).ok();
    }

    /// A reader that counts the bytes pulled through it.
    struct Counting<R> {
        inner: R,
        bytes: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl<R: std::io::Read> std::io::Read for Counting<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.bytes.set(self.bytes.get() + n);
            Ok(n)
        }
    }

    #[test]
    fn the_month_scan_stops_reading_once_every_wanted_id_is_found() {
        // ~8 MB of lines; the two wanted ids sit near the front. A retire line
        // that merely cites the first id must not be taken for the event.
        let wanted = ["01WANTED0000000000000000AA", "01WANTED0000000000000000BB"];
        let mut log = String::new();
        let line = |id: &str, body: &str| {
            let mut event = crate::event::Event::new(
                uuid::Uuid::nil(),
                uuid::Uuid::nil(),
                uuid::Uuid::nil(),
                crate::event::Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
                crate::event::EventKind::Observation,
                "t".to_string(),
                body.to_string(),
            );
            event.id = id.to_string();
            serde_json::to_string(&event).unwrap()
        };
        let mut cite = serde_json::from_str::<serde_json::Value>(&line("01CITE", "")).unwrap();
        cite["links"] = json!([wanted[0]]);
        log.push_str(&format!("{cite}\n"));
        log.push_str("not json but mentions 01WANTED0000000000000000AA\n");
        log.push_str(&format!("{}\n", line(wanted[0], "first body")));
        log.push_str(&format!("{}\n", line(wanted[1], "second body")));
        let filler = "f".repeat(1000);
        for n in 0..8_000 {
            log.push_str(&format!("{}\n", line(&format!("01FILL{n:020}"), &filler)));
        }
        assert!(log.len() > 8_000_000);

        let bytes = std::rc::Rc::new(std::cell::Cell::new(0));
        let reader = std::io::BufReader::new(Counting {
            inner: std::io::Cursor::new(log.clone().into_bytes()),
            bytes: bytes.clone(),
        });
        let mut ids: std::collections::HashSet<String> = wanted.iter().map(|id| id.to_string()).collect();
        let mut found = std::collections::HashMap::new();
        scan_log_for_bodies(reader, &mut ids, &mut found);

        assert!(ids.is_empty(), "ids left unfound: {ids:?}");
        assert_eq!(found[wanted[0]], "first body");
        assert_eq!(found[wanted[1]], "second body");
        assert!(bytes.get() < 100_000, "read {} of {} bytes", bytes.get(), log.len());
    }

    #[test]
    fn the_month_scan_finds_a_late_id_and_leaves_an_absent_one_wanted() {
        let event_line = |id: &str, body: &str| {
            let mut event = crate::event::Event::new(
                uuid::Uuid::nil(),
                uuid::Uuid::nil(),
                uuid::Uuid::nil(),
                crate::event::Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
                crate::event::EventKind::Observation,
                "t".to_string(),
                body.to_string(),
            );
            event.id = id.to_string();
            serde_json::to_string(&event).unwrap()
        };
        let late = "01LATE00000000000000000000";
        let absent = "01ABSENT000000000000000000";
        // A line that cites the absent id without being that event.
        let mut cite = serde_json::from_str::<serde_json::Value>(&event_line("01CITE", "")).unwrap();
        cite["links"] = json!([absent]);
        let mut log = format!("{cite}\n");
        for n in 0..500 {
            log.push_str(&format!("{}\n", event_line(&format!("01FILL{n:020}"), "filler")));
        }
        log.push_str(&event_line(late, "the last line"));

        let mut wanted: std::collections::HashSet<String> =
            [late, absent].iter().map(|id| id.to_string()).collect();
        let mut found = std::collections::HashMap::new();
        scan_log_for_bodies(std::io::Cursor::new(log.into_bytes()), &mut wanted, &mut found);

        assert_eq!(found[late], "the last line");
        assert!(!found.contains_key(absent));
        assert_eq!(wanted.into_iter().collect::<Vec<_>>(), vec![absent.to_string()]);
    }

    #[test]
    fn a_failed_tool_call_reports_inside_the_result() {
        let paths = Paths { data_dir: std::env::temp_dir().join("brain-mcp-test") };
        let result =
            call_tool(&paths, "project", "s", None, &json!({"name": "brain_search", "arguments": {}}));
        assert!(result.is_err(), "missing query must be an error the caller can see");
    }
}
