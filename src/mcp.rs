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
use crate::store::Store;

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
    let session = format!("mcp-{}", std::process::id());

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
            "initialize" => success(id, initialize_result()),
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
            "tools/call" => match call_tool(&paths, &project, &session, &params) {
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
         looks off. It is not free, but it is cheap enough to reach for."
    } else {
        "Costs about 12 seconds of real waiting, 20 at the ceiling, because \
         this machine has no local reranker and the work goes out to a host \
         CLI. Ask for it when the answer is worth that and not otherwise. The \
         first such search also starts a one-time 600 MB download in the \
         background, and the searches after it are the fast kind."
    };
    json!([
        {
            "name": "brain_search",
            "description": "Full-text search this project's memory. Returns matching \
                            observations with their ids, newest-relevant first. Use it \
                            before assuming context is lost: prior sessions in any CLI \
                            wrote here. Pass an id to brain_get for the full body.",
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
                            "Let a cheap model reorder the results by what the \
                             question was actually asking. {rerank_cost} It earns \
                             its keep on a question whose words the entry probably \
                             does not use (\"why did we stop doing X\"), or on a \
                             first search that came back plausible but not right. \
                             Omit it for ordinary lookups."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "brain_seed",
            "description": "One compact block to seed a subagent or a fresh task: the \
                            project's standing lessons first, then pointers relevant \
                            to the task. Paste it into the subagent's prompt; the ids \
                            let it pull full entries with brain_get.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What the subagent will work on, in a phrase.",
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Maximum bytes for the block (default 2048, max 8192).",
                    },
                },
                "required": ["task"],
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
                            wrong but the event itself matters. The user telling you is \
                            one way to learn that; FINDING IT YOURSELF is the common \
                            one - a claim here can be true when written and false a \
                            week later, and nothing else in this system retires it. If \
                            memory says something the code in front of you \
                            contradicts, correcting it is part of the work, not a \
                            favour: the next session is told the same thing otherwise. \
                            Write the replacement with a SHORT FIRST LINE - it becomes \
                            the title - and the detail below it. The original stays in \
                            the log; recall returns your text. Prefer this over \
                            brain_forget when something happened but was described \
                            badly, or was described correctly and has since changed.",
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
            "description": "Mark a memory as stale or unhelpful without deleting it. \
                            Use when the user says a remembered thing is out of date \
                            or was not worth keeping, but is not claiming it is wrong \
                            — for that, brain_correct or brain_forget. Flagged entries \
                            sink in what gets shown and are listed for the user to \
                            review; nothing is destroyed.",
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
            "description": "What sits beside a memory you are already holding. \
                            Given an id, returns other memories whose sessions \
                            named the same files, symbols, or subjects, most \
                            overlap first. Use it when a search result is close \
                            but not the whole story — it answers \"what else \
                            touched this\", which a query for words cannot.",
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
                                        per line. `raw` — live work not yet \
                                        summarized. Also `knowledge`, `note`, \
                                        `page_update`. Omit for all of them mixed, \
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

fn call_tool(paths: &Paths, project: &str, session: &str, params: &Value) -> Result<Value> {
    let name = params.get("name").and_then(Value::as_str).unwrap_or_default();
    let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let store = Store::open(&paths.db())?;

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
            let mut hits = store.search(project, query, topic, pool, crate::store::Recall::Fused)?;

            // Second retrieval stream: a query that names a file or a service
            // finds the work about it even when no title contains the word.
            // Appended rather than interleaved, so text relevance still leads.
            let by_entity = store
                .search_by_entity(project, &crate::consolidate::normalize_entity(query), pool)
                .unwrap_or_default();
            let seen: std::collections::HashSet<String> =
                hits.iter().map(|hit| hit.id.clone()).collect();
            hits.extend(by_entity.into_iter().filter(|hit| !seen.contains(&hit.id)).take(pool));
            hits.truncate(pool);

            if rerank {
                let ladder = crate::summarizer::Ladder::new(&store, &config.summarizer);
                // Borrow the cheap tier of whichever CLI works here.
                let cli = store.project_cli(project)?.unwrap_or_default();
                let model_dir = paths.model_dir_for(crate::rerank::LOCAL_MODEL);
                hits = crate::rerank::rerank(&ladder, &cli, query, &model_dir, hits);
            }
            hits.truncate(limit);

            store.record_recalled(session, hits.iter().map(|hit| hit.id.as_str()))?;
            json!({ "hits": hits, "count": hits.len() })
        }
        "brain_seed" => {
            let task = arguments
                .get("task")
                .and_then(Value::as_str)
                .filter(|t| !t.trim().is_empty())
                .context("brain_seed requires a non-empty `task`")?;
            let budget = arguments
                .get("budget")
                .and_then(Value::as_u64)
                .map_or(crate::inject::SEED_BUDGET, |b| {
                    usize::try_from(b).unwrap_or(crate::inject::SEED_BUDGET).clamp(256, 8192)
                });
            let seed = crate::inject::seed(&store, project, task, budget)?;
            // Seeded ids were surfaced to this session the same as a search
            // hit: the correction gate and the uptake record both need to
            // know that.
            store.record_recalled(session, seed.ids.iter().map(String::as_str))?;
            json!({ "seed_text": seed.text, "ids": seed.ids, "count": seed.ids.len() })
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
            // Not `record_recalled`: asking for a body, having seen only the
            // title, is the one moment an agent says an entry was worth the
            // tokens. Everything else in this file merely offered it.
            store.record_opened(session, events.iter().map(|event| event.id.as_str()))?;
            json!({ "events": events, "count": events.len() })
        }
        "brain_related" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .context("brain_related requires an `id`")?;
            let hits = store.related(project, id, limit_from(&arguments))?;
            store.record_recalled(session, hits.iter().map(|hit| hit.id.as_str()))?;
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
            store.record_recalled(session, hits.iter().map(|hit| hit.id.as_str()))?;
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
                store.already_injected(session, id)? || store.was_recalled(session, id)?,
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
                store.already_injected(session, id)? || store.was_recalled(session, id)?,
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
                store.already_injected(session, id)? || store.was_recalled(session, id)?,
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
            let id = write_note(paths, text, &files)?;
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
fn write_note(paths: &Paths, text: &str, files: &[String]) -> Result<String> {
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
        crate::event::Source { cli: "mcp".to_string(), hook: "note".to_string() },
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
        other => anyhow::bail!(
            "unknown kind `{other}`; known: raw (an untyped observation), \
             session_summary, knowledge, note, page_update"
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
            assert_eq!(tools.len(), 12, "a tool was added or lost");
            for tool in tools {
                assert!(tool.get("name").and_then(Value::as_str).is_some());
                let description = tool.get("description").and_then(Value::as_str).unwrap();
                assert!(description.len() > 40, "description too thin to route on");
                assert_eq!(tool["inputSchema"]["type"], "object");
            }
        }
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

    #[test]
    fn initialize_advertises_tools() {
        let result = initialize_result();
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert!(result["capabilities"].get("tools").is_some());
        assert_eq!(result["serverInfo"]["name"], "rolepod-brain");
    }

    #[test]
    fn a_failed_tool_call_reports_inside_the_result() {
        let paths = Paths { data_dir: std::env::temp_dir().join("brain-mcp-test") };
        let result =
            call_tool(&paths, "project", "s", &json!({"name": "brain_search", "arguments": {}}));
        assert!(result.is_err(), "missing query must be an error the caller can see");
    }
}
