//! Hook worker — the capture path.
//!
//! Spawned by a host CLI's lifecycle hook, reads one event payload on stdin,
//! writes it, exits. Nothing survives the call.
//!
//! Two rules govern everything here:
//!
//! 1. **Never disturb the host.** A capture failure is our problem, not the
//!    user's. Errors go to `brain.log` and the process still exits 0 with a
//!    well-formed acknowledgement, because a hook that errors or prints stray
//!    text degrades the CLI the user is actually trying to work in.
//! 2. **Sanitize before writing.** The scrub happens here, before the event
//!    reaches the log. There is no later stage that can catch a leak.

use std::io::Read;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::{Config, Paths};
use crate::event::{Event, EventKind, EventLog, Source};
use crate::ids::{self, AgentKind};
use crate::inject;
use crate::invocation;
use crate::sanitize::{
    clamp_leaf, leaves_private_open, truncate, truncate_head_tail, Sanitizer, FOOTSTEP_BODY_MAX_BYTES,
    FOOTSTEP_LEAF_MAX_BYTES, LEAF_MAX_BYTES,
};
use crate::maint;
use crate::store::{Ledger, Store};

/// Ceiling for a generated title, before it reaches the primer.
const TITLE_MAX_BYTES: usize = 120;

/// How old unconsolidated work must be before a session opening picks it up.
///
/// Comfortably longer than the consolidation debounce, so an active machine
/// never triggers this path and a returning one always does.
pub const STALE_BACKLOG_SECS: i64 = 15 * 60;

/// How long a stop waits before it asks again whether anything sat idle.
pub const IDLE_SWEEP_DEBOUNCE_SECS: i64 = 15 * 60;

/// How long a session must have been quiet before the idle sweep takes it.
pub const IDLE_SWEEP_AGE_SECS: i64 = 2 * 3600;

/// Should this stop spawn a `consolidate --idle` for work nobody is waiting on?
///
/// A session that goes quiet leaves its backlog until some later `session_start`;
/// every stop is a live hook, so one asking is the cheaper way to finish it.
/// The hook only spends the debounce window - a plain read for most stops, one
/// conditional write for the stop that finds it open. Whether anything is idle
/// is the spawned run's question: no backlog scan runs on the hook path. A
/// store error means no sweep.
fn claim_idle_sweep_window(store: &Store, now: jiff::Timestamp) -> bool {
    store.claim_idle_sweep(now, IDLE_SWEEP_DEBOUNCE_SECS).unwrap_or(false)
}

/// Set in every subprocess we spawn into a host CLI.
///
/// Consolidation shells out to the user's own CLI, and a headless CLI run
/// fires that CLI's lifecycle hooks — which call us straight back. Without
/// this guard the brain captures its own consolidation runs, and those
/// captures then need consolidating. Verified the hard way: a `codex exec`
/// call triggered this machine's `Stop` hooks.
pub const WORKER_ENV: &str = "ROLEPOD_BRAIN_WORKER";

/// Are we running inside our own subprocess?
#[must_use]
pub fn is_worker_child() -> bool {
    std::env::var_os(WORKER_ENV).is_some_and(|value| !value.is_empty())
}

/// The `host_session` pid that stands for "no host CLI found".
const NO_HOST: u32 = 0;

/// What a hook must do to learn the host pid of its session.
#[derive(Debug, PartialEq, Eq)]
enum HostPlan {
    /// Nothing cached: classify, which finds invocation and host together.
    Classify,
    /// Invocation cached but no row (open before the tie existed): classify
    /// once for the host; the row then exists.
    FindHost,
    /// A row exists (a marker included): reuse its pid, no process spawned.
    Reuse(u32),
}

fn host_plan(invocation_cached: bool, peeked_host: Option<u32>) -> HostPlan {
    match (invocation_cached, peeked_host) {
        (false, _) => HostPlan::Classify,
        (true, None) => HostPlan::FindHost,
        (true, Some(pid)) => HostPlan::Reuse(pid),
    }
}

/// Take what the host is sending, and throw it away.
///
/// Capturing nothing is not the same as reading nothing. A hook that returns
/// before draining its stdin closes the pipe under a host that is still
/// writing, and the host reports a failed hook — so the run that promised to
/// leave no trace leaves an error in someone's log. Anything past a pipe
/// buffer is enough to hit it, which is why a small payload never showed it.
///
/// Failures are ignored on purpose: there is nothing to do about them, and
/// this path exists to be silent.
fn drain_stdin(should: bool) {
    if !should {
        return;
    }
    let mut sink = String::new();
    let _ = std::io::stdin().read_to_string(&mut sink);
}

/// Read a payload from stdin and capture it.
///
/// Returns the JSON the host CLI should see on stdout.
///
/// # Errors
/// Returns an error only for conditions the caller should log; the caller is
/// responsible for still exiting 0.
pub fn capture(cli: &str, event_name: &str, stdin_payload: Option<String>) -> Result<String> {
    // An index failure outside a compact window must not cost the host its
    // answer (the primer in particular), but it is still a failure to see in
    // `brain.log`, on the line it always had.
    let mut deferred = None;
    let answer = capture_answering(cli, event_name, stdin_payload, &mut deferred);
    if let Some(error) = deferred {
        log_failure(cli, event_name, &error);
    }
    answer
}

/// Record a capture failure where `brain doctor` will find it. Best-effort by
/// design: if even this fails there is nothing useful left to do.
pub fn log_failure(agent: &str, event: &str, error: &anyhow::Error) {
    use std::io::Write;
    let Ok(paths) = Paths::resolve() else { return };
    if std::fs::create_dir_all(&paths.data_dir).is_err() {
        return;
    }
    let line = format!("{} {agent} {event}: {error:#}\n", jiff::Timestamp::now());
    if let Ok(mut file) =
        std::fs::OpenOptions::new().create(true).append(true).open(paths.log_file())
    {
        let _ = file.write_all(line.as_bytes());
    }
}

/// [`capture`]'s body. A store failure that does not stop the answer goes in
/// `deferred` for the caller to log.
fn capture_answering(
    cli: &str,
    event_name: &str,
    stdin_payload: Option<String>,
    deferred: &mut Option<anyhow::Error>,
) -> Result<String> {
    // Our own subprocess: acknowledge the host and capture nothing.
    if is_worker_child() {
        drain_stdin(stdin_payload.is_none());
        return Ok("{}".to_string());
    }

    // An orchestrator has asked for a clean room: no capture, no injection,
    // no trace. This is a documented public contract, not an internal switch.
    if invocation::silenced() {
        drain_stdin(stdin_payload.is_none());
        return Ok("{}".to_string());
    }

    let raw = match stdin_payload {
        Some(text) => text,
        None => {
            let mut buffer = String::new();
            std::io::stdin().read_to_string(&mut buffer).context("read hook payload")?;
            buffer
        }
    };

    // An empty payload is normal for some hooks; there is nothing to capture
    // but the host still needs its acknowledgement.
    if raw.trim().is_empty() {
        return Ok("{}".to_string());
    }

    let mut payload: Value = serde_json::from_str(&raw).context("parse hook payload")?;

    // A shell command is looked up and never captured: it takes this path
    // before anything that classifies the invocation, runs `ps` or git, or
    // opens the store, and every outcome is an answer.
    if normalize_hook(event_name) == "pre_tool_use"
        && payload.get("tool_name").and_then(Value::as_str) == Some("Bash")
    {
        return Ok(shell_lesson(cli, event_name, &payload).unwrap_or_else(|_| "{}".to_string()));
    }

    let mut hook = normalize_hook(event_name);
    let failed_call = hook == "post_tool_use_failure";
    if let Some(object) = payload.as_object_mut() {
        // The marker is ours alone: a host-sent `failed` must never make a
        // normal call render as FAILED.
        object.remove("failed");
        if failed_call {
            // A failed tool call is the same observation with a result that
            // went wrong, so it joins `post_tool_use` and carries the marker;
            // the host documents the error under different fields, `body_for`
            // keeps them all.
            hook = "post_tool_use".to_string();
            object.insert("failed".to_string(), Value::Bool(true));
        }
    }
    if is_cursor_echo(cli, &hook, &payload) {
        return Ok("{}".to_string());
    }

    let paths = Paths::resolve()?;
    paths.ensure()?;
    let config = Config::load(&paths.config_file())?;
    let sanitizer = Sanitizer::new(&config.sanitize).context("compile sanitizer patterns")?;

    let Some(cwd) = working_directory(&payload) else {
        // We know which events happened but not which project they belong to.
        // Filing them under a guess would put one CLI's memory into another
        // project's brain, and a wrong memory is worse than a missing one.
        return Ok("{}".to_string());
    };
    let scope = ids::resolve_scope(&cwd);

    let cli_kind = AgentKind::parse(cli);
    let session = ids::session_uuid(
        first_string(&payload, &["session_id", "sessionId", "thread_id", "conversationId"])
            .unwrap_or("unknown-session"),
    );
    let delegate = delegate_label(&payload);

    // The one thing a lifecycle hook cannot see and the next session most
    // needs: what the model actually answered. Without it a handoff carries
    // the question and the file that changed, and the answer already given is
    // the part that goes missing - so the next session asks it again.
    //
    // Only the last answer, only at `stop`, bounded and sanitized. The rest of
    // the transcript stays unread and unstored, as it always has: the user's
    // turns are captured verbatim already, and the history is what
    // consolidation reads when it needs it.
    let answer = (hook == "stop")
        .then(|| {
            first_string(&payload, &["transcript_path", "transcriptPath"])
                .filter(|path| is_transcript_of(cli_kind.as_str(), path))
                .and_then(|path| {
                    crate::transcript::last_answer(
                        std::path::Path::new(path),
                        cli_kind.as_str(),
                        &sanitizer,
                    )
                })
        })
        .flatten();

    let mut title = title_for(&hook, &payload);
    if failed_call {
        title = format!("Failed: {title}");
    }
    let title = truncate(&sanitizer.scrub(&title), TITLE_MAX_BYTES);
    // A delegate's tool call is a footstep: it is summarized away and only
    // its shape is worth keeping, so it gets a much smaller body.
    let footstep = is_footstep(delegate.as_deref(), &hook);
    let mut body = event_body(&payload, &sanitizer, footstep);
    let mut title = title;
    if let Some(answer) = answer {
        // The title carries the answer's opening so the pointer line itself
        // says something; the body carries the rest for whoever pulls it.
        title = truncate(&first_line(&answer), TITLE_MAX_BYTES);
        body = answer;
    }
    // A delegate's report is the one thing it produced that the session will
    // want back: a reviewer's findings, a scout's conclusions. Claude Code
    // hands it over whole at `SubagentStop`; the lead's own captures only
    // show that a delegate ran. `scrub_body` bounds it like every body, head
    // and tail kept, so the last finding survives as well as the first.
    if let Some(report) = (hook == "subagent_stop")
        .then(|| first_string(&payload, &["last_assistant_message"]))
        .flatten()
        .filter(|message| !message.trim().is_empty())
    {
        let who = delegate.as_deref().unwrap_or("A subagent");
        title = truncate(
            &sanitizer.scrub(&format!("{who} reported: {}", first_line(report))),
            TITLE_MAX_BYTES,
        );
        body = sanitizer.scrub_body(report);
    }

    // Classified once per session and remembered: working it out costs a
    // process spawn, and the hook budget does not allow one per event.
    //
    // A read-only peek, not `Store::open`: nothing about the store may stand
    // before the log append below, and an open migrates, which waits out the
    // busy timeout when a write lock is held (an index build, the first
    // migration after an upgrade) - longer than the host lets a hook run. A
    // store that cannot be read falls back to classifying again.
    let session_key = session.to_string();
    let remembered = Store::peek_session_invocation(&paths.db(), &session_key);
    let known = remembered.is_some();
    //
    // The host pid is looked up the same way, so every event can refresh the
    // session's `host_session` row without another `ps`. A session with no row
    // yet (open before the tie existed) is classified once to find it.
    //
    // A host that classification cannot find leaves a pid-0 marker row, so
    // that session is never classified again.
    let (invocation, host_pid) = match host_plan(known, known.then(|| Store::peek_session_host(&paths.db(), &session_key)).flatten()) {
        HostPlan::Classify => {
            let (found, pid) = invocation::classify_with_host();
            (found, Some(pid.unwrap_or(NO_HOST)))
        }
        HostPlan::FindHost => (
            invocation::parse(remembered.as_deref().unwrap_or_default()),
            Some(invocation::classify_with_host().1.unwrap_or(NO_HOST)),
        ),
        HostPlan::Reuse(pid) => (invocation::parse(remembered.as_deref().unwrap_or_default()), Some(pid)),
    };

    // A pointer to material consolidation will read and never copy. Claude
    // Code and Codex put it in every payload; Cursor's is accepted when its
    // payload carries one (consolidation finds it by session id otherwise).
    // Not from a delegate: the lead's own hooks record the same path, and
    // the write is last-writer-wins, so a delegate's must never be the last.
    let transcript = first_string(&payload, &["transcript_path", "transcriptPath"])
        .filter(|path| delegate.is_none() && is_transcript_of(cli_kind.as_str(), path));

    let mut event = Event::new(
        scope.workspace_id,
        scope.project_id,
        session,
        Source { cli: cli_kind.as_str().to_string(), hook: hook.clone() },
        EventKind::Observation,
        title,
        body,
    );
    // The same scrub the title and body get. A path is the one field the
    // sanitizer explicitly treats as sensitive by convention - .ssh, .aws,
    // .gnupg - and storing it unscrubbed in a parallel array meant the thing
    // being redacted out of the title sat intact in the column beside it.
    // A failed call changed nothing, so it is not linked to the file it named.
    event.files = if failed_call { Vec::new() } else { files_for(&payload, &scope.root) }
        .iter()
        .map(|path| sanitizer.scrub(path))
        .filter(|path| !path.is_empty())
        .collect();
    if invocation.is_headless() {
        // Tagged, not dropped: a headless run's observations are still true,
        // they are simply worth less than a person's working session, and the
        // primer's floor should be able to say so.
        event
            .extra
            .insert("invocation".to_string(), Value::String(invocation.as_str().to_string()));
    }
    if let Some(label) = &delegate {
        // Same rule, one level down. A subagent's tool calls arrive under the
        // lead's session id - the host gives them no session of their own -
        // and untagged they read as the lead's work: a scout's forty file
        // reads summarized as what this session did. The tag lets the
        // summary keep the delegate's report and drop its footsteps.
        event.agent = Some(label.clone());
    }

    // `pre_tool_use` is an injection surface, not a capture one. It reaches us
    // only for `Read`, and only so that what we know about a file lands before
    // its contents do; capturing it as well would restore the 96% duplication
    // against `post_tool_use` that took it off the capture list to begin with.
    let captured = captures(&hook);
    if captured {
        let project_dir = paths.project_dir(&scope);
        let log = EventLog::open(&project_dir)?;
        // Log first: it is the source of truth, and nothing about the store
        // may come before it. If a store write then fails, the event is still
        // durable: the next consolidation run indexes it from the log, and
        // `brain reindex` rebuilds the lot.
        log.append(&event)?;
    }

    // Only now the store, whose open may wait out the busy timeout. Each
    // failure is reported the way an index failure always was - the caller
    // logs it and the host still gets its acknowledgement - and none can undo
    // the append above. The writes are independent, so one failing does not
    // stop the others; the first error is the one returned.
    //
    // While a compact window is open the database is expected to be held: a
    // hook does not wait it out. It gives up on its writes at the first one
    // that fails - the event is in the log, and the window's catch-up indexes
    // it - and goes on to answer from what it can read.
    let fast = maint::active(&paths);
    let store = if fast {
        match Store::open_waiting(&paths.db(), maint::FAIL_FAST) {
            Ok(store) => store,
            Err(_) => {
                // No store to reset: the wipe waits as a file, as any other.
                if wipes_context(&hook, &payload) {
                    maint::pending_wipe(&paths, &session_key);
                }
                return Ok("{}".to_string());
            }
        }
    } else {
        Store::open(&paths.db())?
    };
    // Set once a store write has failed in the window (or a wipe could not be
    // applied): every later write of this hook is left to the spill files.
    let mut skip = false;
    let mut failed = None;
    if !known {
        failed = store.record_session_invocation(&session_key, invocation.as_str()).err();
        skip = fast && failed.is_some();
    }
    // Every event refreshes the tie, so `ts` is when the session was last seen.
    if let (Some(pid), false) = (host_pid, skip) {
        let at = jiff::Timestamp::now().to_string();
        let tied = store.record_host_session(pid, &session_key, &at).err();
        skip = fast && tied.is_some();
        failed = failed.or(tied);
    }
    if let Some(path) = transcript {
        if !skip {
            skip = store.record_transcript_path(&session_key, path).is_err() && fast;
        }
    }
    if captured && !skip {
        let indexed = store.index_captured(&event);
        skip = fast && indexed.is_err();
        failed = failed.or(indexed.err());
    }
    if !fast {
        *deferred = failed;
    }

    if delegate.is_some() {
        // A subagent shares the lead's session id, so everything past this
        // point - the wipe reset, the consolidation trigger, the file
        // injection - would act on the lead's session on a delegate's
        // behalf. A file injection in particular would spend the lead's
        // budget and mark the file covered for a session that never saw the
        // pointer. What a delegate should know goes in through the brief
        // the lead writes, and nothing else.
        return Ok("{}".to_string());
    }

    // A context wipe keeps the session id but destroys everything the agent
    // knew. Our own de-duplication is keyed to that surviving id, so without
    // this reset the guard that stops us repeating ourselves would instead
    // guarantee amnesia: memory injected before the wipe would be suppressed
    // exactly when it is needed most.
    //
    // A wipe the store could not take is not lost: it waits as a file for the
    // next hook of this session that can write, which applies it here too.
    let wipe_waiting = maint::wipe_pending(&paths, &session_key);
    // A wipe that is still owed leaves the store's dedupe state as it was
    // before the wipe; the primer must not treat it as the agent's memory.
    let mut fresh = false;
    if wipes_context(&hook, &payload) || wipe_waiting {
        let reset = if skip {
            Err(anyhow::anyhow!("the store is held"))
        } else {
            store.reset_injection_state(&session_key)
        };
        if reset.is_ok() {
            if wipe_waiting {
                maint::clear_pending_wipe(&paths, &session_key);
            }
        } else {
            skip = true;
            fresh = true;
            maint::pending_wipe(&paths, &session_key);
        }
    }

    let summoned = summons(
        cli_kind.as_str(),
        &hook,
        invocation.is_headless(),
        || store.has_stale_backlog(STALE_BACKLOG_SECS).unwrap_or(false),
        || !skip && claim_idle_sweep_window(&store, jiff::Timestamp::now()),
    );
    let spill = Spill { paths: &paths, session: &session_key, skip, fresh };
    match summoned.run {
        Some(Run::Session) => spawn_consolidation(Some(&session_key)),
        Some(Run::All) => spawn_consolidation(None),
        None => {}
    }
    if summoned.idle {
        spawn_detached(&["consolidate", "--idle"]);
    }

    if invocation.is_headless() {
        // The whole point: a one-shot run is usually an orchestrated step -
        // a reviewer, a judge - and handing it this project's narrative
        // destroys its independence in a way nothing downstream can see.
        return Ok("{}".to_string());
    }
    // A failed call is capture-only: nothing to say about the file it never read.
    if failed_call || !answer_is_read(cli_kind.as_str(), &payload) {
        return Ok("{}".to_string());
    }
    if hook == "user_prompt_submit" {
        let prompt = first_string(&payload, &["prompt"]).unwrap_or("");
        return Ok(inject_for_prompt(&store, &config, &scope, &session_key, event_name, prompt, &spill));
    }
    Ok(inject_for(&store, &config, &scope, &hook, event_name, &event, &spill))
}

/// The lesson a shell command triggers, as the hook's output.
///
/// The command is only parsed here: never run, never stored, never put in an
/// error. A command no stored lesson names ends on the program-list cache
/// without the store being opened; a missing cache counts as a match. The
/// project comes from the identity cache alone (computing it can run git),
/// and the machine's lessons are always in reach.
fn shell_lesson(cli: &str, event_name: &str, payload: &Value) -> Result<String> {
    const NONE: &str = "{}";
    // Cursor's echo of its own session, and a subagent's call, get nothing -
    // the same rule `inject_for` follows for files.
    if cli != "claude-code"
        || payload.get("cursor_version").is_some()
        || delegate_label(payload).is_some()
    {
        return Ok(NONE.to_string());
    }
    let command = payload
        .get("tool_input")
        .and_then(|input| input.get("command"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let candidates = command_candidates(command);
    if candidates.is_empty() {
        return Ok(NONE.to_string());
    }
    let paths = Paths::resolve()?;
    if let Ok(listed) = std::fs::read_to_string(paths.data_dir.join("lesson-programs")) {
        let known: std::collections::HashSet<&str> = listed.lines().collect();
        if !candidates.iter().any(|c| known.contains(c.split(' ').next().unwrap_or(""))) {
            return Ok(NONE.to_string());
        }
    }
    let Some(cwd) = working_directory(payload) else { return Ok(NONE.to_string()) };
    let machine = ids::machine_id().to_string();
    let project = ids::known_project_id(&cwd).map(|id| id.to_string());
    let mut projects: Vec<&str> = project.as_deref().into_iter().collect();
    projects.push(&machine);

    let session = ids::session_uuid(
        first_string(payload, &["session_id", "sessionId", "thread_id", "conversationId"])
            .unwrap_or("unknown-session"),
    )
    .to_string();
    let store = if maint::active(&paths) {
        Store::open_waiting(&paths.db(), maint::FAIL_FAST)?
    } else {
        Store::open(&paths.db())?
    };
    let config = Config::load(&paths.config_file())?;
    let injection = inject::for_command(&store, &projects, &session, &candidates, &config.injection)?;
    // Not skipped and not fresh on purpose: the store opened just above, and
    // `deliver` spills a failed write itself.
    let spill = Spill { paths: &paths, session: &session, skip: false, fresh: false };
    Ok(deliver(&store, &config, &session, event_name, Some(injection), &spill))
}

/// What a shell command could be triggered by: `program` and `program sub`,
/// lowercase, for each of its first three parts (split on `&&`, `;` and `|`).
/// `sudo`, `env X=Y`, leading assignments and `cd x` are looked through.
fn command_candidates(command: &str) -> Vec<String> {
    let mut end = command.len().min(4096);
    while !command.is_char_boundary(end) {
        end -= 1;
    }
    let text = command[..end].replace("&&", ";").replace('|', ";");
    let mut found: Vec<String> = Vec::new();
    for part in text.split(';').take(3) {
        let mut words = part.split_whitespace().peekable();
        let mut wrapped = false;
        while let Some(word) = words.peek() {
            let skip = if matches!(*word, "sudo" | "env") {
                wrapped = true;
                true
            } else {
                // An assignment, or a flag of the `sudo` / `env` before it.
                word.contains('=') || (wrapped && word.starts_with('-'))
            };
            if !skip {
                break;
            }
            words.next();
        }
        let Some(first) = words.next() else { continue };
        let program = first.rsplit('/').next().unwrap_or(first).to_lowercase();
        let plain = |s: &str| {
            !s.is_empty()
                && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-'))
        };
        if program == "cd" || !plain(&program) {
            continue;
        }
        if let Some(sub) = words.next().map(str::to_lowercase) {
            if !sub.starts_with('-') && plain(&sub) {
                found.push(format!("{program} {sub}"));
            }
        }
        found.push(program);
    }
    found.sort();
    found.dedup();
    found
}

/// Where a hook's ledger writes stand. When `skip` is set the store is held
/// and nothing is tried; a write that fails regardless is kept in
/// `surfaced.jsonl`, to be folded back by consolidation.
struct Spill<'a> {
    paths: &'a Paths,
    session: &'a str,
    skip: bool,
    /// The session's dedupe state is stale (a wipe is owed): read it as empty.
    fresh: bool,
}

/// Is this Cursor's own session, echoed to us through Claude Code's hooks?
///
/// Cursor reads Claude Code's hook config and fires those hooks too, so one
/// Cursor session arrives twice: once from the hooks we wired into Cursor
/// (`cli == "cursor"`) and once as `claude-code`. Only the `cursor_version`
/// key tells the second copy apart. The three per-turn hooks are the
/// duplicated ones; the rest (a delegate's report, session boundaries) have no
/// Cursor-side twin and pass through.
fn is_cursor_echo(cli: &str, hook: &str, payload: &Value) -> bool {
    cli == "claude-code"
        && matches!(hook, "post_tool_use" | "user_prompt_submit" | "stop")
        && payload
            .get("cursor_version")
            .and_then(Value::as_str)
            .is_some_and(|version| !version.trim().is_empty())
}

/// Will anyone read what this hook prints?
///
/// Every CLI with a hook protocol does. OpenCode has none: our plugin spawns
/// the hook itself, and reads the answer only where it waits for one - and
/// says so with `reads_answer`. Everything else it fires and forgets, and so
/// does a plugin written before it read anything. An injection built for one
/// of those reached no model, yet it spent the session budget and marked its
/// pointers as shown.
///
/// The plugin also stops listening at a deadline, `answer_by`, in epoch
/// milliseconds. A hook held past it - behind another writer's lock, most
/// likely, since capture is the part that waits - would otherwise spend the
/// budget on an answer the plugin has already dropped.
fn answer_is_read(cli: &str, payload: &Value) -> bool {
    if cli != "opencode" {
        return true;
    }
    let listening = payload.get("reads_answer").and_then(Value::as_bool) == Some(true);
    let in_time = payload
        .get("answer_by")
        .and_then(Value::as_i64)
        .is_none_or(|by| jiff::Timestamp::now().as_millisecond() <= by);
    listening && in_time
}

/// Which subagent a hook fired inside, if any.
///
/// `agent_id` is the host's own signal that the hook is inside a subagent;
/// `agent_type` alone is not, because a session started with `--agent`
/// carries it on the main thread too. The label is the type - the name the
/// lead dispatched.
fn delegate_label(payload: &Value) -> Option<String> {
    payload.get("agent_id").and_then(Value::as_str).filter(|id| !id.is_empty())?;
    Some(first_string(payload, &["agent_type"]).unwrap_or("subagent").to_string())
}

/// Decide what, if anything, to push back into the model's context.
///
/// Injection failures are silent on purpose. A hook that cannot look up a
/// pointer has still captured the event, and the agent still has the MCP
/// surface; degrading to no injection is strictly better than degrading the
/// session the user is working in.
fn inject_for(
    store: &Store,
    config: &Config,
    scope: &crate::ids::ProjectScope,
    hook: &str,
    event_name: &str,
    event: &Event,
    spill: &Spill<'_>,
) -> String {
    let project = scope.project_id.to_string();
    let session = spill.session;

    let injection = match hook {
        // The one entry point into a fresh context. Compaction arrives here
        // too, as a `session_start` whose source says `compact` - see
        // `wipes_context`, which explains why `post_compact` is not a second
        // route into this arm.
        "session_start" => inject::primer(store, &project, session, &config.injection, spill.fresh).ok(),
        // Both sides of a tool call reach the same file injection, and the
        // first one to arrive wins: `pre_tool_use` for a `Read`, where memory
        // still has time to change what the turn does, and `post_tool_use` for
        // everything else. `record_injected_file` is what keeps the second from
        // repeating the first.
        "pre_tool_use" | "post_tool_use" => event.files.first().and_then(|path| {
            let injection =
                inject::for_file(store, &project, session, path, &event.id, &config.injection)
                    .ok()?;
            // Mark the file covered even when it had nothing, so a file with
            // no memory is not re-queried on every touch.
            if spill.skip || store.record_injected_file(session, path).is_err() {
                maint::spill_surfaced(spill.paths, session, Ledger::File, std::slice::from_ref(path));
            }
            Some(injection)
        }),
        _ => None,
    };
    deliver(store, config, session, event_name, injection, spill)
}

/// The prompt-time push: opt-in (`[injection] prompt_pointers`); off, a
/// prompt gets nothing back, as it always has. The lookup is lexical and
/// time-boxed, and a failure is silent like every injection failure.
fn inject_for_prompt(
    store: &Store,
    config: &Config,
    scope: &crate::ids::ProjectScope,
    session: &str,
    event_name: &str,
    prompt: &str,
    spill: &Spill<'_>,
) -> String {
    if !config.injection.prompt_pointers {
        return "{}".to_string();
    }
    let project = scope.project_id.to_string();
    let injection = inject::for_prompt(store, &project, session, prompt, &config.injection).ok();
    deliver(store, config, session, event_name, injection, spill)
}

/// Record what an injection spent and render it as the hook's output.
fn deliver(
    store: &Store,
    config: &Config,
    session: &str,
    event_name: &str,
    injection: Option<inject::Injection>,
    spill: &Spill<'_>,
) -> String {
    let Some(injection) = injection else { return "{}".to_string() };
    if injection.is_empty() {
        return "{}".to_string();
    }
    let recorded = if spill.skip {
        Err(anyhow::anyhow!("the store is held"))
    } else {
        store.record_injected(
            session,
            &injection.ids,
            injection.in_flight,
            injection.text.len(),
            config.injection.session_budget,
        )
    };
    if recorded.is_err() {
        maint::spill_injected(spill.paths, session, &injection.ids, injection.text.len());
    }
    // Another hook of the same session spent the budget between our read of
    // it and this write. A failed write still injects, as it always has.
    if matches!(recorded, Ok(false)) {
        return "{}".to_string();
    }
    inject::as_hook_output(event_name, &injection)
}

/// Is this hook a capture surface, or only an injection one?
///
/// Nearly all of them are both. `pre_tool_use` is the exception: we ask for it
/// only to get in front of a `Read`, and its payload is the same tool call
/// `post_tool_use` reports a moment later with a result attached. Storing both
/// measured out at 96% duplication across 1,433 real events - double the rows
/// and double the consolidation prompt for no extra fact.
#[must_use]
pub fn captures(hook: &str) -> bool {
    hook != "pre_tool_use"
}

/// Did this event just destroy the agent's context?
///
/// `SessionStart` carries both cases Claude Code actually gives us: `compact`
/// and `clear`. `post_compact` stays answered here because it is true - that
/// event does wipe context - but we no longer register for it. Claude Code
/// rejects `additionalContext` under `PostCompact`, failing the whole hook run
/// and discarding the primer, so injecting there was never anything but a
/// visible error on top of the `SessionStart` that already worked.
#[must_use]
pub fn wipes_context(hook: &str, payload: &Value) -> bool {
    if hook == "post_compact" {
        return true;
    }
    hook == "session_start"
        && payload
            .get("source")
            .and_then(Value::as_str)
            .is_some_and(|source| matches!(source, "compact" | "clear"))
}

/// CLIs whose session end we have actually watched fire.
///
/// Codex was on the wrong side of this list once, on the strength of an event
/// missing from a config file. A probe settled it. Nothing joins this list
/// without the same evidence.
const HAVE_SESSION_END: &[&str] = &["claude-code", "codex"];

/// Should this event kick consolidation?
///
/// A session end is the real boundary: everything that happened is in, and
/// nothing more is coming. `stop` is only a *substitute* boundary, for CLIs
/// that have no session end — and using it where a real one exists means
/// consolidating once per turn instead of once per session, which is model
/// calls the user pays for and did not ask for.
///
/// Compaction counts too, from either side of the fence: it is the last moment
/// a session's detail exists.
#[must_use]
pub fn is_session_boundary(cli: &str, hook: &str) -> bool {
    match hook {
        "session_end" | "pre_compact" => true,
        "stop" => !HAVE_SESSION_END.contains(&cli),
        _ => false,
    }
}

/// What actually triggers consolidation for a CLI, in plain words.
///
/// Reported by `brain doctor`, because "when does it call a model?" is the
/// question people keep having to read source to answer — and the answer
/// genuinely differs per CLI, because their lifecycle surfaces do.
///
/// Stated per CLI rather than derived from the wired event list, because the
/// two are not the same thing: OpenCode's plugin subscribes to the end of an
/// execution and reports it to us as `stop`, so the event list would lie
/// about it.
#[must_use]
pub fn consolidation_triggers(cli: &str) -> &'static str {
    match cli {
        // A real session end, plus compaction on the way out.
        "claude-code" | "codex" => "session end, compaction",
        // No session end; its plugin reports the end of a turn, and
        // compaction, as ours.
        "opencode" => "end of turn, compaction",
        // No session end and no compaction hook: end of turn is all they
        // offer.
        "antigravity" | "cursor" => "end of turn (no session-end event)",
        // Neither a session end nor a turn end reaches us. Its memory is
        // consolidated by the backstop when the next session opens.
        "gemini-cli" => "backstop only (no boundary event reaches us)",
        _ => "backstop only",
    }
}

/// The detached run a hook starts on its way out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Run {
    /// `consolidate --session`, for the session this hook belongs to.
    Session,
    /// `consolidate --all`: the backstop.
    All,
}

/// What a hook starts: at most one run, plus the idle sweep.
#[derive(Debug, Default, PartialEq, Eq)]
struct Summons {
    run: Option<Run>,
    /// `consolidate --idle`, for sessions nobody came back to.
    idle: bool,
}

/// Decide what this hook starts.
///
/// A one-shot run (`claude -p`, `codex exec`) starts nothing, because nothing
/// waits on its consolidation: it never gets a summary (`consolidate_session`
/// settles it as headless, with no model call), and the in-flight list leaves
/// it out (`Store::unconsolidated_sessions`). Yet its start, its stops and its
/// end each started a detached run: 88% of the 1,783 spawns one machine made
/// in a day, nearly all of them standing aside for the run that already held
/// the lock. Its events settle in the next `--all` or idle pass that a
/// person's session starts, at no model cost.
///
/// The two questions are closures so that a one-shot run asks neither: the
/// backlog question is a read on the hook path, and claiming the idle window
/// is a write.
fn summons(
    cli: &str,
    hook: &str,
    headless: bool,
    stale: impl FnOnce() -> bool,
    idle_window: impl FnOnce() -> bool,
) -> Summons {
    if headless {
        return Summons::default();
    }
    let run = if is_session_boundary(cli, hook) {
        // Compaction is the last moment this session's detail exists. Kicking
        // consolidation here means the primer that lands seconds later carries
        // a real narrative rather than a list of raw commands. Detached and
        // idempotent, so an extra run costs nothing.
        Some(Run::Session)
    } else if hook == "session_start" && stale() {
        // The backstop, without a background agent: a session opening is
        // exactly when older unconsolidated work becomes worth finishing,
        // because this session is the one that will read it.
        Some(Run::All)
    } else {
        None
    };
    Summons { run, idle: hook == "stop" && idle_window() }
}

/// Start consolidation in a detached child and return immediately.
///
/// The host CLI is waiting on this hook, so nothing here may block: the child
/// is fully detached from our stdio and outlives us. If it cannot start, the
/// catch-up backstop picks the work up later — this is best-effort by design.
fn spawn_consolidation(session: Option<&str>) {
    match session {
        Some(session) => spawn_detached(&["consolidate", "--session", session]),
        // No session: catch up on whatever is stale, anywhere.
        None => spawn_detached(&["consolidate", "--all"]),
    }
}

/// Run `brain <args>` as a child with no stdio, and do not wait for it.
/// Best-effort: a child that cannot start is simply not started.
fn spawn_detached(args: &[&str]) {
    let Ok(exe) = std::env::current_exe() else { return };
    let _ = detached_command(&exe, args).spawn();
}

/// The command `spawn_detached` runs: no stdio and, on unix, its own process
/// group, so a terminal hang-up aimed at the hook's group cannot kill it.
fn detached_command(exe: &std::path::Path, args: &[&str]) -> std::process::Command {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd
}

/// Which directory this event happened in, or `None` when it cannot be known.
///
/// Host CLIs disagree completely here. Claude Code and Codex send `cwd`.
/// Antigravity sends neither `cwd` nor a populated workspace unless the user
/// passed `--add-dir`, and runs the hook with its working directory set to its
/// OWN config directory — so falling back to the process cwd would file every
/// Antigravity event under a project called "config".
///
/// Hence the last rule: a process cwd that sits inside some CLI's config
/// directory is not a project, and we would rather capture nothing than
/// capture into the wrong brain.
fn working_directory(payload: &Value) -> Option<std::path::PathBuf> {
    if let Some(cwd) = first_string(payload, &["cwd", "workspace_root", "directory"]) {
        return Some(std::path::PathBuf::from(cwd));
    }
    // A list of roots, under whatever the CLI calls it: `workspacePaths` for
    // Antigravity, `workspace_roots` for Cursor. Cursor also sends a `cwd`
    // key, but it arrives empty - which is why the lookup above filters empty
    // strings rather than merely checking for the key.
    for key in ["workspacePaths", "workspace_roots"] {
        if let Some(path) = payload
            .get(key)
            .and_then(Value::as_array)
            .and_then(|paths| paths.first())
            .and_then(Value::as_str)
            .filter(|path| !path.is_empty())
        {
            return Some(std::path::PathBuf::from(path));
        }
    }

    let current = std::env::current_dir().ok()?;
    (!is_cli_config_dir(&current)).then_some(current)
}

/// Is this path inside a host CLI's own configuration directory?
///
/// Both sides are resolved before comparing. `current_dir` hands back the
/// physical path with every symlink already followed, while `$HOME` is
/// whatever the user's shell says it is — and on macOS a temporary or
/// symlinked home differs from its resolved form by a `/private` prefix
/// alone. Comparing the two as written makes this answer `false` for a
/// directory that plainly IS a CLI's config, and every event from that host
/// then lands in a project named `config`.
fn is_cli_config_dir(path: &std::path::Path) -> bool {
    let Some(home) = dirs::home_dir() else { return false };
    let resolve = |path: &std::path::Path| path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    // Both spellings of both sides, because `canonicalize` is not a pure
    // normalisation: it fails on a path that does not exist yet - which a
    // config directory often is - and on Windows it succeeds by returning a
    // verbatim `\\?\C:\...` form. Comparing a resolved home against an
    // unresolved path then matches nothing, and every event inside a CLI's own
    // configuration would be captured as if it were project work. Resolving is
    // still tried first, because a symlinked home is the case it exists for.
    let pairs = [(resolve(path), resolve(&home)), (path.to_path_buf(), home.clone())];
    [".gemini", ".cursor", ".codex", ".claude", ".antigravity", ".config/opencode"]
        .iter()
        .any(|dir| pairs.iter().any(|(path, home)| path.starts_with(home.join(dir))))
}

/// First present, non-empty string among several candidate keys.
fn first_string<'a>(payload: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| payload.get(*key).and_then(Value::as_str))
        .filter(|value| !value.is_empty())
}

/// The tool this event is about, whatever the CLI calls it.
fn tool_name(payload: &Value) -> &str {
    payload
        .get("tool_name")
        .and_then(Value::as_str)
        .or_else(|| payload.pointer("/toolCall/name").and_then(Value::as_str))
        .unwrap_or("")
}

/// What a brain recall result is stored as: it is a copy of rows already in
/// the store, and keeping it would put every recalled row in the index twice.
const BRAIN_RECALL_OMITTED: &str = "[brain recall result omitted]";

/// Brain's own read tools, in every spelling a host gives them
/// (`mcp__plugin_rolepod-brain_brain__brain_search`, `mcp__brain__brain_get`,
/// `MCP:brain_recent`, `brain_brain_related`, `brain_outline`). The writers
/// (note, correct, forget, feedback) and doctor are not recall and stay kept.
fn is_brain_recall(tool: &str) -> bool {
    let lower = tool.to_ascii_lowercase();
    let tail = lower.rsplit(['_', ':']).next().unwrap_or("");
    let head = &lower[..lower.len() - tail.len()];
    matches!(tail, "search" | "get" | "recent" | "timeline" | "related" | "outline")
        && head.strip_suffix("brain_").is_some_and(|before| {
            before.is_empty() || before.ends_with(['_', ':'])
        })
}

/// The tool's arguments, whatever the CLI calls them.
fn tool_input(payload: &Value) -> Option<&Value> {
    payload.get("tool_input").or_else(|| payload.pointer("/toolCall/args"))
}

/// Normalize a hook name to snake_case for the wire format.
///
/// Host CLIs spell the same event differently (`PostToolUse`, `post-tool-use`),
/// and `source.hook` is a filter users will query, so it must not vary by CLI.
#[must_use]
pub fn normalize_hook(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 4);
    for (index, ch) in raw.chars().enumerate() {
        if ch == '-' || ch == '_' || ch == ' ' || ch == '.' {
            out.push('_');
        } else if ch.is_ascii_uppercase() {
            if index > 0 && !out.ends_with('_') {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// Rule-based one-line title.
///
/// This is the zero-LLM tier and a permanent mode, not a placeholder: with the
/// summarizer off these titles are what the primer shows forever, so they are
/// written to be read by a human.
fn title_for(hook: &str, payload: &Value) -> String {
    let str_field = |key: &str| payload.get(key).and_then(Value::as_str).unwrap_or("");

    match hook {
        "user_prompt_submit" => {
            let prompt = str_field("prompt");
            if prompt.is_empty() {
                "Prompt submitted".to_string()
            } else {
                format!("Asked: {}", first_line(prompt))
            }
        }
        "pre_tool_use" | "post_tool_use" | "post_invocation" => {
            let tool = tool_name(payload);
            let input = tool_input(payload);
            // Case-insensitive: Claude Code says `Bash`, Cursor says `Shell`.
            // Matching exactly would have left every Cursor command titled
            // "Used Shell" with the command itself thrown away.
            let shell = tool.eq_ignore_ascii_case("bash") || tool.eq_ignore_ascii_case("shell");
            match (shell, input) {
                (true, Some(input)) => {
                    let command = input.get("command").and_then(Value::as_str).unwrap_or("");
                    describe_command(command)
                }
                (false, Some(input)) if !tool.is_empty() => match tool_path(input) {
                    Some(path) => format!("{tool}: {path}"),
                    None => format!("Used {tool}"),
                },
                _ if !tool.is_empty() => format!("Used {tool}"),
                _ => "Tool call".to_string(),
            }
        }
        "session_start" => {
            let source = str_field("source");
            if source.is_empty() {
                "Session started".to_string()
            } else {
                format!("Session started ({source})")
            }
        }
        "session_end" => {
            let reason = str_field("reason");
            if reason.is_empty() {
                "Session ended".to_string()
            } else {
                format!("Session ended ({reason})")
            }
        }
        "stop" => "Turn finished".to_string(),
        // The report path in `capture` overrides this whenever the host sent
        // the delegate's last message; this is the title when it sent none.
        "subagent_stop" => match delegate_label(payload) {
            Some(who) => format!("{who} finished"),
            None => "Subagent finished".to_string(),
        },
        "pre_compact" => {
            let trigger = str_field("trigger");
            if trigger.is_empty() {
                "Context compacted".to_string()
            } else {
                format!("Context compacted ({trigger})")
            }
        }
        "notification" => {
            let message = str_field("message");
            if message.is_empty() {
                "Notification".to_string()
            } else {
                format!("Notice: {}", first_line(message))
            }
        }
        other => format!("Event: {other}"),
    }
}

/// A delegate's tool call: summarized away later, so only its shape is kept.
fn is_footstep(delegate: Option<&str>, hook: &str) -> bool {
    delegate.is_some() && hook == "post_tool_use"
}

/// The stored body of an event: shaped leaves, the final scrub and 16 KiB
/// cap, then the footstep's own 1 KiB clamp.
fn event_body(payload: &Value, sanitizer: &Sanitizer, footstep: bool) -> String {
    let body = sanitizer.scrub_body(&body_for(payload, sanitizer, footstep));
    if footstep {
        truncate_head_tail(&body, FOOTSTEP_BODY_MAX_BYTES)
    } else {
        body
    }
}

/// The content worth keeping, as compact JSON.
///
/// Deliberately a subset: a hook payload carries a full transcript path, tool
/// schemas, and other noise that would bloat every event without ever being
/// recalled.
fn body_for(payload: &Value, sanitizer: &Sanitizer, footstep: bool) -> String {
    let leaf_max = if footstep { FOOTSTEP_LEAF_MAX_BYTES } else { LEAF_MAX_BYTES };
    let mut kept = serde_json::Map::new();
    // Set once any leaf ends inside an unclosed `<private>`: nothing after it,
    // in this leaf or any later one, is safe to keep.
    let mut private_open = false;
    // Only a failed call keeps the failure fields; anywhere else they are noise
    // and a host-sent `failed` must not pass for ours.
    let failed = payload.get("failed").and_then(Value::as_bool) == Some(true);
    for key in [
        "prompt",
        "tool_name",
        "tool_input",
        "toolCall",
        "terminationReason",
        "tool_response",
        "tool_output",
        "error",
        "is_interrupt",
        "failed",
        "message",
        "reason",
        "source",
        "trigger",
        "custom_instructions",
    ] {
        if !failed && matches!(key, "tool_output" | "error" | "is_interrupt" | "failed") {
            continue;
        }
        if let Some(value) = payload.get(key) {
            if private_open {
                kept.insert(key.to_string(), Value::String("[PRIVATE]".into()));
                continue;
            }
            let mut value = value.clone();
            match key {
                "tool_response" if is_brain_recall(tool_name(payload)) => {
                    value = Value::String(BRAIN_RECALL_OMITTED.into());
                }
                "tool_response" | "tool_output" | "error" => {
                    // `tool_response`, or a failure's `tool_output` / `error`
                    // (a string or an object, by host). The file as it was and
                    // the patch are copies of what the file itself and the edit
                    // input already say.
                    if let Some(object) = value.as_object_mut() {
                        object.remove("originalFile");
                        object.remove("structuredPatch");
                    }
                    shape_leaves(&mut value, sanitizer, leaf_max, &mut private_open);
                }
                "tool_input" if footstep => {
                    shape_leaves(&mut value, sanitizer, leaf_max, &mut private_open);
                }
                "tool_input" => {
                    if let Some(content) = value.as_object_mut().and_then(|o| o.get_mut("content"))
                    {
                        shape_leaves(content, sanitizer, leaf_max, &mut private_open);
                    }
                }
                _ => {}
            }
            kept.insert(key.to_string(), value);
        }
    }
    if kept.is_empty() {
        return String::new();
    }
    serde_json::to_string(&Value::Object(kept)).unwrap_or_default()
}

/// Scrub every string leaf, then clamp it. Scrub first: a clamp that cut a
/// secret in half would leave both halves past the pattern that finds it.
/// `private_open` carries an unclosed `<private>` from one leaf to the rest.
fn shape_leaves(value: &mut Value, sanitizer: &Sanitizer, max: usize, private_open: &mut bool) {
    match value {
        Value::String(text) => {
            if *private_open {
                *text = "[PRIVATE]".to_string();
            } else {
                *private_open = leaves_private_open(text);
                *text = clamp_leaf(&sanitizer.scrub(text), max);
            }
        }
        Value::Array(items) => {
            items.iter_mut().for_each(|v| shape_leaves(v, sanitizer, max, private_open));
        }
        Value::Object(map) => {
            map.values_mut().for_each(|v| shape_leaves(v, sanitizer, max, private_open));
        }
        _ => {}
    }
}

/// File paths this event touched, relative to the project root when possible.
fn files_for(payload: &Value, root: &std::path::Path) -> Vec<String> {
    let mut files = Vec::new();
    if let Some(input) = tool_input(payload) {
        if let Some(path) = tool_path(input) {
            files.push(path);
        }
        // Multi-file tools carry an array of edits.
        if let Some(edits) = input.get("edits").and_then(Value::as_array) {
            for edit in edits {
                if let Some(path) = tool_path(edit) {
                    files.push(path);
                }
            }
        }
    }
    files
        .into_iter()
        .map(|path| relativize(&path, root))
        .filter(|path| !path.is_empty())
        .collect()
}

/// Is this path where the named CLI actually keeps its transcripts?
///
/// A hook payload arrives on stdin from whatever invoked us. Consolidation
/// later reads the path it names and hands the contents to a model, so an
/// unchecked path is a way to ask brain to fetch a file and post it
/// somewhere - the classic confused deputy. Confining it to each CLI's own
/// transcript directory costs nothing: no CLI writes its transcripts
/// anywhere else.
fn is_transcript_of(cli: &str, path: &str) -> bool {
    let Some(home) = dirs::home_dir() else { return false };
    is_transcript_under(&home, cli, path)
}

/// [`is_transcript_of`] against an explicit home, so a test needs no `$HOME`.
fn is_transcript_under(home: &std::path::Path, cli: &str, path: &str) -> bool {
    let roots: &[&str] = match cli {
        "claude-code" => &[".claude/projects"],
        "codex" => &[".codex/sessions"],
        "cursor" => &[".cursor/projects"],
        // The other CLIs write no transcript at all, so any path claiming to
        // be one is by definition not theirs.
        _ => return false,
    };
    // Resolve before comparing: `~/.claude/projects/../../.ssh/id_rsa` is
    // inside the directory only until someone reads it.
    let Ok(resolved) = std::fs::canonicalize(path) else { return false };
    let Some(rest) = roots.iter().find_map(|root| {
        let root = std::fs::canonicalize(home.join(root)).ok()?;
        resolved.strip_prefix(&root).ok().map(std::path::Path::to_path_buf)
    }) else {
        return false;
    };
    if cli != "cursor" {
        return true;
    }
    // Cursor keeps other files under the same tree (rules, mcp state, canvases).
    // Its transcripts are only ever `<project>/agent-transcripts/<id>/<id>.jsonl`.
    let parts: Vec<_> = rest.components().map(|part| part.as_os_str()).collect();
    match parts.as_slice() {
        [_project, dir, id, file] => {
            *dir == "agent-transcripts"
                && std::path::Path::new(file).extension().is_some_and(|ext| ext == "jsonl")
                && std::path::Path::new(file).file_stem() == Some(*id)
        }
        _ => false,
    }
}

/// Pull a filesystem path out of a tool input, whatever the tool calls it.
fn tool_path(input: &Value) -> Option<String> {
    // `AbsolutePath` is Antigravity's spelling; the rest are Claude Code's and
    // Codex's. Every one of these was read off a real captured payload.
    for key in ["file_path", "path", "notebook_path", "filePath", "AbsolutePath"] {
        if let Some(path) = input.get(key).and_then(Value::as_str) {
            if !path.is_empty() {
                return Some(path.to_string());
            }
        }
    }
    None
}

/// Make a path repo-relative so memory survives the checkout moving.
///
/// Both sides are resolved through symlinks first. On macOS a temp or home
/// path routinely arrives as `/var/...` while the project root resolves to
/// `/private/var/...`; comparing them raw silently stores absolute paths, and
/// file-keyed recall then fails to match the same file across sessions.
fn relativize(path: &str, root: &std::path::Path) -> String {
    let resolved = resolve_symlinks(std::path::Path::new(path));
    for candidate in [resolved.as_path(), std::path::Path::new(path)] {
        if let Ok(relative) = candidate.strip_prefix(root) {
            return portable_separators(&relative.to_string_lossy());
        }
    }
    portable_separators(path)
}

/// One spelling of a path, whichever machine recorded it.
///
/// These strings are not used to open anything - they key which memories a
/// file has, they are what a wiki page cites, and they travel between machines
/// through `brain export`. Windows would write `src\main.rs` where every other
/// platform writes `src/main.rs`, and the same file in the same repository
/// would then be two different keys either side of an import. Forward slashes
/// are the portable spelling, and Windows accepts them as input anyway.
///
/// Only separators change. A backslash inside a component on a unix filesystem
/// is a legal character in a name, so this runs on Windows alone.
fn portable_separators(path: &str) -> String {
    if cfg!(windows) { path.replace('\\', "/") } else { path.to_string() }
}

/// Resolve symlinks in the deepest part of a path that exists.
///
/// `canonicalize` fails outright on a path whose leaf is missing — which is
/// every file a tool is about to create — so resolve the closest existing
/// ancestor and re-attach the rest.
fn resolve_symlinks(path: &std::path::Path) -> std::path::PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let mut suffix = Vec::new();
    let mut current = path;
    while let Some(parent) = current.parent() {
        if let Some(name) = current.file_name() {
            suffix.push(name.to_os_string());
        }
        if let Ok(canonical) = parent.canonicalize() {
            let mut out = canonical;
            for part in suffix.iter().rev() {
                out.push(part);
            }
            return out;
        }
        current = parent;
    }
    path.to_path_buf()
}

/// Turn a shell command into something worth reading later.
///
/// A raw command line is mostly bytes nobody recalls: quoted patterns, ranges,
/// pipes, absolute paths. What a person remembers is *what tool ran against
/// what file*. So the title becomes `<verb>: <file>` when both are
/// recognizable, and degrades to the program name rather than guessing.
///
/// Deliberately not clever. Every rule here maps a token we can see to a word
/// that is true of it; nothing infers intent. A wrong title is worse than a
/// plain one, because it will be trusted.
fn describe_command(command: &str) -> String {
    let line = first_line(command);
    if line.is_empty() {
        return "Ran a command".to_string();
    }

    let tokens = shell_tokens(&line);
    let Some(program) = tokens.first().map(String::as_str) else {
        return "Ran a command".to_string();
    };

    // Wrappers say nothing about the work; step past them to the real program.
    let mut index = 0;
    while matches!(
        base_name(&tokens[index]).as_str(),
        "sudo" | "env" | "time" | "rtk" | "nohup" | "command" | "xargs"
    ) && index + 1 < tokens.len()
    {
        index += 1;
    }
    let program = base_name(tokens.get(index).map_or(program, String::as_str));

    let verb = match program.as_str() {
        "sed" | "head" | "tail" | "cat" | "less" | "awk" | "jq" => Some("read"),
        "grep" | "rg" | "ag" | "ack" | "find" | "fd" => Some("search"),
        "pytest" | "jest" | "vitest" | "mocha" => Some("test"),
        "curl" | "wget" | "http" => Some("fetch"),
        _ => None,
    };

    // A subcommand is more informative than the program for tool drivers.
    //
    // "Bare word" is the whole test, and it is what keeps a flag's VALUE from
    // being read as the subcommand: `pnpm --filter @scope/api typecheck`
    // must resolve to `typecheck`, not to the package name. No flag modelling
    // — a token carrying `/`, `@`, `.` or `=` is simply not a subcommand.
    let rest = &tokens[index + 1..];
    let subcommand_at = matches!(
        program.as_str(),
        "cargo" | "git" | "npm" | "pnpm" | "yarn" | "go" | "docker" | "brew" | "gh" | "uv"
    )
    .then(|| rest.iter().position(|token| is_bare_word(token)))
    .flatten();
    let subcommand = subcommand_at.map(|at| rest[at].as_str());

    // Only look for a target AFTER the subcommand. Anything before it belongs
    // to the flags that selected what to run, not to what was operated on.
    let searchable = subcommand_at.map_or(rest, |at| &rest[at + 1..]);
    let target = searchable.iter().rev().find(|token| looks_like_path(token));

    match (verb, subcommand, target) {
        (Some(verb), _, Some(path)) => format!("{verb}: {}", base_name(path)),
        (Some(verb), _, None) => format!("{verb}: {program}"),
        (None, Some(sub), Some(path)) => format!("{program} {sub}: {}", base_name(path)),
        (None, Some(sub), None) => format!("{program} {sub}"),
        (None, None, Some(path)) => format!("{program}: {}", base_name(path)),
        (None, None, None) => format!("Ran {program}"),
    }
}

/// Split a command into tokens, keeping quoted runs together.
///
/// Not a shell parser and not trying to be — it only has to be good enough to
/// stop a quoted regex from being mistaken for a filename.
fn shell_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in line.chars() {
        match (quote, ch) {
            (Some(open), c) if c == open => quote = None,
            (Some(_), c) => current.push(c),
            (None, c @ ('\'' | '"')) => quote = Some(c),
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            // A pipeline or redirect ends the part of the command we describe.
            (None, '|' | '>' | '<' | ';' | '&') => break,
            (None, c) => current.push(c),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// A plain word: a subcommand, never a path, a flag, or a flag's value.
fn is_bare_word(token: &str) -> bool {
    !token.is_empty()
        && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && !token.starts_with('-')
}

/// Does this token name a file rather than a flag or a pattern?
fn looks_like_path(token: &str) -> bool {
    if token.starts_with('-') || token.contains('=') {
        return false;
    }
    // Glob and regex metacharacters mean this is a pattern being searched for,
    // not a file being worked on.
    if token.contains(['*', '?', '{', '}', '[', ']', '(', ')', '$', '^']) {
        return false;
    }
    let name = base_name(token);
    // A dot with a short alphabetic tail is an extension; anything else with a
    // slash is a path. Both are things a person recognizes later.
    name.rsplit_once('.').is_some_and(|(stem, ext)| {
        !stem.is_empty()
            && (1..=5).contains(&ext.len())
            && ext.chars().all(|c| c.is_ascii_alphanumeric())
    }) || token.contains('/')
}

fn base_name(token: &str) -> String {
    token.trim_end_matches('/').rsplit('/').next().unwrap_or(token).to_string()
}

/// First line of a string, trimmed and collapsed.
pub fn first_line(input: &str) -> String {
    input
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_host_plan_spawns_a_process_only_when_nothing_is_known() {
        use super::{host_plan, HostPlan, NO_HOST};
        assert_eq!(host_plan(false, None), HostPlan::Classify);
        assert_eq!(host_plan(true, None), HostPlan::FindHost);
        assert_eq!(host_plan(true, Some(42)), HostPlan::Reuse(42));
        // The marker for an unknown host is a row like any other.
        assert_eq!(host_plan(true, Some(NO_HOST)), HostPlan::Reuse(NO_HOST));
    }

    use super::*;
    use serde_json::json;

    #[test]
    fn a_shell_command_yields_its_program_and_sub_command() {
        assert_eq!(command_candidates("sudo -E env A=1 /usr/bin/Git push origin && ls"), vec!["git", "git push", "ls"]);
        assert_eq!(command_candidates("timeout 5 curl -s x"), vec!["timeout", "timeout 5"]);
        assert_eq!(command_candidates("cd /tmp; cargo test | wc -l"), vec!["cargo", "cargo test", "wc"]);
        assert!(!captures("pre_tool_use"), "a shell pre-event became a capture surface");
        assert!(command_candidates("cd /tmp").is_empty());
        assert!(command_candidates("").is_empty());
        assert_eq!(command_candidates("a; b; c; d"), vec!["a", "b", "c"]);
    }

    #[cfg(unix)]
    #[test]
    fn a_detached_child_gets_its_own_process_group() {
        let mut child = detached_command(std::path::Path::new("/bin/sleep"), &["5"])
            .spawn()
            .unwrap();
        let pgid_of = |pid: u32| -> u32 {
            let out = std::process::Command::new("ps")
                .args(["-o", "pgid=", "-p", &pid.to_string()])
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
        };
        let child_pgid = pgid_of(child.id());
        let own_pgid = pgid_of(std::process::id());
        let _ = child.kill();
        let _ = child.wait();
        assert_eq!(child_pgid, child.id());
        assert_ne!(child_pgid, own_pgid);
    }

    #[test]
    fn a_silenced_process_captures_nothing() {
        let _guard =
            invocation::ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var(invocation::SILENT_ENV, "1");
        let payload = json!({"prompt": "should leave no trace"}).to_string();
        let ack = capture("claude-code", "UserPromptSubmit", Some(payload)).unwrap();
        std::env::remove_var(invocation::SILENT_ENV);
        assert_eq!(ack, "{}");
    }

    #[test]
    fn both_context_wipe_paths_are_recognized() {
        // A host that sends `post_compact` is still telling the truth about
        // context, even though we no longer register for the event.
        assert!(wipes_context("post_compact", &json!({"trigger": "auto"})));
        assert!(wipes_context("session_start", &json!({"source": "clear"})));
        assert!(wipes_context("session_start", &json!({"source": "compact"})));
        // A normal start or resume keeps whatever the agent already had.
        assert!(!wipes_context("session_start", &json!({"source": "startup"})));
        assert!(!wipes_context("session_start", &json!({"source": "resume"})));
        assert!(!wipes_context("session_start", &json!({})));
        assert!(!wipes_context("post_tool_use", &json!({"source": "compact"})));
    }

    #[test]
    fn a_real_session_end_makes_stop_stop_triggering() {
        for cli in ["claude-code", "codex"] {
            assert!(is_session_boundary(cli, "session_end"), "{cli}");
            assert!(is_session_boundary(cli, "pre_compact"), "{cli}");
            // The expensive mistake: `stop` fires every turn.
            assert!(!is_session_boundary(cli, "stop"), "{cli} consolidates per turn");
        }
    }

    #[test]
    fn a_cli_without_a_session_end_still_gets_a_boundary() {
        for cli in ["antigravity", "opencode", "cursor"] {
            assert!(is_session_boundary(cli, "stop"), "{cli} would never consolidate");
            assert!(is_session_boundary(cli, "pre_compact"), "{cli}");
        }
    }

    #[test]
    fn ordinary_events_never_trigger_consolidation() {
        for cli in ["claude-code", "antigravity"] {
            for hook in ["post_tool_use", "user_prompt_submit", "session_start"] {
                assert!(!is_session_boundary(cli, hook), "{cli}/{hook}");
            }
        }
    }

    #[test]
    fn the_reported_triggers_match_the_behaviour() {
        for cli in ["claude-code", "codex"] {
            assert!(consolidation_triggers(cli).starts_with("session end"));
        }
        assert!(consolidation_triggers("antigravity").contains("end of turn"));
        // gemini reaches us with no boundary event at all - saying "end of
        // turn" there would be a health report making something up.
        assert!(consolidation_triggers("gemini-cli").contains("backstop only"));
        assert!(consolidation_triggers("opencode").contains("end of turn"));
    }

    #[test]
    fn a_worker_child_captures_nothing() {
        // Guards the loop: consolidation spawns a host CLI, whose hooks call
        // us, and capturing there would feed the brain its own output.
        let _guard =
            invocation::ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var(WORKER_ENV, "1");
        let payload = json!({"prompt": "should not be captured"}).to_string();
        let ack = capture("claude-code", "UserPromptSubmit", Some(payload)).unwrap();
        std::env::remove_var(WORKER_ENV);
        assert_eq!(ack, "{}");
    }

    #[test]
    fn antigravity_payload_shape_is_understood() {
        // Captured verbatim from a real `agy -p` run on this machine.
        let payload = json!({
            "conversationId": "17e2d461-9aea-442c-9825-6d8c642ad4b6",
            "modelName": "gemini-3.5-flash-low",
            "workspacePaths": ["/repo"],
            "toolCall": {"name": "view_file", "args": {"AbsolutePath": "/repo/src/main.rs"}}
        });
        assert_eq!(tool_name(&payload), "view_file");
        assert_eq!(files_for(&payload, std::path::Path::new("/repo")), vec!["src/main.rs"]);
        assert_eq!(
            working_directory(&payload),
            Some(std::path::PathBuf::from("/repo"))
        );
        assert_eq!(title_for("post_tool_use", &payload), "view_file: /repo/src/main.rs");
    }

    #[test]
    fn a_hook_inside_a_subagent_is_labelled_by_its_type_and_only_then() {
        // `agent_id` is the host's signal that the hook fired inside a
        // subagent. `agent_type` alone is a session started with `--agent`,
        // whose main thread must stay the lead's own work.
        let inside = json!({"agent_id": "agent-def456", "agent_type": "rolepod:universal-reviewer"});
        assert_eq!(delegate_label(&inside).as_deref(), Some("rolepod:universal-reviewer"));
        let unnamed = json!({"agent_id": "agent-def456"});
        assert_eq!(delegate_label(&unnamed).as_deref(), Some("subagent"));
        let agent_session = json!({"agent_type": "security-reviewer", "session_id": "abc"});
        assert_eq!(delegate_label(&agent_session), None);
        assert_eq!(delegate_label(&json!({"agent_id": ""})), None);
        assert_eq!(delegate_label(&json!({})), None);
    }

    #[test]
    fn a_subagent_stop_without_a_message_is_still_titled_by_who_finished() {
        let payload = json!({"agent_id": "agent-1", "agent_type": "Explore"});
        assert_eq!(title_for("subagent_stop", &payload), "Explore finished");
        assert_eq!(title_for("subagent_stop", &json!({})), "Subagent finished");
    }

    #[test]
    fn cursor_payload_shape_is_understood() {
        // Captured verbatim from a real `cursor-agent -p` run: an EMPTY cwd,
        // the project under `workspace_roots`, and Claude-Code-shaped tool
        // fields. The empty cwd is the trap - a key check would have taken it.
        let payload = json!({
            "conversation_id": "a104b5a8-9689-4f7e-a964-991f34c2d470",
            "session_id": "a104b5a8-9689-4f7e-a964-991f34c2d470",
            "cwd": "",
            "workspace_roots": ["/repo"],
            "tool_name": "Shell",
            "tool_input": {"command": "echo hi", "cwd": ""},
            "hook_event_name": "postToolUse"
        });
        assert_eq!(working_directory(&payload), Some(std::path::PathBuf::from("/repo")));
        assert_eq!(tool_name(&payload), "Shell");
        assert_eq!(title_for("post_tool_use", &payload), "Ran echo");
    }

    /// A recorded path reads the same whichever machine wrote it.
    ///
    /// These strings key which memories a file has and travel between machines
    /// through `brain export`. Windows writing `src\\main.rs` where everything
    /// else writes `src/main.rs` makes one file in one repository into two
    /// different keys either side of an import - a brain that quietly forgets
    /// half of what it knows about a file the moment it moves.
    #[test]
    fn a_recorded_path_is_spelled_the_same_on_every_platform() {
        let root = std::env::temp_dir().join(format!("brain-sep-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(root.join("src")).expect("create");
        let file = root.join("src").join("main.rs");
        std::fs::write(&file, "fn main() {}").expect("write");

        let recorded = relativize(&file.to_string_lossy(), &root);
        assert_eq!(recorded, "src/main.rs", "a path was recorded in the local dialect");
        assert!(!recorded.contains('\\'), "a separator survived into a recorded path");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_event_with_no_knowable_project_is_not_captured() {
        // Antigravity without an explicit workspace: no cwd, empty list, and a
        // process cwd inside its own config directory.
        let payload = json!({"conversationId": "abc", "workspacePaths": []});
        let home = dirs::home_dir().unwrap();
        assert!(is_cli_config_dir(&home.join(".gemini/config")));
        assert!(is_cli_config_dir(&home.join(".cursor")));
        assert!(!is_cli_config_dir(&home.join("Project/thing")));
        // With a real workspace it becomes knowable again.
        assert!(working_directory(&json!({"workspacePaths": ["/repo"]})).is_some());
        let _ = payload;
    }

    #[test]
    fn session_ids_are_read_from_every_clis_spelling() {
        for key in ["session_id", "sessionId", "thread_id", "conversationId"] {
            let payload = json!({ key: "0199a1f2-3c4d-7e8f-9012-3456789abcde" });
            assert_eq!(
                first_string(&payload, &["session_id", "sessionId", "thread_id", "conversationId"]),
                Some("0199a1f2-3c4d-7e8f-9012-3456789abcde"),
                "missed {key}"
            );
        }
    }

    #[test]
    fn hook_names_normalize_across_clis() {
        assert_eq!(normalize_hook("PostToolUse"), "post_tool_use");
        assert_eq!(normalize_hook("post-tool-use"), "post_tool_use");
        assert_eq!(normalize_hook("post_tool_use"), "post_tool_use");
        assert_eq!(normalize_hook("UserPromptSubmit"), "user_prompt_submit");
        assert_eq!(normalize_hook("SessionStart"), "session_start");
        // OpenCode spells its events with dots.
        assert_eq!(normalize_hook("tool.execute.after"), "tool_execute_after");
        assert_eq!(normalize_hook("session.created"), "session_created");
    }

    /// OpenCode 2 renamed its tools (`bash` is `shell`, `task` is `subagent`)
    /// and keys a file argument `path`. Nothing here had to change for that -
    /// `shell` and `path` were already spellings we read - and this pins it,
    /// so a later tightening of either match cannot drop OpenCode silently.
    #[test]
    fn opencode_2_tool_calls_title_like_everyone_elses() {
        let shell = json!({"tool_name": "shell", "tool_input": {"command": "cargo test"}});
        assert_eq!(title_for("post_tool_use", &shell), "cargo test");

        let edit = json!({"tool_name": "edit", "tool_input": {"path": "/repo/src/main.rs"}});
        assert_eq!(title_for("post_tool_use", &edit), "edit: /repo/src/main.rs");
    }

    #[test]
    fn titles_describe_edits_and_commands() {
        let edit = json!({"tool_name": "Edit", "tool_input": {"file_path": "/repo/src/main.rs"}});
        assert_eq!(title_for("post_tool_use", &edit), "Edit: /repo/src/main.rs");

        let bash = json!({"tool_name": "Bash", "tool_input": {"command": "cargo test\n--all"}});
        assert_eq!(title_for("post_tool_use", &bash), "cargo test");

        let prompt = json!({"prompt": "  \n why is auth failing? \n more"});
        assert_eq!(title_for("user_prompt_submit", &prompt), "Asked: why is auth failing?");

        let start = json!({"source": "resume"});
        assert_eq!(title_for("session_start", &start), "Session started (resume)");
    }

    #[test]
    fn commands_are_described_by_tool_and_target() {
        // Each of these is the shape of a real captured command, with paths
        // and package names replaced by neutral equivalents.
        for (command, expected) in [
            (
                "sed -n '1,140p' /home/u/proj/apps/admin/src/pages/SettingsPage.tsx",
                "read: SettingsPage.tsx",
            ),
            (
                "rtk grep -n \"scheduleMode|confirm:\" /home/u/proj/apps/admin/src/lib/x.ts",
                "search: x.ts",
            ),
            (
                "awk '/notFound/{print NR\": \"$0}' apps/api/src/index.ts 2>/dev/null | head -20",
                "read: index.ts",
            ),
            ("pnpm --filter @scope/api typecheck 2>&1 | tail -100", "pnpm typecheck"),
            ("cargo build --release", "cargo build"),
            ("git commit -q -m \"a message with spaces\"", "git commit"),
        ] {
            assert_eq!(describe_command(command), expected, "for: {command}");
        }
    }

    #[test]
    fn a_quoted_pattern_is_never_mistaken_for_a_filename() {
        // The pattern contains dots and slashes; the real target is the file.
        let described = describe_command("grep -n 'foo.bar/baz.ts' src/main.rs");
        assert_eq!(described, "search: main.rs");
    }

    #[test]
    fn an_unrecognizable_command_degrades_to_the_program_name() {
        assert_eq!(describe_command("./scripts/weird-thing --go"), "Ran weird-thing");
        assert_eq!(describe_command(""), "Ran a command");
        assert_eq!(describe_command("   "), "Ran a command");
    }

    #[test]
    fn describing_a_command_never_panics_on_anything() {
        for command in [
            "'", "\"unterminated", "| | |", "sudo", "rtk", "cd /tmp && cat > f <<'EOF'",
            "cargo", "--only-flags", "/", "a/b/", "x=1 y=2",
        ] {
            let described = describe_command(command);
            assert!(!described.is_empty(), "empty title for: {command:?}");
        }
    }

    #[test]
    fn titles_never_panic_on_a_payload_missing_everything() {
        let empty = json!({});
        for hook in ["user_prompt_submit", "post_tool_use", "session_start", "stop", "weird"] {
            assert!(!title_for(hook, &empty).is_empty());
        }
    }

    #[test]
    fn files_are_repo_relative() {
        let payload = json!({"tool_input": {"file_path": "/repo/src/main.rs"}});
        let files = files_for(&payload, std::path::Path::new("/repo"));
        assert_eq!(files, vec!["src/main.rs".to_string()]);
    }

    #[test]
    fn relativizing_survives_a_symlinked_root() {
        // The canonical form of the temp dir, versus the path as handed to us.
        let raw = std::env::temp_dir().join("proj/src/main.rs");
        let root = std::env::temp_dir().canonicalize().unwrap().join("proj");
        let payload = json!({"tool_input": {"file_path": raw}});
        assert_eq!(files_for(&payload, &root), vec!["src/main.rs".to_string()]);
    }

    #[test]
    fn files_outside_the_repo_are_kept_absolute() {
        let payload = json!({"tool_input": {"file_path": "/etc/hosts"}});
        let files = files_for(&payload, std::path::Path::new("/repo"));
        assert_eq!(files, vec!["/etc/hosts".to_string()]);
    }

    #[test]
    fn body_keeps_content_and_drops_noise() {
        let payload = json!({
            "prompt": "hello",
            "transcript_path": "/tmp/very/long/path.jsonl",
            "session_id": "abc",
        });
        let body = body_for(&payload, &Sanitizer::builtin(), false);
        assert!(body.contains("hello"));
        assert!(!body.contains("transcript_path"));
    }

    fn shaped(payload: &Value, footstep: bool) -> String {
        body_for(payload, &Sanitizer::builtin(), footstep)
    }

    #[test]
    fn a_brain_recall_result_is_not_stored_again_in_any_spelling() {
        for tool in [
            "mcp__plugin_rolepod-brain_brain__brain_search",
            "mcp__brain__brain_get",
            "MCP:brain_recent",
            "brain_brain_timeline",
            "brain_related",
            "brain_outline",
        ] {
            let payload = json!({
                "tool_name": tool,
                "tool_input": {"id": "obs-4242"},
                "tool_response": {"content": "a recalled row about pineapple"},
            });
            let body = shaped(&payload, false);
            assert!(!body.contains("pineapple"), "{tool}: {body}");
            assert!(body.contains("obs-4242"), "{tool}: {body}");
            assert!(body.contains("[brain recall result omitted]"), "{tool}: {body}");
        }
    }

    #[test]
    fn brain_writers_and_other_tools_keep_their_response() {
        for tool in [
            "mcp__plugin_rolepod-brain_brain__brain_note",
            "brain_correct",
            "brain_forget",
            "brain_feedback",
            "brain_doctor",
            "mcp__other__search",
            "mcp__second-brain__search",
            "mcp__my_brain__get",
            "mcp__gbrain__get",
            "gbrain_get",
            "mcp__second-brain_search",
            "Read",
        ] {
            let payload = json!({
                "tool_name": tool,
                "tool_input": {"id": "x"},
                "tool_response": {"content": "kept pineapple"},
            });
            assert!(shaped(&payload, false).contains("kept pineapple"), "{tool}");
        }
        let payload = json!({
            "tool_name": "mcp__second-brain__search",
            "tool_input": {"id": "x"},
            "tool_response": {"content": "kept pineapple"},
        });
        assert!(shaped(&payload, false).contains(r#""content":"kept pineapple""#));
    }

    #[test]
    fn a_brain_recall_footstep_and_title_keep_the_input() {
        let payload = json!({
            "tool_name": "mcp__brain__brain_search",
            "tool_input": {"query": "q-needle"},
            "tool_response": {"content": "a recalled row about pineapple"},
        });
        let body = shaped(&payload, true);
        assert!(!body.contains("pineapple"), "{body}");
        assert!(body.contains("[brain recall result omitted]"), "{body}");
    }

    #[test]
    fn an_edit_body_drops_the_file_copies_and_keeps_the_input() {
        let payload = json!({
            "tool_name": "Edit",
            "tool_input": {"file_path": "a.rs", "old_string": "x", "new_string": "y"},
            "tool_response": {"originalFile": "o".repeat(50_000), "structuredPatch": [{"lines": ["+y"]}], "ok": true},
        });
        let body = shaped(&payload, false);
        assert!(!body.contains("originalFile") && !body.contains("structuredPatch"));
        assert!(body.contains("tool_input") && body.contains("new_string"));
        assert!(body.len() < 1000);
    }

    #[test]
    fn a_long_stdout_keeps_its_tail_within_the_leaf_cap() {
        let stdout = format!("{}\nERROR: the-tail-marker", "line of output\n".repeat(700));
        let body = shaped(&json!({"tool_response": {"stdout": stdout}}), false);
        let value: Value = serde_json::from_str(&body).unwrap();
        let leaf = value["tool_response"]["stdout"].as_str().unwrap();
        assert!(leaf.len() <= 2150, "{}", leaf.len());
        assert!(leaf.contains("the-tail-marker"));
    }

    #[test]
    fn a_base64_run_is_replaced_by_its_size() {
        let blob = "QUJD".repeat(1280);
        let body = shaped(&json!({"tool_response": {"image": blob}}), false);
        assert!(body.contains("[base64 5120 bytes]"), "{body}");
        assert!(!body.contains("QUJDQUJD"));
    }

    #[test]
    fn a_delegates_footstep_body_stays_small() {
        let payload = json!({
            "tool_name": "Read",
            "tool_input": {"file_path": "a.rs", "content": "i".repeat(4000)},
            "tool_response": {"stdout": format!("{}END", "r".repeat(9000)), "other": "q".repeat(3000)},
        });
        let sanitizer = Sanitizer::builtin();
        assert!(is_footstep(Some("Explore"), "post_tool_use"));
        let body = event_body(&payload, &sanitizer, true);
        assert!(body.len() <= 1126, "{}", body.len());
        // The lead's own call, and a delegate's non-tool event, are not clamped
        // to a footstep.
        assert!(!is_footstep(None, "post_tool_use"));
        assert!(!is_footstep(Some("Explore"), "user_prompt_submit"));
        let lead = event_body(&payload, &sanitizer, false);
        assert!(lead.len() > 2000, "{}", lead.len());
    }

    #[test]
    fn an_unclosed_private_tag_drops_every_later_leaf() {
        let sanitizer = Sanitizer::builtin();
        // In an earlier leaf of the same object, and in an earlier key.
        let same_object = json!({"tool_response": {"stderr": "<private>client", "stdout": "later-visible"}});
        let body = event_body(&same_object, &sanitizer, false);
        assert!(!body.contains("later-visible") && !body.contains("client"), "{body}");
        let across_keys = json!({
            "tool_input": {"content": "x <private>client"},
            "tool_response": {"stdout": "later-visible"},
            "message": "also-later",
        });
        for footstep in [false, true] {
            let body = event_body(&across_keys, &sanitizer, footstep);
            assert!(!body.contains("later-visible") && !body.contains("also-later"), "{body}");
            assert!(!body.contains("client"), "{body}");
        }
        // A balanced region drops only itself.
        let balanced = json!({"tool_response": {"stderr": "<private>c</private>", "stdout": "kept"}});
        assert!(event_body(&balanced, &sanitizer, false).contains("kept"));
    }

    #[test]
    fn a_secret_across_a_leaf_cap_is_redacted_whole() {
        let secret = "ghp_abcdefghijklmnopqrstuvwxyz0123";
        let sanitizer = Sanitizer::builtin();
        for (cap, footstep) in [(LEAF_MAX_BYTES, false), (FOOTSTEP_LEAF_MAX_BYTES, true)] {
            // Place the secret so the cap's head/tail cut would land inside it.
            for offset in [cap / 2 - 40, cap / 2 - 24, cap / 2 - 10] {
                let text = format!("{}{secret}{}", "a ".repeat(offset / 2), "b ".repeat(cap));
                let body = body_for(&json!({"tool_response": {"stdout": text}}), &sanitizer, footstep);
                assert!(!body.contains("ghp_"), "{body}");
                assert!(!body.contains("wxyz0123"), "{body}");
            }
            // And across the tail cut: the secret near the end of the leaf, so
            // the tail window begins inside it.
            let half = (cap - 48) / 2;
            for suffix_len in (half - secret.len() + 1)..half {
                let text = format!("{}{secret}{}", "a ".repeat(cap), "b".repeat(suffix_len));
                let body = body_for(&json!({"tool_response": {"stdout": text}}), &sanitizer, footstep);
                assert!(!body.contains("ghp_") && !body.contains("wxyz0123"), "{suffix_len}: {body}");
            }
        }
    }

    #[test]
    fn a_small_payload_body_is_unchanged() {
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": {"stdout": "a\nb", "stderr": ""},
        });
        let expected = serde_json::to_string(&payload).unwrap();
        assert_eq!(shaped(&payload, false), expected);
    }

    #[test]
    fn an_idle_sweep_spawns_once_per_window() {
        let store = Store::open_memory().unwrap();
        let now = jiff::Timestamp::now();
        assert!(claim_idle_sweep_window(&store, now), "window open: spawn");
        assert!(!claim_idle_sweep_window(&store, now), "window spent: no spawn");
        let inside = now + jiff::SignedDuration::from_secs(IDLE_SWEEP_DEBOUNCE_SECS - 60);
        assert!(!claim_idle_sweep_window(&store, inside), "still inside the window");
        let later = now + jiff::SignedDuration::from_secs(IDLE_SWEEP_DEBOUNCE_SECS + 60);
        assert!(claim_idle_sweep_window(&store, later), "past the window it asks again");
    }

    /// A one-shot run never gets a summary, so none of its hooks may start a
    /// consolidation, and none may pay to ask: the backlog question is a read
    /// on the hook path and the idle window is a write.
    #[test]
    fn a_one_shot_run_summons_nothing_and_reads_nothing() {
        let stale_asked = std::cell::Cell::new(false);
        let window_asked = std::cell::Cell::new(false);
        for cli in ["claude-code", "codex", "cursor", "opencode"] {
            for hook in ["session_start", "session_end", "pre_compact", "stop"] {
                let summoned = summons(
                    cli,
                    hook,
                    true,
                    || {
                        stale_asked.set(true);
                        true
                    },
                    || {
                        window_asked.set(true);
                        true
                    },
                );
                assert_eq!(summoned, Summons::default(), "{cli}/{hook} summoned a run");
            }
        }
        assert!(!stale_asked.get(), "a one-shot run read the backlog");
        assert!(!window_asked.get(), "a one-shot run spent the idle window");

        // A person's session keeps every rule it had.
        let session = Summons { run: Some(Run::Session), idle: false };
        assert_eq!(summons("claude-code", "session_end", false, || true, || true), session);
        assert_eq!(
            summons("claude-code", "session_start", false, || true, || true),
            Summons { run: Some(Run::All), idle: false }
        );
        assert_eq!(summons("claude-code", "session_start", false, || false, || true), Summons::default());
        assert_eq!(summons("cursor", "stop", false, || true, || false), session);
        assert_eq!(
            summons("cursor", "stop", false, || true, || true),
            Summons { run: Some(Run::Session), idle: true }
        );
    }

    #[test]
    fn the_stale_backlog_check_still_counts_rule_based_sessions() {
        let store = Store::open_memory().unwrap();
        let project = uuid::Uuid::new_v4();
        let session = uuid::Uuid::new_v4();
        let mut event = Event::new(
            uuid::Uuid::nil(),
            project,
            session,
            Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
            EventKind::Observation,
            "Edit: a.rs".into(),
            "{}".into(),
        );
        event.ts = (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(3 * 3600)).to_string();
        store.index(&event).unwrap();
        store.record_session_run(&session.to_string(), "p", "01A", "rule-based").unwrap();
        assert!(store.has_stale_backlog(STALE_BACKLOG_SECS).unwrap());
    }

    /// Run one claude-code hook against a fresh store; return how many events landed.
    fn events_after(hook: &str, cursor_version: Option<&str>) -> i64 {
        let _guard =
            invocation::ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let home = std::env::temp_dir().join(format!("brain-cursor-echo-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&home).unwrap();
        std::env::set_var(crate::config::DATA_DIR_ENV, &home);
        let mut payload = json!({
            "session_id": "echo-session",
            "cwd": home.display().to_string(),
            "tool_name": "Read",
            "tool_input": {"file_path": "a.rs"},
        });
        if let Some(version) = cursor_version {
            payload["cursor_version"] = json!(version);
        }
        let ack = capture("claude-code", hook, Some(payload.to_string())).unwrap();
        assert_eq!(ack, "{}");
        let count = Store::open(&home.join("brain.db")).unwrap().count().unwrap();
        std::env::remove_var(crate::config::DATA_DIR_ENV);
        let _ = std::fs::remove_dir_all(&home);
        count
    }

    /// Run `f` against a fresh store under the env lock; the store is open
    /// while `f` runs and the home is gone after.
    fn with_store<T>(f: impl FnOnce(&std::path::Path, &Store) -> T) -> T {
        let _guard =
            invocation::ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let home = std::env::temp_dir().join(format!("brain-failed-call-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&home).unwrap();
        // The embed tests find their staged model through this variable, and
        // they run beside us: lend our home the same models, and put back
        // whatever the variable held.
        crate::embed::tests::use_checkout_model();
        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(
            std::env::temp_dir().join("rolepod-brain-test-home").join("models"),
            home.join("models"),
        );
        let before = std::env::var_os(crate::config::DATA_DIR_ENV);
        std::env::set_var(crate::config::DATA_DIR_ENV, &home);
        let out = f(&home, &Store::open(&home.join("brain.db")).unwrap());
        match before {
            Some(value) => std::env::set_var(crate::config::DATA_DIR_ENV, value),
            None => std::env::remove_var(crate::config::DATA_DIR_ENV),
        }
        let _ = std::fs::remove_dir_all(&home);
        out
    }

    fn tool_payload(home: &std::path::Path, session: &str, tool: &str, extra: Value) -> Value {
        let mut payload = json!({
            "session_id": session,
            "cwd": home.display().to_string(),
            "tool_name": tool,
            "tool_input": {"file_path": "a.rs", "command": "cargo test"},
        });
        for (key, value) in extra.as_object().unwrap() {
            payload[key] = value.clone();
        }
        payload
    }

    fn only_event(store: &Store, session: &str) -> Event {
        store.session_events(&ids::session_uuid(session).to_string()).unwrap().remove(0)
    }

    #[test]
    fn a_failed_file_call_injects_nothing_where_the_same_success_does() {
        with_store(|home, store| {
            let run = |event: &str, session: &str| {
                let payload = tool_payload(home, session, "Edit", json!({"error": "no such file"}));
                capture("claude-code", event, Some(payload.to_string())).unwrap()
            };
            // A prior event on the file gives it memory to inject. It is an
            // Edit: a bare Read is no longer offered as a file hint.
            run("PostToolUse", "seed");
            let control = run("PostToolUse", "control");
            assert!(control.contains("Memory for"), "control must inject: {control}");
            let ack = run("PostToolUseFailure", "failing");
            assert_eq!(ack, "{}", "a failure must inject nothing");
            let session = ids::session_uuid("failing").to_string();
            assert!(!store.file_already_injected(&session, "a.rs").unwrap());
            assert_eq!(store.session_injected_bytes(&session).unwrap(), 0);
        });
    }

    #[test]
    fn every_failure_shape_lands_in_the_body_with_a_failed_title_and_no_files() {
        let shapes = [
            json!({"tool_response": "THE_ERROR_TEXT"}),
            json!({"tool_output": {"error": "THE_ERROR_TEXT"}}),
            json!({"error": "THE_ERROR_TEXT", "is_interrupt": true}),
        ];
        for (n, shape) in shapes.into_iter().enumerate() {
            with_store(|home, store| {
                let payload = tool_payload(home, "s", "Edit", shape.clone());
                capture("claude-code", "PostToolUseFailure", Some(payload.to_string())).unwrap();
                let event = only_event(store, "s");
                assert_eq!(event.source.hook, "post_tool_use");
                assert!(event.title.starts_with("Failed: "), "{n}: {}", event.title);
                assert!(event.files.is_empty(), "{n}: {:?}", event.files);
                let body: Value = serde_json::from_str(&event.body).unwrap();
                assert_eq!(body["failed"], json!(true), "{n}");
                assert!(event.body.contains("THE_ERROR_TEXT"), "{n}: {}", event.body);
            });
        }
    }

    #[test]
    fn a_normal_call_never_carries_failure_fields_even_if_the_host_sends_them() {
        with_store(|home, store| {
            let extra = json!({
                "failed": true, "error": "HOST_ERROR", "tool_output": "HOST_OUT",
                "is_interrupt": true, "tool_response": "fine",
            });
            let payload = tool_payload(home, "s", "Bash", extra);
            capture("claude-code", "PostToolUse", Some(payload.to_string())).unwrap();
            let event = only_event(store, "s");
            assert!(!event.title.starts_with("Failed"), "{}", event.title);
            assert!(!event.files.is_empty());
            let body: Value = serde_json::from_str(&event.body).unwrap();
            for key in ["failed", "error", "tool_output", "is_interrupt"] {
                assert!(body.get(key).is_none(), "{key} kept on a success: {body}");
            }
            assert_eq!(body["tool_response"], json!("fine"));
        });
    }

    #[test]
    fn a_failures_error_is_redacted_and_clamped_like_any_result() {
        with_store(|home, store| {
            let error = format!("ghp_abcdefghijklmnopqrstuvwxyz0123 {}END", "x".repeat(20_000));
            let payload = tool_payload(home, "s", "Bash", json!({"error": error}));
            capture("claude-code", "PostToolUseFailure", Some(payload.to_string())).unwrap();
            let event = only_event(store, "s");
            assert!(!event.body.contains("ghp_"), "secret kept: {}", event.body);
            assert!(event.body.len() < 5_000, "not clamped: {} bytes", event.body.len());
        });
    }

    #[test]
    fn a_failure_echoed_by_cursor_is_dropped_like_its_other_hooks() {
        assert_eq!(events_after("PostToolUseFailure", Some("1.7.0")), 0);
        assert_eq!(events_after("PostToolUseFailure", None), 1);
    }

    #[test]
    fn cursor_echo_through_claude_code_is_not_captured_twice() {
        for hook in ["PostToolUse", "UserPromptSubmit", "Stop"] {
            assert_eq!(events_after(hook, Some("1.7.0")), 0, "{hook} echo must be dropped");
        }
        // Not a Cursor payload: claude-code capture is unchanged.
        assert_eq!(events_after("PostToolUse", None), 1);
        assert_eq!(events_after("PostToolUse", Some("")), 1);
        // A delegate's report is not echoed by Cursor's own hooks.
        assert_eq!(events_after("SubagentStop", Some("1.7.0")), 1);
    }

    const KNOWN: &str = "Retry backoff jitter is capped at thirty seconds";
    const TASK: &str = "how does retry backoff jitter get capped";

    fn scope() -> crate::ids::ProjectScope {
        crate::ids::ProjectScope {
            workspace: "default".into(),
            workspace_id: uuid::Uuid::nil(),
            project: "p".into(),
            project_id: uuid::Uuid::from_u128(7),
            root: std::path::PathBuf::from("/repo"),
        }
    }

    fn event_of(kind: EventKind, title: &str, hook: &str) -> Event {
        Event::new(
            uuid::Uuid::nil(),
            uuid::Uuid::from_u128(7),
            uuid::Uuid::nil(),
            Source { cli: "claude-code".into(), hook: hook.into() },
            kind,
            title.into(),
            String::new(),
        )
    }

    /// A store holding one durable knowledge entry; returns it with the entry's id.
    fn store_with_knowledge() -> (Store, String) {
        let store = Store::open_memory().unwrap();
        let mut entry = event_of(EventKind::Knowledge, KNOWN, "consolidate");
        entry.consolidated = true;
        store.index(&entry).unwrap();
        (store, entry.id)
    }

    fn prompt_hook(store: &Store, config: &Config, prompt: &str) -> String {
        let paths = Paths { data_dir: std::env::temp_dir().join("rolepod-brain-hook-unit") };
        inject_for_prompt(store, config, &scope(), "s1", "UserPromptSubmit", prompt, &Spill { paths: &paths, session: "s1", skip: false, fresh: false })
    }

    fn on() -> Config {
        let mut config = Config::default();
        config.injection.prompt_pointers = true;
        config
    }

    fn context_of(output: &str) -> String {
        let value: Value = serde_json::from_str(output).unwrap();
        value["hookSpecificOutput"]["additionalContext"].as_str().unwrap_or("").to_string()
    }

    #[test]
    fn with_the_flag_off_a_prompt_gets_nothing() {
        let (store, _) = store_with_knowledge();
        assert_eq!(prompt_hook(&store, &Config::default(), TASK), "{}");
    }

    #[test]
    fn an_acknowledgement_or_a_command_is_not_a_task() {
        // Every gated prompt is one that WOULD match a seeded entry were it
        // looked up, so the test goes red with the gate removed.
        let (store, _) = store_with_knowledge();
        let acks = ["ok", "okay", "thanks", "yes", "no", "continue", "โอเค", "ครับ", "ค่ะ", "ต่อ", "ขอบคุณ", "ได้"];
        let mut chatter = event_of(EventKind::Knowledge, &acks.join(" "), "consolidate");
        chatter.consolidated = true;
        store.index(&chatter).unwrap();
        let mut gated: Vec<String> = acks.iter().map(|ack| (*ack).to_string()).collect();
        gated.extend(["  OK \n".into(), "Thanks".into(), String::new(), "   ".into()]);
        gated.extend(
            ["/slash", "<task-notification>", "<scheduled-task", "<command-name>"]
                .iter()
                .map(|lead| format!("{lead} {TASK}")),
        );
        for prompt in &gated {
            assert_eq!(prompt_hook(&store, &on(), prompt), "{}", "{prompt:?}");
            if !prompt.trim().is_empty() {
                // Prove the prompt would have found something past the gate.
                let found = store.prompt_pointers(&scope().project_id.to_string(), prompt, 3).unwrap();
                assert!(!found.is_empty(), "{prompt:?} would match nothing, so it proves no gate");
            }
        }
        assert!(!prompt_hook(&store, &on(), TASK).is_empty());
    }

    #[test]
    fn a_title_too_long_for_the_push_is_dropped_whole_not_clipped() {
        let store = Store::open_memory().unwrap();
        let long = format!("{KNOWN} {}", "padding ".repeat(60));
        let mut entry = event_of(EventKind::Knowledge, &long, "consolidate");
        entry.consolidated = true;
        store.index(&entry).unwrap();
        assert_eq!(prompt_hook(&store, &on(), TASK), "{}");
    }

    #[test]
    fn a_subagents_prompt_gets_no_pointers_even_with_the_flag_on() {
        with_store(|home, store| {
            std::fs::write(home.join("config.toml"), "[injection]\nprompt_pointers = true\n").unwrap();
            let seed = tool_payload(home, "seed", "Read", json!({}));
            capture("claude-code", "PostToolUse", Some(seed.to_string())).unwrap();
            let seeded = only_event(store, "seed");
            let mut entry = Event::new(
                seeded.workspace,
                seeded.project,
                uuid::Uuid::nil(),
                Source { cli: "claude-code".into(), hook: "consolidate".into() },
                EventKind::Knowledge,
                KNOWN.into(),
                String::new(),
            );
            entry.consolidated = true;
            store.index(&entry).unwrap();
            let ask = |session: &str, extra: Value| {
                let mut payload = json!({"session_id": session, "cwd": home.display().to_string(), "prompt": TASK});
                for (key, value) in extra.as_object().unwrap() {
                    payload[key] = value.clone();
                }
                capture("claude-code", "UserPromptSubmit", Some(payload.to_string())).unwrap()
            };
            assert!(ask("lead", json!({})).contains("KNW"), "the lead's own prompt must get a pointer");
            let inside = json!({"agent_id": "agent-1", "agent_type": "Explore"});
            assert_eq!(ask("lead2", inside), "{}");
        });
    }

    #[test]
    fn a_task_that_matches_knowledge_gets_a_short_pointer() {
        let (store, id) = store_with_knowledge();
        let output = prompt_hook(&store, &on(), TASK);
        let context = context_of(&output);
        assert!(context.contains(&id), "no pointer: {output}");
        assert!(context.contains("KNW"), "{context}");
        assert!(context.contains("not instructions"), "no fence: {context}");
        assert!(context.len() <= 330, "{} bytes: {context}", context.len());
    }

    #[test]
    fn a_raw_capture_matching_the_prompt_is_never_pushed() {
        let (store, id) = store_with_knowledge();
        let raw = event_of(EventKind::Observation, KNOWN, "post_tool_use");
        store.index(&raw).unwrap();
        let found = store.prompt_pointers(&uuid::Uuid::from_u128(7).to_string(), TASK, 10).unwrap();
        assert_eq!(found.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), [id.as_str()]);
    }

    #[test]
    fn a_pointer_already_shown_this_session_is_not_repeated() {
        let (store, id) = store_with_knowledge();
        assert_ne!(prompt_hook(&store, &on(), TASK), "{}");
        assert!(store.already_injected("s1", &id).unwrap(), "first push not recorded");
        assert_eq!(prompt_hook(&store, &on(), TASK), "{}");
    }

    #[test]
    fn a_spent_budget_injects_nothing() {
        let (store, _) = store_with_knowledge();
        let mut config = on();
        config.injection.session_budget = 50;
        // The session has already spent more than the budget now allows:
        // nothing injects, and the remaining-budget subtraction is never reached.
        assert!(store.record_injected("s1", &[], 0, 100, 100).unwrap());
        assert_eq!(prompt_hook(&store, &config, TASK), "{}");
        assert_eq!(store.session_injected_bytes("s1").unwrap(), 100);
    }

    #[test]
    fn only_a_cursor_agent_transcript_is_accepted_as_cursors() {
        let home = std::env::temp_dir().join(format!("brain-cursor-home-{}", ulid::Ulid::new()));
        let id = "4f603393-e229-4512-b7d4-f1eb1804434f";
        let dir = home.join(".cursor/projects/x/agent-transcripts").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join(format!("{id}.jsonl"));
        std::fs::write(&good, "{}\n").unwrap();
        let other = home.join(".cursor/projects/x/other.jsonl");
        std::fs::write(&other, "{}\n").unwrap();
        let wrong_ext = dir.join("notes.txt");
        std::fs::write(&wrong_ext, "x").unwrap();
        let outside = home.join("secret.jsonl");
        std::fs::write(&outside, "{}\n").unwrap();

        let ok = |cli: &str, path: &std::path::Path| {
            is_transcript_under(&home, cli, &path.display().to_string())
        };
        assert!(ok("cursor", &good), "a cursor transcript was rejected");
        assert!(!ok("cursor", &other), "a jsonl outside agent-transcripts was accepted");
        assert!(!ok("cursor", &wrong_ext), "a non-jsonl file was accepted");
        let stray = home.join(".cursor/projects/x/agent-transcripts/other.jsonl");
        std::fs::write(&stray, "{}\n").unwrap();
        assert!(!ok("cursor", &stray), "a jsonl not named for its folder was accepted");
        let shallow = home.join(".cursor/projects/agent-transcripts");
        std::fs::create_dir_all(&shallow).unwrap();
        let shallow = shallow.join("x.jsonl");
        std::fs::write(&shallow, "{}\n").unwrap();
        assert!(!ok("cursor", &shallow), "agent-transcripts directly under projects was accepted");
        let deep = dir.join("sub");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join(format!("{id}.jsonl")), "{}\n").unwrap();
        assert!(!ok("cursor", &deep.join(format!("{id}.jsonl"))), "deeper nesting was accepted");
        assert!(!ok("cursor", &outside), "a path outside .cursor was accepted");
        let traversal = home.join(".cursor/projects/x/agent-transcripts").join(id).join("../../../../../secret.jsonl");
        assert!(!ok("cursor", &traversal), "a ../ escape was accepted");
        assert!(!ok("cursor", std::path::Path::new("/nonexistent/a.jsonl")));
        // Another CLI's directory is not cursor's, and cursor's is not theirs.
        assert!(!ok("claude-code", &good));
        assert!(!ok("gemini-cli", &good));

        // A symlink inside the tree that points out of it resolves out of it.
        #[cfg(unix)]
        {
            let link = dir.join("link.jsonl");
            std::os::unix::fs::symlink(&outside, &link).unwrap();
            assert!(!ok("cursor", &link), "a symlink escape was accepted");
        }
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn claude_and_codex_roots_are_unchanged() {
        let home = std::env::temp_dir().join(format!("brain-roots-home-{}", ulid::Ulid::new()));
        for (cli, rel) in [("claude-code", ".claude/projects/p/s.jsonl"), ("codex", ".codex/sessions/s.jsonl")] {
            let path = home.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "{}\n").unwrap();
            assert!(is_transcript_under(&home, cli, &path.display().to_string()), "{cli}");
        }
        std::fs::remove_dir_all(&home).ok();
    }
}
