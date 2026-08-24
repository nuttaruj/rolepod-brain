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
use crate::sanitize::{truncate, Sanitizer};
use crate::store::Store;

/// Ceiling for a generated title, before it reaches the primer.
const TITLE_MAX_BYTES: usize = 120;

/// How old unconsolidated work must be before a session opening picks it up.
///
/// Comfortably longer than the consolidation debounce, so an active machine
/// never triggers this path and a returning one always does.
const STALE_BACKLOG_SECS: i64 = 15 * 60;

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

/// Read a payload from stdin and capture it.
///
/// Returns the JSON the host CLI should see on stdout.
///
/// # Errors
/// Returns an error only for conditions the caller should log; the caller is
/// responsible for still exiting 0.
pub fn capture(cli: &str, event_name: &str, stdin_payload: Option<String>) -> Result<String> {
    // Our own subprocess: acknowledge the host and capture nothing.
    if is_worker_child() {
        return Ok("{}".to_string());
    }

    // An orchestrator has asked for a clean room: no capture, no injection,
    // no trace. This is a documented public contract, not an internal switch.
    if invocation::silenced() {
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

    let payload: Value = serde_json::from_str(&raw).context("parse hook payload")?;

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

    let agent = AgentKind::parse(cli);
    let hook = normalize_hook(event_name);
    let session = ids::session_uuid(
        first_string(&payload, &["session_id", "sessionId", "thread_id", "conversationId"])
            .unwrap_or("unknown-session"),
    );

    let title = truncate(&sanitizer.scrub(&title_for(&hook, &payload)), TITLE_MAX_BYTES);
    let body = sanitizer.scrub_body(&body_for(&payload));

    // Classified once per session and remembered: working it out costs a
    // process spawn, and the hook budget does not allow one per event.
    let store = Store::open(&paths.db())?;
    let session_key = session.to_string();
    let invocation = match store.session_invocation(&session_key)? {
        Some(cached) => invocation::parse(&cached),
        None => {
            let classified = invocation::classify();
            store.record_session_invocation(&session_key, classified.as_str())?;
            classified
        }
    };

    // A pointer to material consolidation will read and never copy. Claude
    // Code and Codex put it in every payload; the other three CLIs write no
    // transcript at all.
    if let Some(path) = first_string(&payload, &["transcript_path", "transcriptPath"]) {
        if is_transcript_of(agent.as_str(), path) {
            let _ = store.record_transcript_path(&session_key, path);
        }
    }

    let mut event = Event::new(
        scope.workspace_id,
        scope.project_id,
        session,
        Source { cli: agent.as_str().to_string(), hook: hook.clone() },
        EventKind::Observation,
        title,
        body,
    );
    // The same scrub the title and body get. A path is the one field the
    // sanitizer explicitly treats as sensitive by convention - .ssh, .aws,
    // .gnupg - and storing it unscrubbed in a parallel array meant the thing
    // being redacted out of the title sat intact in the column beside it.
    event.files = files_for(&payload, &scope.root)
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

    let project_dir = paths.project_dir(&scope);
    let log = EventLog::open(&project_dir)?;
    // Log first: it is the source of truth. If indexing then fails, the event
    // is still durable and `brain reindex` recovers the index.
    log.append(&event)?;

    store.index(&event)?;

    // A context wipe keeps the session id but destroys everything the agent
    // knew. Our own de-duplication is keyed to that surviving id, so without
    // this reset the guard that stops us repeating ourselves would instead
    // guarantee amnesia: memory injected before the wipe would be suppressed
    // exactly when it is needed most.
    if wipes_context(&hook, &payload) {
        store.reset_injection_state(&session_key)?;
    }

    if is_session_boundary(agent.as_str(), &hook) {
        // Compaction is the last moment this session's detail exists. Kicking
        // consolidation here means the primer that lands seconds later carries
        // a real narrative rather than a list of raw commands. Detached and
        // idempotent, so an extra run costs nothing.
        spawn_consolidation(Some(&session_key));
    } else if hook == "session_start" && store.has_stale_backlog(STALE_BACKLOG_SECS).unwrap_or(false)
    {
        // The backstop, without a background agent: a session opening is
        // exactly when older unconsolidated work becomes worth finishing,
        // because this session is the one that will read it.
        spawn_consolidation(None);
    }

    if invocation.is_headless() {
        // The whole point: a one-shot run is usually an orchestrated step -
        // a reviewer, a judge - and handing it this project's narrative
        // destroys its independence in a way nothing downstream can see.
        return Ok("{}".to_string());
    }

    Ok(inject_for(&store, &config, &scope, &session_key, &hook, event_name, &event))
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
    session: &str,
    hook: &str,
    event_name: &str,
    event: &Event,
) -> String {
    let project = scope.project_id.to_string();

    let injection = match hook {
        // Both entry points into a fresh context get the primer: a normal
        // session start, and the moment after a compaction.
        "session_start" | "post_compact" => {
            inject::primer(store, &project, session, &config.injection).ok()
        }
        "post_tool_use" => event.files.first().and_then(|path| {
            let injection =
                inject::for_file(store, &project, session, path, &event.id, &config.injection)
                    .ok()?;
            // Mark the file covered even when it had nothing, so a file with
            // no memory is not re-queried on every touch.
            let _ = store.record_injected_file(session, path);
            Some(injection)
        }),
        _ => None,
    };

    let Some(injection) = injection else { return "{}".to_string() };
    if injection.is_empty() {
        return "{}".to_string();
    }
    let _ = store.record_injected(session, &injection.ids, injection.text.len());
    inject::as_hook_output(event_name, &injection)
}

/// Did this event just destroy the agent's context?
///
/// Two paths, because Claude Code uses different ones: compaction fires
/// `PostCompact` (its docs: "After compaction (receives summary)"), while
/// `/clear` comes through as a `SessionStart` whose `source` says so. Handling
/// only one would leave the other silently amnesiac.
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
/// two are not the same thing: OpenCode's plugin subscribes to `session.idle`
/// and reports it to us as `stop`, so the event list would lie about it.
#[must_use]
pub fn consolidation_triggers(cli: &str) -> &'static str {
    match cli {
        // A real session end, plus compaction on the way out.
        "claude-code" | "codex" => "session end, compaction",
        // No session end; its plugin reports idle, and compaction, as ours.
        "opencode" => "session idle, compaction",
        // No session end and no compaction hook: end of turn is all they
        // offer.
        "antigravity" | "cursor" => "end of turn (no session-end event)",
        // Neither a session end nor a turn end reaches us. Its memory is
        // consolidated by the backstop when the next session opens.
        "gemini-cli" => "backstop only (no boundary event reaches us)",
        _ => "backstop only",
    }
}

/// Start consolidation in a detached child and return immediately.
///
/// The host CLI is waiting on this hook, so nothing here may block: the child
/// is fully detached from our stdio and outlives us. If it cannot start, the
/// catch-up backstop picks the work up later — this is best-effort by design.
fn spawn_consolidation(session: Option<&str>) {
    let Ok(exe) = std::env::current_exe() else { return };
    let args: Vec<&str> = match session {
        Some(session) => vec!["consolidate", "--session", session],
        // No session: catch up on whatever is stale, anywhere.
        None => vec!["consolidate", "--all"],
    };
    let _ = std::process::Command::new(exe)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
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
fn is_cli_config_dir(path: &std::path::Path) -> bool {
    let Some(home) = dirs::home_dir() else { return false };
    [".gemini", ".cursor", ".codex", ".claude", ".antigravity", ".config/opencode"]
        .iter()
        .any(|dir| path.starts_with(home.join(dir)))
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

/// The content worth keeping, as compact JSON.
///
/// Deliberately a subset: a hook payload carries a full transcript path, tool
/// schemas, and other noise that would bloat every event without ever being
/// recalled.
fn body_for(payload: &Value) -> String {
    let mut kept = serde_json::Map::new();
    for key in [
        "prompt",
        "tool_name",
        "tool_input",
        "toolCall",
        "terminationReason",
        "tool_response",
        "message",
        "reason",
        "source",
        "trigger",
        "custom_instructions",
    ] {
        if let Some(value) = payload.get(key) {
            kept.insert(key.to_string(), value.clone());
        }
    }
    if kept.is_empty() {
        return String::new();
    }
    serde_json::to_string(&Value::Object(kept)).unwrap_or_default()
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
    let roots: &[&str] = match cli {
        "claude-code" => &[".claude/projects"],
        "codex" => &[".codex/sessions"],
        // The other CLIs write no transcript at all, so any path claiming to
        // be one is by definition not theirs.
        _ => return false,
    };
    // Resolve before comparing: `~/.claude/projects/../../.ssh/id_rsa` is
    // inside the directory only until someone reads it.
    let Ok(resolved) = std::fs::canonicalize(path) else { return false };
    roots.iter().any(|root| {
        std::fs::canonicalize(home.join(root))
            .is_ok_and(|root| resolved.starts_with(&root))
    })
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
            return relative.to_string_lossy().into_owned();
        }
    }
    path.to_string()
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
    use super::*;
    use serde_json::json;

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
        // Compaction arrives as PostCompact, not as a SessionStart.
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
        assert!(consolidation_triggers("opencode").contains("idle"));
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
        let body = body_for(&payload);
        assert!(body.contains("hello"));
        assert!(!body.contains("transcript_path"));
    }
}
