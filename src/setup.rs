//! `brain setup` — wire the host CLIs once, then never again.
//!
//! Two safety rules, because this edits configuration the user's daily tools
//! depend on:
//!
//! 1. **Dry run by default.** Nothing is written without `--apply`. The plan
//!    printed by a dry run is exactly what `--apply` performs.
//! 2. **Only ever touch our own entries.** Existing hooks belong to other
//!    tools the user chose to install, and clobbering them would be a data
//!    loss they did not ask for. We match on our own marker, remove only
//!    that, and back the file up before writing.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::config::{Config, Paths};
use crate::ids::AgentKind;

/// Substring identifying a hook entry as ours. Present in every command we
/// write, and the only thing `setup` will remove.
const MARKER: &str = "brain hook";

/// Name the MCP server is registered under in both CLIs.
const MCP_SERVER_NAME: &str = "brain";

/// What our plugin is called in every marketplace that carries it.
///
/// The same string keys three different registries - a JSON map, a directory
/// name, and a TOML table - so it is written once here.
const PLUGIN_NAME: &str = "rolepod-brain";

/// How long a host CLI waits for our hook before moving on.
///
/// The unit is not universal: Claude Code and Codex read seconds, Gemini CLI
/// reads milliseconds (verified from a live `~/.gemini/settings.json`, where a
/// working hook is registered with `10000`). Writing `5` there would give us
/// five milliseconds and every capture would be killed.
const HOOK_TIMEOUT_SECS: u32 = 5;
const HOOK_TIMEOUT_MILLIS: u32 = 10_000;

/// launchd label for the consolidation backstop.
const TIMER_LABEL: &str = "dev.rolepod.brain.consolidate";

/// How often the backstop runs. Long on purpose: it is a safety net for work
/// the session-end trigger missed, not the primary path.
const TIMER_INTERVAL_SECS: u32 = 1800;

/// One planned change.
#[derive(Debug)]
pub struct Change {
    pub target: String,
    pub detail: String,
}

/// How a CLI stores its lifecycle hooks.
///
/// Four supported CLIs, four different shapes — there is no common format to
/// converge on, so the writer is explicit about which one it is producing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `{"hooks": {"<Event>": [{"hooks": [{type, command, timeout}]}]}}`.
    /// Claude Code, Codex, and Gemini CLI all use this.
    Grouped,
    /// `{"<namespace>": {"<Event>": ...}}`, where tool events take the grouped
    /// shape and the rest take a bare entry. Antigravity.
    Namespaced,
    /// Not a config file at all: a JavaScript module dropped into a plugin
    /// directory. OpenCode has no hook config; its plugins subscribe to
    /// lifecycle events in code.
    Plugin,
    /// `{"version": 1, "hooks": {"<event>": [{command, timeout}]}}` with no
    /// matcher and no wrapper around handlers. Cursor.
    Flat,
    /// We do not write this CLI's config at all - its own plugin flow does,
    /// and that flow is what grants the hooks permission to run. All `setup`
    /// does is remove any raw entries an earlier version wrote, and print the
    /// command the user should run.
    External,
}

/// A CLI we know how to wire.
pub struct Target {
    pub kind: AgentKind,
    /// Which config shape to write.
    pub layout: Layout,
    /// Events that take `{matcher, hooks:[…]}` rather than a bare entry.
    /// Only consulted for [`Layout::Namespaced`].
    pub grouped_events: &'static [&'static str],
    /// File holding lifecycle hooks.
    pub hooks_file: PathBuf,
    /// Lifecycle events verified to exist for this CLI.
    pub events: &'static [&'static str],
    /// Value written as the hook timeout, in whatever unit this CLI reads.
    pub timeout: u32,
    /// Per-event timeout overrides, for events the CLI caps lower than its
    /// general default.
    pub timeout_overrides: &'static [(&'static str, u32)],
    /// Command that registers an MCP server, if the CLI provides one.
    pub mcp_register: Option<(&'static str, Vec<String>)>,
    /// Standard `{"mcpServers": {…}}` file to register into, for CLIs that
    /// have no `mcp add` command but a format we have actually verified.
    pub mcp_file: Option<PathBuf>,
}

pub fn targets(exe: &Path) -> Result<Vec<Target>> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    let exe = exe.display().to_string();

    Ok(vec![
        Target {
            kind: AgentKind::ClaudeCode,
            layout: Layout::Grouped,
            grouped_events: &[],
            hooks_file: home.join(".claude/settings.json"),
            timeout_overrides: &[],
            // Verified against the installed Claude Code binary's own event
            // names, not documentation.
            //
            // `PreToolUse` is deliberately absent. Measured over 1,433 real
            // captures: 701 pre against 676 post, and the pre title is the
            // same command text with no result attached — 96% pure duplication
            // that doubled storage and every consolidation prompt. The 25
            // pre-without-post events are denied or aborted calls; if that
            // signal is ever wanted, `PermissionRequest` carries it directly.
            events: &[
                "SessionStart",
                "UserPromptSubmit",
                "PostToolUse",
                "Stop",
                "SubagentStop",
                "PreCompact",
                "PostCompact",
                "SessionEnd",
            ],
            timeout: HOOK_TIMEOUT_SECS,
            mcp_file: None,
            mcp_register: Some((
                "claude",
                vec![
                    "mcp".into(),
                    "add".into(),
                    "--scope".into(),
                    "user".into(),
                    MCP_SERVER_NAME.into(),
                    "--".into(),
                    exe.clone(),
                    "mcp".into(),
                ],
            )),
        },
        Target {
            kind: AgentKind::Codex,
            // Codex refuses to run a hook it has not been told to trust, and
            // raw hooks.json entries have no working approval path on the
            // builds we tested - `/hooks` showed nothing to approve. Installing
            // as a plugin does: Codex trusts a plugin's bundled hooks when it
            // materializes them. Verified by outcome - plugin hooks fired
            // without --dangerously-bypass-hook-trust, where raw entries never
            // fired at all.
            layout: Layout::External,
            grouped_events: &[],
            hooks_file: home.join(".codex/hooks.json"),
            // Codex caps the SessionEnd hook at three seconds and prints
            // "clamping SessionEnd hook timeout to 3s" on every session when a
            // config asks for more. Capture still works, but a warning the
            // user sees every time is a cost we impose on them for nothing.
            timeout_overrides: &[("SessionEnd", 3)],
            // `SessionEnd` and `PreCompact` were once believed absent here,
            // because they were missing from this machine's live hooks.json -
            // but that file lists what someone CONFIGURED, not what the CLI
            // supports, and absence there proves nothing. A probe settled it:
            // SessionEnd fires. PreCompact is wired on weaker evidence - the
            // trust store holds a `pre_compact` entry belonging to another
            // tool - and costs nothing if it never fires.
            // `PreToolUse` omitted for the same measured reason as above.
            events: &[
                "SessionStart",
                "UserPromptSubmit",
                "PostToolUse",
                "Stop",
                "SubagentStop",
                "PreCompact",
                "SessionEnd",
            ],
            timeout: HOOK_TIMEOUT_SECS,
            mcp_file: None,
            mcp_register: Some((
                "codex",
                vec![
                    "mcp".into(),
                    "add".into(),
                    MCP_SERVER_NAME.into(),
                    "--".into(),
                    exe,
                    "mcp".into(),
                ],
            )),
        },
        Target {
            kind: AgentKind::parse("gemini-cli"),
            layout: Layout::Grouped,
            grouped_events: &[],
            hooks_file: home.join(".gemini/settings.json"),
            timeout_overrides: &[],
            // Gemini names its lifecycle differently from the other two.
            // Verified from a live settings file that already had working
            // third-party hooks registered on these exact keys.
            //
            // `BeforeTool` is omitted for the same measured reason PreToolUse
            // is: it duplicates the after-event without the result.
            events: &["SessionStart", "BeforeAgent", "AfterAgent", "AfterTool", "PreCompress"],
            timeout: HOOK_TIMEOUT_MILLIS,
            // Gemini CLI keeps MCP servers in the same settings file rather
            // than behind an `mcp add` subcommand, and we do not hand-edit a
            // config format we have not verified. Registration is manual for
            // now; capture still works.
            mcp_file: None,
            mcp_register: None,
        },
        Target {
            kind: AgentKind::parse("antigravity"),
            layout: Layout::Namespaced,
            // Verified by probing a live `agy -p` run: tool events arrive in
            // the grouped shape, the rest as bare entries.
            grouped_events: &["PreToolUse", "PostToolUse"],
            hooks_file: home.join(".gemini/config/hooks.json"),
            timeout_overrides: &[],
            // Antigravity's own embedded docs list exactly five events:
            // PreInvocation, PostInvocation, PreToolUse, PostToolUse, Stop.
            // There is no SessionStart and no user-prompt event, so `Stop` is
            // the only session boundary available; `PostInvocation` carries no
            // content `PostToolUse` does not already have.
            events: &["PostToolUse", "Stop"],
            // SECONDS. Its docs say so outright ("Execution timeout in
            // seconds") and every working hook on this machine uses 3-10 -
            // unlike Gemini CLI, whose lineage it shares but whose unit is
            // milliseconds. Sharing an ancestor is not sharing a contract.
            timeout: HOOK_TIMEOUT_SECS,
            // Antigravity keeps MCP servers in per-plugin mcp_config.json
            // files, a format we have not verified. Capture works regardless.
            mcp_file: None,
            mcp_register: None,
        },
        Target {
            kind: AgentKind::parse("opencode"),
            layout: Layout::Plugin,
            grouped_events: &[],
            // The file we write, not a config we edit.
            // `plugins/`, plural: the directory OpenCode actually reads on
            // this machine, confirmed by the third-party plugin already
            // loading from it. The binary's own strings are ambiguous here.
            hooks_file: home.join(".config/opencode/plugins/rolepod-brain.js"),
            timeout_overrides: &[],
            // Verified against the installed binary's own event names and a
            // working third-party plugin on this machine. `session.idle` is
            // the session boundary; OpenCode has no session-end event.
            events: &["session.created", "session.idle", "session.compacted", "tool.execute.after"],
            timeout: 0,
            mcp_file: None,
            mcp_register: None,
        },
        Target {
            kind: AgentKind::parse("cursor"),
            layout: Layout::Flat,
            grouped_events: &[],
            hooks_file: home.join(".cursor/hooks.json"),
            timeout_overrides: &[],
            // camelCase, a fifth spelling. `postToolUse` covers tool activity
            // on its own: probing showed `afterShellExecution` firing for the
            // SAME execution - same command, same duration, same generation -
            // so wiring both would double-capture every shell call. Plus the
            // prompt event, because what a human types is signal, and `stop`
            // as the only session boundary Cursor offers.
            events: &["beforeSubmitPrompt", "postToolUse", "stop"],
            timeout: HOOK_TIMEOUT_SECS,
            // `~/.cursor/mcp.json` is the standard `{"mcpServers": {…}}`
            // shape, which we have verified on disk - unlike Gemini's and
            // Antigravity's MCP formats, this one we can write safely.
            mcp_file: Some(home.join(".cursor/mcp.json")),
            mcp_register: None,
        },
    ])
}

/// The OpenCode plugin, with our binary path baked in at install time.
///
/// Everything here is wrapped and fire-and-forget on purpose. A plugin runs
/// inside the user's session: an exception thrown from `tool.execute.before`
/// BLOCKS the tool, and any handler that blocks makes the agent feel slow. A
/// memory system may never be the reason someone's editor stalls.
fn opencode_plugin(exe: &Path) -> String {
    format!(
        r#"// rolepod-brain capture for OpenCode. Generated by `brain setup`.
//
// OpenCode has no hook configuration file; plugins subscribe in code. Every
// handler here is fire-and-forget and swallows its own errors: capture must
// never slow down or break the session it is observing.
import {{ spawn }} from "node:child_process"

const BRAIN = {exe:?}

function capture(event, payload) {{
  try {{
    const child = spawn(BRAIN, ["hook", "--cli", "opencode", "--event", event], {{
      stdio: ["pipe", "ignore", "ignore"],
      detached: true,
    }})
    child.on("error", () => {{}})
    child.stdin.on("error", () => {{}})
    child.stdin.end(JSON.stringify(payload))
    child.unref()
  }} catch {{
    /* capture is best effort; the session is not */
  }}
}}

export const RolepodBrainPlugin = async ({{ directory }}) => {{
  // The plugin factory is the only place the project path is available, and
  // it is what lets OpenCode events land in the right brain.
  const cwd = directory || process.cwd()
  let sessionId = null

  return {{
    event: async ({{ event }}) => {{
      try {{
        const type = event?.type
        if (type === "session.created") {{
          sessionId = event?.properties?.info?.id ?? null
          capture("SessionStart", {{ cwd, session_id: sessionId, source: "startup" }})
        }} else if (type === "session.idle") {{
          capture("Stop", {{ cwd, session_id: sessionId }})
        }} else if (type === "session.compacted") {{
          capture("PreCompact", {{ cwd, session_id: sessionId, trigger: "auto" }})
        }}
      }} catch {{
        /* never break the session */
      }}
    }},

    "tool.execute.after": async (input, output) => {{
      try {{
        capture("PostToolUse", {{
          cwd,
          session_id: sessionId,
          tool_name: String(input?.tool ?? ""),
          tool_input: output?.args ?? {{}},
        }})
      }} catch {{
        /* never break a tool call */
      }}
    }},
  }}
}}
"#
    )
}

/// Remove every trace of our wiring, and optionally the memory itself.
///
/// The other half of "it never leaves your machine": something that cannot be
/// fully removed is not really yours. Foreign hooks are untouched, as always -
/// uninstalling us is not licence to disturb anything else.
///
/// # Errors
/// Returns an error when a config exists but cannot be read or written.
pub fn uninstall(apply: bool) -> Result<Vec<Change>> {
    let exe = std::env::current_exe().context("locate our own binary")?;
    let mut changes = Vec::new();

    for target in targets(&exe)? {
        if !cli_present(&target) {
            continue;
        }
        let label = target.kind.as_str().to_string();

        match target.layout {
            Layout::Plugin => {
                if target.hooks_file.is_file() {
                    if apply {
                        std::fs::remove_file(&target.hooks_file)
                            .with_context(|| format!("remove {}", target.hooks_file.display()))?;
                    }
                    changes.push(Change {
                        target: label.clone(),
                        detail: format!(
                            "{} plugin {}",
                            if apply { "removed" } else { "would remove" },
                            target.hooks_file.display()
                        ),
                    });
                }
            }
            Layout::External => changes.push(Change {
                target: label.clone(),
                detail: "remove the plugin yourself: codex plugin remove rolepod-brain".to_string(),
            }),
            _ => changes.extend(strip_hooks(&target, apply)?),
        }

        changes.extend(strip_mcp(&target, apply));
    }

    changes.extend(remove_timer(apply)?);
    Ok(changes)
}

/// Take our entries out of a hook config, leaving everyone else's.
fn strip_hooks(target: &Target, apply: bool) -> Result<Vec<Change>> {
    let mut root = read_json(&target.hooks_file)?;
    let container = match target.layout {
        Layout::Namespaced => MCP_SERVER_NAME,
        _ => "hooks",
    };

    let mut removed = 0usize;
    if let Some(events) = root.get_mut(container).and_then(Value::as_object_mut) {
        for entries in events.values_mut() {
            if let Some(entries) = entries.as_array_mut() {
                let before = entries.len();
                entries.retain(|entry| !is_ours(entry));
                removed += before - entries.len();
            }
        }
        events.retain(|_, entries| entries.as_array().is_none_or(|e| !e.is_empty()));
    }
    // Our whole namespace goes; a grouped config's "hooks" key belongs to
    // everyone and stays.
    if target.layout == Layout::Namespaced {
        if let Some(map) = root.as_object_mut() {
            map.remove(MCP_SERVER_NAME);
        }
    }

    if removed == 0 {
        return Ok(Vec::new());
    }
    if apply {
        let _ = backup(&target.hooks_file)?;
        write_json(&target.hooks_file, &root)?;
    }
    Ok(vec![Change {
        target: target.kind.as_str().to_string(),
        detail: format!(
            "{} {removed} hook(s) from {}",
            if apply { "removed" } else { "would remove" },
            target.hooks_file.display()
        ),
    }])
}

/// Deregister our MCP server, by whichever route registered it.
fn strip_mcp(target: &Target, apply: bool) -> Vec<Change> {
    let label = target.kind.as_str().to_string();

    if let Some(path) = &target.mcp_file {
        let Ok(mut root) = read_json(path) else { return Vec::new() };
        let present = root
            .get("mcpServers")
            .and_then(Value::as_object)
            .is_some_and(|servers| servers.contains_key(MCP_SERVER_NAME));
        if !present {
            return Vec::new();
        }
        if apply {
            if let Some(servers) = root.get_mut("mcpServers").and_then(Value::as_object_mut) {
                servers.remove(MCP_SERVER_NAME);
            }
            let _ = backup(path);
            let _ = write_json(path, &root);
        }
        return vec![Change {
            target: label,
            detail: format!(
                "{} MCP server from {}",
                if apply { "removed" } else { "would remove" },
                path.display()
            ),
        }];
    }

    let Some((program, _)) = &target.mcp_register else { return Vec::new() };
    if !apply {
        return vec![Change {
            target: label,
            detail: format!("would run: {program} mcp remove {MCP_SERVER_NAME}"),
        }];
    }
    match Command::new(program).args(["mcp", "remove", MCP_SERVER_NAME]).output() {
        Ok(output) if output.status.success() => {
            vec![Change { target: label, detail: "deregistered MCP server".to_string() }]
        }
        // Not registered is the desired end state, not a failure.
        _ => Vec::new(),
    }
}

/// Plan (and optionally perform) the wiring.
///
/// # Errors
/// Returns an error when a config file exists but cannot be read or parsed —
/// we refuse to overwrite a file we do not understand.
pub fn run(only: Option<&str>, apply: bool) -> Result<Vec<Change>> {
    let exe = std::env::current_exe().context("locate our own binary")?;
    let mut changes = Vec::new();

    for target in targets(&exe)? {
        if let Some(only) = only {
            if only != "all" && only != target.kind.as_str() {
                continue;
            }
        }
        if !cli_present(&target) {
            changes.push(Change {
                target: target.kind.as_str().to_string(),
                detail: "not installed — skipped".to_string(),
            });
            continue;
        }

        changes.extend(match target.layout {
            Layout::Plugin => install_plugin(&target, &exe, apply)?,
            Layout::External => defer_to_plugin_flow(&target, apply)?,
            _ => wire_hooks(&target, &exe, apply)?,
        });
        changes.extend(register_mcp(&target, apply));
    }

    changes.extend(install_timer(&exe, apply)?);
    changes.extend(write_config_template(apply));
    Ok(changes)
}

/// Every knob, commented out, in the place a person would look for it.
///
/// The config has always been optional and absent by default - correct for
/// install-and-forget, terrible for discovery: the only way to learn a knob
/// existed was the README. This file changes nothing (everything is
/// commented, so parsing it yields exactly the defaults) and teaches
/// everything. Written once, never overwritten: whatever is in this file
/// after that is the user's.
const CONFIG_TEMPLATE: &str = r#"# rolepod-brain configuration.
#
# Everything here is optional and shown at its default. Uncomment a line to
# change it; delete this file to return to all defaults. `brain doctor`
# reports the effective settings.

[summarizer]
# Which CLI's model writes the summaries.
#   auto        borrow the CLI that produced the events, cheapest tier
#   claude-code / codex / gemini   pin one CLI
#   off         permanent rule-based summaries - fully functional, never
#               spends a model call
# mode = "auto"

[summarizer.models]
# Per-CLI model overrides. Memory quality is a spend decision: the default is
# each CLI's cheap tier, and naming a better model here buys better summaries
# at that CLI's price. Per-CLI because model names do not travel between
# vendors. A CLI not named here keeps its cheap default.
# "claude-code" = "sonnet"

[injection]
# Byte ceilings for automatic context injection. Ceilings, not targets: a
# short primer of real signal beats a full one padded with noise.
# primer_budget = 4096
# session_budget = 8192

[consolidation]
# true installs a wall-clock timer as the consolidation backstop. Off by
# default: a hook noticing a stale backlog already catches up, and a timer
# shows up in macOS Login Items as a background agent.
# timer = false

[search]
# true spends one cheap-tier model call per search, reordering results by
# what the query was asking rather than term statistics. A failed or slow
# call leaves the order exactly as the index ranked it.
# rerank = false

[sanitize]
# Additional regexes to redact beyond the built-in credential patterns, and
# substrings that survive redaction.
# extra_patterns = []
# allowlist = []
"#;

/// Leave the template where a curious user will find it. Never overwrite.
fn write_config_template(apply: bool) -> Vec<Change> {
    let Ok(paths) = Paths::resolve() else { return Vec::new() };
    let path = paths.config_file();
    if path.exists() {
        return Vec::new();
    }
    let detail = if apply {
        match std::fs::write(&path, CONFIG_TEMPLATE) {
            Ok(()) => format!("wrote {} (all defaults, commented out)", path.display()),
            Err(error) => format!("could not write the config template: {error}"),
        }
    } else {
        format!("would write {} (all defaults, commented out)", path.display())
    };
    vec![Change { target: "config".to_string(), detail }]
}

/// Install the consolidation backstop.
///
/// A timer is not a daemon: launchd starts a process, it does its work, it
/// exits. Nothing resides. This exists because a hook-triggered consolidation
/// can be missed — a crash, a machine asleep for a week — and the work should
/// still catch up without the user ever running a command.
fn install_timer(exe: &Path, apply: bool) -> Result<Vec<Change>> {
    // Opt-in. Consolidation only has value when a next session reads it, and
    // that session fires hooks - so a hook noticing a stale backlog is the
    // default backstop, and it costs the user nothing visible. A launchd agent
    // costs them a macOS Login Items entry saying "brain can run in the
    // background", which contradicts the product's promise on their own screen.
    let wanted = Config::load(&Paths::resolve()?.config_file())
        .map(|config| config.consolidation.timer)
        .unwrap_or(false);
    if !wanted {
        return remove_timer(apply);
    }
    if !cfg!(target_os = "macos") {
        return Ok(vec![Change {
            target: "timer".to_string(),
            detail: "skipped: only launchd is implemented so far".to_string(),
        }]);
    }
    let Some(home) = dirs::home_dir() else {
        return Ok(vec![Change {
            target: "timer".to_string(),
            detail: "skipped: cannot determine home directory".to_string(),
        }]);
    };

    let plist_path = home.join("Library/LaunchAgents").join(format!("{TIMER_LABEL}.plist"));
    let plist = timer_plist(exe, &home);

    if !apply {
        return Ok(vec![Change {
            target: "timer".to_string(),
            detail: format!("would install {} (every {TIMER_INTERVAL_SECS}s)", plist_path.display()),
        }]);
    }

    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&plist_path, plist)
        .with_context(|| format!("write {}", plist_path.display()))?;

    // Reload so the change takes effect without a logout. `bootout` on a
    // service that was never loaded is an expected failure, not a problem.
    //
    // Note for anyone reading `launchctl print … | grep runs` later: a
    // StartInterval job fires immediately on bootstrap when its interval
    // already elapsed while it was unloaded, so re-running setup inflates that
    // counter. It is not the timer misfiring, and consolidation's debounce
    // makes an extra run cheap.
    let domain = format!("gui/{}", unsafe_uid());
    let _ = Command::new("launchctl").args(["bootout", &format!("{domain}/{TIMER_LABEL}")]).output();
    let loaded = Command::new("launchctl")
        .args(["bootstrap", &domain, &plist_path.to_string_lossy()])
        .output();

    let detail = match loaded {
        Ok(output) if output.status.success() => {
            format!("installed and loaded {}", plist_path.display())
        }
        Ok(output) => format!(
            "wrote {} but launchctl bootstrap failed: {}",
            plist_path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        Err(error) => format!("wrote {} but launchctl is unavailable: {error}", plist_path.display()),
    };
    Ok(vec![Change { target: "timer".to_string(), detail }])
}

/// Unload and delete the background agent, if one is installed.
///
/// Run on every `setup` while the timer is off, so a machine that once had the
/// agent installed loses it - and its Login Items entry - without the user
/// having to know launchctl.
fn remove_timer(apply: bool) -> Result<Vec<Change>> {
    let Some(home) = dirs::home_dir() else { return Ok(Vec::new()) };
    let plist = home.join("Library/LaunchAgents").join(format!("{TIMER_LABEL}.plist"));
    if !plist.is_file() {
        return Ok(vec![Change {
            target: "backstop".to_string(),
            detail: "hook-opportunistic (no background agent)".to_string(),
        }]);
    }
    if !apply {
        return Ok(vec![Change {
            target: "backstop".to_string(),
            detail: format!("would remove the background agent at {}", plist.display()),
        }]);
    }

    let domain = format!("gui/{}", unsafe_uid());
    let _ = Command::new("launchctl")
        .args(["bootout", &format!("{domain}/{TIMER_LABEL}")])
        .output();
    std::fs::remove_file(&plist)
        .with_context(|| format!("remove {}", plist.display()))?;
    Ok(vec![Change {
        target: "backstop".to_string(),
        detail: format!(
            "removed the background agent ({}); backstop is hook-opportunistic",
            plist.display()
        ),
    }])
}

/// The current user id, for the launchd GUI domain.
fn unsafe_uid() -> String {
    // `id -u` avoids a libc dependency for one number.
    Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "501".to_string(), |uid| uid.trim().to_string())
}

fn timer_plist(exe: &Path, home: &Path) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{TIMER_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>consolidate</string>
        <string>--all</string>
    </array>
    <key>StartInterval</key>
    <integer>{TIMER_INTERVAL_SECS}</integer>
    <key>RunAtLoad</key>
    <false/>
    <key>ProcessType</key>
    <string>Background</string>
    <key>LowPriorityIO</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>
"#,
        exe = exe.display(),
        log = home.join(".rolepod-brain/timer.log").display(),
    )
}

/// A CLI counts as present when its config directory exists — the CLI itself
/// may live behind a shim we cannot see on `PATH`.
fn cli_present(target: &Target) -> bool {
    target.hooks_file.parent().is_some_and(Path::is_dir)
}

/// Write (or preview) a plugin file.
///
/// The whole file is ours, so there is no foreign content to preserve and no
/// sweep to perform — uninstalling is deleting it.
fn install_plugin(target: &Target, exe: &Path, apply: bool) -> Result<Vec<Change>> {
    let source = opencode_plugin(exe);
    if !apply {
        return Ok(vec![Change {
            target: target.kind.as_str().to_string(),
            detail: format!(
                "would write plugin {} ({} lifecycle events)",
                target.hooks_file.display(),
                target.events.len()
            ),
        }]);
    }

    let mut changes = Vec::new();
    if let Some(backup) = backup(&target.hooks_file)? {
        changes.push(Change {
            target: target.kind.as_str().to_string(),
            detail: format!("backed up to {}", backup.display()),
        });
    }
    if let Some(parent) = target.hooks_file.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&target.hooks_file, source)
        .with_context(|| format!("write {}", target.hooks_file.display()))?;
    changes.push(Change {
        target: target.kind.as_str().to_string(),
        detail: format!("wrote plugin {}", target.hooks_file.display()),
    });
    Ok(changes)
}

/// Clean up our raw entries and point the user at the CLI's own install flow.
///
/// The raw entries are removed rather than left in place: they never ran, and
/// leaving them would double-capture the day someone did manage to trust them.
fn defer_to_plugin_flow(target: &Target, apply: bool) -> Result<Vec<Change>> {
    let label = target.kind.as_str().to_string();
    let mut changes = Vec::new();

    let mut root = read_json(&target.hooks_file)?;
    let removed = root
        .get_mut("hooks")
        .and_then(Value::as_object_mut)
        .map_or(0, |events| {
            let mut removed = 0usize;
            for groups in events.values_mut() {
                if let Some(groups) = groups.as_array_mut() {
                    let before = groups.len();
                    groups.retain(|group| !is_ours(group));
                    removed += before - groups.len();
                }
            }
            events.retain(|_, groups| groups.as_array().is_none_or(|g| !g.is_empty()));
            removed
        });

    if removed > 0 {
        if apply {
            let _ = backup(&target.hooks_file)?;
            write_json(&target.hooks_file, &root)?;
            changes.push(Change {
                target: label.clone(),
                detail: format!("removed {removed} raw hook entr(y/ies) that never ran"),
            });
        } else {
            changes.push(Change {
                target: label.clone(),
                detail: format!("would remove {removed} raw hook entr(y/ies) that never ran"),
            });
        }
    }

    changes.push(Change {
        target: label,
        detail: "install via the plugin flow (see README): \
                 codex plugin marketplace add nuttaruj/rolepod-brain && \
                 codex plugin add rolepod-brain@rolepod-brain"
            .to_string(),
    });
    Ok(changes)
}

fn wire_hooks(target: &Target, exe: &Path, apply: bool) -> Result<Vec<Change>> {
    let mut root = read_json(&target.hooks_file)?;
    if !root.is_object() {
        root = json!({});
    }

    let mut planned = Vec::new();
    let mut replaced = 0usize;
    let mut swept = Vec::new();

    // Grouped layouts nest everything under "hooks"; namespaced ones give each
    // tool its own top-level key, which is how Antigravity keeps one tool's
    // hooks from colliding with another's.
    let container = match target.layout {
        Layout::Grouped | Layout::Flat => "hooks",
        Layout::Namespaced => MCP_SERVER_NAME,
        // Routed to `install_plugin` before reaching here.
        Layout::Plugin | Layout::External => {
            unreachable!("these targets do not edit a hook config here")
        }
    };
    let hooks = root
        .as_object_mut()
        .expect("root is an object")
        .entry(container)
        .or_insert_with(|| json!({}));
    if !hooks.is_object() {
        anyhow::bail!(
            "{} has a `{container}` key that is not an object; refusing to modify it",
            target.hooks_file.display()
        );
    }

    // Sweep our entries out of events we no longer wire. Without this, an
    // event dropped from the table above keeps firing the old hook forever —
    // the config remembers a decision the code has already reversed.
    if let Some(map) = hooks.as_object_mut() {
        for (event, groups) in map.iter_mut() {
            if target.events.contains(&event.as_str()) {
                continue;
            }
            let Some(groups) = groups.as_array_mut() else { continue };
            let before = groups.len();
            groups.retain(|group| !is_ours(group));
            if groups.len() < before {
                swept.push(event.clone());
            }
        }
        map.retain(|_, groups| groups.as_array().is_none_or(|g| !g.is_empty()));
    }

    for event in target.events {
        let command = format!(
            "{} hook --cli {} --event {}",
            shell_quote(&exe.display().to_string()),
            target.kind.as_str(),
            event
        );
        let timeout = target
            .timeout_overrides
            .iter()
            .find(|(name, _)| name == event)
            .map_or(target.timeout, |(_, seconds)| *seconds);
        let handler = json!({
            "type": "command",
            "command": command,
            "timeout": timeout,
        });
        let grouped = target.layout == Layout::Grouped
            || target.grouped_events.contains(event);
        let entry = if grouped { json!({ "hooks": [handler] }) } else { handler };

        let groups = hooks
            .as_object_mut()
            .expect("hooks is an object")
            .entry((*event).to_string())
            .or_insert_with(|| json!([]));
        let Some(groups) = groups.as_array_mut() else {
            anyhow::bail!(
                "{} has a non-array `hooks.{event}`; refusing to modify it",
                target.hooks_file.display()
            );
        };

        // Remove only our own previous entry, leaving every other tool's
        // hooks exactly where they are.
        let before = groups.len();
        groups.retain(|group| !is_ours(group));
        replaced += before - groups.len();

        groups.push(entry);
        planned.push(*event);
    }

    let mut changes = vec![Change {
        target: target.kind.as_str().to_string(),
        detail: format!(
            "{} {} hook(s) in {}{}{}",
            if apply { "wrote" } else { "would write" },
            planned.len(),
            target.hooks_file.display(),
            if replaced > 0 {
                format!(" (replacing {replaced} existing entr(y/ies) of ours)")
            } else {
                String::new()
            },
            if swept.is_empty() {
                String::new()
            } else {
                format!(" (removed ours from no-longer-wired {})", swept.join(", "))
            }
        ),
    }];

    if target.layout == Layout::Flat {
        // Cursor stamps its config with a schema version. A file we write
        // without it may be ignored outright.
        root.as_object_mut()
            .expect("root is an object")
            .entry("version")
            .or_insert_with(|| json!(1));
    }

    if apply {
        let backup = backup(&target.hooks_file)?;
        if let Some(backup) = backup {
            changes.push(Change {
                target: target.kind.as_str().to_string(),
                detail: format!("backed up to {}", backup.display()),
            });
        }
        write_json(&target.hooks_file, &root)?;
    }
    Ok(changes)
}

fn register_mcp(target: &Target, apply: bool) -> Vec<Change> {
    // The plugin declares the same server. Registering it again would leave
    // two entries pointing at one binary, which is the duplicate-registration
    // bug this project already shipped once for Codex - so when the plugin is
    // installed we withdraw rather than compete with it.
    if plugin_installed(&target.kind) {
        return withdraw_mcp(target, apply);
    }
    if let Some(path) = &target.mcp_file {
        return register_mcp_file(target, path, apply);
    }
    let Some((program, args)) = &target.mcp_register else {
        return Vec::new();
    };
    let rendered = format!("{program} {}", args.join(" "));

    if !apply {
        return vec![Change {
            target: target.kind.as_str().to_string(),
            detail: format!("would run: {rendered}"),
        }];
    }

    // The vendor CLI owns its own MCP config format; shelling out to it is
    // safer than writing that file ourselves.
    match Command::new(program).args(args).output() {
        Ok(output) if output.status.success() => vec![Change {
            target: target.kind.as_str().to_string(),
            detail: format!("registered MCP server `{MCP_SERVER_NAME}`"),
        }],
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{stderr}{}", String::from_utf8_lossy(&output.stdout));
            // Already registered is the state we wanted, so reporting it as a
            // failure would train the user to ignore this command's output.
            if combined.to_lowercase().contains("already exists") {
                return vec![Change {
                    target: target.kind.as_str().to_string(),
                    detail: format!("MCP server `{MCP_SERVER_NAME}` already registered"),
                }];
            }
            vec![Change {
                target: target.kind.as_str().to_string(),
                detail: format!(
                    "MCP registration failed ({}). Run manually: {rendered}\n  {}",
                    output.status,
                    stderr.trim()
                ),
            }]
        }
        Err(error) => vec![Change {
            target: target.kind.as_str().to_string(),
            detail: format!("could not run `{program}` ({error}). Run manually: {rendered}"),
        }],
    }
}

/// Register our stdio server in a standard `{"mcpServers": {…}}` file.
///
/// Only used where that shape has been verified on disk. Other servers in the
/// file are left exactly as they are; we own one key.
fn register_mcp_file(target: &Target, path: &Path, apply: bool) -> Vec<Change> {
    let label = target.kind.as_str().to_string();
    if !apply {
        return vec![Change {
            target: label,
            detail: format!("would register MCP server `{MCP_SERVER_NAME}` in {}", path.display()),
        }];
    }

    let exe = match std::env::current_exe() {
        Ok(exe) => exe.display().to_string(),
        Err(error) => {
            return vec![Change { target: label, detail: format!("cannot locate our binary: {error}") }]
        }
    };

    let result = (|| -> Result<()> {
        let mut root = read_json(path)?;
        if !root.is_object() {
            root = json!({});
        }
        let servers = root
            .as_object_mut()
            .expect("root is an object")
            .entry("mcpServers")
            .or_insert_with(|| json!({}));
        let Some(servers) = servers.as_object_mut() else {
            anyhow::bail!("{} has a non-object `mcpServers`; refusing to modify it", path.display());
        };
        servers.insert(
            MCP_SERVER_NAME.to_string(),
            json!({ "command": exe, "args": ["mcp"] }),
        );
        let _ = backup(path)?;
        write_json(path, &root)
    })();

    match result {
        Ok(()) => vec![Change {
            target: label,
            detail: format!("registered MCP server `{MCP_SERVER_NAME}` in {}", path.display()),
        }],
        Err(error) => vec![Change { target: label, detail: format!("MCP registration failed: {error:#}") }],
    }
}

/// Is our plugin installed for this CLI?
///
/// Each CLI records that fact somewhere different, and none of them agree on
/// a format. What they agree on is the consequence: an installed plugin
/// already supplies the MCP server and the skills, so anything `setup` writes
/// on top of it is a second copy.
#[must_use]
pub fn plugin_installed(kind: &AgentKind) -> bool {
    let Some(home) = dirs::home_dir() else { return false };
    match kind.as_str() {
        // A JSON registry keyed `<plugin>@<marketplace>`.
        "claude-code" => std::fs::read_to_string(home.join(".claude/plugins/installed_plugins.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|root| root.get("plugins")?.as_object().cloned())
            .is_some_and(|plugins| {
                plugins.keys().any(|key| key.starts_with(&format!("{PLUGIN_NAME}@")))
            }),
        // No registry file: an installed plugin is a directory under the
        // marketplace it came from.
        "cursor" => std::fs::read_dir(home.join(".cursor/plugins/cache"))
            .into_iter()
            .flatten()
            .filter_map(std::result::Result::ok)
            .any(|marketplace| marketplace.path().join(PLUGIN_NAME).is_dir()),
        // TOML, and the only CLI where installing the plugin is what makes
        // the hooks run at all.
        "codex" => codex_plugin_installed(&home),
        _ => false,
    }
}

/// Codex records enabled plugins in `~/.codex/config.toml`.
fn codex_plugin_installed(home: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(home.join(".codex/config.toml")) else {
        return false;
    };
    let Ok(config) = toml::from_str::<toml::Value>(&text) else { return false };
    config
        .get("plugins")
        .and_then(toml::Value::as_table)
        .is_some_and(|plugins| {
            plugins.iter().any(|(key, value)| {
                key.starts_with(PLUGIN_NAME)
                    && value.get("enabled").and_then(toml::Value::as_bool) != Some(false)
            })
        })
}

/// Step aside from an MCP registration the plugin now owns.
///
/// Removing our standalone entry rather than leaving it: two registrations of
/// one server is not twice the memory, it is one server listed twice in every
/// tool menu, and the agent has no way to tell they are the same.
fn withdraw_mcp(target: &Target, apply: bool) -> Vec<Change> {
    let label = target.kind.as_str().to_string();
    let detail = |what: String| vec![Change { target: label.clone(), detail: what }];

    if let Some(path) = &target.mcp_file {
        if !apply {
            return detail(format!(
                "plugin installed — would remove our standalone MCP entry from {}",
                path.display()
            ));
        }
        let removed = (|| -> Result<bool> {
            let mut root = read_json(path)?;
            let Some(servers) =
                root.get_mut("mcpServers").and_then(serde_json::Value::as_object_mut)
            else {
                return Ok(false);
            };
            if servers.remove(MCP_SERVER_NAME).is_none() {
                return Ok(false);
            }
            let _ = backup(path)?;
            write_json(path, &root)?;
            Ok(true)
        })();
        return match removed {
            Ok(true) => detail(format!(
                "plugin installed — removed our standalone MCP entry from {}",
                path.display()
            )),
            Ok(false) => detail("plugin supplies the MCP server".to_string()),
            Err(error) => detail(format!("could not remove the standalone MCP entry: {error:#}")),
        };
    }

    let Some((program, _)) = &target.mcp_register else {
        return detail("plugin supplies the MCP server".to_string());
    };
    let args = vec!["mcp".to_string(), "remove".to_string(), "--scope".to_string(),
                    "user".to_string(), MCP_SERVER_NAME.to_string()];
    if !apply {
        return detail(format!("plugin installed — would run: {program} {}", args.join(" ")));
    }
    match Command::new(program).args(&args).output() {
        // Not registered is the state we wanted; the CLI says so on stderr
        // and a non-zero exit, which is not a failure of this step.
        Ok(_) => detail("plugin supplies the MCP server".to_string()),
        Err(error) => detail(format!("could not reach `{program}`: {error}")),
    }
}

/// Does this hook entry belong to us?
///
/// Handles both shapes an entry can take: a group wrapping handlers, and a
/// bare handler. A namespaced config contains both side by side, and missing
/// one shape would mean leaving a stale hook of ours behind on every re-run.
fn is_ours(entry: &Value) -> bool {
    let carries_marker = |value: &Value| {
        value
            .get("command")
            .and_then(Value::as_str)
            .is_some_and(|command| command.contains(MARKER))
    };
    if carries_marker(entry) {
        return true;
    }
    entry
        .get("hooks")
        .and_then(Value::as_array)
        .is_some_and(|handlers| handlers.iter().any(carries_marker))
}

fn read_json(path: &Path) -> Result<Value> {
    if !path.is_file() {
        return Ok(json!({}));
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(&text).with_context(|| {
        format!("{} is not valid JSON; refusing to overwrite it", path.display())
    })
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let mut text = serde_json::to_string_pretty(value).context("serialize config")?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// Copy a config aside before modifying it. Returns the backup path, or
/// `None` when there was nothing to back up.
fn backup(path: &Path) -> Result<Option<PathBuf>> {
    if !path.is_file() {
        return Ok(None);
    }
    let stamp = jiff::Zoned::now().strftime("%Y%m%d-%H%M%S").to_string();
    let backup = path.with_extension(format!(
        "{}.brain-bak.{stamp}",
        path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default()
    ));
    std::fs::copy(path, &backup)
        .with_context(|| format!("back up {} to {}", path.display(), backup.display()))?;
    Ok(Some(backup))
}

/// Quote a path for `/bin/sh` only when it needs it.
fn shell_quote(input: &str) -> String {
    if input
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-'))
    {
        return input.to_string();
    }
    format!("'{}'", input.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_only_our_own_entries() {
        let ours = json!({"hooks": [{"type": "command", "command": "/usr/bin/brain hook --cli codex --event Stop"}]});
        let theirs = json!({"hooks": [{"type": "command", "command": "another-tool capture"}]});
        assert!(is_ours(&ours));
        assert!(!is_ours(&theirs));
        assert!(!is_ours(&json!({})));
    }

    #[test]
    fn an_event_capped_lower_than_the_default_gets_the_cap() {
        // Codex clamps SessionEnd to 3s and warns on every session otherwise.
        let codex = targets(Path::new("/usr/local/bin/brain"))
            .unwrap()
            .into_iter()
            .find(|t| t.kind.as_str() == "codex")
            .expect("codex target");
        assert_eq!(codex.timeout_overrides, &[("SessionEnd", 3)]);
        for (event, seconds) in codex.timeout_overrides {
            assert!(codex.events.contains(event), "override for an unwired event: {event}");
            assert!(*seconds <= codex.timeout, "an override should only ever lower the timeout");
        }
    }

    /// The template must be a no-op as written, and honest about its names.
    ///
    /// Commented-out knobs are only documentation if the names are real: a
    /// typo in the template would send a user editing a key nothing reads,
    /// which is worse than no template. Uncommenting each knob with a
    /// non-default value and asserting the parsed config moved is what keeps
    /// the file and the struct from drifting apart.
    #[test]
    fn the_config_template_is_inert_as_written_and_every_knob_is_real() {
        // As shipped: everything commented, so parsing yields pure defaults.
        let parsed: crate::config::Config =
            toml::from_str(CONFIG_TEMPLATE).expect("the template must be valid TOML");
        let defaults = crate::config::Config::default();
        assert_eq!(parsed.summarizer.mode, defaults.summarizer.mode);
        assert_eq!(parsed.injection.primer_budget, defaults.injection.primer_budget);
        assert_eq!(parsed.injection.session_budget, defaults.injection.session_budget);
        assert_eq!(parsed.consolidation.timer, defaults.consolidation.timer);
        assert_eq!(parsed.search.rerank, defaults.search.rerank);
        assert!(parsed.summarizer.models.is_empty());

        // Every knob uncommented with a NON-default value must actually move
        // the field it claims to control.
        let live = CONFIG_TEMPLATE
            .replace("# mode = \"auto\"", "mode = \"off\"")
            .replace("# \"claude-code\" = \"sonnet\"", "\"claude-code\" = \"sonnet\"")
            .replace("# primer_budget = 4096", "primer_budget = 1")
            .replace("# session_budget = 8192", "session_budget = 2")
            .replace("# timer = false", "timer = true")
            .replace("# rerank = false", "rerank = true")
            .replace("# extra_patterns = []", "extra_patterns = [\"secret-\\\\d+\"]")
            .replace("# allowlist = []", "allowlist = [\"not-a-secret\"]");
        let parsed: crate::config::Config =
            toml::from_str(&live).expect("uncommented template must still parse");
        assert_eq!(parsed.summarizer.mode, "off");
        assert_eq!(parsed.summarizer.models.get("claude-code").map(String::as_str), Some("sonnet"));
        assert_eq!(parsed.injection.primer_budget, 1);
        assert_eq!(parsed.injection.session_budget, 2);
        assert!(parsed.consolidation.timer);
        assert!(parsed.search.rerank);
        assert_eq!(parsed.sanitize.extra_patterns, vec!["secret-\\d+".to_string()]);
        assert_eq!(parsed.sanitize.allowlist, vec!["not-a-secret".to_string()]);
    }

    /// Each CLI records an installed plugin somewhere different, and getting
    /// any of them wrong means a second MCP registration nobody notices.
    #[test]
    fn an_installed_plugin_is_recognised_in_each_cli_s_own_registry() {
        let home = std::env::temp_dir().join(format!("brain-plugin-detect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let _guard = crate::invocation::ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var("HOME").ok();
        std::env::set_var("HOME", &home);

        let claude = AgentKind::parse("claude-code");
        let cursor = AgentKind::parse("cursor");
        let codex = AgentKind::parse("codex");

        // Nothing installed anywhere.
        for kind in [&claude, &cursor, &codex] {
            assert!(!plugin_installed(kind), "{} claimed an absent plugin", kind.as_str());
        }

        // Claude Code: a JSON registry keyed `<plugin>@<marketplace>`.
        std::fs::create_dir_all(home.join(".claude/plugins")).unwrap();
        std::fs::write(
            home.join(".claude/plugins/installed_plugins.json"),
            format!(r#"{{"version":2,"plugins":{{"{PLUGIN_NAME}@{PLUGIN_NAME}":[{{"scope":"user"}}]}}}}"#),
        )
        .unwrap();
        assert!(plugin_installed(&claude));
        assert!(!plugin_installed(&cursor), "one CLI's registry answered for another");

        // Cursor: a directory under the marketplace it came from.
        std::fs::create_dir_all(home.join(".cursor/plugins/cache/somewhere").join(PLUGIN_NAME))
            .unwrap();
        assert!(plugin_installed(&cursor));

        // Codex: a TOML table, and `enabled = false` means not installed.
        std::fs::create_dir_all(home.join(".codex")).unwrap();
        std::fs::write(
            home.join(".codex/config.toml"),
            format!("[plugins.\"{PLUGIN_NAME}@{PLUGIN_NAME}\"]\nenabled = false\n"),
        )
        .unwrap();
        assert!(!plugin_installed(&codex), "a disabled plugin is not an installed one");
        std::fs::write(
            home.join(".codex/config.toml"),
            format!("[plugins.\"{PLUGIN_NAME}@{PLUGIN_NAME}\"]\nenabled = true\n"),
        )
        .unwrap();
        assert!(plugin_installed(&codex));

        // A CLI with no plugin story must never claim one.
        assert!(!plugin_installed(&AgentKind::parse("opencode")));

        match previous {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    /// Every path a plugin manifest names must exist, and the file Claude
    /// Code and Cursor auto-discover must not be Codex's.
    ///
    /// These manifests are read by three different CLIs and by none of our
    /// code, so nothing else here would notice a path that stopped resolving.
    #[test]
    fn the_plugin_manifests_point_at_files_that_exist() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let read = |path: std::path::PathBuf| -> Value {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("missing manifest {}", path.display()));
            serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("{} is not JSON: {error}", path.display()))
        };

        // The marketplace entry has to name the directory the plugin is in.
        let marketplace = read(repo.join(".claude-plugin/marketplace.json"));
        let listed = marketplace["plugins"][0]["source"].as_str().expect("a source path");
        let root = repo.join(listed.trim_start_matches("./"));
        assert!(root.is_dir(), "marketplace names {listed}, which is not a directory");
        assert_eq!(marketplace["plugins"][0]["name"], PLUGIN_NAME);

        // One plugin, three CLIs, three manifests - each of which has to agree
        // on the name the registries are keyed by.
        for manifest in [".claude-plugin", ".cursor-plugin", ".codex-plugin"] {
            let plugin = read(root.join(manifest).join("plugin.json"));
            assert_eq!(plugin["name"], PLUGIN_NAME, "{manifest} disagrees about the name");
        }

        // Codex resolves its paths from the plugin root.
        let codex = read(root.join(".codex-plugin/plugin.json"));
        for key in ["hooks", "mcpServers", "skills"] {
            let path = codex[key].as_str().unwrap_or_else(|| panic!("{key} is not a path"));
            let resolved = root.join(path.trim_start_matches("./"));
            assert!(resolved.exists(), "{key} points at {path}, which does not exist");
        }

        // Claude Code and Cursor auto-discover `hooks/hooks.json` at the
        // plugin root. Ours speaks Codex's event names and tags every capture
        // `--cli codex`, so a file under that name would make those CLIs
        // record their sessions as Codex's - on top of the hooks `setup`
        // already wrote for them.
        assert!(
            !root.join("hooks/hooks.json").exists(),
            "hooks/hooks.json is the name Claude Code and Cursor claim; Codex's must not use it"
        );
    }

    #[test]
    fn each_cli_gets_its_own_timeout_unit() {
        let exe = Path::new("/usr/local/bin/brain");
        let targets = targets(exe).unwrap();
        let gemini = targets
            .iter()
            .find(|t| t.kind.as_str() == "gemini-cli")
            .expect("gemini-cli target");
        assert_eq!(gemini.timeout, HOOK_TIMEOUT_MILLIS, "gemini reads milliseconds");
        let claude = targets
            .iter()
            .find(|t| t.kind.as_str() == "claude-code")
            .expect("claude-code target");
        assert_eq!(claude.timeout, HOOK_TIMEOUT_SECS, "claude code reads seconds");
    }

    #[test]
    fn the_readme_support_table_matches_the_code() {
        // Decision: we claim only what we actually wire. A README that drifts
        // from the table above is a false claim, so the build catches it.
        let readme = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md"),
        )
        .expect("README.md");

        for target in targets(Path::new("/usr/local/bin/brain")).unwrap() {
            let label = match target.kind.as_str() {
                "claude-code" => "Claude Code",
                "codex" => "Codex",
                "gemini-cli" => "Gemini CLI",
                "antigravity" => "Antigravity (`agy`)",
                "opencode" => "OpenCode",
                "cursor" => "Cursor",
                other => panic!("undocumented target: {other}"),
            };
            let row = readme
                .lines()
                .find(|line| line.starts_with(&format!("| {label} |")))
                .unwrap_or_else(|| panic!("{label} is wired but missing from the README table"));
            let claimed = format!("{} lifecycle events", target.events.len());
            assert!(
                row.contains(&claimed),
                "README claims the wrong count for {label}: expected \"{claimed}\" in {row:?}"
            );
        }
    }

    #[test]
    fn no_target_wires_a_before_tool_event() {
        // Measured duplication: the before-event repeats the after-event's
        // command text without its result.
        for target in targets(Path::new("/usr/local/bin/brain")).unwrap() {
            for event in target.events {
                assert!(
                    !matches!(*event, "PreToolUse" | "BeforeTool"),
                    "{} still wires {event}",
                    target.kind
                );
            }
        }
    }

    #[test]
    fn shell_quoting_only_kicks_in_when_needed() {
        assert_eq!(shell_quote("/usr/local/bin/brain"), "/usr/local/bin/brain");
        assert_eq!(shell_quote("/Users/a b/brain"), "'/Users/a b/brain'");
        assert_eq!(shell_quote("/it's/brain"), r"'/it'\''s/brain'");
    }

    #[test]
    fn refuses_to_overwrite_unparseable_config() {
        let dir = std::env::temp_dir().join(format!("brain-setup-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(read_json(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wiring_preserves_foreign_hooks_and_replaces_our_own() {
        let dir = std::env::temp_dir().join(format!("brain-setup-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hooks.json");
        std::fs::write(
            &path,
            serde_json::to_string(&json!({
                "hooks": {
                    "Stop": [
                        {"hooks": [{"type": "command", "command": "other-tool --run"}]},
                        {"hooks": [{"type": "command", "command": "/old/brain hook --cli codex --event Stop"}]}
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let target = Target {
            kind: AgentKind::Codex,
            hooks_file: path.clone(),
            events: &["Stop"],
            timeout: HOOK_TIMEOUT_SECS,
            timeout_overrides: &[],
            layout: Layout::Grouped,
            grouped_events: &[],
            mcp_file: None,
            mcp_register: None,
        };
        wire_hooks(&target, Path::new("/new/brain"), true).unwrap();

        let written = read_json(&path).unwrap();
        let groups = written["hooks"]["Stop"].as_array().unwrap();
        assert_eq!(groups.len(), 2, "foreign hook kept, ours replaced not duplicated");
        assert!(groups.iter().any(|g| g["hooks"][0]["command"] == "other-tool --run"));
        assert!(groups
            .iter()
            .any(|g| g["hooks"][0]["command"]
                .as_str()
                .unwrap()
                .starts_with("/new/brain hook")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dropping_an_event_from_the_table_sweeps_its_stale_hook() {
        let dir = std::env::temp_dir().join(format!("brain-setup-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hooks.json");
        std::fs::write(
            &path,
            serde_json::to_string(&json!({
                "hooks": {
                    // Wired by an older version of us, no longer in the table.
                    "PreToolUse": [
                        {"hooks": [{"type": "command", "command": "/old/brain hook --cli codex --event PreToolUse"}]}
                    ],
                    // Someone else's hook on the same dropped event must stay.
                    "SubagentStart": [
                        {"hooks": [{"type": "command", "command": "other-tool --run"}]}
                    ],
                    "Stop": []
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let target = Target {
            kind: AgentKind::Codex,
            hooks_file: path.clone(),
            events: &["Stop"],
            timeout: HOOK_TIMEOUT_SECS,
            timeout_overrides: &[],
            layout: Layout::Grouped,
            grouped_events: &[],
            mcp_file: None,
            mcp_register: None,
        };
        let changes = wire_hooks(&target, Path::new("/new/brain"), true).unwrap();
        assert!(changes[0].detail.contains("PreToolUse"), "sweep should be reported");

        let written = read_json(&path).unwrap();
        assert!(
            written["hooks"].get("PreToolUse").is_none(),
            "an emptied event should not linger as a dead key"
        );
        assert_eq!(
            written["hooks"]["SubagentStart"][0]["hooks"][0]["command"],
            "other-tool --run",
            "a foreign hook on a dropped event must survive"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_namespaced_config_gets_both_entry_shapes() {
        let dir = std::env::temp_dir().join(format!("brain-setup-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hooks.json");
        // A foreign tool already owns its own namespace here.
        std::fs::write(
            &path,
            serde_json::to_string(&json!({
                "other-tool": {"Stop": [{"type": "command", "command": "other --run"}]}
            }))
            .unwrap(),
        )
        .unwrap();

        let target = Target {
            kind: AgentKind::parse("antigravity"),
            layout: Layout::Namespaced,
            grouped_events: &["PostToolUse"],
            hooks_file: path.clone(),
            events: &["PostToolUse", "Stop"],
            timeout: HOOK_TIMEOUT_SECS,
            timeout_overrides: &[],
            mcp_file: None,
            mcp_register: None,
        };
        wire_hooks(&target, Path::new("/new/brain"), true).unwrap();

        let written = read_json(&path).unwrap();
        // Ours lives under its own namespace, theirs is untouched.
        assert_eq!(written["other-tool"]["Stop"][0]["command"], "other --run");

        // A tool event takes the grouped shape...
        let tool = &written[MCP_SERVER_NAME]["PostToolUse"][0];
        assert!(tool["hooks"][0]["command"].as_str().unwrap().contains("brain hook"));
        // ...and a non-tool event is a bare handler.
        let stop = &written[MCP_SERVER_NAME]["Stop"][0];
        assert!(stop.get("hooks").is_none(), "Stop must not be wrapped: {stop}");
        assert!(stop["command"].as_str().unwrap().contains("brain hook"));
        assert_eq!(stop["timeout"], HOOK_TIMEOUT_SECS, "antigravity reads seconds");

        // Re-running replaces rather than duplicating, in both shapes.
        wire_hooks(&target, Path::new("/new/brain"), true).unwrap();
        let again = read_json(&path).unwrap();
        assert_eq!(again[MCP_SERVER_NAME]["Stop"].as_array().unwrap().len(), 1);
        assert_eq!(again[MCP_SERVER_NAME]["PostToolUse"].as_array().unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_flat_config_keeps_its_schema_version_and_foreign_entries() {
        let dir = std::env::temp_dir().join(format!("brain-setup-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hooks.json");
        std::fs::write(
            &path,
            serde_json::to_string(&json!({
                "version": 1,
                "hooks": {
                    "stop": [{"command": "other-tool summarize"}],
                    "afterFileEdit": [{"command": "other-tool file-edit"}]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let target = Target {
            kind: AgentKind::parse("cursor"),
            layout: Layout::Flat,
            grouped_events: &[],
            hooks_file: path.clone(),
            events: &["stop", "afterFileEdit"],
            timeout: HOOK_TIMEOUT_SECS,
            timeout_overrides: &[],
            mcp_file: None,
            mcp_register: None,
        };
        wire_hooks(&target, Path::new("/new/brain"), true).unwrap();

        let written = read_json(&path).unwrap();
        assert_eq!(written["version"], 1, "the schema version must survive");

        for event in ["stop", "afterFileEdit"] {
            let entries = written["hooks"][event].as_array().unwrap();
            assert_eq!(entries.len(), 2, "{event}: foreign entry lost or ours duplicated");
            // Ours is a bare handler, never wrapped.
            let ours = entries.iter().find(|e| is_ours(e)).unwrap();
            assert!(ours.get("hooks").is_none(), "{event}: Cursor takes flat entries");
            assert!(ours["command"].as_str().unwrap().contains("brain hook"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_two_targets_collide() {
        // Five CLIs, five config shapes, all written by one code path: the
        // realistic failure is a copy-pasted target pointing at another CLI's
        // file, which would silently wire the wrong tool.
        let targets = targets(Path::new("/usr/local/bin/brain")).unwrap();
        let mut kinds = std::collections::HashSet::new();
        let mut files = std::collections::HashSet::new();
        for target in &targets {
            assert!(kinds.insert(target.kind.as_str().to_string()), "duplicate kind");
            assert!(
                files.insert(target.hooks_file.clone()),
                "{} shares a config file with another target",
                target.kind
            );
            assert!(!target.events.is_empty(), "{} wires no events", target.kind);
            assert!(
                target.mcp_file.is_none() || target.mcp_register.is_none(),
                "{} has two MCP registration paths",
                target.kind
            );
        }
    }

    #[test]
    fn ownership_is_recognized_in_both_entry_shapes() {
        assert!(is_ours(&json!({"type": "command", "command": "/x/brain hook --cli codex"})));
        assert!(is_ours(&json!({"hooks": [{"command": "/x/brain hook --cli codex"}]})));
        assert!(!is_ours(&json!({"type": "command", "command": "other-tool"})));
        assert!(!is_ours(&json!({"hooks": [{"command": "another-tool capture"}]})));
    }

    #[test]
    fn antigravity_wires_no_event_its_own_docs_do_not_list() {
        // Its embedded docs enumerate exactly these five.
        const DOCUMENTED: [&str; 5] =
            ["PreInvocation", "PostInvocation", "PreToolUse", "PostToolUse", "Stop"];
        let target = targets(Path::new("/usr/local/bin/brain"))
            .unwrap()
            .into_iter()
            .find(|t| t.kind.as_str() == "antigravity")
            .expect("antigravity target");
        for event in target.events {
            assert!(DOCUMENTED.contains(event), "{event} is not a documented agy event");
        }
    }

    #[test]
    fn dry_run_writes_nothing() {
        let dir = std::env::temp_dir().join(format!("brain-setup-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hooks.json");
        let target = Target {
            kind: AgentKind::Codex,
            hooks_file: path.clone(),
            events: &["Stop"],
            timeout: HOOK_TIMEOUT_SECS,
            timeout_overrides: &[],
            layout: Layout::Grouped,
            grouped_events: &[],
            mcp_file: None,
            mcp_register: None,
        };
        let changes = wire_hooks(&target, Path::new("/new/brain"), false).unwrap();
        assert!(!path.exists(), "dry run must not create the file");
        assert!(changes[0].detail.starts_with("would write"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
