//! How the host CLI was invoked, and what that means for injection.
//!
//! A headless one-shot run (`claude -p`, `codex exec`) is usually not a person
//! working — it is an orchestrated step: a reviewer, a judge, a summarizer.
//! Injecting this project's memory into such a run **contaminates it**: an
//! adversarial reviewer that inherits the author's narrative is no longer
//! independent, and the contamination is invisible in the output.
//!
//! So headless runs get no automatic injection. They may still capture, tagged
//! so the quality floor can weigh them accordingly, and an orchestrator that
//! wants a completely clean room sets [`SILENT_ENV`].

use std::path::Path;

/// Public contract: no injection AND no capture for this process tree.
///
/// Documented and stable — orchestrators are expected to set it directly, so
/// it must not be renamed for internal convenience.
pub const SILENT_ENV: &str = "ROLEPOD_BRAIN_SILENT";

/// How far up the process tree to walk.
///
/// The host CLI is normally our direct parent, but who started the CLI matters
/// too: a codex served to another agent's plugin sits four processes below the
/// plugin that owns it. The whole chain comes from one `ps`, so walking it to
/// the root costs nothing more; the bound only guards against a cycle in a
/// table read mid-change.
///
/// Unix only, because only there is there a process tree to walk.
#[cfg(unix)]
const MAX_ANCESTORS: usize = 32;

/// Where Claude Code keeps the plugins it runs. A process started from here
/// is an agent's plugin - the codex companion's `app-server` broker, say -
/// and whatever CLI it hosts is working for that agent, not for a person.
const AGENT_PLUGIN_DIR: &str = "/.claude/plugins/";

/// What kind of run produced this event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invocation {
    /// A person at a terminal.
    Interactive,
    /// A one-shot programmatic run.
    Headless,
}

impl Invocation {
    #[must_use]
    pub const fn is_headless(self) -> bool {
        matches!(self, Self::Headless)
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Headless => "headless",
        }
    }
}

/// Read back a stored classification.
#[must_use]
pub fn parse(raw: &str) -> Invocation {
    if raw == Invocation::Headless.as_str() {
        Invocation::Headless
    } else {
        Invocation::Interactive
    }
}

/// Has an orchestrator asked us to stay completely out of the way?
#[must_use]
pub fn silenced() -> bool {
    std::env::var_os(SILENT_ENV).is_some_and(|value| {
        // Present-but-empty means unset, not enabled: an exported-but-blank
        // variable is a scripting accident, not an instruction.
        !value.is_empty() && value != "0"
    })
}

/// Classify the current invocation by inspecting the host CLI's own arguments.
///
/// Detection reads the process tree rather than a TTY, and that is deliberate:
/// a hook's own stdio is redirected in BOTH modes, so a TTY test detects
/// nothing about the CLI — verified by probing real runs, where interactive and
/// headless alike showed no terminal on any descriptor. The CLI's argv is the
/// signal that actually differs: `claude` alone versus `claude -p …`.
#[must_use]
pub fn classify() -> Invocation {
    classify_chain(&ancestors())
}

/// The host CLI's own arguments decide first; who started the CLI decides
/// the rest.
///
/// `codex app-server` reads as interactive from its argv - an IDE extension
/// drives one with a person behind it - and so did every review Claude Code
/// delegated through the codex plugin on a real machine: eleven sessions,
/// nearly two thousand captures, each summarized with a model call and each
/// offered to the next session as "what codex was doing". The plugin's broker
/// is the server's grandparent, and no environment variable set per run can
/// reach a server started once at Claude Code's launch. A CLI hosted by
/// another agent, or by an agent's plugin, is a delegate: headless, however
/// it was invoked. An unknown host keeps the interactive default, so an IDE
/// nobody has seen still gets its injection.
fn classify_chain(chain: &[String]) -> Invocation {
    let Some((host, verdict)) = chain
        .iter()
        .enumerate()
        .find_map(|(index, argv)| classify_argv(argv).map(|verdict| (index, verdict)))
    else {
        return Invocation::Interactive;
    };
    if verdict.is_headless() || hosted_by_an_agent(&chain[host + 1..]) {
        Invocation::Headless
    } else {
        Invocation::Interactive
    }
}

/// Is any of these processes another coding agent, or a plugin of one?
fn hosted_by_an_agent(above: &[String]) -> bool {
    above
        .iter()
        .any(|argv| classify_argv(argv).is_some() || argv.contains(AGENT_PLUGIN_DIR))
}

/// Decide from one process's argv, if it is a host CLI we recognize.
///
/// Only the CLI's OWN arguments count. An ancestor shell can carry `-p` for
/// unrelated reasons — a real login shell on this machine runs
/// `/bin/bash --noprofile --norc -p -c …` — so matching a bare flag anywhere in
/// the tree would call every session headless.
fn classify_argv(argv: &str) -> Option<Invocation> {
    let mut words = argv.split_whitespace();
    let program = words.next().map(basename)?;
    let args: Vec<&str> = words.collect();

    let headless = match program {
        "claude" => args.iter().any(|arg| matches!(*arg, "-p" | "--print")),
        // `codex exec` is the one-shot form; the bare command is the TUI.
        "codex" => args.first().is_some_and(|arg| *arg == "exec"),
        // Antigravity's one-shot flag, same family of intent.
        "agy" => args
            .iter()
            .any(|arg| matches!(*arg, "-p" | "--print" | "--prompt") || arg.starts_with("-p=")),
        _ => return None,
    };

    Some(if headless { Invocation::Headless } else { Invocation::Interactive })
}

/// Command lines of our ancestors, nearest first.
///
/// One `ps` for the whole table, then the walk happens in memory. The obvious
/// implementation — one `ps` per ancestor — cost 70 ms on a real hook and blew
/// the 50 ms budget outright. Process spawns are the expensive thing here, so
/// there is exactly one.
#[cfg(unix)]
fn ancestors() -> Vec<String> {
    let Ok(output) = std::process::Command::new("ps").args(["-Ao", "pid=,ppid=,args="]).output()
    else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&output.stdout);

    let mut table = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some((pid, rest)) = line.split_once(char::is_whitespace) else { continue };
        let rest = rest.trim_start();
        let Some((ppid, argv)) = rest.split_once(char::is_whitespace) else { continue };
        let (Ok(pid), Ok(ppid)) = (pid.parse::<u32>(), ppid.trim().parse::<u32>()) else {
            continue;
        };
        table.insert(pid, (ppid, argv.trim().to_string()));
    }

    let mut out = Vec::new();
    let mut pid = std::os::unix::process::parent_id();
    for _ in 0..MAX_ANCESTORS {
        let Some((parent, argv)) = table.get(&pid) else { break };
        out.push(argv.clone());
        if *parent <= 1 {
            break;
        }
        pid = *parent;
    }
    out
}

/// Windows has no `ps` and no way to read a parent pid without `unsafe`.
///
/// The honest answer there is that we do not know, and the caller is built for
/// that: an empty ancestry means invocation is classified from the event alone
/// rather than from who started us. What is lost is telling a headless run
/// apart from an interactive one by inspecting the process tree - which
/// affects how a session is labelled, not whether it is captured.
///
/// The alternatives were both worse. `Get-CimInstance` costs about a second of
/// PowerShell startup inside a 50 ms hook budget, and the Win32 tool help API
/// needs `unsafe`, which this crate forbids.
#[cfg(windows)]
fn ancestors() -> Vec<String> {
    Vec::new()
}

fn basename(path: &str) -> &str {
    Path::new(path).file_name().and_then(std::ffi::OsStr::to_str).unwrap_or(path)
}

/// Serializes the tests that set process-wide environment variables.
///
/// The test runner uses threads, and an environment variable is shared by
/// all of them: a test that silences capture silences every test running
/// beside it. Passing today is a coincidence of scheduling.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_print_mode_is_headless() {
        // Captured verbatim from a real run.
        assert_eq!(
            classify_argv("claude -p --model haiku Reply with exactly: OK"),
            Some(Invocation::Headless)
        );
        assert_eq!(classify_argv("claude --print 'do a thing'"), Some(Invocation::Headless));
    }

    #[test]
    fn interactive_claude_is_not_headless() {
        // Also captured verbatim: an interactive session's argv is bare.
        assert_eq!(classify_argv("claude"), Some(Invocation::Interactive));
        assert_eq!(
            classify_argv("/usr/local/bin/claude --model opus"),
            Some(Invocation::Interactive)
        );
    }

    #[test]
    fn codex_exec_is_headless_but_the_tui_is_not() {
        assert_eq!(
            classify_argv("codex exec -m gpt-5.6-luna 'summarize'"),
            Some(Invocation::Headless)
        );
        assert_eq!(classify_argv("codex"), Some(Invocation::Interactive));
        assert_eq!(classify_argv("codex --model gpt-5.6"), Some(Invocation::Interactive));
    }

    #[test]
    fn an_unrelated_ancestors_flags_are_never_read_as_the_clis() {
        // The real login shell on the author's machine. Matching `-p` anywhere
        // in the tree would call every interactive session headless.
        assert_eq!(classify_argv("/bin/bash --noprofile --norc -p -c export SHELL"), None);
        assert_eq!(classify_argv("-/bin/zsh -l"), None);
        assert_eq!(classify_argv("/sbin/launchd"), None);
        assert_eq!(classify_argv(""), None);
    }

    #[test]
    fn agy_print_forms_are_headless() {
        // agy rejects `-p value`; the prompt attaches to the flag.
        assert_eq!(
            classify_argv("agy --dangerously-skip-permissions -p=hello"),
            Some(Invocation::Headless)
        );
        assert_eq!(classify_argv("agy"), Some(Invocation::Interactive));
    }

    #[test]
    fn a_codex_served_to_another_agents_plugin_is_a_delegate() {
        // The chain under a real hook, nearest first: the shell codex runs
        // hooks in, the codex binary, the npm wrapper, and the broker that
        // Claude Code's codex plugin starts once at launch.
        let chain: Vec<String> = [
            "/opt/codex-resources/zsh/bin/zsh -lc brain hook --cli codex --event SessionStart",
            "/home/me/.nvm/lib/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex app-server",
            "node /home/me/.nvm/bin/codex app-server",
            "/home/me/.nvm/bin/node /home/me/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/app-server-broker.mjs serve",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(classify_chain(&chain), Invocation::Headless);

        // The same server driven by an IDE has a person behind it.
        let ide: Vec<String> = [
            "/opt/codex-resources/zsh/bin/zsh -lc brain hook --cli codex --event SessionStart",
            "/home/me/.nvm/lib/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex app-server",
            "/Applications/Visual Studio Code.app/Contents/Frameworks/Code Helper (Plugin).app/Contents/MacOS/Code Helper (Plugin) --type=utility",
            "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(classify_chain(&ide), Invocation::Interactive);
    }

    #[test]
    fn a_cli_started_by_another_agent_is_a_delegate() {
        // Claude Code's shell tool starting codex: no `exec`, but a coding
        // agent is its grandparent.
        let chain: Vec<String> =
            ["/bin/sh -c codex app-server", "codex app-server", "claude --model opus"]
                .map(String::from)
                .to_vec();
        assert_eq!(classify_chain(&chain), Invocation::Headless);

        // A person's own terminal above their own CLI is not an agent, and
        // `login -pf` is not `claude -p`.
        let person: Vec<String> = [
            "-zsh",
            "/usr/local/bin/codex",
            "login -pf me",
            "/System/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal",
            "/sbin/launchd",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(classify_chain(&person), Invocation::Interactive);

        // No host CLI at all - the test runner, a cron job - is not headless
        // either: that would silence injection for everything unknown.
        assert_eq!(classify_chain(&["/bin/sh".to_string()]), Invocation::Interactive);
        assert_eq!(classify_chain(&[]), Invocation::Interactive);
    }

    #[test]
    fn a_stored_classification_round_trips() {
        for invocation in [Invocation::Headless, Invocation::Interactive] {
            assert_eq!(parse(invocation.as_str()), invocation);
        }
        // An unreadable value must not silently become "headless" and
        // suppress every injection for that session.
        assert_eq!(parse("garbage"), Invocation::Interactive);
    }

    #[test]
    fn silence_needs_a_real_value() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var(SILENT_ENV);
        assert!(!silenced());
        std::env::set_var(SILENT_ENV, "");
        assert!(!silenced(), "an exported-but-blank variable is an accident");
        std::env::set_var(SILENT_ENV, "0");
        assert!(!silenced());
        std::env::set_var(SILENT_ENV, "1");
        assert!(silenced());
        std::env::remove_var(SILENT_ENV);
    }
}
