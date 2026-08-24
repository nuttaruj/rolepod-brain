//! The summarizer ladder — borrowed capability, never borrowed credentials.
//!
//! There is no API-key field in this product and no token in its config. When
//! we need a model we shell out to a CLI the user is already signed into,
//! through that vendor's own supported headless entry point. If every rung
//! fails we fall back to rule-based output, which is a permanent first-class
//! mode rather than a degraded one — the ladder loses quality, never data.
//!
//! Two hazards this module exists to contain:
//!
//! 1. **Recursion.** A headless CLI run fires that CLI's lifecycle hooks,
//!    which call `brain hook` straight back. Every child is spawned with
//!    [`crate::hook::WORKER_ENV`] set so capture short-circuits.
//! 2. **Retry storms.** A CLI that is rate-limited stays rate-limited for a
//!    while. Repeated failures open a circuit breaker and we stay quietly on
//!    rule-based output until the cooldown expires.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::hook::WORKER_ENV;
use crate::store::Store;

/// Consecutive failures before a CLI is taken out of rotation.
const FAILURES_BEFORE_COOLDOWN: i64 = 3;
/// How long a tripped CLI stays out.
const COOLDOWN: Duration = Duration::from_secs(30 * 60);
/// Ceiling on one prompt. Chunking happens above this.
pub const PROMPT_MAX_BYTES: usize = 24 * 1024;
/// Wall-clock ceiling for one summarizer call.
const CALL_TIMEOUT: Duration = Duration::from_secs(180);

/// Rungs a single call may try before giving up on models entirely.
///
/// Two, not "all of them": a prompt the first CLI could not use is often one
/// none of them can, and cascading through every installed CLI would spend a
/// model call each to learn that. One retry covers the case that actually
/// happens - this vendor is rate-limited today, that one is not.
const MAX_RUNGS_PER_CALL: usize = 2;

/// How to reach one CLI's cheap tier.
pub struct CliSpec {
    /// `source.cli` value this spec serves.
    pub cli: &'static str,
    /// Executable name, resolved on `PATH`.
    pub program: &'static str,
    /// Model identifier passed to that CLI.
    pub model: &'static str,
    /// Whether the answer arrives on stdout or in a file we name.
    pub output: OutputMode,
    /// Arguments before the prompt. `{model}` and `{out}` are substituted.
    pub args: &'static [&'static str],
}

/// Where a CLI puts its final answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// The answer is the process's stdout.
    Stdout,
    /// The answer is written to a file we pass in; stdout is progress noise.
    File,
}

/// The model map. One table, checked by `brain doctor`, because model ids rot
/// when a CLI upgrades and a silently-wrong id looks exactly like an outage.
///
/// Every entry here was invoked for real before it was written down.
pub const SPECS: &[CliSpec] = &[
    CliSpec {
        cli: "claude-code",
        program: "claude",
        model: "haiku",
        output: OutputMode::Stdout,
        // A summarizer call is text in, text out - it needs no tool, and a
        // bare `-p` loads the user's whole MCP roster anyway. That is not
        // just startup cost: a worker with tools can ASK for one, and a
        // headless session's permission prompt goes to the user's paired
        // phone - a push notification about a browser tool, from a
        // background job they never see, mid-whatever they were doing.
        // `--strict-mcp-config` with no config empties the MCP list;
        // `--tools=` (the = matters: the flag is variadic and would swallow
        // the prompt) empties the built-in set. Nothing loaded, nothing to
        // ask about.
        args: &["-p", "--model", "{model}", "--strict-mcp-config", "--tools="],
    },
    CliSpec {
        cli: "codex",
        program: "codex",
        model: "gpt-5.6-luna",
        output: OutputMode::File,
        // stdout carries hook chatter and a token counter, so the answer is
        // read from the file rather than scraped out of the stream.
        args: &["exec", "-m", "{model}", "--skip-git-repo-check", "-o", "{out}"],
    },
    CliSpec {
        // Must match what hooks write as `source.cli`, not what the binary is
        // called: this string is looked up against captured events.
        cli: "gemini-cli",
        program: "gemini",
        model: "flash",
        output: OutputMode::Stdout,
        args: &["-m", "{model}", "--skip-trust", "-p"],
    },
];

/// Which CLI a call actually used, for logging and health accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tier {
    /// A host CLI's cheap tier.
    Cli(String),
    /// No model was reachable; the caller must produce rule-based output.
    RuleBased,
}

/// Picks a rung and runs it.
pub struct Ladder<'a> {
    store: &'a Store,
    mode: String,
    /// Per-CLI model overrides from config; a CLI not named keeps its
    /// spec's cheap default.
    models: std::collections::HashMap<String, String>,
    timeout: Duration,
}

impl<'a> Ladder<'a> {
    #[must_use]
    pub fn new(store: &'a Store, config: &crate::config::SummarizerConfig) -> Self {
        Self {
            store,
            mode: config.mode.clone(),
            models: config.models.clone(),
            timeout: CALL_TIMEOUT,
        }
    }

    /// The same ladder on a shorter leash.
    ///
    /// Consolidation runs detached and can afford three minutes. A call made
    /// while someone waits for a search result cannot: past a few seconds the
    /// unranked answer they already had is the better one.
    #[must_use]
    pub fn clone_with_timeout(&self, limit: Duration) -> Ladder<'a> {
        Ladder {
            store: self.store,
            mode: self.mode.clone(),
            models: self.models.clone(),
            timeout: limit,
        }
    }

    /// Is any model allowed at all?
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.mode != "off"
    }

    /// Run `prompt`, preferring the CLI that produced the observations.
    ///
    /// Returns the model's text and which tier answered, or [`Tier::RuleBased`]
    /// with no text when every rung is unavailable. Never returns an error for
    /// an unavailable model: that is an expected state, not a fault.
    ///
    /// # Errors
    /// Returns an error only when the health table cannot be read or written.
    pub fn run<F>(&self, prompt: &str, preferred_cli: &str, usable: F) -> Result<(Tier, String)>
    where
        F: Fn(&str) -> bool,
    {
        if !self.enabled() {
            return Ok((Tier::RuleBased, String::new()));
        }

        let mut attempts = 0usize;
        for spec in self.order(preferred_cli) {
            if attempts >= MAX_RUNGS_PER_CALL {
                break;
            }
            if !self.available(spec)? {
                continue;
            }
            attempts += 1;

            match invoke(spec, self.model_for(spec), prompt, self.timeout) {
                Ok(text) if usable(&text) => {
                    self.store.record_summarizer_success(spec.cli)?;
                    return Ok((Tier::Cli(spec.cli.to_string()), text));
                }
                Ok(text) => {
                    // A CLI whose quota is gone often exits 0 and prints a
                    // banner. That is a failure of this rung, not of the
                    // ladder: it advances, and it counts toward this rung's
                    // breaker exactly like a crash would. Treating it as
                    // "no model available" was the bug - it meant a
                    // rate-limited Claude never fell through to Codex.
                    let detail = unusable_reason(&text);
                    self.store.record_summarizer_failure(spec.cli, &detail)?;
                }
                Err(error) => {
                    self.store
                        .record_summarizer_failure(spec.cli, &format!("{error:#}"))?;
                }
            }
        }
        Ok((Tier::RuleBased, String::new()))
    }

    /// Rungs to try, in order.
    fn order(&self, preferred_cli: &str) -> Vec<&'static CliSpec> {
        // A pinned mode means exactly one rung: the user asked for that CLI,
        // and silently using another would spend the wrong subscription.
        if self.mode != "auto" {
            let pinned = Self::normalize_cli(&self.mode);
            return SPECS.iter().filter(|spec| spec.cli == pinned).collect();
        }
        let preferred_cli = Self::normalize_cli(preferred_cli);
        let mut order: Vec<&CliSpec> = SPECS.iter().filter(|s| s.cli == preferred_cli).collect();
        order.extend(SPECS.iter().filter(|s| s.cli != preferred_cli));
        order
    }

    /// Accept the name a person would write for a CLI we know by another.
    ///
    /// `mode = "gemini"` is what the tool is called; `gemini-cli` is what its
    /// hooks write. Refusing the shorter one would turn a reasonable config
    /// into a summarizer that silently never runs.
    fn normalize_cli(raw: &str) -> &str {
        match raw {
            "gemini" => "gemini-cli",
            other => other,
        }
    }

    /// The model this rung should run: the user's override, or the spec's
    /// cheap default.
    #[must_use]
    pub fn model_for(&self, spec: &CliSpec) -> &str {
        self.models.get(spec.cli).map_or(spec.model, String::as_str)
    }

    /// Is this CLI installed, and not in a cooldown?
    fn available(&self, spec: &CliSpec) -> Result<bool> {
        if !installed(spec.program) {
            return Ok(false);
        }
        Ok(!self.store.summarizer_in_cooldown(spec.cli)?)
    }
}

/// Describe an unusable answer for the health report, without storing it.
///
/// The text is whatever the CLI printed instead of an answer, which may be a
/// login prompt or a quota notice - useful to see, but not worth keeping at
/// length in a breaker row.
fn unusable_reason(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return "empty response".to_string();
    }
    let first = text.lines().find(|line| !line.trim().is_empty()).unwrap_or("").trim();
    format!("unusable answer: {}", crate::sanitize::truncate(first, 160))
}

/// Is a program on `PATH`?
#[must_use]
pub fn installed(program: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        let candidate = dir.join(program);
        candidate.is_file() || candidate.with_extension("exe").is_file()
    })
}

/// Run one CLI once and return its answer.
fn invoke(spec: &CliSpec, model: &str, prompt: &str, timeout: Duration) -> Result<String> {
    // A prompt that begins with a dash would be read as a flag by whichever
    // CLI receives it. The usual guard is a `--` separator, but not every
    // spec here can take one - gemini's prompt is the value of `-p` - so the
    // invariant is enforced on the prompt instead of worked around in argv.
    anyhow::ensure!(
        !prompt.trim_start().starts_with('-'),
        "a prompt may not begin with a dash; it would be parsed as a flag"
    );
    anyhow::ensure!(
        prompt.len() <= PROMPT_MAX_BYTES,
        "prompt is {} bytes, over the {PROMPT_MAX_BYTES}-byte call ceiling",
        prompt.len()
    );

    let out_file: Option<PathBuf> = (spec.output == OutputMode::File).then(|| {
        std::env::temp_dir().join(format!("brain-summary-{}.txt", ulid::Ulid::new()))
    });

    let args: Vec<String> = spec
        .args
        .iter()
        .map(|arg| match *arg {
            "{model}" => model.to_string(),
            "{out}" => out_file.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
            other => other.to_string(),
        })
        .collect();

    let mut command = Command::new(spec.program);
    command
        .args(&args)
        .arg(prompt)
        // Break the hook recursion before the child can start.
        .env(WORKER_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Run somewhere inert: a headless CLI started inside the user's repo
        // may read project instructions we neither need nor want to pay for.
        .current_dir(std::env::temp_dir());

    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {}", spec.program))?;

    let result = wait_with_timeout(&mut child, timeout)?;

    let answer = match (&out_file, spec.output) {
        (Some(path), OutputMode::File) => {
            let text = std::fs::read_to_string(path).unwrap_or_default();
            let _ = std::fs::remove_file(path);
            text
        }
        _ => result.stdout.clone(),
    };

    anyhow::ensure!(
        result.success,
        "{} exited with {}: {}",
        spec.program,
        result.code_display(),
        result.stderr.trim().chars().take(200).collect::<String>()
    );

    Ok(answer)
}

struct CallResult {
    success: bool,
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl CallResult {
    fn code_display(&self) -> String {
        self.code.map_or_else(|| "signal".to_string(), |code| code.to_string())
    }
}

/// Wait for a child, killing it if it overruns.
///
/// A summarizer that hangs must not hold a consolidation run open forever;
/// the events stay unconsolidated and the next trigger tries again.
fn wait_with_timeout(child: &mut std::process::Child, limit: Duration) -> Result<CallResult> {
    // Drain both pipes on their own threads for the whole wait. Reading them
    // only after the child exits deadlocks as soon as it writes more than a
    // pipe buffer holds: it blocks on the write, never exits, and this
    // reports a timeout that never happened.
    let drain = |pipe: Option<std::process::ChildStdout>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(mut pipe) = pipe {
                use std::io::Read;
                let _ = pipe.read_to_string(&mut text);
            }
            text
        })
    };
    let out = drain(child.stdout.take());
    let err = std::thread::spawn({
        let pipe = child.stderr.take();
        move || {
            let mut text = String::new();
            if let Some(mut pipe) = pipe {
                use std::io::Read;
                let _ = pipe.read_to_string(&mut text);
            }
            text
        }
    });

    let start = std::time::Instant::now();
    loop {
        match child.try_wait().context("poll summarizer")? {
            Some(status) => {
                return Ok(CallResult {
                    success: status.success(),
                    code: status.code(),
                    // The child is gone, so both pipes are at EOF and these
                    // threads are already finishing.
                    stdout: out.join().unwrap_or_default(),
                    stderr: err.join().unwrap_or_default(),
                });
            }
            None => {
                if start.elapsed() > limit {
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!("timed out after {}s", limit.as_secs());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// How long a tripped CLI stays out of rotation.
#[must_use]
pub const fn cooldown() -> Duration {
    COOLDOWN
}

/// Failures tolerated before the breaker opens.
#[must_use]
pub const fn failure_threshold() -> i64 {
    FAILURES_BEFORE_COOLDOWN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(mode: &str) -> crate::config::SummarizerConfig {
        crate::config::SummarizerConfig {
            mode: mode.to_string(),
            models: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn a_child_that_writes_more_than_a_pipe_holds_still_finishes() {
        // Reading the pipes only after the child exits deadlocks: the child
        // blocks writing into a full pipe, never exits, and the wait reports
        // a timeout that never happened. A summary is usually small, but a
        // CLI printing a banner or a verbose trace is not.
        let mut child = std::process::Command::new("sh")
            .args(["-c", "head -c 300000 /dev/zero | tr '\\0' 'x'"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let result = wait_with_timeout(&mut child, Duration::from_secs(10))
            .expect("a child that writes a lot is not a timeout");
        assert!(result.success);
        assert_eq!(result.stdout.len(), 300_000, "output was truncated");
    }

    /// The quality knob: a named CLI runs the model the user chose, and
    /// everything unnamed keeps its cheap default - so paying for better
    /// summaries on one CLI can never hand another CLI a model name it
    /// does not recognise.
    #[test]
    fn a_model_override_applies_only_to_the_cli_it_names() {
        let store = Store::open_memory().unwrap();
        let mut config = config("auto");
        config.models.insert("claude-code".to_string(), "sonnet".to_string());
        let ladder = Ladder::new(&store, &config);

        let claude = SPECS.iter().find(|spec| spec.cli == "claude-code").unwrap();
        let codex = SPECS.iter().find(|spec| spec.cli == "codex").unwrap();
        assert_eq!(ladder.model_for(claude), "sonnet");
        assert_eq!(ladder.model_for(codex), "gpt-5.6-luna", "an unnamed CLI keeps its default");
    }

    /// A worker must not be able to raise a permission prompt.
    ///
    /// A headless claude session with tools available can ask to use one,
    /// and that prompt lands on the user's paired phone as a push
    /// notification from a background job. Observed for real: "Claude needs
    /// your permission: Javascript Tool" on a lock screen, from a
    /// consolidation the user never saw. The flags below are what make that
    /// impossible rather than merely unlikely.
    #[test]
    fn the_claude_worker_is_spawned_with_nothing_to_ask_about() {
        let spec = SPECS.iter().find(|spec| spec.cli == "claude-code").expect("claude spec");
        assert!(spec.args.contains(&"--strict-mcp-config"), "MCP servers must not load");
        assert!(
            spec.args.contains(&"--tools="),
            "built-in tools must be disabled, and with `=`: the flag is \
             variadic, and a bare --tools \"\" swallows the prompt argument"
        );
    }

    /// Every summarizer rung must be named the way hooks name it.
    ///
    /// `SPECS.cli` is looked up against `source.cli` on captured events, so a
    /// spec named after its binary instead of its wire name silently stops
    /// the ladder from ever preferring the CLI the user is actually in. That
    /// is invisible in every test that does not compare the two lists.
    #[test]
    fn a_prompt_that_would_be_read_as_a_flag_is_refused() {
        let spec = &SPECS[0];
        let error = invoke(spec, spec.model, "--help me", CALL_TIMEOUT).unwrap_err();
        assert!(error.to_string().contains("begin with a dash"), "{error}");
    }

    #[test]
    fn every_spec_is_named_the_way_setup_names_the_same_cli() {
        let exe = std::path::PathBuf::from("/usr/local/bin/brain");
        let wired: Vec<String> = crate::setup::targets(&exe)
            .expect("targets")
            .iter()
            .map(|target| target.kind.as_str().to_string())
            .collect();
        for spec in SPECS {
            assert!(
                wired.iter().any(|name| name == spec.cli),
                "summarizer knows `{}`, which no host CLI writes: {wired:?}",
                spec.cli
            );
        }
    }

    #[test]
    fn every_spec_substitutes_its_placeholders() {
        for spec in SPECS {
            for arg in spec.args {
                assert!(
                    !arg.contains('{') || matches!(*arg, "{model}" | "{out}"),
                    "unknown placeholder in {}: {arg}",
                    spec.cli
                );
            }
            if spec.output == OutputMode::File {
                assert!(
                    spec.args.contains(&"{out}"),
                    "{} reads from a file but never receives one",
                    spec.cli
                );
            }
        }
    }

    #[test]
    fn auto_mode_prefers_the_cli_that_saw_the_events() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("auto"));
        let order = ladder.order("codex");
        assert_eq!(order[0].cli, "codex");
        assert_eq!(order.len(), SPECS.len(), "every other CLI stays as a fallback");
    }

    #[test]
    fn a_pinned_mode_never_silently_uses_another_subscription() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("claude-code"));
        let order = ladder.order("codex");
        assert_eq!(order.len(), 1);
        assert_eq!(order[0].cli, "claude-code");
    }

    #[test]
    fn off_mode_never_reaches_for_a_model() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("off"));
        assert!(!ladder.enabled());
        let (tier, text) = ladder.run("anything", "claude-code", |_| true).unwrap();
        assert_eq!(tier, Tier::RuleBased);
        assert!(text.is_empty());
    }

    #[test]
    fn the_breaker_opens_after_repeated_failures_and_closes_on_success() {
        let store = Store::open_memory().unwrap();
        for _ in 0..failure_threshold() {
            store.record_summarizer_failure("codex", "rate limited").unwrap();
        }
        assert!(store.summarizer_in_cooldown("codex").unwrap());
        store.record_summarizer_success("codex").unwrap();
        assert!(!store.summarizer_in_cooldown("codex").unwrap());
    }

    #[test]
    fn an_untripped_cli_is_available() {
        let store = Store::open_memory().unwrap();
        store.record_summarizer_failure("codex", "one blip").unwrap();
        assert!(!store.summarizer_in_cooldown("codex").unwrap());
    }

    #[test]
    fn observed_failure_shapes_are_classified_correctly() {
        // Provoked for real on 2026-08-23 with a bogus model id: BOTH CLIs
        // exit non-zero, which the ladder already treats as a hard failure and
        // advances past. Specimens kept so a future CLI change that turns
        // these into exit-0 output is caught by the soft path instead of
        // silently ending consolidation.
        let claude_bad_model =
            "There's an issue with the selected model (no-such-model-xyz). It may not exist";
        let codex_bad_model = "ERROR codex_models_manager::manager: failed to load models cache";
        for text in [claude_bad_model, codex_bad_model] {
            assert!(
                unusable_reason(text).starts_with("unusable answer:"),
                "should be reported as unusable if it ever arrives with exit 0"
            );
        }

        // NOT verified: what either CLI prints when a usage limit is
        // exhausted. It could not be provoked on demand, so the assumed shape
        // - exit 0 with a banner - is exactly that, assumed. The ladder no
        // longer depends on knowing: hard and soft failures both advance.
        let assumed_limit_banner = "You have reached your usage limit.";
        assert!(unusable_reason(assumed_limit_banner).contains("usage limit"));
    }

    #[test]
    fn a_soft_failure_is_described_without_being_stored_whole() {
        let banner = "You have reached your usage limit.\nResets at 3pm.\n";
        let reason = unusable_reason(banner);
        assert!(reason.starts_with("unusable answer:"));
        assert!(reason.contains("usage limit"));
        assert!(!reason.contains("Resets at"), "one line is enough for a breaker row");
        assert_eq!(unusable_reason("   "), "empty response");
    }

    #[test]
    fn the_ladder_tries_at_most_two_rungs() {
        // A prompt no CLI can use should not cost one call per installed CLI.
        assert_eq!(MAX_RUNGS_PER_CALL, 2);
        assert!(MAX_RUNGS_PER_CALL < SPECS.len(), "the bound must actually bind");
    }

    #[test]
    fn an_oversized_prompt_is_refused_before_a_process_is_spawned() {
        let spec = &SPECS[0];
        let error =
            invoke(spec, spec.model, &"x".repeat(PROMPT_MAX_BYTES + 1), CALL_TIMEOUT).unwrap_err();
        assert!(error.to_string().contains("call ceiling"));
    }

    #[test]
    fn a_missing_program_is_not_installed() {
        assert!(!installed("definitely-not-a-real-program-xyz"));
    }
}
