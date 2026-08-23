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
        args: &["-p", "--model", "{model}"],
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
        cli: "gemini",
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
    timeout: Duration,
}

impl<'a> Ladder<'a> {
    #[must_use]
    pub fn new(store: &'a Store, mode: &str) -> Self {
        Self { store, mode: mode.to_string(), timeout: CALL_TIMEOUT }
    }

    /// The same ladder on a shorter leash.
    ///
    /// Consolidation runs detached and can afford three minutes. A call made
    /// while someone waits for a search result cannot: past a few seconds the
    /// unranked answer they already had is the better one.
    #[must_use]
    pub fn clone_with_timeout(&self, limit: Duration) -> Ladder<'a> {
        Ladder { store: self.store, mode: self.mode.clone(), timeout: limit }
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

            match invoke(spec, prompt, self.timeout) {
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
            return SPECS.iter().filter(|spec| spec.cli == self.mode).collect();
        }
        let mut order: Vec<&CliSpec> = SPECS.iter().filter(|s| s.cli == preferred_cli).collect();
        order.extend(SPECS.iter().filter(|s| s.cli != preferred_cli));
        order
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
fn invoke(spec: &CliSpec, prompt: &str, timeout: Duration) -> Result<String> {
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
            "{model}" => spec.model.to_string(),
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
    let start = std::time::Instant::now();
    loop {
        match child.try_wait().context("poll summarizer")? {
            Some(status) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stdout.take() {
                    use std::io::Read;
                    let _ = pipe.read_to_string(&mut stdout);
                }
                if let Some(mut pipe) = child.stderr.take() {
                    use std::io::Read;
                    let _ = pipe.read_to_string(&mut stderr);
                }
                return Ok(CallResult {
                    success: status.success(),
                    code: status.code(),
                    stdout,
                    stderr,
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
        let ladder = Ladder::new(&store, "auto");
        let order = ladder.order("codex");
        assert_eq!(order[0].cli, "codex");
        assert_eq!(order.len(), SPECS.len(), "every other CLI stays as a fallback");
    }

    #[test]
    fn a_pinned_mode_never_silently_uses_another_subscription() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, "claude-code");
        let order = ladder.order("codex");
        assert_eq!(order.len(), 1);
        assert_eq!(order[0].cli, "claude-code");
    }

    #[test]
    fn off_mode_never_reaches_for_a_model() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, "off");
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
        let error = invoke(spec, &"x".repeat(PROMPT_MAX_BYTES + 1), CALL_TIMEOUT).unwrap_err();
        assert!(error.to_string().contains("call ceiling"));
    }

    #[test]
    fn a_missing_program_is_not_installed() {
        assert!(!installed("definitely-not-a-real-program-xyz"));
    }
}
