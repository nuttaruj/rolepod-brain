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

use std::path::{Path, PathBuf};
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

impl CliSpec {
    /// Does this rung actually hand a model name to its CLI?
    ///
    /// Two of them do not, and for them `model` is empty and
    /// `summarizer.models` reaches nothing - a distinction `brain doctor`
    /// has to make too, so the string that decides it lives here rather than
    /// being spelled out again at the reporting end.
    #[must_use]
    pub fn passes_a_model(&self) -> bool {
        self.args.contains(&"{model}")
    }
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
        // Google's replacement for Gemini CLI, and the reason that entry is
        // now last. The binary is `agy`; the name here is the one
        // its hooks write.
        cli: "antigravity",
        program: "agy",
        model: "gemini-3.7-flash-low",
        output: OutputMode::Stdout,
        // `-p` takes the prompt as its VALUE, so it must be the last flag:
        // `agy -p --model X` reads "--model" as the prompt and drops the real
        // one, which the CLI says out loud rather than guessing. Ordering it
        // last is also exactly what `invoke` does - it appends the prompt as
        // the final argument - so the two agree by construction.
        args: &["--model", "{model}", "-p"],
    },
    CliSpec {
        cli: "cursor",
        program: "cursor-agent",
        // Unpinned, like OpenCode below, but for its own reason: Cursor's
        // model list is per-plan. `--list-models` marks `auto` the default on
        // a plan that carries it, and not every plan does - so naming it, or
        // any other id, is a call that fails outright on the plans without it
        // rather than routing around it. Omitting `--model` asks for whatever
        // that install is already set to, the one request every plan can
        // answer. Measured without the flag at 20s, in line with the 18.1s
        // `auto` and 20.3s `composer-2.5` means that replaced each other here
        // before; that was on an account whose default is `auto`, so plans
        // defaulting elsewhere are unmeasured.
        //
        // No `{model}` below to substitute, so `summarizer.models` does not
        // reach this rung. See
        // `a_spec_without_a_model_placeholder_names_no_model`.
        model: "",
        output: OutputMode::Stdout,
        // Two restrictions, both load-bearing. `-p` on its own, in Cursor's
        // own words, "has access to all tools, including write and shell" -
        // and the text we hand it is captured session content, which is data,
        // not instruction. `--mode ask` is its read-only Q&A mode. `--trust`
        // then answers the workspace-trust prompt that would otherwise make
        // every call hang; it applies to the temp directory `invoke` runs in,
        // never the user's repo. `-f` and `--yolo` do what their names say
        // and are not here.
        args: &["-p", "--output-format", "text", "--mode", "ask", "--trust"],
    },
    CliSpec {
        // Second to last. OpenCode is a front end for whatever providers
        // the user has authenticated, so unlike every other rung there is no
        // model we can name that is cheap on all machines - or valid on any
        // given one. It runs on whatever that install is already set to,
        // which is the only choice that cannot spend money the user did not
        // agree to; sitting back here means it is reached only when nothing we
        // can price has answered.
        cli: "opencode",
        program: "opencode",
        // Empty on purpose: no `{model}` to substitute. See
        // `a_spec_without_a_model_placeholder_names_no_model`.
        model: "",
        output: OutputMode::Stdout,
        // Prints a one-line banner naming the model before the answer. It
        // carries no brace, so `extract_json_object` steps over it.
        args: &["run"],
    },
    CliSpec {
        // Must match what hooks write as `source.cli`, not what the binary is
        // called: this string is looked up against captured events.
        //
        // Last on purpose. Google shut this CLI down for individual
        // accounts on 2026-06-18 - it exits 0 and prints an `IneligibleTierError`
        // pointing at Antigravity, which is a rung that looks installed and
        // answers nothing. Enterprise and Code Assist Standard licences were
        // not shut down, and npm is still publishing, so the rung stays for
        // them, at the back, until it is closed to them too - and three
        // failures bench it for half an hour either way.
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
    /// Someone is waiting on this call and its result is a bonus, not the
    /// answer. See [`Ladder::while_waiting`] for what that changes.
    advisory: bool,
}

impl<'a> Ladder<'a> {
    #[must_use]
    pub fn new(store: &'a Store, config: &crate::config::SummarizerConfig) -> Self {
        Self {
            store,
            mode: config.mode.clone(),
            models: config.models.clone(),
            timeout: CALL_TIMEOUT,
            advisory: false,
        }
    }

    /// The same ladder, for a call someone is sitting and waiting on.
    ///
    /// Three things change, all of them consequences of one fact: this call is
    /// advisory. Consolidation runs detached, can afford three minutes, is
    /// worth a second CLI when the first will not answer, and its failures are
    /// real evidence about a CLI. A call holding up a search result is none of
    /// those, so it gets:
    ///
    /// - `limit` instead of three minutes, because past a few seconds the
    ///   answer the caller already had is the better one;
    /// - one rung, and only ever the CLI whose work this is. Consolidation
    ///   may fall through to another vendor because the job has to get done
    ///   and any model that summarises will do; a rerank is a favour asked
    ///   mid-search against a subscription the caller is already waiting on.
    ///   If that one cannot answer, the search keeps the order it had and
    ///   the next search asks again - spending a second vendor's quota to
    ///   reorder a list the caller already holds is a cost nobody asked for;
    /// - no marks on the health table, in either direction. A CLI that
    ///   overran a six-second leash has not been shown to be down - and
    ///   charging it would bench it for thirty minutes of consolidation,
    ///   which had three full minutes to spend and never got to try. The
    ///   breaker is still *read*: a CLI already known to be down is not worth
    ///   the wait.
    #[must_use]
    pub fn while_waiting(&self, limit: Duration) -> Ladder<'a> {
        Ladder {
            store: self.store,
            mode: self.mode.clone(),
            models: self.models.clone(),
            timeout: limit,
            advisory: true,
        }
    }

    /// Rungs this call may try.
    fn rungs(&self) -> usize {
        if self.advisory { 1 } else { MAX_RUNGS_PER_CALL }
    }

    /// Record an outcome against a CLI, unless the call was one it cannot
    /// fairly be judged on.
    fn record(&self, cli: &str, outcome: Result<(), &str>) -> Result<()> {
        if self.advisory {
            return Ok(());
        }
        match outcome {
            Ok(()) => self.store.record_summarizer_success(cli),
            Err(detail) => self.store.record_summarizer_failure(cli, detail),
        }
    }

    /// Is any model allowed at all?
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.mode != "off"
    }

    /// Could this ladder answer right now?
    ///
    /// Mode, installation and the breaker together - the same three questions
    /// [`Ladder::run`] asks before it spends anything, asked without spending
    /// anything. A caller deciding whether to redo degraded work needs to know
    /// that a model is actually reachable: "no CLI is in a cooldown" is also
    /// true of a machine where none is installed, and of `mode = "off"`, and
    /// retrying there is a rewrite that produces the same rule-based page
    /// forever.
    ///
    /// # Errors
    /// Returns an error when the health table cannot be read.
    pub fn could_answer(&self, preferred_cli: &str) -> Result<bool> {
        if !self.enabled() {
            return Ok(false);
        }
        for spec in self.order(preferred_cli) {
            if self.available(spec)? {
                return Ok(true);
            }
        }
        Ok(false)
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
            if attempts >= self.rungs() {
                break;
            }
            if !self.available(spec)? {
                continue;
            }
            attempts += 1;

            match invoke(spec, self.model_for(spec), prompt, self.timeout) {
                Ok(text) if usable(&text) => {
                    self.record(spec.cli, Ok(()))?;
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
                    self.record(spec.cli, Err(&detail))?;
                }
                Err(error) => {
                    self.record(spec.cli, Err(&format!("{error:#}")))?;
                }
            }
        }
        Ok((Tier::RuleBased, String::new()))
    }

    /// Rungs to try, in order.
    fn order(&self, preferred_cli: &str) -> Vec<&'static CliSpec> {
        // An advisory call goes to the CLI whose work this is, or nowhere.
        //
        // Consolidation may fall through to another vendor: the job must get
        // done, and any model that can summarise will do. A rerank is not
        // that. It is a favour asked mid-search, against a subscription the
        // user is already signed into and already waiting on - and if that
        // one cannot answer right now, spending a second vendor's quota to
        // reorder a list the caller already has is a cost they did not ask
        // for. The search returns the order it had, and the next search can
        // try again.
        if self.advisory {
            let preferred = Self::normalize_cli(preferred_cli);
            return SPECS.iter().filter(|spec| spec.cli == preferred).collect();
        }
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
    /// into a summarizer that silently never runs. The same trap is set twice
    /// more by CLIs whose binary is not spelled like their product: someone
    /// pinning `agy` or `cursor-agent` typed the name they invoke.
    fn normalize_cli(raw: &str) -> &str {
        match raw {
            "gemini" => "gemini-cli",
            "agy" => "antigravity",
            "cursor-agent" => "cursor",
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

/// The extensions a program name might really wear on this platform.
///
/// Empty means the name as given. On Windows the list is the reason the whole
/// ladder works there at all: `claude`, `codex` and `gemini` are installed by
/// npm, which writes a `.cmd` shim rather than an executable, and neither
/// `CreateProcess` nor Rust's own `.exe`-only search will find one. Resolving
/// the shim ourselves and handing the full path to `Command` is what closes
/// that - the standard library recognises a `.cmd` and runs it through
/// `cmd.exe` with the escaping that fix required.
#[cfg(windows)]
const PROGRAM_EXTENSIONS: &[&str] = &["", "exe", "cmd", "bat"];
#[cfg(not(windows))]
const PROGRAM_EXTENSIONS: &[&str] = &[""];

/// Where `program` really is on `PATH`, if it is anywhere.
///
/// Returns the full path rather than the bare name, because on Windows the
/// difference between the two is whether the program can be started at all.
#[must_use]
pub fn resolve(program: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(program);
        PROGRAM_EXTENSIONS.iter().find_map(|extension| {
            let candidate = if extension.is_empty() {
                candidate.clone()
            } else {
                candidate.with_extension(extension)
            };
            candidate.is_file().then_some(candidate)
        })
    })
}

/// Is a program on `PATH`?
#[must_use]
pub fn installed(program: &str) -> bool {
    resolve(program).is_some()
}

/// `PATH` for a child, with the program's own directory on the front.
///
/// Returns `None` when there is nothing to add - the caller then leaves the
/// child's environment alone.
///
/// Every CLI in this table except Antigravity's is a Node script installed by
/// npm, and a script starts by asking the kernel to run its shebang - almost
/// always `/usr/bin/env node`. `env` searches `PATH`, so the child needs to
/// find `node`, and OUR `PATH` is whatever the host CLI happened to have.
/// A GUI-launched editor is handed launchd's minimal one: `resolve` still
/// finds `codex` (the hook prepends the directory it lives in), the kernel
/// still reads the shebang, and `env` then fails to find `node` - which
/// surfaces as ENOENT about a file that demonstrably exists.
///
/// The interpreter is installed beside the script that needs it: `node` sits
/// in the same nvm `bin` as the `codex` npm put there, and in the same
/// `/opt/homebrew/bin` as a brewed `gemini`. So the program's own directory,
/// as `PATH` spelled it, is the answer.
///
/// Following the symlink first is the plausible-looking version that does not
/// work. `codex` in an nvm `bin` points at
/// `lib/node_modules/@openai/codex/bin/`, and Homebrew's `gemini` points into
/// a `Cellar` - package directories, with no interpreter in either. Measured
/// on both, rather than reasoned about: the un-followed directory starts the
/// program and the resolved one does not.
fn interpreter_path(program: &Path) -> Option<std::ffi::OsString> {
    let dir = program.parent()?;
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut dirs: Vec<PathBuf> = vec![dir.to_path_buf()];
    dirs.extend(std::env::split_paths(&current));
    std::env::join_paths(dirs).ok()
}

/// A directory the child can actually start in.
///
/// `current_dir` is not a hint. If the directory does not exist the spawn
/// fails with ENOENT - the same errno a missing program gives, about a program
/// that is sitting right there - and it fails that way for every rung at once,
/// scripts and native binaries alike, because no interpreter is involved. That
/// is how this was found: four CLIs reporting a missing file, two of them
/// Mach-O binaries with no shebang to blame.
///
/// A host CLI can hand its hooks a `TMPDIR` that no longer exists, so the
/// directory is created rather than assumed. On failure this is an error, not
/// a fallback to the current directory: running a headless CLI inside the
/// user's repo is the thing the inert directory exists to prevent, and doing
/// it silently to keep a summary alive is the wrong trade.
fn inert_dir(candidate: PathBuf) -> Result<PathBuf> {
    std::fs::create_dir_all(&candidate)
        .with_context(|| format!("no usable working directory at {}", candidate.display()))?;
    Ok(candidate)
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

    // The resolved path, not the bare name. On Windows these are npm `.cmd`
    // shims and a bare name reaches nothing; everywhere else this is the same
    // file `PATH` would have found, just named in full.
    let program = resolve(spec.program)
        .with_context(|| format!("{} is not on PATH", spec.program))?;
    let workdir = inert_dir(std::env::temp_dir())?;
    let mut command = Command::new(&program);
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
        .current_dir(&workdir);
    if let Some(path) = interpreter_path(&program) {
        command.env("PATH", path);
    }

    let mut child = command.spawn().with_context(|| {
        // "No such file or directory" here is never about the program: it was
        // resolved to an existing file a line ago. Two other things wear the
        // same errno, and the message names both - and now names the
        // directory too, because the first time this happened the report did
        // not carry enough to tell them apart and the cause was never pinned
        // down. A script's interpreter missing from the child's PATH is one.
        // The working directory not existing is the other, and that one fails
        // every rung identically, native binaries included.
        format!(
            "spawn {} ({}) in {} - if this says no such file, it is the \
             script's interpreter or that directory, not the program",
            spec.program,
            program.display(),
            workdir.display()
        )
    })?;

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
    fn a_child_is_never_told_to_start_in_a_directory_that_is_not_there() {
        // The failure this closes reports itself as a missing program. A
        // `current_dir` that does not exist makes `spawn` return ENOENT, which
        // reads exactly like the executable is gone - and it does that to
        // every rung at once, including native binaries with no interpreter to
        // suspect. A host CLI handing its hooks a stale `TMPDIR` is enough.
        let root = std::env::temp_dir().join(format!("brain-inert-{}", ulid::Ulid::new()));
        assert!(!root.exists(), "the fixture must start from nothing");

        let dir = inert_dir(root.join("deeper")).expect("a missing directory is created, not fatal");
        assert!(dir.is_dir(), "inert_dir returned a path the child still cannot enter");

        // Idempotent: the ordinary case is a directory that already exists.
        assert!(inert_dir(dir.clone()).is_ok());

        // And it is an error rather than a silent fallback to the user's repo,
        // which is the whole reason the child is sent elsewhere.
        let file = root.join("a-file");
        std::fs::write(&file, b"x").unwrap();
        assert!(inert_dir(file.join("under-a-file")).is_err());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_child_can_find_the_interpreter_its_program_needs() {
        // The failure this closes, reproduced on this machine: with only
        // `/usr/bin:/bin` on PATH, the full path to an npm-installed `codex`
        // fails with `env: node: No such file or directory` - about a file
        // that is demonstrably there. Put its own directory on the front and
        // the same command prints `codex-cli 0.147.0`, because that is where
        // npm also put `node`.
        let dir = std::env::temp_dir().join(format!("brain-interp-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let program = dir.join("some-cli");
        std::fs::write(&program, "#!/bin/sh\necho hi\n").unwrap();

        let path = interpreter_path(&program).expect("a program in a directory has a directory");
        let dirs: Vec<PathBuf> = std::env::split_paths(&path).collect();
        assert_eq!(dirs.first(), Some(&dir), "the program's own directory is not searched first");
        assert!(dirs.len() > 1, "the rest of PATH was dropped rather than extended");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_spec_without_a_model_placeholder_names_no_model() {
        // The trap this closes: a spec whose args never substitute `{model}`
        // still has a `model` field, and `model_for` still returns whatever a
        // user puts in `summarizer.models`. Writing a model there - in the
        // spec or in config - would look like a pinned cheap tier and change
        // nothing about what runs. An empty default makes the report say so
        // out loud instead.
        for spec in SPECS {
            assert_eq!(
                spec.passes_a_model(),
                !spec.model.is_empty(),
                "{}: a model that is never passed, or a placeholder with nothing to put in it",
                spec.cli
            );
        }
    }

    #[test]
    fn a_binary_may_be_pinned_by_the_name_it_is_invoked_with() {
        // Nobody types "antigravity" at a shell; they type `agy`. A mode
        // pinned to the spelling they know must not silently produce an empty
        // ladder.
        let store = Store::open_memory().unwrap();
        for (typed, spec) in [("agy", "antigravity"), ("cursor-agent", "cursor"), ("gemini", "gemini-cli")]
        {
            let ladder = Ladder::new(&store, &config(typed));
            let order = ladder.order("claude-code");
            assert_eq!(order.len(), 1, "`{typed}` matched no rung");
            assert_eq!(order[0].cli, spec);
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

    /// A rerank asks the CLI whose work it is, or nobody.
    ///
    /// Consolidation is allowed to fall through to another vendor: the job
    /// has to get done and any model that summarises will do. A rerank is a
    /// favour asked mid-search. If the CLI the user is already signed into
    /// and already waiting on cannot answer, spending a second vendor's quota
    /// to reorder a list the caller already holds is a cost they did not ask
    /// for. The search keeps the order it had, and the next one tries again.
    #[test]
    fn an_advisory_call_never_spends_another_vendors_quota() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("auto"));
        let waiting = ladder.while_waiting(Duration::from_secs(20));

        assert_eq!(
            waiting.order("codex").iter().map(|spec| spec.cli).collect::<Vec<_>>(),
            vec!["codex"],
            "an advisory call reached past the CLI whose work it is"
        );
        // Consolidation keeps its fallbacks.
        assert_eq!(ladder.order("codex").len(), SPECS.len());

        // And when that CLI is out of rotation, the advisory call has nowhere
        // to go - which is the point. It must not quietly become a call to
        // whichever vendor happens to be up.
        for _ in 0..failure_threshold() {
            store.record_summarizer_failure("codex", "rate limited").unwrap();
        }
        assert!(store.summarizer_in_cooldown("codex").unwrap());
        let (tier, text) = waiting.run("anything", "codex", |_| true).unwrap();
        assert_eq!(tier, Tier::RuleBased, "a benched CLI was substituted for another");
        assert!(text.is_empty());
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
    fn off_mode_could_never_answer() {
        // The caller that asks this is deciding whether to redo degraded work.
        // Reading the breaker alone would say yes here - `off` never ran, so
        // it never failed, so nothing is in a cooldown - and the answer would
        // still be rule-based every time.
        let store = Store::open_memory().unwrap();
        assert!(!Ladder::new(&store, &config("off")).could_answer("claude-code").unwrap());
    }

    #[test]
    fn a_mode_that_matches_no_rung_could_not_answer() {
        // A pinned mode naming a CLI this table does not carry - a typo, or a
        // rung dropped in a later version while the config that named it
        // stayed on disk. The order is empty, so nothing can run, and a
        // caller deciding whether to redo degraded work must be told that
        // rather than left to retry forever. Pinned rather than `auto` so the
        // answer does not depend on what is installed on the test machine.
        let store = Store::open_memory().unwrap();
        assert!(!Ladder::new(&store, &config("windsurf")).could_answer("codex").unwrap());
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

    /// A second CLI is worth waiting for when the work is detached. It is not
    /// worth doubling a search's wait to improve an ordering that was already
    /// good enough to return.
    #[test]
    fn a_call_someone_is_waiting_on_asks_one_cli_only() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("auto"));
        assert_eq!(ladder.rungs(), MAX_RUNGS_PER_CALL);
        assert_eq!(ladder.while_waiting(Duration::from_secs(6)).rungs(), 1);
    }

    /// The bug this exists to prevent: a six-second leash is not evidence
    /// about a CLI that consolidation gives three minutes. Charging one for
    /// the other benched a working CLI for half an hour.
    #[test]
    fn an_advisory_call_leaves_no_mark_on_the_breaker() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("auto"));
        let waiting = ladder.while_waiting(Duration::from_secs(6));

        for _ in 0..failure_threshold() + 2 {
            waiting.record("codex", Err("timed out after 6s")).unwrap();
        }
        assert!(
            !store.summarizer_in_cooldown("codex").unwrap(),
            "an advisory timeout took a CLI out of consolidation"
        );

        // Nor in the other direction: a fast rerank is not a clean bill of
        // health for a CLI consolidation has been failing against.
        for _ in 0..failure_threshold() {
            store.record_summarizer_failure("claude-code", "rate limited").unwrap();
        }
        waiting.record("claude-code", Ok(())).unwrap();
        assert!(
            store.summarizer_in_cooldown("claude-code").unwrap(),
            "an advisory success cleared a breaker it never earned"
        );

        // The same ladder, doing the work it is judged on, still counts.
        for _ in 0..failure_threshold() {
            ladder.record("gemini-cli", Err("rate limited")).unwrap();
        }
        assert!(store.summarizer_in_cooldown("gemini-cli").unwrap());
    }

    #[test]
    fn an_oversized_prompt_is_refused_before_a_process_is_spawned() {
        let spec = &SPECS[0];
        let error =
            invoke(spec, spec.model, &"x".repeat(PROMPT_MAX_BYTES + 1), CALL_TIMEOUT).unwrap_err();
        assert!(error.to_string().contains("call ceiling"));
    }

    /// A host CLI installed by npm is found, and can actually be started.
    ///
    /// This is the one thing about this platform that is not shared with the
    /// others, and the end-to-end suite does not run here to cover it. npm
    /// writes `claude.cmd` rather than `claude.exe`; `CreateProcess` cannot
    /// start a `.cmd`, and Rust's own search only ever appends `.exe`. Both
    /// halves are asserted - that `resolve` returns the shim, and that the
    /// shim spawned through the returned path runs and answers - because
    /// finding a program you then cannot execute is the failure this guards.
    #[cfg(windows)]
    #[test]
    fn an_npm_shim_is_found_and_can_be_run() {
        let dir = std::env::temp_dir().join(format!("brain-shim-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).expect("create");
        let name = "brain-fake-host-cli";
        std::fs::write(dir.join(format!("{name}.cmd")), "@echo off\r\necho shim-answered\r\n")
            .expect("write shim");

        let previous = std::env::var_os("PATH");
        std::env::set_var("PATH", &dir);
        let found = resolve(name);
        let installed_says = installed(name);
        match previous {
            Some(value) => std::env::set_var("PATH", value),
            None => std::env::remove_var("PATH"),
        }

        let found = found.unwrap_or_else(|| panic!("{name}.cmd was not found on PATH"));
        assert_eq!(
            found.extension().and_then(std::ffi::OsStr::to_str),
            Some("cmd"),
            "resolved {} rather than the shim",
            found.display()
        );
        assert!(installed_says, "resolve found it and installed disagreed");

        let output = Command::new(&found).output().expect("spawn the shim");
        let said = String::from_utf8_lossy(&output.stdout);
        assert!(said.contains("shim-answered"), "the shim did not run: {said:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_program_is_not_installed() {
        assert!(!installed("definitely-not-a-real-program-xyz"));
    }
}
