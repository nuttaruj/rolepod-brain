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
    /// The model id we tested, passed to that CLI. It is the floor and the
    /// way back: a newer id found by [`refresh_models`] replaces it only when
    /// it belongs to `family` and is numbered higher, and a call that fails on
    /// the newer id is tried again on this one.
    pub model: &'static str,
    /// Whether the answer arrives on stdout or in a file we name.
    pub output: OutputMode,
    /// Arguments before the prompt. `{model}` and `{out}` are substituted.
    pub args: &'static [&'static str],
    /// How to tell a newer model of the same cheap tier from every other
    /// model, for a CLI whose ids move on. `None` where the id is an alias the
    /// CLI keeps current itself, or where no id is passed at all.
    pub family: Option<Family>,
}

/// The ids that count as "the same cheap tier, newer", and where to list them.
pub struct Family {
    /// Matches a model id of the tier; capture group 1 is its version, dotted
    /// numbers (`6`, `5.6`, `3.10`).
    pub pattern: &'static str,
    /// Where the CLI's list of models comes from.
    pub listing: Listing,
}

/// Where a CLI says which models it can run.
pub enum Listing {
    /// Codex keeps the list it last fetched in `models_cache.json`, so reading
    /// it costs no call and no login.
    CodexCache,
    /// A command whose stdout has one `<id><TAB><name>` line per model.
    Command(&'static [&'static str]),
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
/// Every entry here was invoked for real before it was written down. Where a
/// rung has a `family`, its `model` is a pin: the tested floor and the way
/// back, not necessarily what runs - a newer id of the same tier found by
/// [`refresh_models`] runs instead, until a call on it fails.
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
        args: &[
            "-p",
            "--model",
            "{model}",
            "--strict-mcp-config",
            "--tools=",
            // Every call otherwise leaves a saved session holding brain's
            // prompt in the user's history (8k+ of them on one machine).
            "--no-session-persistence",
            // A summarizer worker is not an agent the user started, so the
            // host must not announce it as one. Without this, a machine
            // draining a backlog sent its owner one push notification per
            // summary - to the phone, carrying the text of the summary.
        ],
        // `haiku` is an alias the CLI keeps pointing at its newest Haiku.
        family: None,
    },
    CliSpec {
        cli: "codex",
        program: "codex",
        model: "gpt-5.6-luna",
        output: OutputMode::File,
        // stdout carries hook chatter and a token counter, so the answer is
        // read from the file rather than scraped out of the stream.
        //
        // The two `-c`/`-s` overrides exist because `codex exec` inherits the
        // user's config.toml: one machine's default was effort `max` with a
        // full-access sandbox, and a JSON summary of 3 KB of events ran
        // 38-77s there, spending up to 3.6k reasoning tokens per call.
        // Summarising is not a reasoning task, and a worker that calls no
        // tool needs no sandbox wider than read-only.
        args: &[
            "exec",
            // No rollout file per call: ~1k a week otherwise.
            "--ephemeral",
            "-m",
            "{model}",
            "-c",
            "model_reasoning_effort=\"low\"",
            "-s",
            "read-only",
            "--skip-git-repo-check",
            "-o",
            "{out}",
        ],
        family: Some(Family {
            pattern: r"^gpt-(\d+(?:\.\d+)*)-luna$",
            listing: Listing::CodexCache,
        }),
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
        family: Some(Family {
            pattern: r"^gemini-(\d+(?:\.\d+)*)-flash-low$",
            listing: Listing::Command(&["models"]),
        }),
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
        family: None,
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
        family: None,
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
        // `flash` is an alias the CLI keeps pointing at its newest Flash.
        family: None,
    },
];

/// Which CLI a call actually used, for logging and health accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tier {
    /// A host CLI's cheap tier.
    Cli(String),
    /// No model was reachable; the caller must produce rule-based output.
    RuleBased,
    /// Nothing worth a model's time happened: the session opened, ran a
    /// command or two that touched nothing, and closed. Settled without a
    /// call, without a page, and without a summary for the primer to carry.
    Quiet,
    /// A one-shot run - `claude -p`, `codex exec`, a codex served to another
    /// agent's plugin. Its captures stay in the log and answer `brain_recent`
    /// and search; its outcome already reached memory through the session
    /// that delegated it. Settled the way a quiet session is: no call, no
    /// page, no summary, and never a line about work left in flight.
    Headless,
}

/// What a call is for and whose it is, for the `summarizer_calls` ledger.
#[derive(Debug, Clone, Copy)]
pub struct CallContext<'a> {
    /// `consolidate`, `merge`, `synthesis`, `ingest` or `ingest-merge`.
    pub purpose: &'a str,
    /// The session (or, for synthesis, the project) the call works for.
    pub session: &'a str,
}

/// The words a timed-out child's error starts with; the ledger reads them
/// back to tell a timeout from a crash.
const TIMED_OUT: &str = "timed out after";

/// Picks a rung and runs it.
pub struct Ladder<'a> {
    store: &'a Store,
    mode: String,
    /// Per-CLI model overrides from config; a CLI not named keeps its
    /// spec's cheap default.
    models: std::collections::HashMap<String, String>,
    /// Newer models found by [`refresh_models`], read once when the ladder is
    /// made so a call never queries the store for them.
    discovered: Discovered,
    /// CLIs whose found model failed during this ladder's life while their pin
    /// answered. The store remembers it for a week; this makes the rest of
    /// the run believe it without another failed call per session.
    fell_back: std::cell::RefCell<std::collections::HashSet<&'static str>>,
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
            discovered: Discovered::read(store, jiff::Timestamp::now()),
            fell_back: std::cell::RefCell::default(),
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
            discovered: self.discovered.clone(),
            fell_back: self.fell_back.clone(),
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
    /// with no text when every rung is unavailable, or when a rung answered in
    /// JSON of the wrong shape (a content failure: no further rung is tried and
    /// no vendor is charged). Never returns an error for
    /// an unavailable model: that is an expected state, not a fault.
    ///
    /// # Errors
    /// Returns an error only when the health table cannot be read or written.
    pub fn run<F>(
        &self,
        ctx: &CallContext<'_>,
        prompt: &str,
        preferred_cli: &str,
        usable: F,
    ) -> Result<(Tier, String)>
    where
        F: Fn(&str) -> bool,
    {
        self.run_with(ctx, prompt, preferred_cli, usable, |spec| self.available(spec), invoke)
    }

    /// [`Ladder::run`] with the two things that touch the machine - is this
    /// CLI reachable, and running it - handed in, so a test can drive every
    /// outcome without a CLI installed.
    fn run_with<F, A, I>(
        &self,
        ctx: &CallContext<'_>,
        prompt: &str,
        preferred_cli: &str,
        usable: F,
        available: A,
        invoke: I,
    ) -> Result<(Tier, String)>
    where
        F: Fn(&str) -> bool,
        A: Fn(&CliSpec) -> Result<bool>,
        I: Fn(&CliSpec, &str, &str, Duration) -> Result<String>,
    {
        if !self.enabled() {
            return Ok((Tier::RuleBased, String::new()));
        }

        // A prompt over the ceiling is the caller's bug, not a sick CLI: no rung
        // could take it, so say so once instead of walking every one, and leave
        // every breaker alone.
        if prompt.len() > PROMPT_MAX_BYTES {
            let first = self.order(preferred_cli).first().map(|spec| spec.cli).unwrap_or_default();
            self.ledger(&crate::store::SummarizerCall {
                session: ctx.session.to_string(),
                purpose: ctx.purpose.to_string(),
                cli: first.to_string(),
                model: String::new(),
                prompt_bytes: prompt.len() as u64,
                answer_bytes: 0,
                ms: 0,
                outcome: "oversize".to_string(),
            });
            return Ok((Tier::RuleBased, String::new()));
        }

        let mut attempts = 0usize;
        for spec in self.order(preferred_cli) {
            if attempts >= self.rungs() {
                break;
            }
            if !available(spec)? {
                continue;
            }
            attempts += 1;

            let (model, origin) = self.model_choice(spec);
            let mut attempt = self.attempt(ctx, spec, model, prompt, &usable, &invoke);
            // A model we found by listing, not one we tested, may be listed
            // and still not runnable on this account. When it fails for any
            // reason but the content, the tested pin gets the same call inside
            // this rung, and only the pin's outcome is the rung's. If the pin
            // answers, the found model is the culprit and is set aside; if it
            // fails too, the CLI is down and nothing about the model is
            // learned. A call someone is waiting on gets neither retry nor
            // verdict, as with the breaker. A timeout is a non-content failure
            // like any other, so a found model that hangs holds this rung for
            // two timeouts at most, and is set aside if the pin answers.
            if origin == Origin::Newest && !self.advisory && !attempt.good && !attempt.content_failure
            {
                let retry = self.attempt(ctx, spec, spec.model, prompt, &usable, &invoke);
                if retry.good {
                    self.fell_back.borrow_mut().insert(spec.cli);
                    let mark = format!("{model} {}", jiff::Timestamp::now());
                    // A mark that cannot be written costs one more retry next
                    // run; it must not cost this answer.
                    let _ = self.store.set_state(&bad_key(spec.cli), &mark);
                }
                attempt = retry;
            }
            let Attempt { result, good, content_failure } = attempt;

            match result {
                Ok(text) if good => {
                    self.record(spec.cli, Ok(()))?;
                    return Ok((Tier::Cli(spec.cli.to_string()), text));
                }
                Ok(_) if content_failure => {
                    // The CLI answered, in JSON, with something that does not
                    // fit this prompt. That is this session's content, not a
                    // sick vendor: no breaker mark, and no second rung, since
                    // the next CLI would be handed the same prompt. The caller
                    // writes the rule-based floor and counts the attempt.
                    return Ok((Tier::RuleBased, String::new()));
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

    /// One call to one CLI on one model, with its ledger row written.
    fn attempt<F, I>(
        &self,
        ctx: &CallContext<'_>,
        spec: &CliSpec,
        model: &str,
        prompt: &str,
        usable: &F,
        invoke: &I,
    ) -> Attempt
    where
        F: Fn(&str) -> bool,
        I: Fn(&CliSpec, &str, &str, Duration) -> Result<String>,
    {
        let started = std::time::Instant::now();
        let result = invoke(spec, model, prompt, self.timeout);
        // The bail in `wait_with_timeout` is unwrapped, so it leads the
        // message; a CLI's own stderr is embedded later and must not match.
        let good = matches!(&result, Ok(text) if usable(text));
        let content_failure = matches!(&result, Ok(text) if !good && is_wrong_shape_json(text));
        let (outcome, answer_bytes) = match &result {
            Ok(text) if good => ("ok", text.len()),
            Ok(text) if content_failure => ("unparseable", text.len()),
            Ok(text) => ("unusable", text.len()),
            Err(error) if error.to_string().starts_with(TIMED_OUT) => ("timeout", 0),
            Err(_) => ("spawn_error", 0),
        };
        self.ledger(&crate::store::SummarizerCall {
            session: ctx.session.to_string(),
            purpose: ctx.purpose.to_string(),
            cli: spec.cli.to_string(),
            model: model.to_string(),
            prompt_bytes: prompt.len() as u64,
            answer_bytes: answer_bytes as u64,
            ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            outcome: outcome.to_string(),
        });
        Attempt { result, good, content_failure }
    }

    /// One row in `summarizer_calls`, unless the call was advisory (a rerank
    /// has its own table). A ledger that cannot be written never changes what
    /// the ladder answers, so the write's error is dropped.
    fn ledger(&self, call: &crate::store::SummarizerCall) {
        if self.advisory {
            return;
        }
        let _ = self.store.record_summarizer_call(call);
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

    /// The model this rung should run: the user's override, else the newest
    /// model of the cheap tier [`refresh_models`] found, else the spec's pin.
    #[cfg(test)]
    fn model_for(&self, spec: &CliSpec) -> &str {
        self.model_choice(spec).0
    }

    /// [`Ladder::model_for`], and which of the three it was.
    fn model_choice(&self, spec: &CliSpec) -> (&str, Origin) {
        let (model, origin) = choose_model(spec, &self.models, &self.discovered);
        if origin == Origin::Newest && self.fell_back.borrow().contains(spec.cli) {
            return (spec.model, Origin::BuiltIn);
        }
        (model, origin)
    }

    /// Is this CLI installed, and not in a cooldown?
    fn available(&self, spec: &CliSpec) -> Result<bool> {
        if !installed(spec.program) {
            return Ok(false);
        }
        Ok(!self.store.summarizer_in_cooldown(spec.cli)?)
    }
}

/// Is this answer a JSON object a caller rejected, as opposed to a CLI
/// reporting trouble?
///
/// After one code fence is stripped the text must start with `{` and parse.
/// An object with an `error` key is how a vendor reports a quota or auth
/// failure in JSON, so it stays a vendor failure.
fn is_wrong_shape_json(text: &str) -> bool {
    let mut body = text.trim();
    if let Some(rest) = body.strip_prefix("```") {
        // Drop an optional language tag; a newline is not required.
        body = rest.trim_start_matches(|c: char| c.is_ascii_alphanumeric()).trim();
        body = body.strip_suffix("```").unwrap_or(body).trim();
    }
    if !body.starts_with('{') {
        return false;
    }
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(serde_json::Value::Object(map)) => !map.contains_key("error"),
        _ => false,
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

/// Where `program` really is, if it is anywhere: `PATH` first, then the
/// directories CLIs actually install into.
///
/// Returns the full path rather than the bare name, because on Windows the
/// difference between the two is whether the program can be started at all.
///
/// The fallback exists because of who spawns us. Consolidation runs from a
/// hook, and a GUI-launched host hands its hooks launchd's minimal `PATH`.
/// A real machine showed what that costs: claude primary (native installer,
/// `~/.local/bin`, visible), codex working (an nvm `bin`, invisible) — when
/// claude went down, every call skipped codex as "not installed" and floored
/// every session to rule-based. The CLI was there the whole time; only the
/// hook's `PATH` could not see it.
#[must_use]
pub fn resolve(program: &str) -> Option<std::path::PathBuf> {
    let in_dir = |dir: &Path| {
        let candidate = dir.join(program);
        PROGRAM_EXTENSIONS.iter().find_map(|extension| {
            let candidate = if extension.is_empty() {
                candidate.clone()
            } else {
                candidate.with_extension(extension)
            };
            candidate.is_file().then_some(candidate)
        })
    };
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .find_map(|dir| in_dir(&dir))
        .or_else(|| fallback_dirs().iter().find_map(|dir| in_dir(dir)))
}

/// Where the CLIs in the table land when the caller's `PATH` misses them.
///
/// Only directories under the user's home: native installers
/// (`~/.local/bin` — claude, cursor-agent, agy) and every nvm version's
/// `bin`, newest first (codex and gemini are npm packages, and nvm is where
/// npm usually puts them). System-wide directories like `/opt/homebrew/bin`
/// are deliberately absent even though a brewed CLI can live there: a test
/// that redirects `HOME` must be able to hide every CLI on the machine it
/// runs on, and an absolute directory in this list would leak the host's
/// real CLIs into every fixture. Windows hooks inherit the user's real
/// environment, so it gets no list of its own.
fn fallback_dirs() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let mut dirs = vec![home.join(".local/bin")];
    if let Ok(entries) = std::fs::read_dir(home.join(".nvm/versions/node")) {
        let mut versions: Vec<PathBuf> =
            entries.flatten().map(|entry| entry.path().join("bin")).collect();
        versions.sort();
        versions.reverse();
        dirs.extend(versions);
    }
    dirs
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

/// Variables that tell a child it is part of the session that spawned it.
///
/// A summarizer worker is not the user's agent. Left in place, these make the
/// host treat it as one: the finished-agent notification a real machine
/// received named the project of the session whose hook happened to start the
/// run, for work belonging to a different project entirely, once per summary
/// in a backlog. They also changed the child's behaviour in ways that made
/// three separate experiments here measure nothing, because a `claude` started
/// from inside a `claude` is not the same program.
///
/// The `ORCA_` family is what taught this lesson properly. Orca hosts the
/// terminal the session runs in and installs a `Stop` hook that posts each
/// finished run to a local port so it can be announced. Its hook already
/// declines to report other people's background work - it checks for
/// `DEVIN_PROJECT_DIR` and `CLAUDE_JOB_DIR` and leaves - and it needs
/// `ORCA_AGENT_HOOK_PORT`, `ORCA_AGENT_HOOK_TOKEN` and `ORCA_PANE_KEY` to do
/// anything at all. Passing those through made every summariser look like the
/// user's own pane finishing a turn, and a machine draining a backlog sent its
/// owner one push notification per summary, to the phone, carrying the text of
/// the summary. Dropping them lets Orca's own guard do the rest; nothing of
/// Orca's is touched.
///
/// Only identity is removed. Credentials are not: a user who set
/// `ANTHROPIC_API_KEY` meant it, and deciding otherwise would move their spend
/// without asking.
fn host_session_vars() -> Vec<std::ffi::OsString> {
    std::env::vars_os()
        .map(|(key, _)| key)
        .filter(|key| key.to_str().is_some_and(is_host_session_var))
        .collect()
}

/// Split out from the sweep above so it can be tested without touching the
/// process environment, which no test can do without racing every other one.
fn is_host_session_var(name: &str) -> bool {
    const PREFIXES: &[&str] =
        &["CLAUDE_CODE_", "CODEX_COMPANION_", "CURSOR_SESSION_", "ORCA_", "DEVIN_"];
    const EXACT: &[&str] =
        &["CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_PLUGIN_DATA", "CLAUDE_PROJECT_DIR"];
    PREFIXES.iter().any(|prefix| name.starts_with(prefix)) || EXACT.contains(&name)
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
    // Absolute, because it is also the child's `PWD`: a relative `TMPDIR`
    // would have the child resolve it again from inside the directory itself.
    std::path::absolute(&candidate)
        .with_context(|| format!("no usable working directory at {}", candidate.display()))
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

    let mut child = spawn_headless(spec.program, &args, Some(prompt))?;

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

/// Start one of the table's programs the way every call to it starts: found
/// the way [`resolve`] finds it, told it is a worker, run somewhere inert.
/// `tail` is the prompt, which goes last.
fn spawn_headless(
    name: &str,
    args: &[String],
    tail: Option<&str>,
) -> Result<std::process::Child> {
    // The resolved path, not the bare name. On Windows these are npm `.cmd`
    // shims and a bare name reaches nothing; everywhere else this is the same
    // file `PATH` would have found, just named in full.
    let program = resolve(name).with_context(|| format!("{name} is not on PATH"))?;
    let workdir = inert_dir(std::env::temp_dir())?;
    let mut command = Command::new(&program);
    for key in host_session_vars() {
        command.env_remove(key);
    }
    command.args(args);
    if let Some(tail) = tail {
        command.arg(tail);
    }
    command
        // Break the hook recursion before the child can start.
        .env(WORKER_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Run somewhere inert: a headless CLI started inside the user's repo
        // may read project instructions we neither need nor want to pay for.
        .current_dir(&workdir)
        // `current_dir` leaves `PWD` naming the repo the hook fired in, and
        // `opencode run` trusts `PWD` over its real directory: it chdirs back
        // and files the session under the user's project.
        .env("PWD", &workdir);
    if let Some(path) = interpreter_path(&program) {
        command.env("PATH", path);
    }

    command.spawn().with_context(|| {
        // "No such file or directory" here is never about the program: it was
        // resolved to an existing file a line ago. Two other things wear the
        // same errno, and the message names both - and now names the
        // directory too, because the first time this happened the report did
        // not carry enough to tell them apart and the cause was never pinned
        // down. A script's interpreter missing from the child's PATH is one.
        // The working directory not existing is the other, and that one fails
        // every rung identically, native binaries included.
        format!(
            "spawn {name} ({}) in {} - if this says no such file, it is the \
             script's interpreter or that directory, not the program",
            program.display(),
            workdir.display()
        )
    })
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
                    anyhow::bail!("{TIMED_OUT} {}s", limit.as_secs());
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

/// What one call returned, classified once for the ladder to act on.
struct Attempt {
    result: Result<String>,
    /// The caller accepted the answer.
    good: bool,
    /// The CLI answered in JSON that does not fit the prompt.
    content_failure: bool,
}

/// Where a call's model came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The user named it in `summarizer.models`.
    Config,
    /// The newest model of the tier, found by [`refresh_models`].
    Newest,
    /// The spec's tested pin.
    BuiltIn,
}

/// How long a lookup answers for.
const LOOKUP_EVERY_SECS: i64 = crate::store::DAY_SECS;
/// How long a model that failed is left alone before it gets another chance.
const FAILED_FOR_SECS: i64 = 7 * crate::store::DAY_SECS;
/// Longest a `models` listing may take.
const LISTING_TIMEOUT: Duration = Duration::from_secs(15);

fn model_key(cli: &str) -> String {
    format!("summarizer_model:{cli}")
}

fn bad_key(cli: &str) -> String {
    format!("summarizer_model_bad:{cli}")
}

fn checked_key(cli: &str) -> String {
    format!("summarizer_model_checked:{cli}")
}

/// The models [`refresh_models`] left in the store, split by whether a call
/// has since shown them not to run.
#[derive(Debug, Default, Clone)]
pub struct Discovered {
    /// CLI to the newest id of its tier, minus the ones marked failed.
    newest: std::collections::HashMap<String, String>,
    /// CLI to the id found and marked failed, within the last week.
    failed: std::collections::HashMap<String, String>,
}

impl Discovered {
    /// Read what the store holds: a few keys, no process, no network.
    ///
    /// A mark names the id that failed, so a different id found since is not
    /// held back by it; and a stored id that is no longer above the pin (the
    /// pin was raised in a newer build) is ignored rather than trusted.
    #[must_use]
    pub fn read(store: &Store, now: jiff::Timestamp) -> Self {
        let mut found = Self::default();
        for spec in SPECS.iter().filter(|spec| spec.family.is_some()) {
            let Ok(Some(id)) = store.state(&model_key(spec.cli)) else { continue };
            if !is_newer_than_pin(spec, &id) {
                continue;
            }
            let marked = store
                .state(&bad_key(spec.cli))
                .ok()
                .flatten()
                .is_some_and(|mark| mark_holds(&mark, &id, now));
            let side = if marked { &mut found.failed } else { &mut found.newest };
            side.insert(spec.cli.to_string(), id);
        }
        found
    }

    /// The found model a call has shown not to run, if that is why a rung is
    /// on its pin.
    #[must_use]
    pub fn failed_model(&self, cli: &str) -> Option<&str> {
        self.failed.get(cli).map(String::as_str)
    }
}

/// Is this mark (`<id> <timestamp>`) about `id`, and recent enough to count?
fn mark_holds(mark: &str, id: &str, now: jiff::Timestamp) -> bool {
    let Some((marked, at)) = mark.split_once(' ') else { return false };
    marked == id
        && at
            .parse::<jiff::Timestamp>()
            .is_ok_and(|at| now.as_second() - at.as_second() < FAILED_FOR_SECS)
}

/// The model a rung runs and where it came from: the user's override, then the
/// newest found model, then the pin.
#[must_use]
pub fn choose_model<'a>(
    spec: &CliSpec,
    overrides: &'a std::collections::HashMap<String, String>,
    found: &'a Discovered,
) -> (&'a str, Origin) {
    if let Some(model) = overrides.get(spec.cli) {
        return (model, Origin::Config);
    }
    if let Some(model) = found.newest.get(spec.cli) {
        return (model, Origin::Newest);
    }
    (spec.model, Origin::BuiltIn)
}

/// The dotted number a family's pattern captures from `id`.
fn version_of(pattern: &regex::Regex, id: &str) -> Option<Vec<u64>> {
    pattern.captures(id)?.get(1)?.as_str().split('.').map(|part| part.parse().ok()).collect()
}

fn is_newer_than_pin(spec: &CliSpec, id: &str) -> bool {
    let Some(family) = &spec.family else { return false };
    pick_newest(family, spec.model, &[id.to_string()]).is_some()
}

/// The id of the family numbered highest, if any is numbered above the pin.
///
/// Numbers compare part by part, so `3.10` follows `3.9` and `6.1` follows
/// `6`. Nothing at or below the pin is returned: the pin is the tested floor,
/// and a lookup is only ever allowed to move up from it.
fn pick_newest(family: &Family, pin: &str, ids: &[String]) -> Option<String> {
    let pattern = regex::Regex::new(family.pattern).ok()?;
    let floor = version_of(&pattern, pin)?;
    ids.iter()
        .filter_map(|id| Some((version_of(&pattern, id)?, id)))
        .filter(|(version, _)| *version > floor)
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, id)| id.clone())
}

/// The ids codex lists for use: shown in its picker and not on their way out.
///
/// `None` when the text is not a model cache at all.
fn parse_codex_cache(text: &str) -> Option<Vec<String>> {
    let root: serde_json::Value = serde_json::from_str(text).ok()?;
    let models = root.get("models")?.as_array()?;
    Some(
        models
            .iter()
            .filter(|model| model["visibility"] == "list" && model["upgrade"].is_null())
            .filter_map(|model| model["slug"].as_str().map(str::to_string))
            .collect(),
    )
}

/// The ids in a `<id><TAB><name>` listing; the banner line has no tab.
fn parse_listing(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(id, _)| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .collect()
}

fn codex_cache_path() -> Option<PathBuf> {
    let home = std::env::var_os("CODEX_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".codex")))?;
    Some(home.join("models_cache.json"))
}

/// Run a listing command and read its ids. `None` on any failure, a timeout
/// included: a lookup that did not happen is not a lookup that found nothing.
fn list_models(spec: &CliSpec, args: &[&str]) -> Option<Vec<String>> {
    let args: Vec<String> = args.iter().map(|arg| (*arg).to_string()).collect();
    let mut child = spawn_headless(spec.program, &args, None).ok()?;
    let result = wait_with_timeout(&mut child, LISTING_TIMEOUT).ok()?;
    result.success.then(|| parse_listing(&result.stdout))
}

/// Every model this CLI lists, or `None` when it could not be asked.
fn discover(spec: &CliSpec) -> Option<Vec<String>> {
    let ids = match &spec.family.as_ref()?.listing {
        Listing::CodexCache => {
            parse_codex_cache(&std::fs::read_to_string(codex_cache_path()?).ok()?)?
        }
        Listing::Command(args) => list_models(spec, args)?,
    };
    // A listing that names nothing is a CLI that is not signed in or not
    // itself, not a verdict that no model exists.
    (!ids.is_empty()).then_some(ids)
}

/// Look for a newer model of each installed CLI's cheap tier, at most once a
/// day per CLI, and keep the answer in the store for [`Ladder::new`].
///
/// Only consolidation calls this: a lookup can start a process, and the hooks
/// and the MCP server must stay as quick as they are.
///
/// # Errors
/// Returns an error when the store cannot be read or written.
pub fn refresh_models(store: &Store) -> Result<()> {
    // A unit test that reaches a consolidation run must not read the host's
    // codex cache or start its `agy`; the lookup has its own tests through
    // `refresh_models_with`.
    if cfg!(test) {
        return Ok(());
    }
    refresh_models_with(store, jiff::Timestamp::now(), |spec| installed(spec.program), discover)
}

/// [`refresh_models`] with the clock, the installed check and the lookup
/// handed in, so a test can drive them without a CLI.
fn refresh_models_with(
    store: &Store,
    now: jiff::Timestamp,
    is_installed: impl Fn(&CliSpec) -> bool,
    list: impl Fn(&CliSpec) -> Option<Vec<String>>,
) -> Result<()> {
    for spec in SPECS {
        let Some(family) = &spec.family else { continue };
        if !is_installed(spec) {
            continue;
        }
        let looked_up = store
            .state(&checked_key(spec.cli))?
            .and_then(|at| at.parse::<jiff::Timestamp>().ok());
        if looked_up.is_some_and(|at| now.as_second() - at.as_second() < LOOKUP_EVERY_SECS) {
            continue;
        }
        // A failed lookup leaves both the pick and the date alone, so the
        // next consolidation asks again instead of waiting out a day.
        let Some(ids) = list(spec) else { continue };
        match pick_newest(family, spec.model, &ids) {
            Some(id) => store.set_state(&model_key(spec.cli), &id)?,
            None => store.clear_state(&model_key(spec.cli))?,
        }
        store.set_state(&checked_key(spec.cli), &now.to_string())?;
    }
    Ok(())
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

    /// `codex exec` inherits the user's config.toml. Measured on one machine
    /// whose default is effort `max` and a full-access sandbox: a 3 KB
    /// summary ran 38-77s and spent up to 3.6k reasoning tokens. The worker
    /// pins what it actually needs instead of inheriting what the user set
    /// for themselves.
    #[test]
    fn the_codex_worker_pins_low_effort_and_a_read_only_sandbox() {
        let spec = SPECS.iter().find(|spec| spec.cli == "codex").unwrap();
        let args = spec.args.join(" ");
        assert!(args.contains("-c model_reasoning_effort=\"low\""), "{args}");
        assert!(args.contains("-s read-only"), "{args}");
        assert!(args.contains("--skip-git-repo-check"), "{args}");
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

    /// A summarizer call must not leave a saved session in the host CLI's
    /// history: thousands of them piled up, each holding brain's prompt.
    #[test]
    fn claude_and_codex_workers_persist_no_session() {
        let claude = SPECS.iter().find(|spec| spec.cli == "claude-code").expect("claude spec");
        assert!(claude.args.contains(&"--no-session-persistence"));
        let codex = SPECS.iter().find(|spec| spec.cli == "codex").expect("codex spec");
        let exec = codex.args.iter().position(|a| *a == "exec").expect("exec");
        assert_eq!(codex.args[exec + 1], "--ephemeral", "an `exec` flag, so right after `exec`");
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
    fn a_worker_does_not_wear_the_identity_of_the_session_that_started_it() {
        // Measured, not reasoned about. Orca hosts the terminal the session
        // runs in and installs a `Stop` hook that posts every finished run to
        // a local port so it can be announced; it needs `ORCA_AGENT_HOOK_PORT`,
        // `ORCA_AGENT_HOOK_TOKEN` and `ORCA_PANE_KEY`, and it already declines
        // to report other tools' background work by checking for
        // `DEVIN_PROJECT_DIR` and `CLAUDE_JOB_DIR`. Passing its variables
        // through made every summariser look like the user's own pane
        // finishing a turn, one phone notification per summary, each carrying
        // the text of the summary.
        //
        // Dropping them lets Orca's own guard do the rest. Nothing of Orca's is
        // touched - an earlier attempt to solve this by editing its hook script
        // left a `.bak-brain-guard` beside a file that no longer differed from
        // it, because the tool had since replaced its own script.
        for name in [
            "ORCA_AGENT_HOOK_PORT",
            "ORCA_AGENT_HOOK_TOKEN",
            "ORCA_PANE_KEY",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDECODE",
            "CLAUDE_PLUGIN_DATA",
            "DEVIN_PROJECT_DIR",
        ] {
            assert!(is_host_session_var(name), "{name} would still reach the worker");
        }

        // Credentials are not identity. A user who set one meant it, and
        // taking it away would move their spend without asking.
        for name in ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "PATH", "HOME", "AWS_PROFILE"] {
            assert!(!is_host_session_var(name), "{name} must survive");
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

        // Absolute even from a relative `TMPDIR`, because it becomes `PWD`.
        assert!(inert_dir(PathBuf::from(".")).unwrap().is_absolute());

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
        let (tier, text) = waiting.run(&CallContext { purpose: "test", session: "" }, "anything", "codex", |_| true).unwrap();
        assert_eq!(tier, Tier::RuleBased, "a benched CLI was substituted for another");
        assert!(text.is_empty());
    }

    /// Every non-advisory invoke leaves one row: the ones that worked and
    /// every way one can fail. The ladder's answer is what it always was.
    #[test]
    fn every_invoke_leaves_one_row_and_clear_leaves_them_be() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("codex"));
        let ctx = CallContext { purpose: "consolidate", session: "s1" };
        let replies = std::cell::RefCell::new(vec![
            Ok("good answer".to_string()),
            Ok("usage limit reached".to_string()),
            Err(anyhow::anyhow!("{TIMED_OUT} 180s")),
            Err(anyhow::anyhow!("codex exited with 1: boom")),
            // A CLI's own stderr may say "timed out after": that is the CLI's
            // network, not this ladder's clock.
            Err(anyhow::anyhow!("codex exited with 1: request timed out after 30s")),
        ]);
        let mut tiers = Vec::new();
        for _ in 0..5 {
            let (tier, _) = ladder
                .run_with(
                    &ctx,
                    "prompt",
                    "codex",
                    |text| text.starts_with("good"),
                    |_| Ok(true),
                    |_, _, _, _| replies.borrow_mut().remove(0),
                )
                .unwrap();
            tiers.push(tier);
        }
        assert_eq!(tiers[0], Tier::Cli("codex".to_string()));
        assert!(tiers[1..].iter().all(|tier| *tier == Tier::RuleBased));

        let calls = store.summarizer_calls_since(60).unwrap();
        let outcomes: Vec<&str> = calls.iter().map(|call| call.outcome.as_str()).collect();
        assert_eq!(outcomes, ["ok", "unusable", "timeout", "spawn_error", "spawn_error"]);
        assert!(calls.iter().all(|c| c.purpose == "consolidate" && c.session == "s1"));
        assert!(calls.iter().all(|c| c.cli == "codex" && c.prompt_bytes == 6));
        assert_eq!(calls[0].answer_bytes, 11);
        assert_eq!(calls[2].answer_bytes, 0);

        // An advisory call is `rerank_runs`' business, not this table's.
        let waiting = ladder.while_waiting(Duration::from_secs(6));
        waiting
            .run_with(&ctx, "p", "codex", |_| true, |_| Ok(true), |_, _, _, _| Ok("x".to_string()))
            .unwrap();
        assert_eq!(store.summarizer_calls_since(60).unwrap().len(), 5);

        store.clear().unwrap();
        assert_eq!(store.summarizer_calls_since(60).unwrap().len(), 5, "clear() ate the ledger");
    }

    /// Drive one `run_with` over `reply` on an auto ladder where every CLI is
    /// reachable; returns the tier, how many rungs were invoked, the store.
    fn run_one_reply(reply: &str) -> (Tier, usize, Store) {
        let store = Store::open_memory().unwrap();
        let invoked = std::cell::Cell::new(0usize);
        let tier = {
            let ladder = Ladder::new(&store, &config("auto"));
            let ctx = CallContext { purpose: "consolidate", session: "s1" };
            ladder
                .run_with(
                    &ctx,
                    "prompt",
                    "codex",
                    |text| {
                        serde_json::from_str::<serde_json::Value>(text)
                            .ok()
                            .and_then(|v| v["summary"].as_str().map(|s| !s.trim().is_empty()))
                            .unwrap_or(false)
                    },
                    |_| Ok(true),
                    |_, _, _, _| {
                        invoked.set(invoked.get() + 1);
                        Ok(reply.to_string())
                    },
                )
                .unwrap()
                .0
        };
        (tier, invoked.get(), store)
    }

    /// A model that answered in JSON of the wrong shape did its job badly; it
    /// is not down. Charging the vendor would bench it for thirty minutes
    /// because one session's content did not fit.
    #[test]
    fn a_json_answer_of_the_wrong_shape_is_not_a_vendor_failure() {
        for reply in [
            "```json\n{\"wrong\":1}\n```",
            "```{\"wrong\":1}```",
            "{\"wrong\":1}",
            "{\"summary\":\"  \"}",
        ] {
            let (tier, invoked, store) = run_one_reply(reply);
            assert_eq!(tier, Tier::RuleBased, "{reply}");
            assert_eq!(invoked, 1, "{reply}: a second rung was spent on a content failure");
            assert!(store.summarizer_health().unwrap().is_empty(), "{reply}: the vendor was charged");
            let calls = store.summarizer_calls_since(60).unwrap();
            assert_eq!(calls.len(), 1, "{reply}");
            assert_eq!(calls[0].outcome, "unparseable", "{reply}");
        }
    }

    #[test]
    fn garbage_and_error_objects_still_count_against_the_vendor() {
        for reply in ["garbage", "{\"error\":{\"message\":\"quota exceeded\"}}", "{not json"] {
            let (tier, invoked, store) = run_one_reply(reply);
            assert_eq!(tier, Tier::RuleBased);
            assert_eq!(invoked, 2, "{reply}: the ladder did not try a second rung");
            let health = store.summarizer_health().unwrap();
            assert!(health.iter().all(|row| row.failures == 1) && health.len() == MAX_RUNGS_PER_CALL, "{reply}");
            let calls = store.summarizer_calls_since(60).unwrap();
            assert!(calls.iter().all(|call| call.outcome == "unusable"), "{reply}");
        }
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
        let (tier, text) = ladder.run(&CallContext { purpose: "test", session: "" }, "anything", "claude-code", |_| true).unwrap();
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
    fn an_oversized_prompt_stops_the_ladder_at_one_row() {
        let store = Store::open_memory().unwrap();
        let ladder = Ladder::new(&store, &config("auto"));
        let ctx = CallContext { purpose: "clean", session: "p" };
        let spawned = std::cell::Cell::new(0);
        let (tier, text) = ladder
            .run_with(
                &ctx,
                &"x".repeat(PROMPT_MAX_BYTES + 1),
                "claude-code",
                |_| true,
                |_| Ok(true),
                |_, _, _, _| {
                    spawned.set(spawned.get() + 1);
                    Ok("never".to_string())
                },
            )
            .unwrap();
        assert_eq!((tier, text.as_str()), (Tier::RuleBased, ""));
        assert_eq!(spawned.get(), 0);
        let calls = store.summarizer_calls_since(60).unwrap();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!((calls[0].purpose.as_str(), calls[0].outcome.as_str()), ("clean", "oversize"));
        // No CLI was marked sick for the caller's mistake.
        assert!(!store.summarizer_in_cooldown("claude-code").unwrap());
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

    fn spec_of(cli: &str) -> &'static CliSpec {
        SPECS.iter().find(|spec| spec.cli == cli).unwrap()
    }

    fn ids(list: &[&str]) -> Vec<String> {
        list.iter().map(|id| (*id).to_string()).collect()
    }

    /// A `models_cache.json` as codex writes it: the cheap tier twice over, a
    /// dearer tier numbered higher, a hidden luna, and a luna on its way out.
    const CODEX_CACHE: &str = r#"{"fetched_at":"x","models":[
        {"slug":"gpt-5.6-luna","visibility":"list","upgrade":null},
        {"slug":"gpt-6-luna","visibility":"list","upgrade":null},
        {"slug":"gpt-6.1-sol","visibility":"list","upgrade":null},
        {"slug":"gpt-7-luna","visibility":"hide","upgrade":null},
        {"slug":"gpt-6.2-luna","visibility":"list",
         "upgrade":{"model":"gpt-6.3-sol","retirement_at":"2026-10-14T19:00:00Z"}}
    ]}"#;

    #[test]
    fn the_newest_model_of_the_pinned_tier_wins_and_nothing_else_does() {
        let codex = spec_of("codex");
        let family = codex.family.as_ref().unwrap();

        // Hidden and retiring models are not offered; a higher number in
        // another tier (`sol`) is not the same tier.
        let listed = parse_codex_cache(CODEX_CACHE).expect("a cache that parses");
        assert!(!listed.contains(&"gpt-7-luna".to_string()), "a hidden model was offered");
        assert!(!listed.contains(&"gpt-6.2-luna".to_string()), "a retiring model was offered");
        assert_eq!(pick_newest(family, codex.model, &listed), Some("gpt-6-luna".to_string()));

        // Never below the pin, and the pin itself is not news.
        assert_eq!(pick_newest(family, codex.model, &ids(&["gpt-5.5-luna"])), None);
        assert_eq!(pick_newest(family, codex.model, &ids(&["gpt-5.6-luna"])), None);

        // Versions compare number by number: 3.10 is after 3.9, 6.1 after 6.
        let antigravity = spec_of("antigravity").family.as_ref().unwrap();
        assert_eq!(
            pick_newest(
                antigravity,
                "gemini-3.7-flash-low",
                &ids(&["gemini-3.9-flash-low", "gemini-3.10-flash-low", "gemini-3.10-flash-high"])
            ),
            Some("gemini-3.10-flash-low".to_string())
        );
        assert_eq!(
            pick_newest(family, "gpt-6-luna", &ids(&["gpt-6-luna", "gpt-6.1-luna"])),
            Some("gpt-6.1-luna".to_string())
        );
    }

    #[test]
    fn a_cache_that_is_missing_or_unreadable_names_no_models() {
        assert!(parse_codex_cache("not json").is_none());
        assert!(parse_codex_cache("{}").is_none());
    }

    #[test]
    fn the_antigravity_listing_is_read_by_its_tab_columns() {
        let out = "Fetching available models...\n\
                   gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n\
                   gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n\
                   gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n\
                   claude-sonnet-5\tClaude Sonnet 5\n\
                   a line with no tab\n";
        let listed = parse_listing(out);
        assert_eq!(listed.len(), 4, "{listed:?}");
        let family = spec_of("antigravity").family.as_ref().unwrap();
        assert_eq!(
            pick_newest(family, "gemini-3.7-flash-low", &listed),
            Some("gemini-3.8-flash-low".to_string())
        );
    }

    #[test]
    fn a_pin_is_a_member_of_its_own_family() {
        for spec in SPECS {
            if let Some(family) = &spec.family {
                let pattern = regex::Regex::new(family.pattern).unwrap();
                assert!(pattern.is_match(spec.model), "{}: pin {} is outside its family", spec.cli, spec.model);
            }
        }
    }

    fn found_in(store: &Store) -> Discovered {
        Discovered::read(store, jiff::Timestamp::now())
    }

    #[test]
    fn the_model_comes_from_config_then_the_newest_then_the_pin() {
        let store = Store::open_memory().unwrap();
        let codex = spec_of("codex");
        let none = std::collections::HashMap::new();

        // Nothing discovered, nothing configured: every CLI runs its pin.
        for spec in SPECS {
            assert_eq!(choose_model(spec, &none, &found_in(&store)), (spec.model, Origin::BuiltIn));
        }

        store.set_state("summarizer_model:codex", "gpt-6-luna").unwrap();
        assert_eq!(choose_model(codex, &none, &found_in(&store)), ("gpt-6-luna", Origin::Newest));
        let ladder = Ladder::new(&store, &config("auto"));
        assert_eq!(ladder.model_for(codex), "gpt-6-luna");
        assert_eq!(ladder.model_for(spec_of("claude-code")), "haiku", "another CLI is untouched");

        let mut overrides = std::collections::HashMap::new();
        overrides.insert("codex".to_string(), "gpt-5.6-sol".to_string());
        assert_eq!(choose_model(codex, &overrides, &found_in(&store)), ("gpt-5.6-sol", Origin::Config));
    }

    #[test]
    fn a_newest_model_marked_failed_is_skipped_until_the_mark_lapses_or_the_id_changes() {
        let store = Store::open_memory().unwrap();
        let codex = spec_of("codex");
        let none = std::collections::HashMap::new();
        let now = jiff::Timestamp::now();
        let days_ago = |days: i64| now - jiff::SignedDuration::from_secs(days * 24 * 3600);
        store.set_state("summarizer_model:codex", "gpt-6-luna").unwrap();

        store.set_state("summarizer_model_bad:codex", &format!("gpt-6-luna {}", days_ago(1))).unwrap();
        let found = Discovered::read(&store, now);
        assert_eq!(choose_model(codex, &none, &found), ("gpt-5.6-luna", Origin::BuiltIn));
        assert_eq!(found.failed_model("codex"), Some("gpt-6-luna"));

        // A different id was found since: the mark was about the old one.
        store.set_state("summarizer_model:codex", "gpt-6.1-luna").unwrap();
        assert_eq!(choose_model(codex, &none, &Discovered::read(&store, now)), ("gpt-6.1-luna", Origin::Newest));

        // And after a week the same id gets another chance.
        store.set_state("summarizer_model:codex", "gpt-6-luna").unwrap();
        store.set_state("summarizer_model_bad:codex", &format!("gpt-6-luna {}", days_ago(8))).unwrap();
        let found = Discovered::read(&store, now);
        assert_eq!(choose_model(codex, &none, &found), ("gpt-6-luna", Origin::Newest));
        assert_eq!(found.failed_model("codex"), None);
    }

    /// Run `run_with` on a codex-pinned ladder where `invoke` answers by the
    /// model it is handed; returns the tier, answer, models tried and the store.
    fn run_on_codex(
        advisory: bool,
        reply: impl Fn(&str) -> Result<String>,
        usable: impl Fn(&str) -> bool,
    ) -> (Tier, String, Vec<String>, Store) {
        let store = Store::open_memory().unwrap();
        store.set_state("summarizer_model:codex", "gpt-6-luna").unwrap();
        let tried = std::cell::RefCell::new(Vec::new());
        let (tier, text) = {
            let ladder = Ladder::new(&store, &config("codex"));
            let ladder = if advisory { ladder.while_waiting(Duration::from_secs(6)) } else { ladder };
            ladder
                .run_with(
                    &CallContext { purpose: "consolidate", session: "s1" },
                    "prompt",
                    "codex",
                    usable,
                    |_| Ok(true),
                    |_, model, _, _| {
                        tried.borrow_mut().push(model.to_string());
                        reply(model)
                    },
                )
                .unwrap()
        };
        (tier, text, tried.into_inner(), store)
    }

    #[test]
    fn a_newest_model_that_fails_while_the_pin_answers_is_marked_and_left_alone() {
        let reply = |model: &str| match model {
            "gpt-6-luna" => Err(anyhow::anyhow!("codex exited with 1: unknown model")),
            _ => Ok("good answer".to_string()),
        };
        let (tier, text, tried, store) = run_on_codex(false, reply, |text| text.starts_with("good"));

        assert_eq!(tier, Tier::Cli("codex".to_string()));
        assert_eq!(text, "good answer");
        assert_eq!(tried, ["gpt-6-luna", "gpt-5.6-luna"]);
        assert!(
            store.state("summarizer_model_bad:codex").unwrap().is_some_and(|mark| mark.starts_with("gpt-6-luna ")),
            "the failed id was not marked"
        );
        let calls = store.summarizer_calls_since(60).unwrap();
        let rows: Vec<(&str, &str)> = calls.iter().map(|c| (c.model.as_str(), c.outcome.as_str())).collect();
        assert_eq!(rows, [("gpt-6-luna", "spawn_error"), ("gpt-5.6-luna", "ok")]);
        let health = store.summarizer_health().unwrap();
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].failures, 0, "the pin answered: no strike for the CLI");

        // The same ladder goes straight to the pin from now on.
        let ladder = Ladder::new(&store, &config("codex"));
        assert_eq!(ladder.model_for(spec_of("codex")), "gpt-5.6-luna");
    }

    #[test]
    fn the_ladder_remembers_a_fallback_for_the_rest_of_its_run() {
        let store = Store::open_memory().unwrap();
        store.set_state("summarizer_model:codex", "gpt-6-luna").unwrap();
        let ladder = Ladder::new(&store, &config("codex"));
        let tried = std::cell::RefCell::new(Vec::new());
        for _ in 0..2 {
            ladder
                .run_with(
                    &CallContext { purpose: "consolidate", session: "s1" },
                    "prompt",
                    "codex",
                    |_| true,
                    |_| Ok(true),
                    |_, model, _, _| {
                        tried.borrow_mut().push(model.to_string());
                        if model == "gpt-6-luna" { Err(anyhow::anyhow!("boom")) } else { Ok("x".to_string()) }
                    },
                )
                .unwrap();
        }
        assert_eq!(tried.into_inner(), ["gpt-6-luna", "gpt-5.6-luna", "gpt-5.6-luna"]);
    }

    #[test]
    fn when_the_pin_fails_too_the_cli_failed_once_and_nothing_is_marked() {
        let (tier, _, tried, store) =
            run_on_codex(false, |_| Err(anyhow::anyhow!("codex exited with 1: down")), |_| true);

        assert_eq!(tier, Tier::RuleBased);
        assert_eq!(tried, ["gpt-6-luna", "gpt-5.6-luna"]);
        assert_eq!(store.state("summarizer_model_bad:codex").unwrap(), None);
        let health = store.summarizer_health().unwrap();
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].failures, 1, "one failed call is one strike, not two");
        assert_eq!(store.summarizer_calls_since(60).unwrap().len(), 2);
    }

    #[test]
    fn a_content_failure_on_the_newest_model_is_not_retried() {
        let (tier, _, tried, store) = run_on_codex(false, |_| Ok("{\"wrong\":1}".to_string()), |_| false);

        assert_eq!(tier, Tier::RuleBased);
        assert_eq!(tried, ["gpt-6-luna"]);
        assert_eq!(store.state("summarizer_model_bad:codex").unwrap(), None);
    }

    #[test]
    fn an_advisory_call_never_retries_on_the_pin_or_marks_a_model() {
        let (tier, _, tried, store) =
            run_on_codex(true, |_| Err(anyhow::anyhow!("codex exited with 1: down")), |_| true);

        assert_eq!(tier, Tier::RuleBased);
        assert_eq!(tried, ["gpt-6-luna"], "a favour asked mid-search was asked twice");
        assert_eq!(store.state("summarizer_model_bad:codex").unwrap(), None);
    }

    #[test]
    fn models_are_looked_up_at_most_once_a_day_and_a_failed_lookup_does_not_count() {
        let store = Store::open_memory().unwrap();
        let now = jiff::Timestamp::now();
        let later = |hours: i64| now + jiff::SignedDuration::from_secs(hours * 3600);
        let looked_up = std::cell::RefCell::new(Vec::new());
        let codex_only = |spec: &CliSpec| spec.cli == "codex";
        let lists = |spec: &CliSpec| {
            looked_up.borrow_mut().push(spec.cli);
            Some(ids(&["gpt-5.6-luna", "gpt-6-luna"]))
        };

        refresh_models_with(&store, now, codex_only, lists).unwrap();
        assert_eq!(store.state("summarizer_model:codex").unwrap().as_deref(), Some("gpt-6-luna"));
        // Only the installed CLI with a family is looked up.
        assert_eq!(*looked_up.borrow(), ["codex"]);

        refresh_models_with(&store, later(23), codex_only, lists).unwrap();
        assert_eq!(looked_up.borrow().len(), 1, "looked up twice inside a day");
        refresh_models_with(&store, later(25), codex_only, lists).unwrap();
        assert_eq!(looked_up.borrow().len(), 2);

        // A lookup that fails keeps what was known and is tried again next time.
        refresh_models_with(&store, later(50), codex_only, |_| None).unwrap();
        assert_eq!(store.state("summarizer_model:codex").unwrap().as_deref(), Some("gpt-6-luna"));
        refresh_models_with(&store, later(51), codex_only, lists).unwrap();
        assert_eq!(looked_up.borrow().len(), 3, "a failed lookup was counted as done");

        // A list with nothing newer than the pin forgets the old pick.
        refresh_models_with(&store, later(80), codex_only, |_| Some(ids(&["gpt-5.5-luna"]))).unwrap();
        assert_eq!(store.state("summarizer_model:codex").unwrap(), None);
    }

    #[test]
    fn a_missing_program_is_not_installed() {
        assert!(!installed("definitely-not-a-real-program-xyz"));
    }
}
