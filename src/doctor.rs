//! `brain doctor` — prove the wiring works, or say exactly what is missing.
//!
//! A memory system that silently stops capturing is worse than one that was
//! never installed, because the user keeps trusting it. This command exists so
//! that failure is always one command away from being visible.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::config::{Config, Paths};
use crate::store::Store;

/// One check and its outcome.
pub struct Check {
    pub ok: bool,
    pub name: String,
    pub detail: String,
}

impl Check {
    fn pass(name: &str, detail: impl Into<String>) -> Self {
        Self { ok: true, name: name.to_string(), detail: detail.into() }
    }
    fn fail(name: &str, detail: impl Into<String>) -> Self {
        Self { ok: false, name: name.to_string(), detail: detail.into() }
    }
}

/// Run every check.
///
/// # Errors
/// Returns an error only when the data directory cannot be resolved; every
/// other problem is reported as a failed check, because the point of this
/// command is to enumerate problems rather than stop at the first.
pub fn run() -> Result<Vec<Check>> {
    let paths = Paths::resolve()?;
    let mut checks = Vec::new();

    checks.push(if paths.data_dir.is_dir() {
        Check::pass("data directory", paths.data_dir.display().to_string())
    } else {
        Check::fail(
            "data directory",
            format!("{} does not exist — run `brain setup --apply`", paths.data_dir.display()),
        )
    });

    match Config::load(&paths.config_file()) {
        Ok(config) => checks.push(Check::pass(
            "config",
            format!(
                "summarizer={} primer_budget={}B session_budget={}B",
                config.summarizer.mode,
                config.injection.primer_budget,
                config.injection.session_budget
            ),
        )),
        Err(error) => checks.push(Check::fail("config", error.to_string())),
    }

    match Store::open(&paths.db()) {
        Ok(store) => {
            let total = store.count().unwrap_or(0);
            let by_cli = store.counts_by_cli().unwrap_or_default();
            if total == 0 {
                checks.push(Check::fail(
                    "capture",
                    "no events indexed yet — start a session in a wired CLI, then re-run",
                ));
            } else {
                let breakdown = by_cli
                    .iter()
                    .map(|(cli, count)| format!("{cli}={count}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                checks.push(Check::pass("capture", format!("{total} events ({breakdown})")));
            }
        }
        Err(error) => checks.push(Check::fail("index", error.to_string())),
    }

    checks.extend(split_tree_check(&paths));
    checks.push(injection_check(&paths));
    checks.push(taxonomy_check(&paths));
    checks.extend(summarizer_checks(&paths));
    checks.push(semantic_check(&paths));
    checks.push(reranker_check(&paths));
    checks.extend(hook_checks());
    checks.extend(trigger_checks());
    checks.push(timer_check());
    checks.push(resident_check());
    checks.push(error_log_check(&paths.log_file()));

    Ok(checks)
}

/// How much memory has actually been classified.
///
/// A model that quietly stops emitting `kind` degrades the primer from a typed
/// index back to a flat list, and nothing else would report it — the answers
/// still parse, the summaries still land. This is the check that would notice.
fn taxonomy_check(paths: &Paths) -> Check {
    let Ok(store) = Store::open(&paths.db()) else {
        return Check::fail("taxonomy", "index unreadable");
    };
    let Ok(counts) = store.topic_counts() else {
        return Check::fail("taxonomy", "could not read topic counts");
    };
    if counts.is_empty() {
        return Check::pass("taxonomy", "nothing classified yet (consolidation assigns it)");
    }
    let detail = counts
        .iter()
        .map(|(topic, count)| format!("{topic}={count}"))
        .collect::<Vec<_>>()
        .join(" ");
    Check::pass("taxonomy", detail)
}

/// Is there exactly one wiki tree?
///
/// Two ways to get a second one, both observed on a real machine in one day:
/// the vault gets renamed inside Obsidian (which renames the real directory,
/// so the next hook finds nothing and starts a fresh tree), or a hook races
/// a layout migration and recreates the legacy home it was mid-write to.
/// Either way, new memory quietly lands in a tree recall never reads - a
/// split brain, and nothing else in this report would show it.
fn split_tree_check(paths: &Paths) -> Vec<Check> {
    let pretty = paths.data_dir.join(crate::config::WIKI_DIR);
    let legacy = paths.data_dir.join(crate::config::LEGACY_WIKI_DIR);
    if pretty.is_dir() && legacy.is_dir() {
        return vec![Check::fail(
            "wiki tree",
            format!(
                "both {} and {} exist — capture reads only the first. Merge them \
                 (brain import --merge can help), then re-run brain reindex",
                pretty.display(),
                legacy.display()
            ),
        )];
    }
    // A `default/` directory inside the wiki that still holds projects is
    // the same disease at the next level down: `brain reindex` fixes it.
    let stale = paths.wiki().join("default");
    if stale.is_dir() && std::fs::read_dir(&stale).is_ok_and(|mut dir| dir.next().is_some()) {
        return vec![Check::fail(
            "wiki tree",
            "a legacy default/ level still holds projects — run brain reindex",
        )];
    }
    Vec::new()
}

/// What automatic injection has actually cost, measured rather than assumed.
///
/// The design commits to a per-session ceiling; this is where that promise is
/// checked against reality instead of against the config file.
fn injection_check(paths: &Paths) -> Check {
    let Ok(store) = Store::open(&paths.db()) else {
        return Check::fail("injection", "index unreadable");
    };
    let config = Config::load(&paths.config_file()).unwrap_or_default();
    let Ok((sessions, total, worst)) = store.injection_stats() else {
        return Check::fail("injection", "could not read injection stats");
    };

    if sessions == 0 {
        return Check::pass("injection", "nothing injected yet");
    }
    let mean = total / sessions;
    let budget = i64::try_from(config.injection.session_budget).unwrap_or(i64::MAX);
    // The worst number on record has no age of its own - a session's spend
    // is permanent once written - so without this a bug fixed today reads
    // identically to one happening right now. The age is reported, not used
    // to downgrade the check: this really did happen, and saying when is
    // more honest than either hiding it or crying wolf forever.
    let age = store
        .worst_injection_at()
        .ok()
        .flatten()
        .and_then(|ts| failure_age(Some(&ts)))
        .map_or_else(String::new, |age| format!(" ({age})"));
    let detail = format!("{sessions} session(s), mean {mean}B, worst {worst}B{age}, cap {budget}B");
    if worst > budget {
        Check::fail("injection", format!("{detail} - OVER BUDGET"))
    } else {
        Check::pass("injection", detail)
    }
}

/// Human-readable age of the last failure.
fn failure_age(at: Option<&str>) -> Option<String> {
    let at: jiff::Timestamp = at?.parse().ok()?;
    let seconds = jiff::Timestamp::now().as_second() - at.as_second();
    Some(match seconds {
        ..=90 => "just now".to_string(),
        s if s < 3600 => format!("{}m ago", s / 60),
        s if s < 86_400 => format!("{}h ago", s / 3600),
        s => format!("{}d ago", s / 86_400),
    })
}

/// Which summarizer rungs are actually reachable.
///
/// Model ids rot when a CLI upgrades, and a wrong id fails exactly like an
/// outage. This lists the table so a rotted entry is visible rather than
/// silently degrading every session to rule-based output.
/// Whether a local reranker is present, and what its absence costs.
///
/// Absence is the ordinary state and never a failure: reranking is off by
/// default, the weights are fetched only when someone first asks for one, and
/// a build for a target `ort` publishes no binaries for cannot use them at
/// all. What this reports is which of those a machine is in — because the
/// difference between "reranking takes 1.5 seconds" and "reranking takes
/// twelve" is otherwise invisible until someone waits through it.
fn reranker_check(paths: &Paths) -> Check {
    let dir = paths.model_dir_for(crate::rerank::LOCAL_MODEL);
    if !cfg!(feature = "local-rerank") {
        // Not a fault and not a missing install - only a build that was asked
        // to leave the code out. Reranking still works, through the CLI, at
        // the price it has always cost.
        return Check::pass(
            "reranker",
            "not built into this binary - reranking uses the host CLI (~12s)",
        );
    }
    if crate::rerank::local_is_ready(&dir) {
        return Check::pass("reranker", format!("local, ready ({})", dir.display()));
    }
    if dir.with_extension("fetching").exists() {
        return Check::pass("reranker", "downloading - the CLI answers until it lands");
    }
    // Naming the missing half is worth the extra line. Weights without a
    // runtime is what an upgrade from a statically linked build looks like,
    // and it is otherwise indistinguishable from having downloaded nothing.
    #[cfg(feature = "local-rerank")]
    if dir.join(crate::xencoder::WEIGHTS_FILE).is_file() {
        return Check::pass(
            "reranker",
            "weights are here but ONNX Runtime is not - the next rerank fetches it",
        );
    }
    // Windows can load the runtime but has no installer script to go and get
    // it, so the sentence that is true everywhere else - that asking once
    // starts the download - would be a promise nothing keeps.
    if cfg!(windows) {
        return Check::pass(
            "reranker",
            "not fetched yet - reranking uses the host CLI until the model is placed by hand",
        );
    }
    Check::pass(
        "reranker",
        "not fetched yet - the first rerank asks the CLI and starts the download",
    )
}

/// How much of the corpus can be searched by meaning rather than by words.
///
/// Worth reporting because it is the one part of recall that lags capture on
/// purpose: vectors are written by consolidation, so a database that has just
/// been imported, reindexed, or upgraded is briefly keyword-only. Someone
/// wondering why a search feels shallow should be able to see that here rather
/// than guess at it.
fn semantic_check(paths: &Paths) -> Check {
    let Ok(store) = Store::open(&paths.db()) else {
        return Check::fail("semantic", "index unreadable");
    };
    let Ok((embedded, total)) = store.vector_coverage() else {
        return Check::fail("semantic", "coverage query failed");
    };
    // A model that has not arrived yet is not a broken install. Recall keeps
    // four of its five rankings without it - words, declared entities, shared
    // neighbours, and substring matching - so this is the one thing `doctor`
    // reports as missing rather than failed, with the command that fixes it.
    let model = paths.model_dir();
    if !model.join(crate::embed::WEIGHTS_FILE).is_file() {
        // Windows has no installer script of its own yet, so pointing a
        // Windows reader at a shell pipeline is worse than saying nothing -
        // it is a command that cannot work, printed by a tool they are
        // consulting because something already did not. Name the files and
        // where they live instead.
        let how = if cfg!(windows) {
            format!(
                "Download model-int8.safetensors and tokenizer.json from the latest \
                 release at https://github.com/nuttaruj/rolepod-brain/releases and put \
                 them in {}",
                model.display()
            )
        } else {
            "Fetch it once: curl -fsSL \
             https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh \
             | sh -s -- --model-only"
                .to_string()
        };
        return Check::pass(
            "semantic",
            format!(
                "the embedding model is not in {} yet, so search is running on words, \
                 entities and neighbours but not meaning. {how}",
                model.display()
            ),
        );
    }
    // Ask the model to load before reporting on an index it produced. A
    // coverage number is a fact about the past; whether anything can still be
    // embedded or searched is a fact about this binary.
    if let Err(error) = crate::embed::readiness() {
        return Check::fail("semantic", format!("{error} — {embedded} of {total} embedded"));
    }
    let dims = crate::embed::DIMS;
    if total == 0 {
        return Check::pass("semantic", format!("nothing captured yet ({dims} dims ready)"));
    }
    #[allow(clippy::cast_precision_loss)]
    let percent = embedded as f64 / total as f64 * 100.0;
    let detail = format!("{embedded} of {total} event(s) embedded ({percent:.0}%, {dims} dims)");
    if embedded == 0 {
        // Not a failure: keyword search still works, and the next
        // consolidation fills this in without anyone doing anything.
        return Check::pass("semantic", format!("{detail} — pending first consolidation"));
    }
    Check::pass("semantic", detail)
}

/// How one rung should read in the health line, overrides applied.
///
/// A rung with no model of ours runs on whatever that CLI is configured for.
/// Printing `opencode=` reads as a broken lookup; saying so reads as the fact
/// it is. The override is consulted only for rungs that pass a model at all -
/// reading it for the others would print a name the call never sends, which is
/// the same report about a machine that does not exist, arrived at from the
/// other direction.
fn model_label(spec: &crate::summarizer::CliSpec, overrides: &HashMap<String, String>) -> String {
    if !spec.passes_a_model() {
        return format!("{}=(its own default)", spec.cli);
    }
    let model = overrides.get(spec.cli).map_or(spec.model, String::as_str);
    format!("{}={model}", spec.cli)
}

fn summarizer_checks(paths: &Paths) -> Vec<Check> {
    let mut checks = Vec::new();
    // Effective models, overrides applied: a report that shows the spec's
    // default while config runs something else is a report about a machine
    // that does not exist.
    let summarizer_cfg = Config::load(&paths.config_file()).unwrap_or_default().summarizer;
    let (installed, missing): (Vec<_>, Vec<_>) = crate::summarizer::SPECS
        .iter()
        .partition(|spec| crate::summarizer::installed(spec.program));
    let a_model_is_installed = !installed.is_empty();
    let installed: Vec<String> =
        installed.iter().map(|spec| model_label(spec, &summarizer_cfg.models)).collect();

    if installed.is_empty() {
        checks.push(Check::fail(
            "summarizer",
            "no supported CLI on PATH — consolidation stays rule-based",
        ));
    } else {
        // The rungs that are NOT here are stated, not omitted. A row that
        // lists only what exists invites filling the silence: a machine with
        // Codex the desktop app and no `codex` binary read as having a
        // fallback it did not have, and the outage that followed looked like
        // the ladder refusing to cascade.
        let mut detail = installed.join(" ");
        if !missing.is_empty() {
            let absent: Vec<&str> = missing.iter().map(|spec| spec.cli).collect();
            detail.push_str(&format!(" — not on this machine: {}", absent.join(", ")));
        }
        checks.push(Check::pass("summarizer", detail));
    }

    // A rung the current table no longer names - renamed or dropped - cannot
    // fail again, because nothing will ever record success OR failure
    // against that key again. Reporting it is not a live warning; it is
    // orphaned state outliving the identity that wrote it, exactly the class
    // of bug the "gemini" -> "gemini-cli" rename itself produced: that rename
    // is what orphaned this row in the first place.
    let live_rungs: Vec<&str> = crate::summarizer::SPECS.iter().map(|spec| spec.cli).collect();

    if let Ok(store) = Store::open(&paths.db()) {
        if let Some(check) = consolidation_check(
            &store.consolidation_tiers().unwrap_or_default(),
            &summarizer_cfg.mode,
            a_model_is_installed,
        ) {
            checks.push(check);
        }
        for health in store.summarizer_health().unwrap_or_default() {
            if health.failures == 0 || !live_rungs.contains(&health.cli.as_str()) {
                continue;
            }
            let cli = health.cli;
            let failures = health.failures;
            let cooling = store.summarizer_in_cooldown(&cli).unwrap_or(false);
            let age = failure_age(health.last_failed_at.as_deref());
            let last_error = health.last_error;
            // A per-CLI breaker only clears when that CLI is next used, so a
            // rung nobody has exercised keeps its last error indefinitely.
            // Without an age, a failure fixed hours ago is indistinguishable
            // from one happening now, and a report like that gets ignored.
            checks.push(Check::fail(
                &format!("summarizer: {cli}"),
                format!(
                    "{failures} consecutive failure(s){}{}: {}",
                    if cooling { ", in cooldown" } else { "" },
                    age.map_or(String::new(), |age| format!(", last {age}")),
                    last_error.unwrap_or_default()
                ),
            ));
        }
    }
    checks
}

/// Who has actually answered consolidation, from each session's last run.
///
/// The `capture` row counts events each CLI *produced*; nothing in the report
/// said which CLI *answered*. That silence got filled: a reader took
/// `codex=1` under capture as "the ladder never reached codex" and called the
/// fallback broken, while the proof it worked sat in
/// `session_state.last_tier`, where only a SQL query would find it. The
/// report states it instead.
///
/// One shape IS a live warning: every session floored at rule-based while a
/// model is installed and enabled. That is what a hook running under a PATH
/// that hides every CLI looks like from the inside, and nothing else in the
/// report can see it - the summarizer row checks THIS process's PATH, which
/// is usually a terminal's, not the hook's.
fn consolidation_check(
    tiers: &[(String, i64)],
    mode: &str,
    a_model_is_installed: bool,
) -> Option<Check> {
    if tiers.is_empty() {
        // The capture row already says nothing has happened yet.
        return None;
    }
    let tally = tiers
        .iter()
        .map(|(tier, count)| format!("{tier}={count}"))
        .collect::<Vec<_>>()
        .join(" ");
    let stuck =
        mode != "off" && a_model_is_installed && tiers.iter().all(|(tier, _)| tier == "rule-based");
    if stuck {
        return Some(Check::fail(
            "consolidation",
            format!(
                "{tally} — a model is installed and enabled, yet no session has ever been \
                 answered by one; the hook likely runs under a PATH that cannot see any CLI"
            ),
        ));
    }
    Some(Check::pass("consolidation", format!("answered by: {tally} (each session's last run)")))
}

/// Is our wiring actually present, for every CLI we know how to wire?
///
/// Derived from the same target table `setup` writes from, so a newly
/// supported CLI cannot be silently absent from the health report — the gap
/// that would let capture be broken for one CLI while doctor stayed green.
fn hook_checks() -> Vec<Check> {
    let Ok(exe) = std::env::current_exe() else {
        return vec![Check::fail("hooks", "cannot locate our own binary")];
    };
    let Ok(targets) = crate::setup::targets(&exe) else {
        return vec![Check::fail("hooks", "cannot resolve home directory")];
    };

    let path_var = std::env::var_os("PATH").unwrap_or_default();
    targets
        .into_iter()
        // A CLI that is not installed is not a problem to report.
        .filter(crate::setup::config_dir_present)
        .map(|target| {
            let name = format!("hooks: {}", target.kind);
            let path = &target.hooks_file;

            // A directory without the CLI is an IDE leftover or an old
            // install, not something to wire — and never a FAIL: this can run
            // inside an MCP server spawned with a minimal PATH, and a red row
            // about the machine's PATH would teach people to ignore red rows.
            if !crate::setup::binary_present(&target, &path_var) {
                return Check::pass(
                    &name,
                    format!(
                        "`{}` is not on PATH — nothing captures here, and `brain setup` \
                         skips it",
                        target.binaries.join("`/`"),
                    ),
                );
            }

            // Codex installs through its own plugin flow, which is also what
            // grants the hooks permission to run - so the question is whether
            // that plugin is installed, not whether a config file mentions us.
            if target.layout == crate::setup::Layout::External {
                return if crate::setup::plugin_installed(&target.kind) {
                    Check::pass(&name, "installed and enabled as a Codex plugin")
                } else {
                    Check::fail(
                        &name,
                        "not installed. Codex will not run hooks it has not trusted, and it \
                         trusts a plugin's bundled hooks: `codex plugin marketplace add \
                         nuttaruj/rolepod-brain && codex plugin add rolepod-brain@rolepod-brain`",
                    )
                };
            }

            // A plugin target is a file we own outright: present or not.
            if target.layout == crate::setup::Layout::Plugin {
                return if path.is_file() {
                    Check::pass(&name, format!("plugin installed: {}", path.display()))
                } else {
                    Check::fail(&name, "plugin missing — run `brain setup --apply`")
                };
            }

            let Ok(text) = std::fs::read_to_string(path) else {
                return Check::fail(
                    &name,
                    format!("{} not readable — run `brain setup --apply`", path.display()),
                );
            };
            let Ok(root) = serde_json::from_str::<Value>(&text) else {
                return Check::fail(&name, format!("{} is not valid JSON", path.display()));
            };

            // Grouped and flat layouts nest under "hooks"; namespaced ones put
            // us under our own key.
            let container = match target.layout {
                crate::setup::Layout::Namespaced => root.get("brain"),
                _ => root.get("hooks"),
            };
            let wired: Vec<String> = container
                .and_then(Value::as_object)
                .map(|events| {
                    events
                        .iter()
                        .filter(|(_, entries)| {
                            serde_json::to_string(entries)
                                .unwrap_or_default()
                                .contains("brain hook")
                        })
                        .map(|(event, _)| event.clone())
                        .collect()
                })
                .unwrap_or_default();

            if wired.is_empty() {
                // An installed plugin carries the hooks itself, and `setup`
                // stands down for it on purpose. Reporting that as "not wired,
                // run setup" would be a failure about a working machine, and
                // the remedy it names does nothing — which teaches people to
                // stop reading this output.
                if let Some(events) = crate::setup::plugin_hook_events(&target.kind) {
                    return Check::pass(
                        &name,
                        format!("{} event(s) via the plugin: {}", events.len(), events.join(", ")),
                    );
                }
                return Check::fail(
                    &name,
                    format!("not wired in {} — run `brain setup --apply`", path.display()),
                );
            }

            Check::pass(&name, format!("{} event(s): {}", wired.len(), wired.join(", ")))
        })
        .collect()
}

/// When each wired CLI actually calls a model.
///
/// A model call is the one thing here that costs the user something, so what
/// causes one should not require reading the source to find out.
fn trigger_checks() -> Vec<Check> {
    let Ok(exe) = std::env::current_exe() else { return Vec::new() };
    let Ok(targets) = crate::setup::targets(&exe) else { return Vec::new() };
    targets
        .into_iter()
        .filter(|target| {
            crate::setup::config_dir_present(target)
                && crate::setup::binary_present(
                    target,
                    &std::env::var_os("PATH").unwrap_or_default(),
                )
        })
        .map(|target| {
            let cli = target.kind.as_str().to_string();
            Check::pass(
                &format!("consolidates: {cli}"),
                crate::hook::consolidation_triggers(&cli),
            )
        })
        .collect()
}

/// The backstop is hook-opportunistic, and nothing else may exist.
///
/// The wall-clock timer feature is removed. A machine an older version
/// installed it on still has the launchd job, though - and an orphaned job
/// that wakes a binary which no longer knows why is exactly what this report
/// exists to surface.
fn timer_check() -> Check {
    let Some(home) = dirs::home_dir() else {
        return Check::fail("backstop", "cannot determine home directory");
    };
    let plist = home.join("Library/LaunchAgents/dev.rolepod.brain.consolidate.plist");
    if plist.is_file() {
        return Check::fail(
            "backstop",
            format!(
                "a launchd job from an older version is still installed ({}) - run `brain setup --apply` to remove it",
                plist.display()
            ),
        );
    }
    Check::pass(
        "backstop",
        "hook-opportunistic - a session opening finishes stale work; nothing registered to run in the background",
    )
}

/// What of ours is running, if anything.
///
/// Named for what it reports rather than what it promises. It used to be
/// called "no resident process", which read as a contradiction the moment
/// anything was running - `no resident process  7 live` sent the first person
/// to read it looking for a leak. There is no daemon to assert the absence of;
/// the honest line is the count and what those processes are.
///
/// This check runs inside a `brain` process, so it must not count itself.
fn resident_check() -> Check {
    let Some(running) = running_brains() else {
        return Check::fail("processes", "could not list processes");
    };
    let me = std::process::id();
    let stray: Vec<String> =
        running.into_iter().filter(|pid| *pid != me).map(|pid| format!("pid {pid}")).collect();

    if stray.is_empty() {
        Check::pass("processes", "none running")
    } else {
        Check::pass(
            "processes",
            format!(
                "{} MCP server(s), one per open session — {}",
                stray.len(),
                stray.join(", ")
            ),
        )
    }
}

/// Every `brain` process on this machine, or `None` if we could not ask.
///
/// Two spellings of one question. `ps` reports a command that may be a path,
/// so the name is taken from its last component; `tasklist` reports an image
/// name that is already bare and already carries `.exe`. Neither needs a
/// crate, and both are present on a machine that can run this at all.
#[cfg(unix)]
fn running_brains() -> Option<Vec<u32>> {
    let output = std::process::Command::new("ps").args(["-Ao", "pid=,comm="]).output().ok()?;
    Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let mut parts = line.trim().splitn(2, char::is_whitespace);
                let pid: u32 = parts.next()?.parse().ok()?;
                let command = parts.next()?.trim();
                let name = Path::new(command).file_name()?.to_string_lossy();
                (name == "brain").then_some(pid)
            })
            .collect(),
    )
}

#[cfg(windows)]
fn running_brains() -> Option<Vec<u32>> {
    // `/NH` drops the header, `/FO CSV` quotes every field, and the two we
    // want are the first: image name, then pid.
    let output = std::process::Command::new("tasklist")
        .args(["/FO", "CSV", "/NH", "/FI", "IMAGENAME eq brain.exe"])
        .output()
        .ok()?;
    Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let mut fields = line.split("\",\"");
                let name = fields.next()?.trim_start_matches('"');
                if !name.eq_ignore_ascii_case("brain.exe") {
                    return None;
                }
                fields.next()?.parse().ok()
            })
            .collect(),
    )
}

/// Recent capture failures. Hooks never print to the host CLI, so this file is
/// the only place they surface.
fn error_log_check(path: &Path) -> Check {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Check::pass("capture errors", "none recorded");
    };
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    if lines.is_empty() {
        return Check::pass("capture errors", "none recorded");
    }
    let recent = lines.iter().rev().take(3).rev().copied().collect::<Vec<_>>().join("\n    ");
    Check::fail(
        "capture errors",
        format!("{} in {}\n    {recent}", lines.len(), path.display()),
    )
}

/// Render checks for a terminal. Returns the report and whether all passed.
#[must_use]
pub fn render(checks: &[Check]) -> (String, bool) {
    let mut out = String::new();
    let mut all_ok = true;
    for check in checks {
        if !check.ok {
            all_ok = false;
        }
        let mark = if check.ok { "ok  " } else { "FAIL" };
        let _ = writeln!(out, "{mark} {:<20} {}", check.name, check.detail);
    }
    (out, all_ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_marks_failures_and_reports_overall_status() {
        let checks = vec![
            Check::pass("a", "fine"),
            Check::fail("b", "broken"),
        ];
        let (out, ok) = render(&checks);
        assert!(!ok);
        assert!(out.contains("ok   a"));
        assert!(out.contains("FAIL b"));
    }

    #[test]
    fn all_passing_reports_ok() {
        let (_, ok) = render(&[Check::pass("a", "fine")]);
        assert!(ok);
    }

    #[test]
    fn a_rung_that_passes_no_model_reports_its_own_default_whatever_config_says() {
        // The report has to survive a user naming a model for a rung that
        // never sends one. Showing that name would claim the ladder runs
        // something it does not - and the fix is not to drop the override
        // silently but to say which rungs it can reach, which is what the
        // label does.
        let unpinned = crate::summarizer::SPECS
            .iter()
            .find(|spec| !spec.passes_a_model())
            .expect("at least one rung runs on its CLI's own default");

        let mut overrides = HashMap::new();
        assert_eq!(
            model_label(unpinned, &overrides),
            format!("{}=(its own default)", unpinned.cli)
        );

        overrides.insert(unpinned.cli.to_string(), "composer-2.5".to_string());
        assert_eq!(
            model_label(unpinned, &overrides),
            format!("{}=(its own default)", unpinned.cli),
            "an override on a rung with no `{{model}}` argument must not be reported as running"
        );

        let pinned = crate::summarizer::SPECS
            .iter()
            .find(|spec| spec.passes_a_model())
            .expect("at least one rung names a model");
        let mut overrides = HashMap::new();
        overrides.insert(pinned.cli.to_string(), "sonnet".to_string());
        assert_eq!(model_label(pinned, &overrides), format!("{}=sonnet", pinned.cli));
    }

    #[test]
    fn missing_error_log_is_a_pass_not_a_failure() {
        let check = error_log_check(Path::new("/nonexistent/brain.log"));
        assert!(check.ok);
    }

    #[test]
    fn the_consolidation_row_names_the_tier_that_answered() {
        let tiers = vec![("claude-code".to_string(), 41), ("codex".to_string(), 3)];
        let check = consolidation_check(&tiers, "auto", true).expect("sessions exist");
        assert!(check.ok);
        assert!(check.detail.contains("codex=3"), "{}", check.detail);
    }

    #[test]
    fn nothing_consolidated_yet_adds_no_consolidation_row() {
        assert!(consolidation_check(&[], "auto", true).is_none());
    }

    #[test]
    fn every_session_stuck_on_rule_based_fails_only_when_a_model_could_answer() {
        let stuck = vec![("rule-based".to_string(), 7)];
        assert!(
            !consolidation_check(&stuck, "auto", true).expect("row").ok,
            "a model nobody ever reaches is the reported outage, not health"
        );
        // mode off: rule-based is what the user asked for.
        assert!(consolidation_check(&stuck, "off", true).expect("row").ok);
        // No CLI installed: the summarizer row already fails, and louder.
        assert!(consolidation_check(&stuck, "auto", false).expect("row").ok);
        // One model answer anywhere proves the ladder reaches a CLI.
        let mixed = vec![("rule-based".to_string(), 7), ("codex".to_string(), 1)];
        assert!(consolidation_check(&mixed, "auto", true).expect("row").ok);
    }
}
