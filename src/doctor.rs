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
                "summarizer={} rerank={} primer_budget={}B session_budget={}B",
                config.summarizer.mode,
                config.search.rerank,
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

    checks.extend(cursor_overlap_check(&paths));
    checks.extend(split_tree_check(&paths));
    checks.push(injection_check(&paths));
    checks.push(taxonomy_check(&paths));
    checks.extend(summarizer_checks(&paths));
    checks.push(semantic_check(&paths));
    checks.push(retention_check(&paths));
    let store = Store::open(&paths.db()).ok();
    let space = store.as_ref().and_then(|store| crate::maint::status(&paths, store).ok());
    checks.extend(space.map(|status| space_check(&status)));
    checks.push(reranker_check(&paths));
    checks.push(hub_check(&paths));
    if let Some(store) = &store {
        checks.push(Check::pass("knowledge cleanup", crate::clean::summary(store)));
        checks.extend(hub_answers_check(&store.rerank_runs().unwrap_or_default()));
    }
    checks.push(update_check(&paths, store.as_ref()));
    checks.extend(hook_checks());
    checks.extend(trigger_checks());
    checks.push(timer_check());
    checks.push(resident_check(&paths));
    checks.push(sync_check(&paths));
    checks.push(team_check(&paths));
    checks.push(wiki_check(&paths));
    checks.push(error_log_check(&paths.log_file()));

    Ok(checks)
}

/// Sessions since `cutoff` (an RFC 3339 timestamp) captured under both the
/// `cursor` and the `claude-code` label.
fn double_captured_sessions(
    conn: &rusqlite::Connection,
    cutoff: &str,
) -> rusqlite::Result<i64> {
    conn.query_row(
        "SELECT COUNT(*) FROM (
             SELECT session FROM events WHERE ts >= ?1 GROUP BY session
             HAVING SUM(cli = 'cursor') > 0 AND SUM(cli = 'claude-code') > 0
         )",
        [cutoff],
        |row| row.get(0),
    )
}

/// Info line: Cursor sessions the store holds twice, under both labels.
///
/// Cursor also fires Claude Code's hooks, which `capture` now drops; sessions
/// recorded before that, or by an older binary, still show here. A zero or an
/// unreadable store says nothing.
fn cursor_overlap_check(paths: &Paths) -> Option<Check> {
    let conn = rusqlite::Connection::open_with_flags(
        paths.db(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .ok()?;
    let cutoff = jiff::Timestamp::now()
        .checked_sub(jiff::SignedDuration::from_secs(7 * 24 * 3600))
        .ok()?
        .to_string();
    let sessions = double_captured_sessions(&conn, &cutoff).ok()?;
    (sessions > 0).then(|| {
        Check::pass(
            "cursor duplicate capture",
            format!("{sessions} session(s) in the last 7 days hold both `cursor` and `claude-code` events"),
        )
    })
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

/// The shared rerank hub: whether it is wanted, whether it is up, and what
/// it holds. Information only - the hub starts itself with the next reranked
/// search and leaves when idle, so nothing here asks for a command.
#[cfg(not(unix))]
fn hub_check(_paths: &Paths) -> Check {
    Check::pass("hub", "off - not available on this platform")
}

#[cfg(unix)]
fn hub_check(paths: &Paths) -> Check {
    use crate::hub::client;
    if client::mode(paths) == client::HubMode::Off {
        return Check::pass("hub", "off - each session reranks in its own process");
    }
    if let Some(hub) = client::status(paths) {
        let mb = |kb: u64| kb / 1024;
        let footprint = hub.footprint_kb.map_or_else(|| "unknown".to_string(), |kb| format!("{} MB", mb(kb)));
        let mut detail = format!(
            "pid {} · build {} · up {} · footprint {footprint} · {} session(s) in 10 min · model {} · {} restart(s)/24h",
            hub.pid,
            crate::hub::printable(&hub.build),
            uptime_label(hub.uptime_ms / 1000),
            hub.sessions,
            if hub.model_loaded { "loaded" } else { "not loaded" },
            client::deaths_within(paths, 24 * 3600 * 1000),
        );
        if hub.retiring {
            detail.push_str(" · retiring for a newer build");
        }
        return Check::pass("hub", detail);
    }
    if let Some(left) = client::lockout_remaining(paths) {
        return Check::pass(
            "hub",
            format!(
                "not running - it died repeatedly, so it is not restarted for {}s; searches keep index order meanwhile",
                left.as_secs()
            ),
        );
    }
    Check::pass("hub", "not running - starts with the next reranked search, leaves when idle")
}

#[cfg(unix)]
fn uptime_label(secs: u64) -> String {
    match secs {
        0..=119 => format!("{secs}s"),
        120..=7199 => format!("{}m", secs / 60),
        _ => format!("{}h", secs / 3600),
    }
}

/// Warn when most of the latest reranks got no answer from the hub. Silent
/// otherwise, and silent on too few runs to say.
#[cfg(not(unix))]
fn hub_answers_check(_runs: &[crate::store::RerankRun]) -> Option<Check> {
    None
}

#[cfg(unix)]
fn hub_answers_check(runs: &[crate::store::RerankRun]) -> Option<Check> {
    let last = &runs[runs.len().saturating_sub(20)..];
    let missed: Vec<&str> = last.iter().map(|run| run.reason.as_str()).filter(|r| r.starts_with(crate::hub::reason::PREFIX)).collect();
    if last.len() < 4 || missed.len() * 2 <= last.len() {
        return None;
    }
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for reason in missed.iter().copied() {
        match counts.iter_mut().find(|(seen, _)| *seen == reason) {
            Some((_, n)) => *n += 1,
            None => counts.push((reason, 1)),
        }
    }
    let by_reason = counts.iter().map(|(r, n)| format!("{r} x{n}")).collect::<Vec<_>>().join(", ");
    Some(Check::pass(
        "hub answers",
        format!(
            "warn: {} of the last {} reranks got no answer from the hub ({by_reason}) - those searches kept index order",
            missed.len(),
            last.len()
        ),
    ))
}

/// Automatic retention: the window, when a pass last finished, how many bodies
/// the index has let go of, and how many the rule would drop now.
///
/// Information only. Pending is counted over one bounded window of rowids from
/// the pass's cursor, since reading the whole table costs seconds on a large
/// store, so it is a floor; a pass that finished within the last day left
/// nothing pending that the next one would not take, so then no count is made.
fn retention_check(paths: &Paths) -> Check {
    let Ok(store) = Store::open(&paths.db()) else {
        return Check::fail("retention", "index unreadable");
    };
    let Ok(config) = Config::load(&paths.config_file()) else {
        return Check::fail("retention", "config unreadable, so retention is not running");
    };
    let dropped = store.retention_dropped().unwrap_or(0);
    let last = store.retention_done_at().ok().flatten().unwrap_or_else(|| "never".to_string());
    let Some(days) = config.retention.effective_days() else {
        return Check::pass(
            "retention",
            format!("off (retention.days = 0) · last pass {last} · dropped so far {dropped}"),
        );
    };
    let now = jiff::Timestamp::now();
    let spent = store
        .retention_done_at()
        .ok()
        .flatten()
        .and_then(|at| at.parse::<jiff::Timestamp>().ok())
        .is_some_and(|at| now.as_second() - at.as_second() < crate::consolidate::RETENTION_REPEAT_SECS);
    let pending = if spent {
        "none until the next pass".to_string()
    } else {
        now.checked_sub(jiff::SignedDuration::from_secs(i64::from(days) * 24 * 3600))
            .ok()
            .zip(store.retention_cursor().ok())
            .and_then(|(cutoff, cursor)| {
                store.retention_pending(&cutoff.to_string(), cursor, crate::consolidate::RETENTION_COUNT_SPAN).ok()
            })
            .map_or_else(|| "unknown".to_string(), |pending| format!("at least {pending}"))
    };
    Check::pass(
        "retention",
        format!("{days} days · last pass {last} · dropped so far {dropped} · pending {pending}"),
    )
}

/// The index file's space: when it was last rewritten and how much a rewrite
/// would give back. Always shown, because the rewrite happens by itself; it is
/// a warning only when the index has wanted one for a fortnight, with the last
/// reason one was put off.
fn space_check(status: &crate::maint::CompactStatus) -> Check {
    let last = status.last.as_deref().map_or("never", |at| at.split('T').next().unwrap_or(at));
    let mut detail = format!(
        "last compact {last} · about {} MB reclaimable · runs automatically",
        status.reclaimable / (1024 * 1024)
    );
    if status.overdue {
        let why = status.skip.as_deref().unwrap_or("no attempt recorded");
        detail = format!("warn: {detail} · wanted for over 14 days · last put off: {why}");
    }
    Check::pass("index space", detail)
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
            "It is being fetched automatically; a consolidation run retries daily \
             until it arrives."
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
///
/// The model is the one a call would send, followed by why: the user's
/// config, the newest of its tier that consolidation found, or the built-in
/// pin - and when the pin is in use because a found model failed, which one.
fn model_label(
    spec: &crate::summarizer::CliSpec,
    overrides: &HashMap<String, String>,
    found: &crate::summarizer::Discovered,
) -> String {
    use crate::summarizer::Origin;
    if !spec.passes_a_model() {
        return format!("{}=(its own default)", spec.cli);
    }
    let (model, origin) = crate::summarizer::choose_model(spec, overrides, found);
    let why = match (origin, found.failed_model(spec.cli)) {
        (Origin::Config, _) => "config".to_string(),
        (Origin::Newest, _) => "newest".to_string(),
        (Origin::BuiltIn, Some(failed)) => format!("built-in; {failed} failed"),
        (Origin::BuiltIn, None) => "built-in".to_string(),
    };
    format!("{}={model} ({why})", spec.cli)
}

/// The "summarizer calls" row: information, not a verdict - what the
/// summarizer has cost in the last day. Absent when it made no call.
fn spend_check(today: &[crate::store::SummarizerCall]) -> Option<Check> {
    if today.is_empty() {
        return None;
    }
    let failed = today.iter().filter(|call| call.outcome != "ok").count();
    Some(Check::pass(
        "summarizer calls",
        format!("{} in the last 24h, {failed} not ok (see `brain stats`)", today.len()),
    ))
}

/// The "failing sessions" row: sessions consolidation keeps failing on, with
/// the last reason. A warning, not a failure - the rest of memory is fine, and
/// a parked session is retried once a day, up to three times.
/// Absent when none has failed twice.
fn failing_check(failing: &[(String, i64, String)]) -> Option<Check> {
    if failing.is_empty() {
        return None;
    }
    let parked = failing.iter().filter(|(_, n, _)| *n >= Store::PARK_AFTER).count();
    let lines: Vec<String> = failing
        .iter()
        .take(3)
        .map(|(session, attempts, error)| format!("{session} x{attempts}: {error}"))
        .collect();
    Some(Check::pass(
        "failing sessions",
        format!(
            "warn: {} session(s) failing, {parked} parked after {} attempts (parked ones are retried daily, up to 3 times)\n    {}",
            failing.len(),
            Store::PARK_AFTER,
            lines.join("\n    ")
        ),
    ))
}

/// Observations older than this, still unsummarized, are worth a warning.
const BACKLOG_WARN_SECS: i64 = 2 * 3600;

/// The "backlog" row: how old the oldest observation still waiting for a
/// summary is, parked sessions not counted (they wait on a person). A warning
/// past two hours, never a failure; absent when nothing is waiting or it is
/// young. Hours of backlog is the shape of a run that keeps yielding.
fn backlog_check(oldest_pending: Option<&str>) -> Option<Check> {
    let at: jiff::Timestamp = oldest_pending?.parse().ok()?;
    let age = jiff::Timestamp::now().as_second() - at.as_second();
    if age <= BACKLOG_WARN_SECS {
        return None;
    }
    Some(Check::pass(
        "consolidation backlog",
        format!(
            "warn: the oldest unsummarized observation is {}h old (the next session start catches it up)",
            age / 3600
        ),
    ))
}

/// The "consolidation runs" row: invocations in the last day, how many stood
/// aside for another run, and the longest one that did work. Information.
fn runs_check(runs: &[crate::store::ConsolidationRun]) -> Option<Check> {
    if runs.is_empty() {
        return None;
    }
    let yielded = runs.iter().filter(|run| run.yielded).count();
    let failed = runs.iter().filter(|run| run.error.is_some()).count();
    let longest = runs
        .iter()
        .filter(|run| !run.yielded)
        .filter_map(|run| {
            let (start, end) =
                (run.started.parse::<jiff::Timestamp>().ok()?, run.ended.parse::<jiff::Timestamp>().ok()?);
            Some(end.as_second() - start.as_second())
        })
        .max();
    Some(Check::pass(
        "consolidation runs",
        format!(
            "{} in the last 24h, {yielded} yielded, {failed} errored{}",
            runs.len(),
            longest.map_or(String::new(), |secs| format!(", longest pass {}m{}s", secs / 60, secs % 60))
        ),
    ))
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
    // Read, never refreshed: a lookup can start a CLI, and doctor reports what
    // consolidation last found rather than going to find out.
    let store = Store::open(&paths.db()).ok();
    let found = store.as_ref().map_or_else(crate::summarizer::Discovered::default, |store| {
        crate::summarizer::Discovered::read(store, jiff::Timestamp::now())
    });
    let installed: Vec<String> =
        installed.iter().map(|spec| model_label(spec, &summarizer_cfg.models, &found)).collect();

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

    if let Some(store) = &store {
        if let Some(check) = consolidation_check(
            &store.consolidation_tiers().unwrap_or_default(),
            &summarizer_cfg.mode,
            a_model_is_installed,
        ) {
            checks.push(check);
        }
        // Information, not a verdict: what the summarizer has cost today.
        let today = store.summarizer_calls_since(crate::store::DAY_SECS).unwrap_or_default();
        checks.extend(spend_check(&today));
        checks.extend(failing_check(&store.failing_sessions(2).unwrap_or_default()));
        checks.extend(backlog_check(store.oldest_pending_ts().unwrap_or_default().as_deref()));
        checks.extend(runs_check(
            &store.consolidation_runs_since(crate::store::DAY_SECS).unwrap_or_default(),
        ));
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

            // A plugin target is a file we own outright: present or not - and
            // if present, the shape the installed CLI will load. OpenCode 2
            // refuses the file OpenCode 1 loaded, and says so only in its log.
            if target.layout == crate::setup::Layout::Plugin {
                return if !path.is_file() {
                    Check::fail(&name, "plugin missing — run `brain setup --apply`")
                } else if !crate::setup::plugin_source_is_current(path) {
                    Check::fail(
                        &name,
                        format!(
                            "plugin {} has no default export, which OpenCode 2 requires to \
                             load it — the next consolidation run rewrites it",
                            path.display()
                        ),
                    )
                } else if !crate::setup::plugin_source_delivers(path) {
                    Check::fail(
                        &name,
                        format!(
                            "plugin {} only captures: OpenCode sessions get no memory pushed \
                             — the next consolidation run rewrites it",
                            path.display()
                        ),
                    )
                } else {
                    Check::pass(&name, format!("plugin installed: {}", path.display()))
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
                "a launchd job from an older version is still installed ({}) - the next consolidation run removes it",
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
fn resident_check(paths: &Paths) -> Check {
    let Some(running) = running_brains() else {
        return Check::fail("processes", "could not list processes");
    };
    let me = std::process::id();
    // The hub is one of these processes too; the lock names it.
    #[cfg(unix)]
    let hub = hub_pid(paths, &running);
    #[cfg(not(unix))]
    let hub: Option<u32> = {
        let _ = paths;
        None
    };
    let stray: Vec<String> = running
        .into_iter()
        .filter(|pid| *pid != me && Some(*pid) != hub)
        .map(|pid| format!("pid {pid}"))
        .collect();

    let mut parts = Vec::new();
    if let Some(pid) = hub {
        parts.push(format!("1 hub (pid {pid})"));
    }
    if !stray.is_empty() {
        parts.push(format!("{} MCP server(s), one per open session — {}", stray.len(), stray.join(", ")));
    }
    if parts.is_empty() {
        Check::pass("processes", "none running")
    } else {
        Check::pass("processes", parts.join("; "))
    }
}

/// The hub's pid, when the lock names a process that is running.
#[cfg(unix)]
fn hub_pid(paths: &Paths, running: &[u32]) -> Option<u32> {
    std::fs::read_to_string(paths.data_dir.join(crate::hub::endpoint::LOCK_FILE))
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok())
        .filter(|pid| running.contains(pid))
}

/// `[[dd-]hh:]mm:ss` as `ps -o etime=` prints it, in seconds.
#[cfg(unix)]
fn parse_etime(text: &str) -> Option<u64> {
    let (days, clock) = match text.trim().split_once('-') {
        Some((d, rest)) => (d.parse::<u64>().ok()?, rest),
        None => (0, text.trim()),
    };
    let mut secs = 0;
    for part in clock.split(':') {
        secs = secs * 60 + part.parse::<u64>().ok()?;
    }
    Some(days * 86400 + secs)
}

/// How many `brain mcp` sessions started before `installed_secs` (unix
/// seconds) and are still running: they keep the build they began with.
/// The hub and this process are not sessions. `None` when `ps` cannot say.
#[cfg(unix)]
fn sessions_on_old_build(paths: &Paths, installed_secs: i64, now: i64) -> Option<usize> {
    let running = running_brains()?;
    let hub = hub_pid(paths, &running);
    let me = std::process::id();
    let started_before = u64::try_from(now.saturating_sub(installed_secs)).ok()?;
    let mut count = 0;
    for pid in running.into_iter().filter(|pid| *pid != me && Some(*pid) != hub) {
        let Ok(out) = std::process::Command::new("ps").args(["-o", "etime=,command=", "-p", &pid.to_string()]).output() else {
            continue;
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let mut tokens = text.split_whitespace();
        let (Some(age), Some(_exe)) = (tokens.next().and_then(parse_etime), tokens.next()) else { continue };
        if tokens.next() == Some("mcp") && age > started_before {
            count += 1;
        }
    }
    Some(count)
}

/// `5s`, `12m`, `3h`, `2d`.
fn age_label(secs: i64) -> String {
    match secs.max(0) {
        s @ 0..=119 => format!("{s}s"),
        s @ 120..=7199 => format!("{}m", s / 60),
        s @ 7200..=172_799 => format!("{}h", s / 3600),
        s => format!("{}d", s / 86400),
    }
}

/// The auto-update row: mode, what was installed and when, the last check and
/// skip, bad versions, and sessions still on the old build. Information only
/// and local: it reads `schema_state` and files, never the network, and it
/// names no command (the updater runs by itself). A rollback or a bad version
/// is a `warn:`, never a failure.
fn update_check(paths: &Paths, store: Option<&Store>) -> Check {
    use crate::update::{Readiness, STATE_BAD, STATE_CHECKED_AT, STATE_INSTALLED, STATE_PREV, STATE_SKIP};
    let config = Config::load(&paths.config_file()).unwrap_or_default();
    let state = |key: &str| store.and_then(|s| s.state(key).ok().flatten());
    let now = crate::update::now_secs();
    let ago = |secs: i64| format!("{} ago", age_label(now - secs));
    let mut warn = false;
    let mut parts = vec![match crate::update::readiness(&config) {
        Readiness::Off(why) => format!("off ({why})"),
        Readiness::NoKey => "auto, waiting for a signed release".to_string(),
        Readiness::Elsewhere => "auto, but installed another way, so not updated here".to_string(),
        Readiness::Ready => "auto".to_string(),
    }];

    let installed = state(STATE_INSTALLED).unwrap_or_default();
    let mut it = installed.split_whitespace();
    let installed_at = match (it.next(), it.next().and_then(|n| n.parse::<i64>().ok())) {
        (Some(version), Some(ms)) if ms > 0 => {
            let prev = state(STATE_PREV).map(|p| format!(", was {p}")).unwrap_or_default();
            parts.push(format!("installed {version} {}{prev}", ago(ms / 1000)));
            Some(ms / 1000)
        }
        (Some(version), Some(_)) => {
            parts.push(format!("running {version} after a rollback"));
            None
        }
        _ => None,
    };
    match state(STATE_CHECKED_AT).and_then(|n| n.parse::<i64>().ok()) {
        Some(at) => parts.push(format!("checked {}", ago(at))),
        None => parts.push("never checked".to_string()),
    }
    if let Some(skip) = state(STATE_SKIP) {
        let (reason, at) = skip.split_once('@').unwrap_or((&skip, ""));
        let when = at.parse::<i64>().map(|s| format!(" {}", ago(s))).unwrap_or_default();
        if matches!(reason, "rolled-back" | "rollback-unavailable") {
            warn = true;
        }
        parts.push(format!("last skip: {reason}{when}"));
    }
    let bad = crate::update::bad_versions(&state(STATE_BAD).unwrap_or_default(), &paths.data_dir);
    if !bad.is_empty() {
        warn = true;
        parts.push(format!("marked bad: {}", bad.join(" ")));
    }
    #[cfg(unix)]
    if let Some(at) = installed_at {
        if let Some(n) = sessions_on_old_build(paths, at, now).filter(|n| *n > 0) {
            let was = state(STATE_PREV).unwrap_or_else(|| "the old version".to_string());
            parts.push(format!("{n} session(s) still on {was} until they close"));
        }
    }
    #[cfg(not(unix))]
    let _ = installed_at;
    let detail = parts.join(" · ");
    Check::pass("update", if warn { format!("warn: {detail}") } else { detail })
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

/// Whether this brain syncs anywhere, which is off unless the owner said so.
fn sync_check(paths: &Paths) -> Check {
    let config = Config::load(&paths.config_file()).unwrap_or_default();
    let Some(dir) = config.sync.dir else {
        return Check::pass(
            "sync",
            "off - memory stays on this machine (`brain sync init <dir>` to opt in)",
        );
    };
    if !paths.data_dir.join("sync.key").is_file() {
        return Check::fail(
            "sync",
            format!("{} configured but sync.key is missing - run `brain sync init`", dir.display()),
        );
    }
    if !dir.is_dir() {
        return Check::fail("sync", format!("configured dir {} does not exist", dir.display()));
    }
    let bundles = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().ends_with(".brain.enc"))
                .count()
        })
        .unwrap_or(0);
    Check::pass("sync", format!("{} - {bundles} bundle(s), key present", dir.display()))
}

/// A team, when there is one: where it publishes and whose name this machine puts on it.
fn team_check(paths: &Paths) -> Check {
    let config = Config::load(&paths.config_file()).unwrap_or_default();
    let Some(dir) = config.team.dir else {
        return Check::pass("team", "none - lessons stay on this machine (`brain team init` to share)");
    };
    let Some(author) = config.team.author.filter(|name| !name.trim().is_empty()) else {
        return Check::fail("team", "configured without a name - run `brain team init <dir> --name \"…\"`");
    };
    if !paths.data_dir.join("team.key").is_file() {
        return Check::fail(
            "team",
            format!("{} configured but team.key is missing - run `brain team init`", dir.display()),
        );
    }
    if !dir.is_dir() {
        return Check::fail("team", format!("folder {} does not exist", dir.display()));
    }
    let bundles = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().ends_with(".team.enc"))
                .count()
        })
        .unwrap_or(0);
    Check::pass(
        "team",
        format!("{} - {bundles} bundle(s), publishing as {author}; lessons only", dir.display()),
    )
}

/// The vault as a wiki: does every link land, and can every page be reached?
///
/// The lint a wiki needs and the one thing search cannot tell you: a page
/// with no way in is invisible to a person even when an agent finds it.
/// Links are resolved the way the pages write them - relative to the
/// project directory - and then the way Obsidian falls back, by a unique
/// file name anywhere in the vault.
fn wiki_check(paths: &Paths) -> Check {
    let wiki = paths.wiki();
    if !wiki.is_dir() {
        return Check::pass("wiki", "nothing consolidated yet");
    }
    let Some(lint) = crate::consolidate::lint_wiki(&wiki) else {
        return Check::fail("wiki", format!("{} could not be read", wiki.display()));
    };
    let detail = format!(
        "{} page(s), {} unresolved link(s), {} orphan(s) - old-style links are relinked automatically",
        lint.pages, lint.unresolved, lint.orphans
    );
    if lint.unresolved == 0 && lint.orphans == 0 {
        Check::pass("wiki", format!("{} page(s), every link resolves, every page reachable", lint.pages))
    } else {
        Check::pass("wiki", detail)
    }
}

/// Recent capture failures. Hooks never print to the host CLI, so this file is
/// the only place they surface.
fn error_log_check(path: &Path) -> Check {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Check::pass("capture errors", "none recorded");
    };
    error_log_summary(&text, &path.display().to_string(), jiff::Timestamp::now().as_second())
}

/// Only the last week counts: older lines are history, and `brain.log` is set
/// aside by the daily upkeep once its first line is a month old. A line with
/// no timestamp of its own belongs to the one above it.
const ERROR_LOG_WINDOW_SECS: i64 = 7 * 24 * 3600;

fn error_log_summary(text: &str, name: &str, now: i64) -> Check {
    let mut at: Option<i64> = None;
    let mut recent: Vec<&str> = Vec::new();
    let mut newest: Option<i64> = None;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        if let Some(stamp) = line.split_whitespace().next().and_then(|t| t.parse::<jiff::Timestamp>().ok()) {
            at = Some(stamp.as_second());
        }
        // No timestamp anywhere yet: counted, since hiding a failure is worse.
        if at.is_none_or(|at| now.saturating_sub(at) <= ERROR_LOG_WINDOW_SECS) {
            recent.push(line);
            newest = at.or(newest);
        }
    }
    if recent.is_empty() {
        return Check::pass("capture errors", "none in the last 7 days");
    }
    let age = newest.map_or_else(String::new, |at| {
        let hours = now.saturating_sub(at) / 3600;
        format!(", newest {hours}h ago")
    });
    let shown = recent.iter().rev().take(3).rev().copied().collect::<Vec<_>>().join("\n    ");
    Check::fail(
        "capture errors",
        format!("{} in the last 7 days{age} in {name}\n    {shown}", recent.len()),
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

    fn space(last: Option<&str>, mb: u64, wanted: bool, overdue: bool, skip: Option<&str>) -> Check {
        space_check(&crate::maint::CompactStatus {
            last: last.map(str::to_string),
            reclaimable: mb * 1024 * 1024,
            wanted,
            overdue,
            skip: skip.map(str::to_string),
        })
    }

    #[test]
    fn the_wiki_row_says_the_unresolved_count_once() {
        let dir = std::env::temp_dir().join(format!("brain-doctor-wiki-{}", ulid::Ulid::new()));
        let paths = Paths { data_dir: dir.clone() };
        let wiki = paths.wiki();
        std::fs::create_dir_all(wiki.join("proj")).unwrap();
        std::fs::write(wiki.join("proj/a.md"), "see [[nowhere]]\n").unwrap();
        let check = wiki_check(&paths);
        assert!(check.detail.contains("unresolved link(s)"), "{}", check.detail);
        assert!(!check.detail.contains("old-style links;"), "{}", check.detail);
        assert!(!check.detail.contains("left are not"), "{}", check.detail);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_space_row_is_always_there_and_never_names_a_command() {
        let never = space(None, 0, false, false, None);
        assert!(never.ok);
        assert_eq!(never.detail, "last compact never · about 0 MB reclaimable · runs automatically");

        let due = space(None, 294, true, false, None);
        assert!(due.ok && due.detail.contains("about 294 MB reclaimable"), "{}", due.detail);

        let done = space(Some("2026-10-08T03:04:05Z"), 1, false, false, None);
        assert!(done.detail.starts_with("last compact 2026-10-08 ·"), "{}", done.detail);

        for row in [&never, &due, &done] {
            assert!(!row.detail.contains("brain "), "a command in an automatic row: {}", row.detail);
        }
    }

    #[test]
    fn an_overdue_compact_warns_with_the_reason_it_was_put_off() {
        let row = space(None, 300, true, true, Some("low-disk: 900 MB free, 5000 MB needed@2026-10-01T00:00:00Z"));
        assert!(row.ok, "a warning is not a failure");
        assert!(row.detail.starts_with("warn: "), "{}", row.detail);
        assert!(row.detail.contains("low-disk"), "{}", row.detail);
        assert!(!row.detail.contains("brain "), "{}", row.detail);
    }

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
    fn the_failing_row_names_the_session_its_attempts_and_the_reason() {
        assert!(failing_check(&[]).is_none());
        let row = failing_check(&[
            ("s-1".into(), 3, "write page: disk full".into()),
            ("s-2".into(), 2, "timed out".into()),
        ])
        .unwrap();
        assert!(row.ok, "a failing session warns, it does not fail the report");
        assert!(row.detail.contains("warn: 2 session(s) failing, 1 parked"), "{}", row.detail);
        assert!(row.detail.contains("s-1 x3: write page: disk full"), "{}", row.detail);
    }

    #[test]
    fn the_backlog_row_warns_only_past_two_hours() {
        let ago = |secs: i64| {
            (jiff::Timestamp::now() - jiff::SignedDuration::from_secs(secs)).to_string()
        };
        assert!(backlog_check(None).is_none());
        assert!(backlog_check(Some(&ago(3600))).is_none(), "an hour is not stale");
        let row = backlog_check(Some(&ago(3 * 3600 + 60))).unwrap();
        assert!(row.ok, "a warning, not a failure");
        assert!(row.detail.starts_with("warn: the oldest unsummarized observation is 3h old"), "{}", row.detail);
    }

    #[test]
    fn the_runs_row_counts_yields_errors_and_the_longest_pass() {
        let run = |yielded: bool, secs: i64, error: Option<&str>| {
            let now = jiff::Timestamp::now();
            crate::store::ConsolidationRun {
                started: (now - jiff::SignedDuration::from_secs(secs)).to_string(),
                ended: now.to_string(),
                mode: "all".into(),
                yielded,
                sessions: 0,
                events: 0,
                failed: 0,
                rule_based: 0,
                error: error.map(str::to_string),
            }
        };
        assert!(runs_check(&[]).is_none());
        let row = runs_check(&[run(true, 1, None), run(false, 125, None), run(false, 5, Some("x"))]).unwrap();
        assert_eq!(row.detail, "3 in the last 24h, 1 yielded, 1 errored, longest pass 2m5s");
    }

    #[test]
    fn the_spend_row_counts_the_day_and_what_did_not_end_ok() {
        let call = |outcome: &str| crate::store::SummarizerCall {
            session: "s".into(),
            purpose: "consolidate".into(),
            cli: "codex".into(),
            model: "m".into(),
            prompt_bytes: 1,
            answer_bytes: 1,
            ms: 1,
            outcome: outcome.into(),
        };
        assert!(spend_check(&[]).is_none());
        let row = spend_check(&[call("ok"), call("timeout"), call("unusable")]).unwrap();
        assert_eq!(row.name, "summarizer calls");
        assert!(row.detail.contains("3 in the last 24h, 2 not ok"), "{}", row.detail);
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

        let found = crate::summarizer::Discovered::default();
        let mut overrides = HashMap::new();
        assert_eq!(
            model_label(unpinned, &overrides, &found),
            format!("{}=(its own default)", unpinned.cli)
        );

        overrides.insert(unpinned.cli.to_string(), "composer-2.5".to_string());
        assert_eq!(
            model_label(unpinned, &overrides, &found),
            format!("{}=(its own default)", unpinned.cli),
            "an override on a rung with no `{{model}}` argument must not be reported as running"
        );

        let pinned = crate::summarizer::SPECS
            .iter()
            .find(|spec| spec.passes_a_model())
            .expect("at least one rung names a model");
        let mut overrides = HashMap::new();
        overrides.insert(pinned.cli.to_string(), "sonnet".to_string());
        assert_eq!(model_label(pinned, &overrides, &found), format!("{}=sonnet (config)", pinned.cli));
    }

    #[test]
    fn a_rung_says_which_model_it_runs_and_why() {
        let codex = crate::summarizer::SPECS.iter().find(|spec| spec.cli == "codex").unwrap();
        let none = HashMap::new();
        let store = Store::open_memory().unwrap();
        let read = |store: &Store| crate::summarizer::Discovered::read(store, jiff::Timestamp::now());

        assert_eq!(model_label(codex, &none, &read(&store)), "codex=gpt-5.6-luna (built-in)");

        store.set_state("summarizer_model:codex", "gpt-6-luna").unwrap();
        assert_eq!(model_label(codex, &none, &read(&store)), "codex=gpt-6-luna (newest)");

        let mut overrides = HashMap::new();
        overrides.insert("codex".to_string(), "gpt-5.6-sol".to_string());
        assert_eq!(model_label(codex, &overrides, &read(&store)), "codex=gpt-5.6-sol (config)");

        let mark = format!("gpt-6-luna {}", jiff::Timestamp::now());
        store.set_state("summarizer_model_bad:codex", &mark).unwrap();
        assert_eq!(
            model_label(codex, &none, &read(&store)),
            "codex=gpt-5.6-luna (built-in; gpt-6-luna failed)"
        );
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

    #[test]
    fn only_sessions_under_both_labels_in_the_window_count() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE events (session TEXT, cli TEXT, ts TEXT);
             INSERT INTO events VALUES ('both', 'cursor', '2026-10-04T00:00:00Z');
             INSERT INTO events VALUES ('both', 'claude-code', '2026-10-04T00:00:01Z');
             INSERT INTO events VALUES ('cursor-only', 'cursor', '2026-10-04T00:00:00Z');
             INSERT INTO events VALUES ('claude-only', 'claude-code', '2026-10-04T00:00:00Z');
             INSERT INTO events VALUES ('old', 'cursor', '2026-09-01T00:00:00Z');
             INSERT INTO events VALUES ('old', 'claude-code', '2026-09-01T00:00:00Z');",
        )
        .unwrap();
        assert_eq!(double_captured_sessions(&conn, "2026-09-28T00:00:00Z").unwrap(), 1);
    }

    #[test]
    fn only_the_last_seven_days_of_the_log_fail_the_row() {
        let now = jiff::Timestamp::now().as_second();
        let at = |ago: i64| jiff::Timestamp::from_second(now - ago).unwrap();
        let day = 24 * 3600;
        let old = format!("{} hook x: old\n", at(9 * day));
        let check = error_log_summary(&old, "brain.log", now);
        assert!(check.ok, "{}", check.detail);

        let both = format!("{old}{} hook x: new\n  continued\n", at(3 * 3600));
        let check = error_log_summary(&both, "brain.log", now);
        assert!(!check.ok);
        assert!(check.detail.starts_with("2 in the last 7 days, newest 3h ago"), "{}", check.detail);
    }

    #[test]
    fn no_row_tells_the_user_to_run_what_runs_by_itself() {
        let source = include_str!("doctor.rs");
        let tests = source.find("#[cfg(test)]").unwrap();
        for command in ["brain compact", "brain consolidate --all", "re-points old links", "--session X --force", "Fetch it once"] {
            assert!(!source[..tests].contains(command), "doctor still says: {command}");
        }
    }
}
