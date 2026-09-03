//! rolepod-brain — install-and-forget memory for AI coding agents.
//!
//! One binary, three entry modes, nothing resident between them:
//!
//! - `brain hook`   — spawned by a host CLI's lifecycle hook, captures, exits.
//! - `brain mcp`    — stdio MCP server, lives exactly as long as one session.
//! - everything else — operator commands run by hand.

#![forbid(unsafe_code)]

mod config;
mod consolidate;
mod doctor;
mod embed;
mod event;
mod history;
mod hook;
mod ids;
mod inject;
mod invocation;
mod mcp;
mod portable;
mod rerank;
#[cfg(feature = "local-rerank")]
mod xencoder;
mod revise;
mod sanitize;
mod setup;
mod store;
mod tokenize;
mod transcript;
mod sync;
mod summarizer;

use std::io::Write;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use crate::config::Paths;
use crate::event::EventLog;
use crate::store::Store;

#[derive(Parser)]
#[command(
    name = "brain",
    version,
    about = "Install-and-forget memory for AI coding agents. No resident process."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Capture one lifecycle event from stdin. Invoked by host CLI hooks.
    Hook {
        /// Which CLI is calling, e.g. `claude-code`.
        #[arg(long)]
        cli: String,
        /// Lifecycle event name, in whatever spelling the CLI uses.
        #[arg(long)]
        event: String,
    },
    /// Run the stdio MCP server. Invoked by host CLI MCP configuration.
    Mcp,
    /// Consolidate pending observations into wiki pages.
    Consolidate {
        /// Limit to one session id.
        #[arg(long)]
        session: Option<String>,
        /// Every project on this machine, not just this directory's.
        #[arg(long)]
        all: bool,
        /// Ignore the debounce and the minimum-pending rule.
        #[arg(long)]
        force: bool,
    },
    /// Wire the host CLIs. Prints a plan; use --apply to perform it.
    Setup {
        /// Limit to one CLI (`claude-code`, `codex`) instead of all.
        #[arg(long)]
        cli: Option<String>,
        /// Perform the changes instead of only printing them.
        #[arg(long)]
        apply: bool,
    },
    /// Check that capture, storage, and wiring are actually working.
    Doctor,
    /// What a wiki page used to say, from the wiki's own git history.
    History {
        /// Part of the page's path or name, e.g. `decisions` or a session date.
        query: String,
        /// Show what each revision changed, not just when it happened.
        #[arg(long)]
        diff: bool,
    },
    /// Remove brain from every CLI. Prints a plan; use --apply to perform it.
    Uninstall {
        /// Perform the changes instead of only printing them.
        #[arg(long)]
        apply: bool,
        /// Also delete ~/.rolepod-brain, including all memory. Asks first.
        #[arg(long)]
        wipe: bool,
    },
    /// Rebuild the SQLite index from the event log.
    Reindex,
    /// Search this project's memory from the terminal.
    Search {
        /// FTS5 query.
        query: String,
        /// Maximum hits.
        #[arg(short = 'k', long, default_value_t = 10)]
        limit: usize,
        /// Narrow to one kind: decision, bugfix, feature, discovery, config, test.
        #[arg(long)]
        topic: Option<String>,
        /// Rerank the hits by a small model, the way `brain_search` over MCP does.
        /// Default follows `[search] rerank` in config.
        #[arg(long, overrides_with = "no_rerank")]
        rerank: bool,
        /// Keep the index's own order even when config says rerank.
        #[arg(long)]
        no_rerank: bool,
    },
    /// Sync this brain with your other machines through a folder you own.
    Sync {
        #[command(subcommand)]
        action: Option<SyncAction>,
    },
    /// Drop the bodies of old, never-surfaced observations; keep every title and link.
    Retire {
        /// Only observations older than this many months.
        #[arg(long = "older-than-months", default_value_t = 6)]
        older_than_months: u32,
        /// Actually retire. Without it, measure and report only.
        #[arg(long)]
        apply: bool,
    },
    /// One compact block to seed a subagent: lessons, then task-relevant pointers.
    Seed {
        /// What the subagent will work on, in a phrase.
        task: String,
        /// Maximum bytes for the block.
        #[arg(long, default_value_t = inject::SEED_BUDGET)]
        budget: usize,
    },
    /// What this brain has actually done. Local counters; nothing is sent anywhere.
    Stats,
    /// Write this brain to an archive, for moving it to another machine.
    Export {
        /// Archive path. Defaults to brain-<date>.tar.gz here.
        file: Option<String>,
    },
    /// Restore a brain from an archive written by `brain export`.
    Import {
        /// Archive path.
        file: String,
        /// Add the incoming memory to what is already here.
        #[arg(long, conflicts_with = "replace")]
        merge: bool,
        /// Set the existing brain aside and take the incoming one.
        #[arg(long)]
        replace: bool,
    },
    /// Withdraw a memory. Appends a tombstone; nothing is deleted.
    Forget {
        /// Event id, as shown by `brain search`. Omit when using --entity.
        id: Option<String>,
        /// Withdraw every memory mentioning this instead of one id.
        /// Prints what would go; use --apply to perform it.
        #[arg(long)]
        entity: Option<String>,
        /// Perform an --entity withdrawal instead of only listing it.
        #[arg(long)]
        apply: bool,
    },
    /// Replace what a memory says, keeping the original in the log.
    Correct {
        /// Event id, as shown by `brain search`.
        id: String,
        /// What it should say instead.
        text: String,
    },
    /// Show where this directory's memory lives.
    Where {
        /// Print only where the embedding model belongs, and nothing else.
        ///
        /// `bootstrap.sh` asks for this rather than rebuilding the path
        /// itself, so the installer and the binary can never disagree about
        /// where the model goes.
        #[arg(long)]
        models: bool,
        /// Print only where the reranker belongs, and nothing else.
        ///
        /// Same reason as `--models`: the installer asks rather than
        /// rebuilding the path, so the two can never disagree.
        #[arg(long)]
        reranker: bool,
    },
}

#[derive(clap::Subcommand)]
enum SyncAction {
    /// Point this brain at a shared folder and mint the key.
    Init {
        /// A directory your machines already share (iCloud, Dropbox, NAS...).
        dir: String,
    },
}

fn main() {
    let cli = Cli::parse();

    // The hook path has its own error discipline: it must never disturb the
    // host CLI, whatever happens.
    if let Commands::Hook { cli: agent, event } = &cli.command {
        run_hook(agent, event);
        return;
    }

    if let Err(error) = run(cli.command) {
        eprintln!("brain: {error:#}");
        std::process::exit(1);
    }
}

/// Capture path. Always exits 0 with a well-formed acknowledgement.
fn run_hook(agent: &str, event: &str) {
    match hook::capture(agent, event, None) {
        Ok(ack) => println!("{ack}"),
        Err(error) => {
            log_capture_failure(agent, event, &error);
            // The host still needs a valid acknowledgement, or it logs our
            // silence as a malformed hook response on every single call.
            println!("{{}}");
        }
    }
}

/// Record a capture failure where `brain doctor` will find it. Best-effort by
/// design: if even this fails there is nothing useful left to do.
fn log_capture_failure(agent: &str, event: &str, error: &anyhow::Error) {
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

fn run(command: Commands) -> Result<()> {
    match command {
        Commands::Hook { .. } => unreachable!("handled in main"),
        Commands::Mcp => mcp::serve(),
        Commands::Consolidate { session, all, force } => {
            let outcome = consolidate::run(session.as_deref(), all, force)?;
            if outcome.embedded > 0 {
                println!("Embedded {} event(s) for semantic search.", outcome.embedded);
            }
            if outcome.adopted > 0 {
                println!(
                    "Adopted {} hand-edited page(s) into memory.",
                    outcome.adopted
                );
            }
            if outcome.yielded {
                println!(
                    "Another consolidation run is working; this one stood aside. \
                     Nothing is lost - the work stays pending for whichever run holds it."
                );
            } else if outcome.sessions == 0 {
                println!(
                    "Nothing to consolidate ({} session(s) waiting for more events or a debounce).",
                    outcome.skipped
                );
            } else {
                println!(
                    "Consolidated {} session(s), {} event(s) via {}.",
                    outcome.sessions,
                    outcome.events,
                    outcome.tiers.join(", ")
                );
            }
            Ok(())
        }
        Commands::Setup { cli, apply } => {
            let changes = setup::run(cli.as_deref(), apply)?;
            if changes.is_empty() {
                println!("Nothing to do: no supported CLI found.");
                return Ok(());
            }
            for change in &changes {
                println!("{:<12} {}", change.target, change.detail);
            }
            if !apply {
                println!("\nDry run. Re-run with --apply to perform these changes.");
            }
            Ok(())
        }
        Commands::Doctor => {
            let checks = doctor::run()?;
            let (report, all_ok) = doctor::render(&checks);
            print!("{report}");
            if !all_ok {
                std::process::exit(1);
            }
            Ok(())
        }
        Commands::Uninstall { apply, wipe } => uninstall(apply, wipe),
        Commands::Reindex => reindex(),
        Commands::Search { query, limit, topic, rerank, no_rerank } => {
            let rerank = if rerank {
                Some(true)
            } else if no_rerank {
                Some(false)
            } else {
                None
            };
            search(&query, limit, topic.as_deref(), rerank)
        }
        Commands::History { query, diff } => history::report(&query, diff),
        Commands::Sync { action } => match action {
            Some(SyncAction::Init { dir }) => {
                println!("{}", sync::init(&dir)?);
                Ok(())
            }
            None => {
                let outcome = sync::run()?;
                if outcome.pulled == 0 && outcome.skipped.is_empty() {
                    println!("no peer bundles yet - this machine published its own");
                } else {
                    println!(
                        "pulled {} bundle(s), {} new event(s)",
                        outcome.pulled, outcome.gained
                    );
                }
                for name in &outcome.skipped {
                    println!("skipped {name} - wrong key or corrupt (not fatal)");
                }
                println!("pushed {} KB, encrypted", outcome.pushed_bytes / 1024);
                Ok(())
            }
        },
        Commands::Retire { older_than_months, apply } => {
            let outcome = revise::retire(older_than_months, apply)?;
            let kb = outcome.bytes / 1024;
            if outcome.count == 0 {
                println!(
                    "Nothing to retire: no consolidated observation older than \
                     {older_than_months} month(s) has gone unseen."
                );
            } else if apply {
                println!(
                    "Retired {} observation bodies; {kb} KB left the index.",
                    outcome.count
                );
                println!("Titles, topics, files and links kept; the log keeps everything.");
            } else {
                println!(
                    "Would retire {} observation bodies, freeing {kb} KB from the index.",
                    outcome.count
                );
                println!(
                    "Kept either way: id, timestamp, title, topic, files, links. \
                     The log is untouched."
                );
                println!("Re-run with --apply to retire them.");
            }
            Ok(())
        }
        Commands::Seed { task, budget } => {
            let paths = Paths::resolve()?;
            let store = Store::open(&paths.db())?;
            let scope = ids::resolve_scope(&std::env::current_dir().unwrap_or_default());
            let seed =
                inject::seed(&store, &scope.project_id.to_string(), &task, budget.clamp(256, 8192))?;
            if seed.text.is_empty() {
                println!("Nothing to seed yet - no lessons and no matches for the task.");
            } else {
                print!("{}", seed.text);
            }
            Ok(())
        }
        Commands::Stats => stats(),
        Commands::Export { file } => {
            let path = file.map_or_else(portable::default_archive, std::path::PathBuf::from);
            println!("{}", portable::export(&path)?);
            Ok(())
        }
        Commands::Import { file, merge, replace } => {
            let policy = match (merge, replace) {
                (true, _) => portable::Existing::Merge,
                (_, true) => portable::Existing::Replace,
                _ => portable::Existing::Refuse,
            };
            println!("{}", portable::import(std::path::Path::new(&file), policy)?);
            // The index is rebuilt rather than shipped, which also proves on
            // arrival that the log really is the source of truth.
            reindex()?;
            let checks = doctor::run()?;
            let (report, _) = doctor::render(&checks);
            print!("{report}");
            Ok(())
        }
        Commands::Forget { id, entity, apply } => match (id, entity) {
            (Some(id), None) => {
                let outcome = revise::forget(&id)?;
                println!("Forgot {id} — \"{}\"", outcome.target_title);
                println!(
                    "Recorded as {}. The log keeps both; recall no longer shows it.",
                    outcome.id
                );
                Ok(())
            }
            (None, Some(entity)) => {
                let outcomes = revise::forget_entity(&entity, apply)?;
                if outcomes.is_empty() {
                    println!("Nothing mentions `{entity}` in this project.");
                    return Ok(());
                }
                for outcome in &outcomes {
                    println!("{}  {}", outcome.id, outcome.target_title);
                }
                if apply {
                    println!(
                        "\nWithdrew {} memor(y/ies) mentioning `{entity}`. The log keeps every one.",
                        outcomes.len()
                    );
                } else {
                    println!(
                        "\n{} memor(y/ies) mention `{entity}`. Matching is by text, so a mention \
                         under another name is not listed. Re-run with --apply to withdraw these.",
                        outcomes.len()
                    );
                }
                Ok(())
            }
            (Some(_), Some(_)) => {
                anyhow::bail!("pass an id or --entity, not both")
            }
            (None, None) => anyhow::bail!("pass an event id, or --entity NAME"),
        },
        Commands::Correct { id, text } => {
            let outcome = revise::correct(&id, &text)?;
            println!("Corrected {id} — was \"{}\"", outcome.target_title);
            println!("Recorded as {}. Recall now returns your text instead.", outcome.id);
            Ok(())
        }
        Commands::Where { models, reranker } => where_am_i(models, reranker),
    }
}

/// Rebuild every index row from the log.
///
/// This is the command that proves the log is the source of truth: deleting
/// `brain.db` must cost nothing but the time this takes.
fn reindex() -> Result<()> {
    let paths = Paths::resolve()?;
    paths.ensure()?;

    // Layout first, so the index and the hubs below are rebuilt against the
    // homes projects will actually live in.
    for moved in consolidate::migrate_layout(&paths)? {
        println!("moved {moved}");
    }

    let store = Store::open(&paths.db())?;
    store.clear()?;

    let mut indexed = 0usize;
    let mut skipped = 0usize;
    let mut projects = 0usize;

    for (_, project_dir) in consolidate::known_projects(&paths)? {
        let log = EventLog::open(&project_dir)?;
        let (events, bad_lines) = log.read_all()?;
        skipped += bad_lines;
        if events.is_empty() {
            continue;
        }
        projects += 1;
        for event in &events {
            store.index(event)?;
            indexed += 1;
        }
    }

    // Hub notes are derived from the pages exactly as the index is derived
    // from the log, and they were only ever refreshed when a project happened
    // to consolidate something. A project with no pending work would keep
    // whatever hub state it had - or none at all, if its last consolidation
    // predated hubs existing. Rebuilding derived state is what this command is
    // for, so it rebuilds these too.
    let mut hubs = 0usize;
    for (scope, dir) in consolidate::known_projects(&paths).unwrap_or_default() {
        if consolidate::write_hubs(&dir, &scope, &store).is_ok() {
            hubs += 1;
        }
    }

    println!("Reindexed {indexed} event(s) from {projects} project(s).");
    println!("Rebuilt hub notes for {hubs} project(s).");
    if skipped > 0 {
        println!("Skipped {skipped} unreadable line(s); the rest of the log was unaffected.");
    }
    Ok(())
}

fn search(query: &str, limit: usize, topic: Option<&str>, rerank: Option<bool>) -> Result<()> {
    let paths = Paths::resolve()?;
    let scope = ids::resolve_scope(&std::env::current_dir().unwrap_or_default());
    let store = Store::open(&paths.db())?;
    let config = config::Config::load(&paths.config_file())?;
    // The same default as `brain_search` over MCP: a terminal and an agent
    // asking the same question get the same order, or the terminal is the
    // place a reranked answer cannot be reproduced.
    let rerank = rerank.unwrap_or(config.search.rerank);
    // A typo'd topic must not read as "nothing remembered": say so and search
    // everything, rather than returning an empty page the user cannot explain.
    let scoped = topic.and_then(event::normalize_topic);
    if let (Some(asked), None) = (topic, scoped) {
        eprintln!(
            "Unknown topic `{asked}` — searching everything. Known: {}",
            event::TOPICS.join(", ")
        );
    }
    let project = scope.project_id.to_string();
    // A reranker is worth a wider pool to choose from, as over MCP.
    let pool = if rerank { rerank::LOCAL_POOL.max(limit) } else { limit };
    let mut hits = store.search(&project, query, scoped, pool, store::Recall::Fused)?;
    if rerank {
        let ladder = summarizer::Ladder::new(&store, &config.summarizer);
        let cli = store.project_cli(&project)?.unwrap_or_default();
        let model_dir = paths.model_dir_for(rerank::LOCAL_MODEL);
        hits = rerank::rerank(&ladder, &cli, query, &model_dir, hits);
    }
    hits.truncate(limit);

    if hits.is_empty() {
        let where_ = scoped.map_or(String::new(), |topic| format!(" under topic `{topic}`"));
        println!("No matches in {}{where_}.", scope.project);
        return Ok(());
    }
    for hit in hits {
        println!("{}  {}  {:<12} {}", hit.id, &hit.ts[..hit.ts.len().min(19)], hit.cli, hit.title);
        if !hit.snippet.is_empty() {
            println!("    {}", hit.snippet.replace('\n', " "));
        }
    }
    Ok(())
}

/// Remove the wiring, and on request the memory.
///
/// The memory is deleted only behind a typed confirmation. Everything else
/// here is reversible by re-running setup; that is not, and a flag is too
/// small a gesture for the only irreversible thing this program can do.
fn uninstall(apply: bool, wipe: bool) -> Result<()> {
    let changes = setup::uninstall(apply)?;
    if changes.is_empty() {
        println!("Nothing wired; nothing to remove.");
    }
    for change in &changes {
        println!("{:<12} {}", change.target, change.detail);
    }

    if !apply {
        println!("\nDry run. Re-run with --apply to perform these changes.");
        if wipe {
            println!("--wipe would then ask before deleting your memory.");
        }
        return Ok(());
    }

    if wipe {
        let paths = Paths::resolve()?;
        println!("\nThis deletes {} and every memory in it.", paths.data_dir.display());
        println!("It cannot be undone. Type DELETE to confirm: ");
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer).context("read confirmation")?;
        if answer.trim() == "DELETE" {
            std::fs::remove_dir_all(&paths.data_dir)
                .with_context(|| format!("remove {}", paths.data_dir.display()))?;
            println!("Deleted {}.", paths.data_dir.display());
        } else {
            println!("Left {} untouched.", paths.data_dir.display());
        }
    } else {
        let paths = Paths::resolve()?;
        println!("\nYour memory is still at {}.", paths.data_dir.display());
        println!("Delete it with --wipe, or by hand; nothing else refers to it.");
    }
    Ok(())
}

/// Report what this brain has done, from its own counters.
///
/// Local only, and deliberately: these numbers exist so the user can judge
/// whether the thing is earning its keep, which is not a question anyone else
/// needs the answer to.
fn stats() -> Result<()> {
    let paths = Paths::resolve()?;
    let store = Store::open(&paths.db())?;

    println!("Captured");
    let by_cli = store.counts_by_cli().unwrap_or_default();
    let total: i64 = by_cli.iter().map(|(_, count)| count).sum();
    println!("  {total} event(s) total");
    for (cli, count) in &by_cli {
        println!("    {cli:<14} {count}");
    }

    println!("\nConsolidated");
    let tiers = store.consolidation_tiers().unwrap_or_default();
    if tiers.is_empty() {
        println!("  nothing yet");
    }
    for (tier, count) in &tiers {
        // The split that matters: a model wrote it, or a rule did.
        println!("    {tier:<14} {count} session(s)");
    }

    println!("\nRevised");
    let (forgotten, corrected) = store.revision_counts().unwrap_or((0, 0));
    println!("  {forgotten} forgotten, {corrected} corrected");

    println!("\nInjection");
    let (sessions, bytes, worst) = store.injection_stats().unwrap_or((0, 0, 0));
    if sessions == 0 {
        println!("  nothing injected yet");
    } else {
        println!("  {sessions} session(s), {} B mean, {worst} B worst", bytes / sessions);
    }

    // The question a byte budget cannot answer: was any of it used? Reported
    // only once there is something to compare against - a zero here would
    // otherwise be unreadable, meaning either "nobody used it" or "we have not
    // been counting long enough", which are opposite conclusions.
    let recalls = store.recall_count().unwrap_or(0);
    let (pushed, pulled) = store.injection_uptake().unwrap_or((0, 0));
    if recalls == 0 {
        println!("  uptake not measurable yet: no recall has been recorded");
        println!("  (counting started when this brain was upgraded, not at first capture)");
    } else {
        // "recalled", not "read": until 0.28.5 this counted any way an id
        // came back - a search hit counted the same as a body actually
        // fetched - and the line claimed the stronger of the two. The
        // opened-in-full number below is the one that means what it says.
        println!("  {pulled} of {pushed} injected pointer(s) came back through recall");
        // What search offered against what an agent chose to open. The only
        // relevance signal here that no model supplied.
        if let Ok((offered, opened)) = store.recall_precision() {
            if offered > 0 {
                println!("  {opened} of {offered} recalled entr(ies) were opened in full");
            }
        }
        // Split out because the two carry different verdicts. Summaries
        // nobody pulls means the primer describes the wrong things; work in
        // flight nobody pulls means the lines reserved for it are the wrong
        // number - and the reserve cannot be argued about without this.
        let (flight_pushed, flight_pulled) = store.in_flight_uptake().unwrap_or((0, 0));
        if flight_pushed > 0 {
            println!(
                "    of those, {flight_pulled} of {flight_pushed} were work still \
                 unsummarized when it was handed over"
            );
        }
        println!("  {recalls} recall result(s) handed to agents");
    }

    println!("\nNothing above leaves this machine.");
    Ok(())
}

fn where_am_i(models_only: bool, reranker_only: bool) -> Result<()> {
    let paths = Paths::resolve()?;
    if reranker_only {
        println!("{}", paths.model_dir_for(rerank::LOCAL_MODEL).display());
        return Ok(());
    }
    if models_only {
        println!("{}", paths.model_dir().display());
        return Ok(());
    }
    let scope = ids::resolve_scope(&std::env::current_dir().unwrap_or_default());
    println!("project    {} ({})", scope.project, scope.project_id);
    println!("workspace  {} ({})", scope.workspace, scope.workspace_id);
    println!("root       {}", scope.root.display());
    println!("memory     {}", paths.project_dir(&scope).display());
    println!("index      {}", paths.db().display());
    println!("model      {}", paths.model_dir().display());
    println!("reranker   {}", paths.model_dir_for(rerank::LOCAL_MODEL).display());
    Ok(())
}
