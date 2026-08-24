//! Forgetting and correcting — the two things memory could not do.
//!
//! Memory that can only be born is a liability. A summary written by a cheap
//! model can be wrong, and a wrong memory is injected into every future
//! session with exactly the confidence of a right one. Worse, the failure mode
//! is invisible: nobody re-reads the wiki looking for sentences that were
//! never true.
//!
//! Neither operation edits or deletes anything. A tombstone and a correction
//! are ordinary appended lines that happen to be *about* an earlier line. The
//! log stays the append-only truth — including the fact that something was
//! withdrawn, and what it said before — while the derived index stops showing
//! it. That is also why replaying the log reproduces both: a correction always
//! sorts after what it corrects, because ULIDs are time-ordered.

use anyhow::{Context, Result};

use crate::config::Paths;
use crate::event::{Event, EventKind, EventLog, Source};
use crate::ids;
use crate::store::Store;

/// What a revision did, for the caller to report.
pub struct Outcome {
    pub id: String,
    pub target_title: String,
}

/// Withdraw an entry from memory.
///
/// # Errors
/// Returns an error when the id is unknown, already forgotten, or the write
/// fails.
pub fn forget(id: &str) -> Result<Outcome> {
    let (paths, store, scope) = context()?;
    let (_, title) = store
        .event_summary(id)?
        .with_context(|| format!("no memory with id {id}"))?;
    anyhow::ensure!(store.event_exists(id)?, "{id} is already forgotten");

    let mut event = Event::new(
        scope.workspace_id,
        scope.project_id,
        uuid::Uuid::nil(),
        Source { cli: "brain".to_string(), hook: "forget".to_string() },
        EventKind::Tombstone,
        // Deliberately says nothing about what was withdrawn. A tombstone
        // quoting its target puts the forgotten text straight back into
        // search results and the primer - which is the whole thing being
        // undone. The link carries the identity; the words do not.
        "Withdrew a memory".to_string(),
        String::new(),
    );
    event.links = vec![id.to_string()];
    // A tombstone is the final word on its target; there is nothing for a
    // summarizer to improve about it.
    event.consolidated = true;

    let log = EventLog::open(&paths.project_dir(&scope))?;
    log.append(&event)?;
    store.index(&event)?;

    Ok(Outcome { id: event.id, target_title: title })
}

/// Withdraw every memory that mentions one thing.
///
/// The third forgetting primitive, after purge-by-id and supersede: "forget
/// everything about this customer / this key / this repository", where the
/// caller does not know - and should not have to enumerate - the ids.
///
/// **Siblings must survive.** Entities are recorded per session, so tombstoning
/// whole sessions would destroy every unrelated memory those sessions also
/// hold. This matches at the event level instead: only entries whose own text
/// mentions the thing are withdrawn.
///
/// Two-step by design, like `setup`: the first call lists what would go, and
/// nothing happens without `apply`. A bulk withdrawal the caller cannot
/// preview is a bulk withdrawal they cannot verify.
///
/// Honest about its reach: this is lexical. A memory that refers to the same
/// entity by another name, or in another script, is not found - published
/// measurements put deterministic matching in the single digits on
/// obfuscated identifiers. It withdraws what it lists; it does not promise
/// the list is complete.
///
/// # Errors
/// Returns an error when the query or a write fails.
pub fn forget_entity(name: &str, apply: bool) -> Result<Vec<Outcome>> {
    anyhow::ensure!(!name.trim().is_empty(), "name something to forget");
    let (paths, store, scope) = context()?;
    let project = scope.project_id.to_string();
    let targets = store.search(&project, name, None, ENTITY_FORGET_MAX)?;

    if !apply {
        return Ok(targets
            .into_iter()
            .map(|hit| Outcome { id: hit.id, target_title: hit.title })
            .collect());
    }

    let log = EventLog::open(&paths.project_dir(&scope))?;
    let mut done = Vec::new();
    for hit in targets {
        if !store.event_exists(&hit.id)? {
            continue;
        }
        let mut event = Event::new(
            scope.workspace_id,
            scope.project_id,
            uuid::Uuid::nil(),
            Source { cli: "brain".to_string(), hook: "forget".to_string() },
            EventKind::Tombstone,
            // Says nothing about its target, for the same reason single
            // withdrawal does not: a tombstone that names what it removed
            // puts the removed thing back into search.
            "Withdrew a memory".to_string(),
            String::new(),
        );
        event.links = vec![hit.id.clone()];
        event.consolidated = true;
        log.append(&event)?;
        store.index(&event)?;
        done.push(Outcome { id: hit.id, target_title: hit.title });
    }
    Ok(done)
}

/// Ceiling on one bulk withdrawal.
///
/// A cap the user can see beats an unbounded sweep they cannot: if a name
/// matches more than this, the right move is a narrower name, not a larger
/// blast radius.
const ENTITY_FORGET_MAX: usize = 200;

/// Replace what an entry says.
///
/// # Errors
/// Returns an error when the id is unknown or forgotten, the replacement is
/// empty, or the write fails.
pub fn correct(id: &str, text: &str) -> Result<Outcome> {
    anyhow::ensure!(!text.trim().is_empty(), "a correction needs text");
    let (paths, store, scope) = context()?;
    let (_, title) = store
        .event_summary(id)?
        .with_context(|| format!("no memory with id {id}"))?;
    anyhow::ensure!(
        store.event_exists(id)?,
        "{id} was forgotten; there is nothing to correct"
    );

    // A correction is human-written, but it goes through the same scrub as
    // anything else: someone fixing a summary may well paste the thing the
    // summary got wrong, secrets included.
    let config = crate::config::Config::load(&paths.config_file())?;
    let sanitizer = crate::sanitize::Sanitizer::new(&config.sanitize)
        .context("compile sanitizer patterns")?;
    let body = sanitizer.scrub_body(text);
    let headline = crate::sanitize::truncate(&crate::hook::first_line(&body), 120);

    let mut event = Event::new(
        scope.workspace_id,
        scope.project_id,
        uuid::Uuid::nil(),
        // `correct` in the hook slot is what tells indexing this note is about
        // another event rather than a standalone one.
        Source { cli: "brain".to_string(), hook: "correct".to_string() },
        EventKind::Note,
        headline,
        body,
    );
    event.links = vec![id.to_string()];
    event.consolidated = true;

    let log = EventLog::open(&paths.project_dir(&scope))?;
    log.append(&event)?;
    store.index(&event)?;

    Ok(Outcome { id: event.id, target_title: title })
}

fn context() -> Result<(Paths, Store, ids::ProjectScope)> {
    let paths = Paths::resolve()?;
    paths.ensure()?;
    let store = Store::open(&paths.db())?;
    let scope = ids::resolve_scope(&std::env::current_dir().unwrap_or_default());
    Ok((paths, store, scope))
}
