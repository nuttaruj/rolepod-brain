//! A team brain: the distillate, and nothing else.
//!
//! Sessions are personal. What a person asked, what they ran, what their
//! agent narrated back - none of it crosses to a teammate, ever, and that is
//! the design rather than a setting. What crosses is the tier that already
//! had to survive several sessions to exist at all: the rules, gotchas,
//! decisions and procedures under `knowledge/`. One member's hard-won "vitest
//! has to run file-by-file here" is worth the whole feature; their Tuesday is
//! not.
//!
//! Everything underneath is Mode A's, unchanged: the same XChaCha20-Poly1305
//! bundle, the same folder the members already share, the same id-keyed union
//! of append-only logs. A team is that transport plus three things:
//!
//! **A content gate.** Only `EventKind::Knowledge` events this store itself
//! minted are published, and every one is scrubbed twice on the way out: the
//! capture sanitizer again, then a boundary scrub for the things that are
//! harmless inside one person's memory and are not theirs to send - home
//! directories, which carry a real name, and email addresses. A distilled
//! lesson quotes the sessions it came from, and the boundary between two
//! people is exactly where that must stop.
//!
//! **A name.** Joining asks for one, and every published entry carries it. A
//! rule whose author cannot be asked "why?" is a rule nobody can retire.
//!
//! **No reach into anyone else's memory.** Only additive knowledge is
//! published - never a tombstone, never a correction - so no bundle can
//! delete or rewrite what another member wrote. What arrives is indexed
//! `team = 1`, which every path that rewrites knowledge skips. Disagreeing
//! means publishing your own better lesson, not editing theirs.
//!
//! Off until `brain team init` is run, like everything else that leaves this
//! machine.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{Config, Paths};
use crate::event::{Event, EventKind, EventLog};
use crate::ids::ProjectScope;
use crate::store::Store;

/// Where published knowledge lands, under the vault root. Keyed by project
/// id, because the project id is what two members of a team share - it is
/// derived from the repository's root commit, so the same repo is the same
/// project on both machines whatever either of them called the directory.
pub const DIR: &str = "_team";

/// Bundle filename suffix. Distinct from Mode A's, so one folder can carry
/// both without either reading the other's files.
const BUNDLE_SUFFIX: &str = ".team.enc";

/// What one team sync did.
pub struct Outcome {
    pub pulled: usize,
    pub skipped: Vec<String>,
    pub gained: usize,
    pub published: usize,
    pub pushed_bytes: u64,
}

/// Join (or create) a team: point at a shared folder, mint the key, take a
/// name to sign with.
///
/// # Errors
/// Returns an error when the name is empty, the directory cannot be used, or
/// the config cannot be written.
pub fn init(dir: &str, author: &str) -> Result<String> {
    let author = author.trim();
    anyhow::ensure!(
        !author.is_empty(),
        "a team needs a name to sign your entries with: `brain team init <dir> --name \"Your Name\"`"
    );
    let paths = Paths::resolve()?;
    paths.ensure()?;
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let dir = dir.canonicalize().with_context(|| format!("resolve {}", dir.display()))?;

    let mut config = Config::load(&paths.config_file())?;
    config.team.dir = Some(dir.clone());
    config.team.author = Some(author.to_string());
    let rendered = toml::to_string(&config).context("render config")?;
    std::fs::write(paths.config_file(), rendered).context("write config")?;

    let key_path = paths.data_dir.join("team.key");
    let minted = crate::sync::mint_key(&key_path)?;

    Ok(format!(
        "team folder: {}\nsigning as: {author}\nkey: {} ({})\n\n\
         What crosses: rules, gotchas, decisions and procedures - the lessons\n\
         that already survived several sessions. What never does: your\n\
         sessions, your prompts, your captures, your notes.\n\n\
         To add a teammate:\n  \
         1. they run `brain team init <same folder> --name \"Their Name\"`\n  \
         2. they REPLACE their {} with this one - the key is what makes\n     \
            a folder of bundles one team\n  \
         3. both run `brain team`\n\n\
         Nothing is published until you run `brain team`, and nobody can\n\
         delete or rewrite what you publish.",
        dir.display(),
        key_path.display(),
        if minted { "minted now - share it with your team" } else { "already present" },
        key_path.display(),
    ))
}

/// Pull every teammate's bundle, then publish our own knowledge.
///
/// # Errors
/// Returns an error when the team is not configured, the key is missing, or
/// the push fails. A bundle that will not open is reported and skipped.
pub fn run() -> Result<Outcome> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file())?;
    let dir = config.team.dir.as_ref().context(
        "no team on this machine - run `brain team init <shared folder> --name \"Your Name\"` first",
    )?;
    anyhow::ensure!(dir.is_dir(), "team folder {} does not exist", dir.display());
    let author = config.team.author.clone().unwrap_or_default();
    anyhow::ensure!(
        !author.trim().is_empty(),
        "no name to sign with - run `brain team init` again with --name"
    );
    let key = crate::sync::read_key(&paths.data_dir.join("team.key"), "brain team init")?;
    let origin = crate::ids::origin().context("this store has no origin id yet")?;
    let store = Store::open(&paths.db())?;
    let sanitizer =
        crate::sanitize::Sanitizer::new(&config.sanitize).context("compile sanitizer patterns")?;

    let mut outcome =
        Outcome { pulled: 0, skipped: Vec::new(), gained: 0, published: 0, pushed_bytes: 0 };

    // Pull first, so what we publish is measured against what the team
    // already knows.
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let path = entry.context("read team folder entry")?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else { continue };
        let Some(stem) = name.strip_suffix(BUNDLE_SUFFIX) else { continue };
        if stem == origin {
            continue;
        }
        let sealed = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let Ok(plain) = crate::sync::open_bundle(&key, &sealed) else {
            outcome.skipped.push(name.to_string());
            continue;
        };
        outcome.pulled += 1;
        outcome.gained += absorb(&paths, &String::from_utf8_lossy(&plain))?;
    }

    // Index and render whatever is now on the team shelf - including a
    // bundle pulled by an earlier run that this machine has not read yet.
    let indexed = index_team_logs(&paths, &store)?;
    for (scope, project_dir) in crate::consolidate::known_projects(&paths)? {
        let _ = crate::consolidate::write_hubs(&project_dir, &scope, &store);
    }

    // Publish: our own knowledge, scrubbed again, signed.
    let mut lines = Vec::new();
    for (_, project_dir) in crate::consolidate::known_projects(&paths)? {
        let (events, _) = EventLog::open(&project_dir)?.read_all()?;
        for event in events {
            let Some(published) = publishable(&event, &origin, &author, &sanitizer) else {
                continue;
            };
            lines.push(serde_json::to_string(&published).context("render published event")?);
        }
    }
    outcome.published = lines.len();
    if !lines.is_empty() {
        let plain = format!("{}\n", lines.join("\n"));
        let sealed = crate::sync::seal_bundle(&key, plain.as_bytes())?;
        outcome.pushed_bytes = sealed.len() as u64;
        let target = dir.join(format!("{origin}{BUNDLE_SUFFIX}"));
        let tmp = dir.join(format!("{origin}{BUNDLE_SUFFIX}.tmp"));
        std::fs::write(&tmp, sealed).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &target)
            .with_context(|| format!("publish {}", target.display()))?;
    }
    let _ = indexed;
    Ok(outcome)
}

/// A home directory in any of the shapes the three platforms write it.
/// The tail is kept - `~/dev/app/src/x.rs` says everything useful - and the
/// account name, which is usually a person's real name, is not sent.
static HOME_DIR: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?i)(/Users/|/home/|[A-Z]:\\Users\\)[^/\\\s:]+")
        .expect("home pattern compiles")
});

/// An address is a person, not a lesson.
static EMAIL: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
        .expect("email pattern compiles")
});

/// The second scrub, run only on what leaves for a teammate.
///
/// Deliberately not part of the capture sanitizer: inside one person's own
/// memory a home path is useful and an address may be the whole point of a
/// note. Across the boundary neither is ours to send, and a distilled lesson
/// carries both without anyone noticing - it was written from sessions that
/// were full of them.
fn boundary_scrub(text: &str) -> String {
    let without_homes = HOME_DIR.replace_all(text, "~");
    EMAIL.replace_all(&without_homes, "[REDACTED]").into_owned()
}

/// The content gate: is this event ours to publish, and what does it look
/// like once it is fit to leave?
///
/// `None` for everything that is not a lesson this store itself minted -
/// which is every capture, every session summary, every note, every
/// document, and everything a teammate published to us. `Some` for a
/// knowledge entry, with title and body scrubbed a second time and the
/// author's name attached.
///
/// Links are dropped deliberately: they name the session summaries the
/// lesson was drawn from, and those ids mean nothing on another machine
/// while telling it how much work the author did and when.
fn publishable(
    event: &Event,
    origin: &str,
    author: &str,
    sanitizer: &crate::sanitize::Sanitizer,
) -> Option<Event> {
    if event.kind != EventKind::Knowledge {
        return None;
    }
    // Ours, and only ours. `origin` is stamped at append time; an event
    // written before that field existed is ours by the same token, because
    // only our own log is read here and a teammate's entries live under
    // `_team/`.
    if event.origin.as_deref().is_some_and(|stamped| stamped != origin) {
        return None;
    }
    let mut published = event.clone();
    published.title = boundary_scrub(&sanitizer.scrub(&event.title));
    published.body = boundary_scrub(&sanitizer.scrub_body(&event.body));
    published.files.clear();
    published.links.clear();
    published.origin = Some(origin.to_string());
    published.extra.insert("author".to_string(), serde_json::Value::from(author));
    Some(published)
}

/// Merge one bundle's lines into the team shelf, keyed by project id.
///
/// Returns how many lines were new. Anything that is not a knowledge event
/// is dropped on arrival as well as on departure: a bundle is written by
/// someone else's software, and the gate that matters is the one on the
/// receiving side.
fn absorb(paths: &Paths, plain: &str) -> Result<usize> {
    let mut by_project: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for line in plain.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(event) = serde_json::from_str::<Event>(line) else { continue };
        if event.kind != EventKind::Knowledge {
            continue;
        }
        by_project
            .entry(event.project.to_string())
            .or_default()
            .push((event.id.clone(), line.to_string()));
    }

    let mut gained = 0;
    for (project, incoming) in by_project {
        let dir = paths.wiki().join(DIR).join(&project).join("events");
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let path = dir.join("knowledge.jsonl");
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        let mut ids: Vec<String> = existing
            .lines()
            .filter_map(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()?
                    .get("id")?
                    .as_str()
                    .map(str::to_string)
            })
            .collect();
        let mut lines: Vec<String> =
            existing.lines().filter(|line| !line.trim().is_empty()).map(str::to_string).collect();
        for (id, line) in incoming {
            if ids.contains(&id) {
                continue;
            }
            ids.push(id);
            lines.push(line);
            gained += 1;
        }
        lines.sort();
        std::fs::write(&path, format!("{}\n", lines.join("\n")))
            .with_context(|| format!("write {}", path.display()))?;
    }
    Ok(gained)
}

/// Index every team log into the store and render what a person reads.
///
/// Returns how many entries were indexed. Called by `brain team` and by
/// `brain reindex`, which is what keeps the team shelf derived state like
/// everything else here.
///
/// # Errors
/// Returns an error when a log cannot be read or a page cannot be written.
pub fn index_team_logs(paths: &Paths, store: &Store) -> Result<usize> {
    let root = paths.wiki().join(DIR);
    if !root.is_dir() {
        return Ok(0);
    }
    let mut indexed = 0;
    for entry in std::fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let project_dir = entry.context("read team dir entry")?.path();
        if !project_dir.is_dir() {
            continue;
        }
        let (events, _) = EventLog::open(&project_dir)?.read_all()?;
        for event in &events {
            if event.kind != EventKind::Knowledge {
                continue;
            }
            store.index_team_event(event)?;
            indexed += 1;
        }
        write_pages(&project_dir, &events)?;
    }
    Ok(indexed)
}

/// One published lesson as a hub lists it.
pub struct TeamPage {
    pub title: String,
    pub author: String,
    /// Vault-relative link target.
    pub link: String,
}

/// The team pages that belong to one project, for its hub.
#[must_use]
pub fn pages_for(project_dir: &Path, scope: &ProjectScope) -> Vec<TeamPage> {
    // `_team/<id>` sits at the vault root, and a project directory is one or
    // two levels under it.
    let mut root = project_dir.to_path_buf();
    let mut pages = Vec::new();
    for _ in 0..3 {
        let candidate = root.join(DIR).join(scope.project_id.to_string());
        if candidate.is_dir() {
            let Ok((events, _)) = EventLog::open(&candidate).and_then(|log| log.read_all()) else {
                return pages;
            };
            let depth = project_dir.strip_prefix(&root).map_or(1, |rest| rest.components().count());
            let up = "../".repeat(depth);
            for event in events {
                if event.kind != EventKind::Knowledge {
                    continue;
                }
                pages.push(TeamPage {
                    title: event.title.clone(),
                    author: author_of(&event),
                    link: format!(
                        "{up}{DIR}/{}/{}/{}",
                        scope.project_id,
                        kind_dir(&event),
                        crate::ids::slugify(&event.title)
                    ),
                });
            }
            pages.sort_by(|a, b| a.title.cmp(&b.title));
            return pages;
        }
        let Some(parent) = root.parent().map(Path::to_path_buf) else { break };
        root = parent;
    }
    pages
}

fn author_of(event: &Event) -> String {
    event
        .extra
        .get("author")
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .unwrap_or("unsigned")
        .to_string()
}

fn kind_dir(event: &Event) -> String {
    let kind = if event.source.hook.is_empty() { "lesson" } else { event.source.hook.as_str() };
    match kind {
        "decision" => "decisions".to_string(),
        other => format!("{other}s"),
    }
}

/// Render one team log as pages a person can read.
fn write_pages(project_dir: &Path, events: &[Event]) -> Result<()> {
    for event in events {
        if event.kind != EventKind::Knowledge {
            continue;
        }
        let dir = project_dir.join(kind_dir(event));
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let path = dir.join(format!("{}.md", crate::ids::slugify(&event.title)));
        let mut page = String::new();
        let _ = writeln!(
            page,
            "---\ntitle: {}\ntags: [knowledge, team, {}]\nauthor: {}\n---\n",
            crate::consolidate::yaml_scalar(&event.title),
            event.source.hook,
            crate::consolidate::yaml_scalar(&author_of(event))
        );
        let _ = writeln!(page, "# {}\n", event.title);
        let _ = writeln!(page, "{}\n", event.body);
        let _ = writeln!(
            page,
            "Published by **{}** on {}. This is a teammate's entry: read it, \
             and if you disagree, publish your own rather than editing theirs.",
            author_of(event),
            &event.ts[..event.ts.len().min(10)]
        );
        std::fs::write(&path, page).with_context(|| format!("write {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Source;
    use uuid::Uuid;

    fn knowledge(title: &str, body: &str, hook: &str, origin: Option<&str>) -> Event {
        let mut event = Event::new(
            Uuid::nil(),
            Uuid::nil(),
            Uuid::nil(),
            Source { cli: "brain".into(), hook: hook.into() },
            EventKind::Knowledge,
            title.into(),
            body.into(),
        );
        event.origin = origin.map(str::to_string);
        event.links = vec!["01SOURCE0000000000000000".to_string()];
        event.files = vec!["src/auth.rs".to_string()];
        event
    }

    fn sanitizer() -> crate::sanitize::Sanitizer {
        crate::sanitize::Sanitizer::new(&crate::sanitize::SanitizeConfig::default()).unwrap()
    }

    #[test]
    fn only_our_own_lessons_are_published_and_they_leave_signed_and_scrubbed() {
        let clean = sanitizer();
        let mine = knowledge(
            "vitest runs file-by-file here",
            "Set by /Users/someone/dev/app; ask alex@example.com. Token: AKIAIOSFODNN7EXAMPLE",
            "gotcha",
            Some("me"),
        );
        let published = publishable(&mine, "me", "Alex", &clean).expect("our own lesson publishes");
        assert_eq!(published.extra.get("author").and_then(|v| v.as_str()), Some("Alex"));
        assert!(!published.body.contains("/Users/someone"), "a home directory crossed: {}", published.body);
        assert!(published.body.contains("~/dev/app"), "the useful tail was thrown away: {}", published.body);
        assert!(!published.body.contains("alex@example.com"), "an address crossed: {}", published.body);
        assert!(!published.body.contains("AKIAIOSFODNN7EXAMPLE"), "the capture sanitizer did not run: {}", published.body);
        assert!(published.links.is_empty(), "session ids must not travel");
        assert!(published.files.is_empty(), "a teammate's file layout is not ours to send");

        // A teammate's entry is never republished by us.
        let theirs = knowledge("their lesson", "body", "rule", Some("them"));
        assert!(publishable(&theirs, "me", "Alex", &clean).is_none());
    }

    #[test]
    fn the_boundary_scrub_takes_the_name_and_keeps_the_path() {
        assert_eq!(boundary_scrub("/Users/jane/dev/app/src/x.rs"), "~/dev/app/src/x.rs");
        assert_eq!(boundary_scrub("/home/jane-doe/app"), "~/app");
        assert_eq!(boundary_scrub(r"C:\Users\Jane\app"), r"~\app");
        assert_eq!(boundary_scrub("mail j.doe+ci@team.example.org now"), "mail [REDACTED] now");
        // Everything else is left exactly as written.
        let plain = "vitest must run file-by-file; see src/config.ts and https://vitest.dev";
        assert_eq!(boundary_scrub(plain), plain);
    }

    #[test]
    fn nothing_but_a_lesson_ever_crosses() {
        let clean = sanitizer();
        for kind in [
            EventKind::Observation,
            EventKind::SessionSummary,
            EventKind::Source,
            EventKind::Note,
            EventKind::PageUpdate,
            EventKind::Tombstone,
            EventKind::Retire,
        ] {
            let mut event = knowledge("t", "b", "gotcha", Some("me"));
            event.kind = kind;
            assert!(
                publishable(&event, "me", "Alex", &clean).is_none(),
                "{kind:?} must never be published to a team"
            );
        }
        // And a tombstone that arrives anyway is dropped on the way in, so
        // no bundle can withdraw what someone else wrote.
        let data = std::env::temp_dir().join(format!("brain-team-{}", ulid::Ulid::new()));
        let paths = Paths { data_dir: data.clone() };
        std::fs::create_dir_all(paths.wiki()).unwrap();
        let mut tombstone = knowledge("withdraw", "", "forget", Some("them"));
        tombstone.kind = EventKind::Tombstone;
        let lesson = knowledge("their lesson", "body", "rule", Some("them"));
        let bundle = format!(
            "{}\n{}\n",
            serde_json::to_string(&lesson).unwrap(),
            serde_json::to_string(&tombstone).unwrap()
        );
        assert_eq!(absorb(&paths, &bundle).unwrap(), 1, "only the lesson is absorbed");
        let shelf = std::fs::read_to_string(
            paths.wiki().join(DIR).join(Uuid::nil().to_string()).join("events/knowledge.jsonl"),
        )
        .unwrap();
        assert!(shelf.contains("their lesson"));
        assert!(!shelf.contains("\"kind\":\"tombstone\""), "a withdrawal crossed the boundary");
        // Absorbing the same bundle twice adds nothing.
        assert_eq!(absorb(&paths, &bundle).unwrap(), 0);
        std::fs::remove_dir_all(&data).ok();
    }

    #[test]
    fn a_published_page_says_who_wrote_it_and_that_it_is_not_yours_to_edit() {
        let dir = std::env::temp_dir().join(format!("brain-teampage-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut lesson = knowledge("Always lint before committing", "Because CI is slow.", "rule", Some("them"));
        lesson.extra.insert("author".to_string(), serde_json::Value::from("Sam"));
        write_pages(&dir, std::slice::from_ref(&lesson)).unwrap();

        let page = std::fs::read_to_string(dir.join("rules/always-lint-before-committing.md")).unwrap();
        assert!(page.contains("author: Sam"), "{page}");
        assert!(page.contains("tags: [knowledge, team, rule]"), "{page}");
        assert!(page.contains("publish your own rather than editing theirs"), "{page}");

        // An entry with no name still says so rather than looking like ours.
        let anonymous = knowledge("A rule", "body", "rule", Some("them"));
        write_pages(&dir, std::slice::from_ref(&anonymous)).unwrap();
        assert!(std::fs::read_to_string(dir.join("rules/a-rule.md")).unwrap().contains("unsigned"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
