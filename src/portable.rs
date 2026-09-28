//! Moving a brain between machines.
//!
//! This is the necessary consequence of never syncing: if the product will not
//! move memory for you, it has to make moving it yourself trivial. The new
//! laptop needs an answer, and "copy a folder and hope" is not one.
//!
//! What travels is the wiki — the append-only logs and the pages — plus the
//! config. The SQLite index deliberately does not: it is derived, it is the
//! largest file, and shipping it would invite a mismatch between an index and
//! a log that disagree. `import` rebuilds it, which also proves on arrival
//! that the log really is the source of truth.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::config::Paths;

/// What an import may cost, whatever the archive claims.
///
/// Each is a multiple of a real brain measured on 2026-09-28 - 3,839 members
/// in 293 folders at most five deep, a longest path of 183 bytes, a largest
/// log of 213 MB, 665 MB unpacked from 153 MB (4.3 to one) - so
/// an honest archive clears them by a wide margin, and a crafted one cannot
/// fill the disk, the inode table, or the afternoon.
struct Limits {
    /// Files and folders both, counting the folders tar makes on the way to a
    /// member: nesting is how a small archive would create millions of them.
    members: usize,
    path_bytes: usize,
    depth: usize,
    /// One log is read whole to be merged, so one log is what memory bounds.
    log_bytes: u64,
    /// Never unpack more than this, however large the archive.
    unpacked_bytes: u64,
    /// Nor more than this many times the archive's own size - what a
    /// compression bomb cannot hide...
    ratio: u64,
    /// ...except that a small archive may always reach this much, since a
    /// few kilobytes of repetitive text can compress past any fixed ratio.
    ratio_floor: u64,
    /// Per tar run: listing, measuring and unpacking each get this long.
    deadline: Duration,
}

const LIMITS: Limits = Limits {
    members: 200_000,
    path_bytes: 512,
    depth: 32,
    log_bytes: 2 << 30,
    unpacked_bytes: 32 << 30,
    ratio: 100,
    ratio_floor: 64 << 20,
    deadline: Duration::from_secs(600),
};

/// How an import should treat a brain that already exists here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Existing {
    /// Refuse. The default, because the alternative is silent data loss.
    Refuse,
    /// Add the incoming logs to what is here.
    Merge,
    /// Move the current brain aside and take the incoming one whole.
    Replace,
}

/// Write the wiki and config to a tarball.
///
/// # Errors
/// Returns an error when there is nothing to export or `tar` fails.
pub fn export(archive: &Path) -> Result<String> {
    export_members(archive, true)
}

/// The sync bundle: the wiki tree (which holds the logs) and nothing else.
///
/// A machine's config is its own - budgets, models and the sync dir itself
/// differ per machine, and syncing them would fight the user's settings on
/// every pull.
pub fn export_wiki_only(archive: &Path) -> Result<String> {
    export_members(archive, false)
}

fn export_members(archive: &Path, include_config: bool) -> Result<String> {
    let paths = Paths::resolve()?;
    let wiki = paths.wiki();
    anyhow::ensure!(wiki.is_dir(), "no wiki at {} to export", wiki.display());

    let wiki_member = wiki
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| crate::config::WIKI_DIR.to_string());
    let mut members = vec![wiki_member];
    if include_config && paths.config_file().is_file() {
        members.push("config.toml".to_string());
    }
    // An import takes files and directories only, so a link in the vault
    // would make every archive of it one nobody can import - and a sync
    // bundle every other machine refuses on every run. Said here instead, on
    // the one machine that can fix it.
    for member in &members {
        refuse_links(&paths.data_dir.join(member), &[".git", ".obsidian"]).context(
            "an archive carries files and directories only: replace the link with a copy \
             of what it points at, or move it out of the wiki",
        )?;
    }

    // The vault's git history and the editor's UI state never travel. Both
    // belong to the machine that made them: a history is a different one on
    // every machine, and grafting one over another writes into `.git/objects`,
    // whose files are read-only - a sync between two machines that had both
    // committed anything failed there with `Permission denied`. Size says the
    // same thing more plainly: one real vault's `.git` is 109 MB against 250 MB
    // of memory, and it would ride in every bundle, every sync.
    let mut args = vec![
        "-czf".to_string(),
        archive.display().to_string(),
        "--exclude".to_string(),
        ".git".to_string(),
        "--exclude".to_string(),
        ".obsidian".to_string(),
        "-C".to_string(),
        paths.data_dir.display().to_string(),
    ];
    args.extend(members.iter().cloned());
    run("tar", &args).context("write the archive")?;

    let size = std::fs::metadata(archive).map(|meta| meta.len()).unwrap_or(0);
    Ok(format!(
        "Exported {} ({} KB). Contains the log and pages; the index is rebuilt on \
         import, and the vault's git history stays with this machine.",
        archive.display(),
        size / 1024
    ))
}

/// Unpack a tarball into this machine's brain.
///
/// # Errors
/// Returns an error when the archive is missing, a brain already exists and no
/// policy was chosen, or unpacking fails.
pub fn import(archive: &Path, existing: Existing) -> Result<String> {
    import_archive(archive, existing, true).map(|(message, _)| message)
}

/// The import a sync loop runs: the wiki only, also saying how many events
/// were new, so the loop can report without parsing its own message back.
///
/// A machine's config is its own (see [`export_wiki_only`]), so a bundle
/// carrying one is refused rather than allowed to repoint this machine's
/// sync folder or loosen its sanitizer.
///
/// # Errors
/// As [`import`].
pub fn import_counted(archive: &Path, existing: Existing) -> Result<(String, usize)> {
    import_archive(archive, existing, false)
}

fn import_archive(
    archive: &Path,
    existing: Existing,
    with_config: bool,
) -> Result<(String, usize)> {
    anyhow::ensure!(archive.is_file(), "no archive at {}", archive.display());
    let paths = Paths::resolve()?;
    paths.ensure()?;

    let wiki = paths.wiki();
    let occupied = wiki.is_dir()
        && std::fs::read_dir(&wiki)
            .map(|mut entries| entries.any(|entry| entry.is_ok()))
            .unwrap_or(false);

    anyhow::ensure!(
        !(occupied && existing == Existing::Refuse),
        "a brain already exists at {}.\n\n\
         Choose what should happen to it:\n  \
         --merge    add the incoming memory to it (safe: entries are ULID-keyed)\n  \
         --replace  set it aside and take the incoming one instead",
        wiki.display()
    );

    // Every pass reads one private copy. The path someone named can change
    // between the checks and the unpacking, and checking one file and then
    // unpacking another checks nothing.
    //
    // Unpacked somewhere else first, too. `tar -xzf` straight into the data
    // directory REPLACES same-named files, and the whole point of a named
    // marker is that the same project on two machines has the same project
    // id, the same directory, and the same events/YYYY-MM.jsonl - so a
    // merge that let tar win would silently destroy the local month. The
    // logs are the source of truth and are not in the wiki's git history,
    // so there would be nothing to recover from.
    //
    // Named for the moment and for this run alike: two imports - or a sync
    // and an import - can start in the same second, and neither may unpack
    // the other's archive or clear the other's staging.
    let stamp = jiff::Zoned::now().strftime("%Y%m%d-%H%M%S").to_string();
    let run = ulid::Ulid::new();
    let copy = paths.data_dir.join(format!("import.{stamp}-{run}.tar.gz"));
    let staging = paths.data_dir.join(format!("import.staging.{stamp}-{run}"));
    let mut notes = Vec::new();
    let merged = unpack_and_graft(archive, &copy, &staging, &paths, with_config, || {
        if occupied && existing == Existing::Replace {
            // Moved, never deleted. An import that destroys the memory it was
            // meant to restore is the worst possible outcome of this command.
            // And moved only now, once the archive has been unpacked and
            // checked, so a refused archive leaves the brain where it was.
            let aside = paths
                .data_dir
                .join(format!("wiki.replaced.{}", jiff::Zoned::now().strftime("%Y%m%d-%H%M%S")));
            std::fs::rename(&wiki, &aside)
                .with_context(|| format!("move {} aside", wiki.display()))?;
            notes.push(format!("previous wiki moved to {}", aside.display()));
        }
        Ok(())
    });
    // Both are scratch space; leaving them behind would look like a second
    // brain sitting next to the real one.
    let _ = std::fs::remove_dir_all(&staging);
    let _ = std::fs::remove_file(&copy);
    let counts = merged?;

    notes.push(format!(
        "merged into {} ({} file(s), {} new event(s))",
        paths.data_dir.display(),
        counts.files,
        counts.events
    ));

    Ok((notes.join("; "), counts.events))
}

/// Copy, check, unpack, check again, make room, graft - each step only once
/// the one before it has passed.
fn unpack_and_graft(
    archive: &Path,
    copy: &Path,
    staging: &Path,
    paths: &Paths,
    with_config: bool,
    make_room: impl FnOnce() -> Result<()>,
) -> Result<Grafted> {
    // Opened once, judged on that handle, and copied for exactly the length
    // it had then: a path that turns into a pipe, or a file that keeps
    // growing, cannot make the copy itself unbounded.
    let mut source =
        std::fs::File::open(archive).with_context(|| format!("open {}", archive.display()))?;
    let meta = source.metadata().with_context(|| format!("read {}", archive.display()))?;
    anyhow::ensure!(meta.is_file(), "{} is not a file", archive.display());
    anyhow::ensure!(
        meta.len() <= LIMITS.unpacked_bytes,
        "{} is {} bytes, more than an import unpacks at all",
        archive.display(),
        meta.len()
    );
    let mut target =
        std::fs::File::create(copy).with_context(|| format!("create {}", copy.display()))?;
    std::io::copy(&mut std::io::Read::take(&mut source, meta.len()), &mut target)
        .with_context(|| format!("copy {}", archive.display()))?;
    // What tar refuses is not the same on every platform: bsdtar rejects a
    // `..` member, GNU tar historically extracts it. An archive is attacker
    // controlled the moment someone is talked into importing one, so the
    // check belongs here rather than in whichever tar is installed.
    let measured = refuse_escaping_members(copy, with_config, &LIMITS)?;

    std::fs::create_dir_all(staging).with_context(|| format!("create {}", staging.display()))?;
    let args = [
        "-xzf".to_string(),
        copy.display().to_string(),
        "-C".to_string(),
        staging.display().to_string(),
    ];
    match tar_bounded(&args, false, measured, LIMITS.deadline).context("unpack the archive")? {
        Ran::Finished { .. } => {}
        // `-x` writes nothing to stdout; a tar that does is not one to trust.
        Ran::PastLimit => anyhow::bail!("tar wrote to stdout while unpacking"),
    }
    // What landed, judged again on disk rather than on the listing's word:
    // the same names, no links, and no more bytes than were measured on the
    // way through - a sparse member is where the two could part.
    refuse_unexpected_members(staging, with_config)?;
    let unpacked = refuse_links(staging, &[])?;
    anyhow::ensure!(
        unpacked <= measured,
        "the archive unpacked to {unpacked} bytes, past the {measured} it was measured at"
    );

    make_room()?;
    graft(staging, &paths.data_dir)
}

/// Refuse anything unpacked beside the wiki (and `config.toml`, when the
/// caller takes one) - the listing's allowlist, held to what is on disk.
fn refuse_unexpected_members(staging: &Path, with_config: bool) -> Result<()> {
    for entry in std::fs::read_dir(staging).with_context(|| format!("read {}", staging.display()))? {
        let name = entry.context("read entry")?.file_name();
        let expected = name == crate::config::WIKI_DIR
            || name == crate::config::LEGACY_WIKI_DIR
            || (with_config && name == "config.toml");
        anyhow::ensure!(expected, "the archive unpacked {} outside the wiki", name.to_string_lossy());
    }
    Ok(())
}

/// What a graft did, for the caller to report.
struct Grafted {
    files: usize,
    events: usize,
}

/// Move an unpacked archive into place without losing what is already there.
///
/// Pages are regenerated from the log, so the incoming copy simply wins.
/// Logs are the source of truth and are merged line by line instead: ids are
/// ULIDs, so the union of two logs is well-defined, ordered, and identical
/// whichever machine performs it.
fn graft(staging: &Path, data_dir: &Path) -> Result<Grafted> {
    let mut counts = Grafted { files: 0, events: 0 };
    let mut stack = vec![staging.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let entry = entry.context("read entry")?;
            let path = entry.path();
            // The entry's own type, not its target's: `is_dir` and `copy` both
            // follow a symlink. `refuse_links` has already been over staging;
            // this keeps graft from following one whoever calls it.
            let kind = entry.file_type().context("read entry type")?;
            if kind.is_dir() {
                stack.push(path);
                continue;
            }
            anyhow::ensure!(kind.is_file(), "{} is not a regular file", path.display());
            let relative = path.strip_prefix(staging).unwrap_or(&path);
            // Both wiki names map onto whichever this machine uses. Without
            // this, importing an archive from a pre-0.12 install onto a
            // migrated machine unpacks a second tree under the old name -
            // one the resolution in `Paths::wiki` would never look at again.
            let dest = data_dir.join(normalize_wiki_member(data_dir, relative));
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
            counts.files += 1;
            // Case-blind, because the filesystem may be: on macOS and Windows
            // an incoming `2026-09.JSONL` is the local `2026-09.jsonl`, and
            // copying it over the log instead of merging would erase it.
            if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"))
                && dest.is_file()
            {
                // Merging reads the log whole; one past this is not a log.
                let size = entry.metadata().map(|meta| meta.len()).unwrap_or(u64::MAX);
                anyhow::ensure!(
                    size <= LIMITS.log_bytes,
                    "{} is {size} bytes, past the {} a log may be",
                    relative.display(),
                    LIMITS.log_bytes
                );
                counts.events += union_logs(&path, &dest)?;
            } else {
                std::fs::copy(&path, &dest)
                    .with_context(|| format!("write {}", dest.display()))?;
            }
        }
    }
    Ok(counts)
}

/// Map either wiki directory name onto the one this machine uses.
fn normalize_wiki_member(data_dir: &Path, relative: &Path) -> PathBuf {
    let mut parts = relative.components();
    let Some(first) = parts.next() else { return relative.to_path_buf() };
    let first = first.as_os_str().to_string_lossy();
    if first != crate::config::WIKI_DIR && first != crate::config::LEGACY_WIKI_DIR {
        return relative.to_path_buf();
    }
    let current = Paths { data_dir: data_dir.to_path_buf() }
        .wiki()
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| crate::config::WIKI_DIR.to_string());
    PathBuf::from(current).join(parts.as_path())
}

/// Merge one incoming log into an existing one, keyed by event id.
///
/// Returns how many lines the local log did not already have. Sorting by id
/// is sorting by time, because ULIDs are time-ordered - so the merged log
/// reads in the order things actually happened on both machines.
fn union_logs(incoming: &Path, local: &Path) -> Result<usize> {
    let mut lines: Vec<String> = Vec::new();
    // A set, not a list: a month's log is tens of thousands of lines, and a
    // list searched once per line made merging one into itself take minutes.
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut added = 0usize;
    for (path, is_local) in [(local, true), (incoming, false)] {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            match line_id(line) {
                Some(id) => {
                    if ids.insert(id) {
                        lines.push(line.to_string());
                        if !is_local {
                            added += 1;
                        }
                    }
                }
                // A line we cannot read is still someone's data. Keeping it
                // costs a duplicate at worst; dropping it is unrecoverable.
                None => lines.push(line.to_string()),
            }
        }
    }
    // Cached: the key is a JSON parse, and a plain sort would run it on both
    // sides of every comparison.
    lines.sort_by_cached_key(|line| line_id(line));
    std::fs::write(local, format!("{}\n", lines.join("\n")))
        .with_context(|| format!("write {}", local.display()))?;
    Ok(added)
}

fn line_id(line: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("id")?
        .as_str()
        .map(str::to_string)
}

/// Refuse an archive that reaches anywhere an export does not.
///
/// An import is a file someone was sent. Absolute paths and `..` components
/// would let unpacking touch a hook config, a shell profile, or anything else
/// outside the data directory. Inside it is no safer: the reranker's runtime
/// under `models/` is loaded as native code, a hook under the vault's `.git`
/// runs on the next consolidation, and `sync.key` decides who can read the
/// next sync. So a member must sit under the wiki - or be `config.toml`, when
/// the caller takes one - and never under `.git` or `.obsidian`, which an
/// export never carries.
///
/// A link does the same damage in the other direction: a member with an
/// ordinary name that is a symlink to `~/.ssh/id_rsa` is copied into the
/// brain by following it. Nothing brain exports is a link, a device or a
/// pipe, so only files and directories are let through.
///
/// Returns how many bytes the archive unpacks to, measured by letting tar
/// write every member to a counter that stops it at the limit - the budget
/// the unpacked tree is held to afterwards.
fn refuse_escaping_members(archive: &Path, with_config: bool, limits: &Limits) -> Result<u64> {
    use std::path::Component;
    let listing = list_archive(archive, "-tzf", limits)?;
    let members = listing.lines().count();
    anyhow::ensure!(
        members <= limits.members,
        "archive holds {members} members; an import takes at most {}",
        limits.members
    );
    // The folders tar creates on the way to each member, which the listing
    // does not count. Collected only up to the limit, so the counting cannot
    // cost what it exists to prevent.
    let mut folders: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for member in listing.lines() {
        anyhow::ensure!(
            member.len() <= limits.path_bytes,
            "archive contains a path longer than {} bytes: {member}",
            limits.path_bytes
        );
        let member = member.trim();
        let mut names = Vec::new();
        for part in Path::new(member).components() {
            match part {
                Component::Normal(name) => names.push(name),
                Component::CurDir => {}
                // `..`, a root, or a Windows drive: each one reaches out.
                _ => anyhow::bail!("archive contains an unsafe path: {member}"),
            }
        }
        anyhow::ensure!(
            names.len() <= limits.depth,
            "archive contains a path deeper than {} levels: {member}",
            limits.depth
        );
        let mut folder = PathBuf::new();
        for name in names.iter().take(names.len().saturating_sub(1)) {
            folder.push(name);
            if folders.insert(folder.clone()) {
                anyhow::ensure!(
                    members + folders.len() <= limits.members,
                    "archive would create more than {} files and folders",
                    limits.members
                );
            }
        }
        let top = names.first().map(|name| name.to_string_lossy()).unwrap_or_default();
        let expected = top == crate::config::WIKI_DIR
            || top == crate::config::LEGACY_WIKI_DIR
            || (with_config && top == "config.toml" && names.len() == 1);
        anyhow::ensure!(
            expected && !names.iter().any(|name| is_hidden(name)),
            "archive contains a path outside the wiki{}: {member}",
            if with_config { " and config.toml" } else { "" }
        );
    }
    // The verbose listing opens each line with the member's mode string, and
    // its first character is the type in GNU tar and bsdtar alike: `-` a file,
    // `d` a directory, `l` a symlink, `h` a hard link. bsdtar prints a hard
    // link whose header claims a size as `-`, but it will not unpack one to a
    // target outside the archive, so what slips past here stays inside it.
    for line in list_archive(archive, "-tvzf", limits)?.lines() {
        anyhow::ensure!(
            line.starts_with('-') || line.starts_with('d'),
            "archive contains a member that is not a regular file or directory: {line}"
        );
    }

    let packed = std::fs::metadata(archive).map(|meta| meta.len()).unwrap_or(0);
    let budget =
        limits.unpacked_bytes.min(limits.ratio_floor.max(packed.saturating_mul(limits.ratio)));
    let args = ["-xzOf".to_string(), archive.display().to_string()];
    match tar_bounded(&args, false, budget, limits.deadline).context("measure the archive")? {
        Ran::Finished { wrote, .. } => Ok(wrote),
        Ran::PastLimit => anyhow::bail!(
            "the archive unpacks to more than {budget} bytes - past {}x its own {packed}, \
             or {} in all",
            limits.ratio,
            limits.unpacked_bytes
        ),
    }
}

/// Is this a name for `.git` or `.obsidian`, as the filesystem will read it?
///
/// Not only the exact name: macOS and Windows fold case, so `.GIT` is `.git`;
/// Windows drops trailing dots and spaces, so `.git.` is too, and may answer
/// to an 8.3 short name like `GIT~1`; and `.git::$INDEX_ALLOCATION` names the
/// directory through an NTFS stream. APFS folds further than ASCII - `.obſidian`,
/// with a long s, is `.obsidian` there - so a dot-name that is not plain ASCII
/// is refused outright. No name brain writes has a colon, a tilde in that
/// place, or a non-ASCII dot-name, so refusing them costs nothing real.
fn is_hidden(name: &std::ffi::OsStr) -> bool {
    let name = name.to_string_lossy().to_ascii_lowercase();
    let trimmed = name.trim_end_matches(['.', ' ']);
    trimmed == ".git"
        || trimmed == ".obsidian"
        || name.starts_with("git~")
        || name.starts_with("obsidi~")
        || name.contains(':')
        || (name.starts_with('.') && !name.is_ascii())
}

/// Refuse a tree holding anything but files and directories.
///
/// Walked without following anything, so a symlink is seen as one. On import
/// this runs over what tar actually unpacked, and catches a symlink whatever
/// the local tar made of the archive. A hard link unpacks as an ordinary file
/// and cannot be told apart here; the listing check is what refuses those.
fn refuse_links(root: &Path, skip: &[&str]) -> Result<u64> {
    let mut bytes = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let meta = std::fs::symlink_metadata(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let kind = meta.file_type();
        if kind.is_dir() {
            for entry in
                std::fs::read_dir(&path).with_context(|| format!("read {}", path.display()))?
            {
                let entry = entry.context("read entry")?;
                let name = entry.file_name();
                if skip.iter().any(|skipped| name == *skipped) {
                    continue;
                }
                anyhow::ensure!(
                    !is_hidden(&name),
                    "{} is under .git or .obsidian, which an archive never carries",
                    entry.path().display()
                );
                stack.push(entry.path());
            }
            continue;
        }
        anyhow::ensure!(kind.is_file(), "{} is not a regular file or directory", path.display());
        bytes += meta.len();
    }
    Ok(bytes)
}

fn list_archive(archive: &Path, flags: &str, limits: &Limits) -> Result<String> {
    // Room for every member the limits allow, with its verbose columns.
    let most = (limits.members as u64) * (limits.path_bytes as u64 + 256);
    let args = [flags.to_string(), archive.display().to_string()];
    match tar_bounded(&args, true, most, limits.deadline)
        .with_context(|| format!("cannot read {}", archive.display()))?
    {
        Ran::Finished { kept, .. } => Ok(String::from_utf8_lossy(&kept).into_owned()),
        Ran::PastLimit => anyhow::bail!(
            "the listing of {} runs past {most} bytes, more than {} members could fill",
            archive.display(),
            limits.members
        ),
    }
}

/// How a bounded tar run ended, when it did not fail.
enum Ran {
    /// It finished, having written this much - and this, when asked to keep it.
    Finished { wrote: u64, kept: Vec<u8> },
    /// It was stopped for writing more than the limit.
    PastLimit,
}

/// Run tar on an archive someone else made, within bounds.
///
/// Killed at the deadline, or the moment it has written more than `limit`
/// bytes to stdout - reading on past that is exactly the cost the limit is
/// there to refuse. Passing the limit is an answer, not a failure: the caller
/// knows what the limit meant and says so. A deadline or a failing tar is an
/// error.
fn tar_bounded(args: &[String], keep: bool, limit: u64, deadline: Duration) -> Result<Ran> {
    use std::io::{ErrorKind, Read};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut child = Command::new("tar")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("run tar")?;
    let over = Arc::new(AtomicBool::new(false));
    let reader = {
        let over = Arc::clone(&over);
        let mut stdout = child.stdout.take().context("tar stdout")?;
        std::thread::spawn(move || {
            let (mut seen, mut kept, mut buf) = (0u64, Vec::new(), vec![0u8; 64 * 1024]);
            loop {
                let n = match stdout.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    Err(_) => break,
                };
                seen += n as u64;
                if seen > limit {
                    over.store(true, Ordering::Relaxed);
                    break;
                }
                if keep {
                    kept.extend_from_slice(&buf[..n]);
                }
            }
            (seen, kept)
        })
    };
    let errors = {
        let mut stderr = child.stderr.take().context("tar stderr")?;
        std::thread::spawn(move || {
            // Enough to say what went wrong; the rest is drained, not kept,
            // so a tar that complains forever neither blocks nor fills memory.
            let mut text = Vec::new();
            let _ = (&mut stderr).take(64 * 1024).read_to_end(&mut text);
            let _ = std::io::copy(&mut stderr, &mut std::io::sink());
            String::from_utf8_lossy(&text).into_owned()
        })
    };

    let started = Instant::now();
    let status = loop {
        let exited = match child.try_wait() {
            Ok(exited) => exited,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error).context("wait for tar");
            }
        };
        if let Some(status) = exited {
            break Some(status);
        }
        if over.load(Ordering::Relaxed) || started.elapsed() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let (wrote, kept) = reader.join().map_err(|_| anyhow::anyhow!("tar output reader panicked"))?;
    let stderr = errors.join().unwrap_or_default();
    // Before the exit status: a tar stopped for writing too much dies of a
    // closed pipe, and that is not the story to tell.
    if over.load(Ordering::Relaxed) {
        return Ok(Ran::PastLimit);
    }
    let status = status
        .with_context(|| format!("tar did not finish within {} s", deadline.as_secs()))?;
    anyhow::ensure!(status.success(), "tar failed: {}", stderr.trim());
    Ok(Ran::Finished { wrote, kept })
}

fn run(program: &str, args: &[String]) -> Result<()> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("run {program}"))?;
    anyhow::ensure!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

/// Default archive name, for a user who did not pick one.
#[must_use]
pub fn default_archive() -> PathBuf {
    PathBuf::from(format!("brain-{}.tar.gz", jiff::Zoned::now().strftime("%Y%m%d")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_is_never_exported() {
        // It is derived, it is the biggest file, and shipping it invites an
        // index and a log that disagree.
        let source = std::fs::read_to_string("src/portable.rs").unwrap();
        assert!(!source.contains("\"brain.db\""), "the index must not be in the archive");
    }

    #[test]
    fn neither_the_history_nor_the_editors_state_is_ever_exported() {
        // Both are per-machine derived state, and grafting a history over
        // another machine's writes into read-only `.git/objects`.
        let source = std::fs::read_to_string("src/portable.rs").unwrap();
        for excluded in ["\".git\"", "\".obsidian\""] {
            assert!(
                source.contains(&format!("{excluded}.to_string()")),
                "{excluded} is not excluded from the archive"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_link_left_in_staging_is_refused_before_the_graft() {
        // Whatever a tar makes of an archive, a link left in staging must not
        // reach the graft, which would copy it into the brain by following it.
        let base = std::env::temp_dir().join(format!("brain-graft-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let staging = base.join("staging");
        std::fs::create_dir_all(staging.join("wiki")).unwrap();
        std::fs::write(staging.join("wiki/page.md"), "inside").unwrap();
        assert!(refuse_links(&staging, &[]).is_ok(), "files and directories are refused");

        let secret = base.join("secret.md");
        std::fs::write(&secret, "outside").unwrap();
        std::os::unix::fs::symlink(&secret, staging.join("wiki/leak.md")).unwrap();
        let refused = refuse_links(&staging, &[]).unwrap_err();
        assert!(refused.to_string().contains("not a regular file"), "{refused}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_archive_may_hold_the_wiki_and_nothing_an_export_would_not() {
        // Inside the data directory is not safe either: the reranker's runtime
        // is loaded as native code, and a hook under the vault's `.git` runs
        // on the next consolidation.
        let base = std::env::temp_dir().join(format!("brain-members-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let archive_of = |name: &str, members: &[&str]| {
            let tree = base.join(name);
            for member in members {
                let path = tree.join(member);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, "x").unwrap();
            }
            let archive = base.join(format!("{name}.tar.gz"));
            let mut args = vec!["-czf".to_string(), archive.display().to_string()];
            args.extend(["-C".to_string(), tree.display().to_string()]);
            args.extend(members.iter().map(ToString::to_string));
            run("tar", &args).unwrap();
            archive
        };
        let wiki = |rest: &str| format!("{}/{rest}", crate::config::WIKI_DIR);

        let page = wiki("project/pages/a.md");
        let ordinary = archive_of("ordinary", &[&page, "wiki/old.md"]);
        assert!(refuse_escaping_members(&ordinary, false, &LIMITS).is_ok());

        let with_config = archive_of("config", &[&page, "config.toml"]);
        assert!(refuse_escaping_members(&with_config, true, &LIMITS).is_ok(), "a full import takes it");
        assert!(refuse_escaping_members(&with_config, false, &LIMITS).is_err(), "a sync must not");

        let hook = wiki(".git/hooks/pre-commit");
        let folded = wiki(".GIT/hooks/post-commit");
        let plugin = wiki(".obsidian/plugins/x/main.js");
        for member in [
            "models/bge-reranker-v2-m3-int8/libonnxruntime.dylib",
            "sync.key",
            &hook,
            &folded,
            &plugin,
        ] {
            let archive = archive_of(&format!("bad{}", member.len()), &[&page, member]);
            let refused = refuse_escaping_members(&archive, true, &LIMITS).unwrap_err();
            assert!(refused.to_string().contains("outside the wiki"), "{member}: {refused}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_archive_past_any_limit_is_refused_before_it_is_unpacked() {
        // The same checks at a scale a test can reach: few members, short
        // paths, a megabyte unpacked.
        let small = Limits {
            members: 3,
            path_bytes: 64,
            depth: 4,
            log_bytes: 1 << 20,
            unpacked_bytes: 1 << 20,
            ratio: 100,
            ratio_floor: 1 << 20,
            deadline: Duration::from_secs(60),
        };
        let base = std::env::temp_dir().join(format!("brain-limits-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let archive_of = |name: &str, files: &[(String, Vec<u8>)]| {
            let tree = base.join(name);
            for (member, body) in files {
                let path = tree.join(member);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, body).unwrap();
            }
            let archive = base.join(format!("{name}.tar.gz"));
            let mut args = vec!["-czf".to_string(), archive.display().to_string()];
            args.extend(["-C".to_string(), tree.display().to_string()]);
            args.extend(files.iter().map(|(member, _)| member.clone()));
            run("tar", &args).unwrap();
            archive
        };
        let page = |name: &str| (format!("{}/{name}", crate::config::WIKI_DIR), b"x".to_vec());

        let fits = archive_of("fits", &[page("a.md"), page("b.md")]);
        assert!(refuse_escaping_members(&fits, false, &small).is_ok());

        let many = archive_of("many", &[page("a.md"), page("b.md"), page("c.md"), page("d.md")]);
        let refused = refuse_escaping_members(&many, false, &small).unwrap_err();
        assert!(format!("{refused:#}").contains("members"), "{refused:#}");

        let long = archive_of("long", &[page(&"x".repeat(80))]);
        let refused = refuse_escaping_members(&long, false, &small).unwrap_err();
        assert!(format!("{refused:#}").contains("longer than"), "{refused:#}");

        let deep = archive_of("deep", &[page("a/b/c/d.md")]);
        let refused = refuse_escaping_members(&deep, false, &small).unwrap_err();
        assert!(format!("{refused:#}").contains("deeper than"), "{refused:#}");

        // Two members, but four folders on the way to them: past three.
        let nested = archive_of("nested", &[page("a/x.md"), page("b/y.md")]);
        let refused = refuse_escaping_members(&nested, false, &small).unwrap_err();
        assert!(format!("{refused:#}").contains("files and folders"), "{refused:#}");

        // Eight megabytes of zeros pack into a few kilobytes: a bomb, in
        // miniature, and measured without ever landing on disk.
        let zeros = (format!("{}/zeros.md", crate::config::WIKI_DIR), vec![0u8; 8 << 20]);
        let bomb = archive_of("bomb", &[zeros]);
        let refused = refuse_escaping_members(&bomb, false, &small).unwrap_err();
        assert!(format!("{refused:#}").contains("unpacks to more than"), "{refused:#}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn git_and_obsidian_are_recognised_by_every_name_the_filesystem_answers_to() {
        use std::ffi::OsStr;
        for name in [
            ".git",
            ".GIT",
            ".Git.",
            ".git ",
            "GIT~1",
            ".git::$INDEX_ALLOCATION",
            ".Obsidian",
            "OBSIDI~1",
            ".ob\u{17f}idian",
        ] {
            assert!(is_hidden(OsStr::new(name)), "{name} passed");
        }
        // What a real vault holds beside them must still travel.
        for name in [".gitignore", ".gitattributes", ".DS_Store", "digit~less.md", "หน้า.md"] {
            assert!(!is_hidden(OsStr::new(name)), "{name} was refused");
        }
    }

    #[test]
    fn a_log_named_in_capitals_is_merged_not_copied_over() {
        // On a case-blind filesystem `x.JSONL` is `x.jsonl`; copying it over
        // would erase the local log. On a case-sensitive one it is a second
        // file, and the local log is untouched either way.
        let base = std::env::temp_dir().join(format!("brain-jsonl-case-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (staging, data_dir) = (base.join("staging"), base.join("data"));
        let events = |root: &Path| root.join(crate::config::WIKI_DIR).join("p/events");
        std::fs::create_dir_all(events(&staging)).unwrap();
        std::fs::create_dir_all(events(&data_dir)).unwrap();
        std::fs::write(events(&data_dir).join("2026-09.jsonl"), "{\"id\":\"01A\"}\n").unwrap();
        std::fs::write(events(&staging).join("2026-09.JSONL"), "{\"id\":\"01B\"}\n").unwrap();

        graft(&staging, &data_dir).unwrap();
        let local = std::fs::read_to_string(events(&data_dir).join("2026-09.jsonl")).unwrap();
        assert!(local.contains("01A"), "the local log was overwritten: {local}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_default_archive_name_carries_the_date() {
        let name = default_archive().display().to_string();
        assert!(name.starts_with("brain-20"));
        assert!(name.ends_with(".tar.gz"));
    }
}
