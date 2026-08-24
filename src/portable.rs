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
use std::process::Command;

use anyhow::{Context, Result};

use crate::config::Paths;

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
    let paths = Paths::resolve()?;
    let wiki = paths.wiki();
    anyhow::ensure!(wiki.is_dir(), "no wiki at {} to export", wiki.display());

    let wiki_member = wiki
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| crate::config::WIKI_DIR.to_string());
    let mut members = vec![wiki_member];
    if paths.config_file().is_file() {
        members.push("config.toml".to_string());
    }

    let mut args = vec!["-czf".to_string(), archive.display().to_string(), "-C".to_string(),
                        paths.data_dir.display().to_string()];
    args.extend(members.iter().cloned());
    run("tar", &args).context("write the archive")?;

    let size = std::fs::metadata(archive).map(|meta| meta.len()).unwrap_or(0);
    Ok(format!(
        "Exported {} ({} KB). Contains the log and pages; the index is rebuilt on import.",
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
    anyhow::ensure!(archive.is_file(), "no archive at {}", archive.display());
    let paths = Paths::resolve()?;
    paths.ensure()?;

    let wiki = paths.wiki();
    let occupied = wiki.is_dir()
        && std::fs::read_dir(&wiki)
            .map(|mut entries| entries.any(|entry| entry.is_ok()))
            .unwrap_or(false);

    let mut notes = Vec::new();
    match (occupied, existing) {
        (true, Existing::Refuse) => anyhow::bail!(
            "a brain already exists at {}.\n\n\
             Choose what should happen to it:\n  \
             --merge    add the incoming memory to it (safe: entries are ULID-keyed)\n  \
             --replace  set it aside and take the incoming one instead",
            wiki.display()
        ),
        (true, Existing::Replace) => {
            // Moved, never deleted. An import that destroys the memory it was
            // meant to restore is the worst possible outcome of this command.
            let aside = paths
                .data_dir
                .join(format!("wiki.replaced.{}", jiff::Zoned::now().strftime("%Y%m%d-%H%M%S")));
            std::fs::rename(&wiki, &aside)
                .with_context(|| format!("move {} aside", wiki.display()))?;
            notes.push(format!("previous wiki moved to {}", aside.display()));
        }
        (true, Existing::Merge) | (false, _) => {}
    }

    // Unpack somewhere else first. `tar -xzf` straight into the data
    // directory REPLACES same-named files, and the whole point of a named
    // marker is that the same project on two machines has the same project
    // id, the same directory, and the same events/YYYY-MM.jsonl - so a
    // merge that let tar win would silently destroy the local month. The
    // logs are the source of truth and are not in the wiki's git history,
    // so there would be nothing to recover from.
    // What tar refuses is not the same on every platform: bsdtar rejects a
    // `..` member, GNU tar historically extracts it. An archive is attacker
    // controlled the moment someone is talked into importing one, so the
    // check belongs here rather than in whichever tar is installed.
    refuse_escaping_members(archive)?;

    let staging = paths
        .data_dir
        .join(format!("import.staging.{}", jiff::Zoned::now().strftime("%Y%m%d-%H%M%S")));
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("create {}", staging.display()))?;
    let unpacked = run(
        "tar",
        &[
            "-xzf".to_string(),
            archive.display().to_string(),
            "-C".to_string(),
            staging.display().to_string(),
        ],
    )
    .context("unpack the archive");
    let merged = unpacked.and_then(|()| graft(&staging, &paths.data_dir));
    // Staging is scratch space; leaving it behind would look like a second
    // brain sitting next to the real one.
    let _ = std::fs::remove_dir_all(&staging);
    let counts = merged?;

    notes.push(format!(
        "merged into {} ({} file(s), {} new event(s))",
        paths.data_dir.display(),
        counts.files,
        counts.events
    ));

    Ok(notes.join("; "))
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
            let path = entry.context("read entry")?.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
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
            if path.extension().is_some_and(|ext| ext == "jsonl") && dest.is_file() {
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
    let mut ids: Vec<String> = Vec::new();
    let mut added = 0usize;
    for (path, is_local) in [(local, true), (incoming, false)] {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            match line_id(line) {
                Some(id) if ids.contains(&id) => {}
                Some(id) => {
                    ids.push(id);
                    lines.push(line.to_string());
                    if !is_local {
                        added += 1;
                    }
                }
                // A line we cannot read is still someone's data. Keeping it
                // costs a duplicate at worst; dropping it is unrecoverable.
                None => lines.push(line.to_string()),
            }
        }
    }
    lines.sort_by_key(|line| line_id(line));
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

/// Refuse an archive that would write outside where it is unpacked.
///
/// Absolute paths and `..` components are the whole attack: an import is a
/// file someone was sent, and unpacking it must not be able to touch a
/// hook config, a shell profile, or anything else outside the data
/// directory.
fn refuse_escaping_members(archive: &Path) -> Result<()> {
    let listing = Command::new("tar")
        .args(["-tzf", &archive.display().to_string()])
        .output()
        .context("list the archive")?;
    anyhow::ensure!(
        listing.status.success(),
        "cannot read {}: {}",
        archive.display(),
        String::from_utf8_lossy(&listing.stderr).trim()
    );
    for member in String::from_utf8_lossy(&listing.stdout).lines() {
        let member = member.trim();
        let escapes = member.starts_with('/')
            || member.starts_with("~/")
            || Path::new(member)
                .components()
                .any(|part| part == std::path::Component::ParentDir);
        anyhow::ensure!(!escapes, "archive contains an unsafe path: {member}");
    }
    Ok(())
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
    fn a_default_archive_name_carries_the_date() {
        let name = default_archive().display().to_string();
        assert!(name.starts_with("brain-20"));
        assert!(name.ends_with(".tar.gz"));
    }
}
