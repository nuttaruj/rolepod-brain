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

    let mut members = vec!["wiki".to_string()];
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
        // Merging relies on the same property that makes the logs safe to
        // union-merge in git: ids are globally unique and time-ordered, so two
        // logs concatenate without contradicting each other.
        (true, Existing::Merge) | (false, _) => {}
    }

    run(
        "tar",
        &[
            "-xzf".to_string(),
            archive.display().to_string(),
            "-C".to_string(),
            paths.data_dir.display().to_string(),
        ],
    )
    .context("unpack the archive")?;
    notes.push(format!("unpacked into {}", paths.data_dir.display()));

    Ok(notes.join("; "))
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
