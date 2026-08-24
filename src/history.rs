//! What the wiki used to say.
//!
//! Every consolidation commits the pages it wrote, so the record of how a
//! memory changed its mind already exists in full - there was simply no way
//! to ask for it. This is the cheapest temporal answer available: not "when
//! was this true in the world", which nothing here knows, but "when did
//! memory start saying so, and what did it say before".
//!
//! Read-only, and local: `git log` against a repository with no remotes.
//! Nothing here can send anything anywhere.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::Paths;

/// One recorded version of a page.
pub struct Revision {
    pub commit: String,
    pub when: String,
    pub subject: String,
}

/// Pages whose path or title contains `query`, best match first.
///
/// Matching is on the path rather than the contents, because the question
/// this answers is "how did THIS page change" - the caller already knows
/// which page they mean, and searching text is what `brain search` is for.
fn matching_pages(wiki: &Path, query: &str) -> Vec<PathBuf> {
    let needle = query.to_ascii_lowercase();
    let mut found = Vec::new();
    let mut stack = vec![wiki.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                // `.git` holds the history; it is not part of it.
                if path.file_name().is_some_and(|name| name == ".git") {
                    continue;
                }
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "md")
                && path.to_string_lossy().to_ascii_lowercase().contains(&needle)
            {
                found.push(path);
            }
        }
    }
    // Shortest path first: a hub note beats a session page that merely
    // mentions the same words in its filename.
    found.sort_by_key(|path| (path.as_os_str().len(), path.clone()));
    found
}

/// Every recorded version of one page, newest first.
///
/// # Errors
/// Returns an error when git cannot be run.
pub fn revisions(wiki: &Path, page: &Path) -> Result<Vec<Revision>> {
    let relative = page.strip_prefix(wiki).unwrap_or(page);
    let output = std::process::Command::new("git")
        .args(["log", "--format=%h%x1f%ad%x1f%s", "--date=format:%Y-%m-%d %H:%M", "--"])
        .arg(relative)
        .current_dir(wiki)
        .output()
        .context("run git log")?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut parts = line.split('\u{1f}');
            Some(Revision {
                commit: parts.next()?.to_string(),
                when: parts.next()?.to_string(),
                subject: parts.next().unwrap_or_default().to_string(),
            })
        })
        .collect())
}

/// What one revision changed, as a diff.
///
/// # Errors
/// Returns an error when git cannot be run.
pub fn diff(wiki: &Path, page: &Path, commit: &str) -> Result<String> {
    let relative = page.strip_prefix(wiki).unwrap_or(page);
    let output = std::process::Command::new("git")
        .args(["show", "--format=", "--unified=1", commit, "--"])
        .arg(relative)
        .current_dir(wiki)
        .output()
        .context("run git show")?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Print the history of whatever page `query` names.
///
/// # Errors
/// Returns an error when the data directory cannot be resolved.
pub fn report(query: &str, show_diff: bool) -> Result<()> {
    let paths = Paths::resolve()?;
    let wiki = paths.wiki();
    if !wiki.join(".git").exists() {
        println!("The wiki has no history yet — it gets one on the first consolidation.");
        return Ok(());
    }

    let pages = matching_pages(&wiki, query);
    let Some(page) = pages.first() else {
        println!("No page matching `{query}`.");
        return Ok(());
    };

    println!("{}", page.strip_prefix(&wiki).unwrap_or(page).display());
    if pages.len() > 1 {
        println!("({} other page(s) also match; showing the closest)", pages.len() - 1);
    }

    let revisions = revisions(&wiki, page)?;
    if revisions.is_empty() {
        println!("\nNo recorded revisions — written, but never committed.");
        return Ok(());
    }
    println!();
    for revision in &revisions {
        println!("{}  {}  {}", revision.commit, revision.when, revision.subject);
        if show_diff {
            let body = diff(&wiki, page, &revision.commit)?;
            for line in body.lines().filter(|line| {
                (line.starts_with('+') || line.starts_with('-')) && !line.starts_with("+++")
                    && !line.starts_with("---")
            }) {
                println!("    {line}");
            }
            println!();
        }
    }
    if !show_diff && revisions.len() > 1 {
        println!("\n{} revision(s). Add --diff to see what changed.", revisions.len());
    }
    Ok(())
}
