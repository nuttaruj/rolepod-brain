//! Project and session identity.
//!
//! Reduced to the identity rules this project needs and re-expressed against
//! our own error type and marker filename.
//!
//! Two rules decide where an event lands:
//!
//! 1. A `.rolepod-brain.toml` marker in the nearest ancestor directory wins.
//!    This is how monorepos, linked worktrees, and work/personal splits get an
//!    explicit answer instead of a guess.
//! 2. Otherwise the project is the main git repository root (so every worktree
//!    of one repo shares one brain), falling back to `$cwd` outside a repo.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Marker filename looked up in every ancestor of `$cwd`.
pub const MARKER_FILE: &str = ".rolepod-brain.toml";

/// Namespace for deriving stable v5 UUIDs from paths and names.
/// Fixed forever: changing it re-keys every existing brain.
const NAMESPACE: Uuid = Uuid::from_u128(0x726f_6c65_706f_645f_6272_6169_6e5f_7631);

/// Which CLI produced an event.
///
/// Only variants this project has actually wired and tested get a name; every
/// other client is `Other(String)` and still captures with its raw identifier
/// preserved. Adding a variant means adding a tested `setup` path for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AgentKind {
    ClaudeCode,
    Codex,
    #[serde(untagged)]
    Other(String),
}

impl AgentKind {
    /// Canonical wire string. Round-trips through [`AgentKind::parse`].
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
            Self::Other(raw) => raw.as_str(),
        }
    }

    /// Parse a CLI identifier. Unknown values are preserved verbatim rather
    /// than collapsed into a single `other` bucket, so `source.cli` stays a
    /// usable filter for clients we have not wired yet.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        match raw {
            "claude-code" | "claude" => Self::ClaudeCode,
            "codex" => Self::Codex,
            other => Self::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for AgentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Resolved location of one event: which brain, which project inside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectScope {
    /// Workspace name as written by the operator (`default` when unset).
    pub workspace: String,
    /// Stable workspace UUID, derived from the name.
    pub workspace_id: Uuid,
    /// Human-facing project name — repo directory name, or the marker's value.
    pub project: String,
    /// Stable project UUID, derived from the resolved root path or marker name.
    pub project_id: Uuid,
    /// Absolute path this scope was resolved from.
    pub root: PathBuf,
}

impl ProjectScope {
    /// Directory name used under `wiki/<workspace>/`.
    ///
    /// The UUID is the identity; this is the on-disk label. It carries the
    /// slug so the wiki stays greppable and an 8-hex suffix so two projects
    /// with the same basename in different parents never collide.
    #[must_use]
    pub fn dir_name(&self) -> String {
        let short = self.project_id.simple().to_string();
        format!("{}--{}", slugify(&self.project), &short[..8])
    }
}

/// Marker file contents. Every field optional — an empty marker still pins the
/// project to that directory, which is the common monorepo case.
#[derive(Debug, Default, Deserialize)]
struct Marker {
    #[serde(default)]
    project: MarkerProject,
}

#[derive(Debug, Default, Deserialize)]
struct MarkerProject {
    workspace: Option<String>,
    name: Option<String>,
}

/// Resolve the project scope for a working directory.
///
/// Never fails: an unreadable or malformed marker is ignored in favour of the
/// git-root rule, because losing an observation is worse than filing it under
/// a slightly wrong name.
#[must_use]
pub fn resolve_scope(cwd: &Path) -> ProjectScope {
    let start = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());

    let (root, marker) = match find_marker(&start) {
        Some((dir, marker)) => (dir, marker),
        None => (git_root(&start).unwrap_or_else(|| start.clone()), Marker::default()),
    };

    let workspace = marker
        .project
        .workspace
        .filter(|w| !w.trim().is_empty())
        .unwrap_or_else(|| "default".to_string());

    let marker_named = marker.project.name.as_ref().is_some_and(|n| !n.trim().is_empty());
    let project = marker
        .project
        .name
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| {
            root.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unnamed".to_string())
        });

    // Identity follows the path by default, so a project needs no setup to be
    // remembered separately from its neighbours.
    //
    // A marker that NAMES the project pins identity to that name instead. That
    // is what makes memory portable: the same repository checked out at a
    // different path on another machine is the same project, which is exactly
    // what someone restoring a brain onto a new laptop means. Without it, an
    // imported brain would sit beside the work rather than attach to it.
    //
    // The cost is that renaming a marker orphans that project's memory. Naming
    // it is an explicit act, so that trade is the user's to make.
    let project_id = match marker_named {
        true => Uuid::new_v5(&NAMESPACE, format!("project:{project}").as_bytes()),
        false => Uuid::new_v5(&NAMESPACE, root.to_string_lossy().as_bytes()),
    };
    let workspace_id = Uuid::new_v5(&NAMESPACE, workspace.as_bytes());

    ProjectScope { workspace, workspace_id, project, project_id, root }
}

/// Walk from `start` upwards looking for a marker file.
fn find_marker(start: &Path) -> Option<(PathBuf, Marker)> {
    for dir in start.ancestors() {
        let candidate = dir.join(MARKER_FILE);
        if candidate.is_file() {
            let marker = std::fs::read_to_string(&candidate)
                .ok()
                .and_then(|text| toml::from_str::<Marker>(&text).ok())
                .unwrap_or_default();
            return Some((dir.to_path_buf(), marker));
        }
    }
    None
}

/// Find the main repository root, resolving linked worktrees to their origin
/// so every worktree of one repo shares a single brain.
fn git_root(start: &Path) -> Option<PathBuf> {
    for dir in start.ancestors() {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Some(dir.to_path_buf());
        }
        if dot_git.is_file() {
            // Linked worktree: `.git` is a file containing `gitdir: <path>`,
            // pointing at `<main>/.git/worktrees/<name>`. Two parents up from
            // there is the main `.git`, whose parent is the main checkout.
            if let Some(main) = std::fs::read_to_string(&dot_git)
                .ok()
                .and_then(|text| worktree_main_root(&text))
            {
                return Some(main);
            }
            return Some(dir.to_path_buf());
        }
    }
    None
}

/// Parse `gitdir: …/.git/worktrees/<name>` into the main checkout path.
fn worktree_main_root(gitdir_file: &str) -> Option<PathBuf> {
    let raw = gitdir_file.trim().strip_prefix("gitdir:")?.trim();
    let path = Path::new(raw);
    let mut ancestors = path.ancestors();
    // <main>/.git/worktrees/<name> -> worktrees -> .git
    ancestors.next()?;
    ancestors.next()?;
    let dot_git = ancestors.next()?;
    if dot_git.file_name()? != ".git" {
        return None;
    }
    Some(dot_git.parent()?.to_path_buf())
}

/// Derive a stable session UUID from a CLI-supplied session identifier.
///
/// Claude Code already emits a UUID; anything else (Codex thread ids, ad-hoc
/// strings) is hashed into the same namespace so the column type never varies.
#[must_use]
pub fn session_uuid(raw: &str) -> Uuid {
    Uuid::try_parse(raw).unwrap_or_else(|_| Uuid::new_v5(&NAMESPACE, raw.as_bytes()))
}

/// Lowercase, dash-separated, filesystem-safe rendering of a name.
#[must_use]
pub fn slugify(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_dash = true; // leading dashes suppressed
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_is_filesystem_safe() {
        assert_eq!(slugify("My Project!"), "my-project");
        assert_eq!(slugify("  ../etc/passwd "), "etc-passwd");
        assert_eq!(slugify("###"), "unnamed");
    }

    #[test]
    fn agent_kind_round_trips_and_preserves_unknown() {
        assert_eq!(AgentKind::parse("claude-code").as_str(), "claude-code");
        assert_eq!(AgentKind::parse("codex").as_str(), "codex");
        assert_eq!(AgentKind::parse("opencode").as_str(), "opencode");
    }

    #[test]
    fn session_uuid_passes_through_real_uuids() {
        let raw = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
        assert_eq!(session_uuid(raw).to_string(), raw);
    }

    #[test]
    fn session_uuid_is_stable_for_non_uuids() {
        assert_eq!(session_uuid("codex-thread-7"), session_uuid("codex-thread-7"));
        assert_ne!(session_uuid("codex-thread-7"), session_uuid("codex-thread-8"));
    }

    #[test]
    fn worktree_gitdir_resolves_to_main_checkout() {
        let parsed = worktree_main_root("gitdir: /home/u/proj/.git/worktrees/feature\n");
        assert_eq!(parsed, Some(PathBuf::from("/home/u/proj")));
    }

    #[test]
    fn a_named_marker_pins_identity_across_paths() {
        // The migration case: the same repository at a different path on
        // another machine must be the same project, or an imported brain sits
        // beside the work instead of attaching to it.
        let base = std::env::temp_dir().join(format!("brain-ids-{}", uuid::Uuid::new_v4()));
        let here = base.join("machine-a/repo");
        let there = base.join("machine-b/somewhere-else/repo");
        for dir in [&here, &there] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join(MARKER_FILE), "[project]\nname = \"acme-api\"\n").unwrap();
        }

        let a = resolve_scope(&here);
        let b = resolve_scope(&there);
        assert_eq!(a.project_id, b.project_id, "a named project must survive its path");
        assert_eq!(a.project, "acme-api");

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn without_a_marker_identity_still_follows_the_path() {
        let base = std::env::temp_dir().join(format!("brain-ids-{}", uuid::Uuid::new_v4()));
        let one = base.join("one/repo");
        let two = base.join("two/repo");
        for dir in [&one, &two] {
            std::fs::create_dir_all(dir).unwrap();
        }
        assert_ne!(
            resolve_scope(&one).project_id,
            resolve_scope(&two).project_id,
            "two unrelated checkouts named `repo` must not share memory"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn dir_name_disambiguates_same_basename() {
        let a = resolve_scope(Path::new("/tmp"));
        assert!(a.dir_name().contains("--"));
        assert_eq!(a.dir_name().len(), slugify(&a.project).len() + 2 + 8);
    }
}
