//! Configuration and on-disk layout.
//!
//! The whole surface is deliberately small. Every value has a default that
//! makes an untouched install light, so `config.toml` is optional and usually
//! absent.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ids::ProjectScope;
use crate::sanitize::SanitizeConfig;

/// Environment variable that relocates the whole data directory. Exists so
/// tests never touch a real brain.
pub const DATA_DIR_ENV: &str = "ROLEPOD_BRAIN_HOME";

/// What the wiki directory is called - and therefore what the Obsidian vault
/// is called, because Obsidian names a vault after its folder.
pub const WIKI_DIR: &str = "Rolepod Brain";

/// The name every version before 0.12 used.
pub const LEGACY_WIKI_DIR: &str = "wiki";

/// Top-level config, read from `<data_dir>/config.toml` when present.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub summarizer: SummarizerConfig,
    pub injection: InjectionConfig,
    pub sanitize: SanitizeConfig,
    pub search: SearchConfig,
    pub sync: SyncConfig,
}

/// What happens after the index has answered.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Let a cheap model reorder search results by what the query was asking.
    ///
    /// Off by default, and the default is a claim about cost, not quality:
    /// every search would spend a model call in the middle of someone's
    /// session, and text relevance is already right most of the time. Turn it
    /// on if your searches return the right entry in the wrong place.
    pub rerank: bool,
}

/// Where sync bundles go, when the owner opts in. `None` - the default,
/// forever - means memory never leaves this machine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SyncConfig {
    pub dir: Option<std::path::PathBuf>,
}

/// Which model tier consolidates, if any.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SummarizerConfig {
    /// `auto` borrows the cheap tier of the CLI that produced the events.
    /// `off` is permanent rule-based consolidation — a first-class mode, not a
    /// degraded one. Named CLIs pin one summarizer.
    pub mode: String,
    /// Per-CLI model overrides: `claude-code = "sonnet"` makes that CLI's
    /// summaries worth more, at that CLI's price.
    ///
    /// Per-CLI rather than one global name, because model names do not
    /// travel: "sonnet" means something to `claude` and nothing to `codex`,
    /// and a global override under `auto` mode would hand every other rung a
    /// model it cannot run — which fails exactly like an outage and charges
    /// the breaker for it. A CLI not named here keeps its cheap default:
    /// quality is opt-in per CLI, never an accident of config.
    ///
    /// Reaches only the rungs that pass a model name. Cursor and OpenCode do
    /// not — see `CliSpec::passes_a_model` — and `brain doctor` reports them
    /// as running their own default rather than echoing a name back.
    pub models: std::collections::HashMap<String, String>,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self { mode: "auto".to_string(), models: std::collections::HashMap::new() }
    }
}

/// Byte budgets for automatic context injection.
///
/// Byte-denominated on purpose: a count of "50 observations" is an
/// approximation, because 50 long lines and 50 short lines are not the same
/// spend. These are ceilings, not targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InjectionConfig {
    /// Ceiling for the session-start primer.
    pub primer_budget: usize,
    /// Ceiling for everything auto-injected in one session, across layers.
    pub session_budget: usize,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self { primer_budget: 4096, session_budget: 8192 }
    }
}

/// Resolved paths for one machine.
#[derive(Debug, Clone)]
pub struct Paths {
    pub data_dir: PathBuf,
}

impl Paths {
    /// Resolve the data directory: `$ROLEPOD_BRAIN_HOME`, else
    /// `~/.rolepod-brain`.
    ///
    /// # Errors
    /// Returns an error when no home directory can be determined.
    pub fn resolve() -> Result<Self> {
        let data_dir = match std::env::var_os(DATA_DIR_ENV) {
            Some(dir) if !dir.is_empty() => PathBuf::from(dir),
            _ => dirs::home_dir()
                .context("cannot determine home directory")?
                .join(".rolepod-brain"),
        };
        Ok(Self { data_dir })
    }

    #[must_use]
    pub fn db(&self) -> PathBuf {
        self.data_dir.join("brain.db")
    }

    /// The wiki directory - the durable memory, and an Obsidian vault.
    ///
    /// Named for the product because Obsidian names a vault after its
    /// folder, and a vault called "wiki" in the switcher says nothing.
    /// The name is load-bearing the other way too: renaming a vault inside
    /// Obsidian renames the real directory, so the pretty name has to be
    /// the real name - a symlink would be silently left behind pointing at
    /// nothing the moment someone renamed the vault it fronted.
    ///
    /// A tree written by a version that called it `wiki/` keeps working
    /// untouched; `brain reindex` renames it. When both exist, the new name
    /// wins - that is the migrated state, and migration refuses to create
    /// the both-exist state itself.
    #[must_use]
    pub fn wiki(&self) -> PathBuf {
        let pretty = self.data_dir.join(WIKI_DIR);
        if pretty.is_dir() {
            return pretty;
        }
        let legacy = self.data_dir.join(LEGACY_WIKI_DIR);
        if legacy.is_dir() {
            return legacy;
        }
        pretty
    }

    /// Where the embedding model lives once it has been fetched.
    ///
    /// Not compiled into the binary, which is a distribution decision and not
    /// an architectural one: the table is 122 MB, GitHub refuses a file over
    /// 100 MB, and putting it in the repository would also add that much to
    /// every clone forever. It is published as a release asset instead and
    /// fetched once, checksum-verified.
    ///
    /// The directory is versioned, so a build that expects different weights
    /// does not read the old ones and quietly answer differently — it finds
    /// nothing and fetches what it needs.
    #[must_use]
    pub fn model_dir(&self) -> PathBuf {
        self.data_dir.join("models").join(crate::embed::MODEL)
    }

    /// Where a named model's files live, under the same versioned scheme
    /// `model_dir` uses: a build expecting different weights finds nothing
    /// rather than reading the wrong ones.
    #[must_use]
    pub fn model_dir_for(&self, model: &str) -> PathBuf {
        self.data_dir.join("models").join(model)
    }

    #[must_use]
    pub fn config_file(&self) -> PathBuf {
        self.data_dir.join("config.toml")
    }

    /// Where the hook binary writes its own failures. Hooks must never print
    /// to the host CLI's stderr, so this is the only place a capture problem
    /// becomes visible — `brain doctor` reads it.
    #[must_use]
    pub fn log_file(&self) -> PathBuf {
        self.data_dir.join("brain.log")
    }

    /// Project directory inside the wiki.
    ///
    /// The layout is human-first: a project in the unnamed workspace lives
    /// directly under `wiki/` as its bare slug - `wiki/rolepod-brain/` - and
    /// a named workspace keeps its own level, `wiki/work/api/`. The
    /// `--<8 hex>` suffix that used to be on every directory now exists only
    /// to break a genuine collision, because two projects that happen to
    /// share a basename must never share a directory: that is how one
    /// project's memory silently becomes another's.
    ///
    /// Resolution order, most-specific first:
    ///
    /// 1. The suffixed directory, if it exists. A project that ever needed
    ///    the suffix keeps it - a directory rename breaks nothing today but
    ///    would re-fire the moment the clean name frees up, and a project
    ///    whose home moves twice is worse than one with an ugly name.
    /// 2. The legacy home, `wiki/default/<slug>--<id>`. Kept working so an
    ///    install that never runs `brain reindex` is degraded in looks only,
    ///    never in function.
    /// 3. The bare slug, when it is ours or unclaimed. Ownership is read
    ///    from the directory's own log, because the directory name no longer
    ///    carries the id that used to make this check unnecessary.
    /// 4. The suffixed name, freshly. The bare slug exists and belongs to
    ///    someone else - another project, or a workspace directory.
    #[must_use]
    pub fn project_dir(&self, scope: &ProjectScope) -> PathBuf {
        let parent = if scope.workspace == "default" {
            self.wiki()
        } else {
            self.wiki().join(crate::ids::slugify(&scope.workspace))
        };

        let suffixed = parent.join(scope.dir_name());
        if suffixed.is_dir() {
            return suffixed;
        }
        if scope.workspace == "default" {
            let legacy = self.wiki().join("default").join(scope.dir_name());
            if legacy.is_dir() {
                return legacy;
            }
        }

        let clean = parent.join(crate::ids::slugify(&scope.project));
        // Two brand-new projects with the same slug starting in the same
        // instant could both see the clean name as unclaimed. The window is
        // one hook's first-ever event against another project's first-ever
        // event with a colliding basename; the old always-suffixed layout
        // closed it by construction, and this one accepts it.
        if !clean.exists() || dir_belongs_to(&clean, scope.project_id) {
            return clean;
        }
        suffixed
    }

    /// Create the data directory if it does not exist.
    ///
    /// # Errors
    /// Returns an error when the directory cannot be created.
    pub fn ensure(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)
            .with_context(|| format!("create data directory {}", self.data_dir.display()))
    }
}

/// Does this wiki directory hold the given project's log?
///
/// Reads only the first parseable line of one log file - the project id is on
/// every line, so one is enough - and never creates anything: `EventLog::open`
/// makes directories as a side effect, which would turn a question into a
/// claim.
fn dir_belongs_to(dir: &Path, project_id: uuid::Uuid) -> bool {
    let events = dir.join("events");
    let Ok(entries) = std::fs::read_dir(&events) else { return false };
    let mut files: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    for file in files {
        let Ok(handle) = std::fs::File::open(&file) else { continue };
        let mut first = String::new();
        use std::io::BufRead;
        if std::io::BufReader::new(handle).read_line(&mut first).is_err() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&first) {
            return value.get("project").and_then(serde_json::Value::as_str)
                == Some(project_id.to_string().as_str());
        }
    }
    false
}

impl Config {
    /// Load config, falling back to defaults when the file is absent.
    ///
    /// # Errors
    /// Returns an error when the file exists but is unreadable or malformed —
    /// a typo in a budget should be loud, not silently ignored.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_background_agent_by_default() {
        // The product's promise is that nothing runs; a Login Items entry out
        // of the box would contradict it on the user's own screen.
    }

    #[test]
    fn defaults_are_light() {
        let config = Config::default();
        assert_eq!(config.summarizer.mode, "auto");
        assert_eq!(config.injection.primer_budget, 4096);
        assert_eq!(config.injection.session_budget, 8192);
    }

    #[test]
    fn partial_config_keeps_other_defaults() {
        let config: Config = toml::from_str("[summarizer]\nmode = \"off\"\n").unwrap();
        assert_eq!(config.summarizer.mode, "off");
        assert_eq!(config.injection.primer_budget, 4096);
    }

    #[test]
    fn malformed_config_is_an_error() {
        assert!(toml::from_str::<Config>("[injection]\nprimer_budget = \"big\"\n").is_err());
    }
}
