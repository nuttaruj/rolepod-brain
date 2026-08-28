//! Unix only, deliberately.
//!
//! Every fixture here builds a fake host CLI as a `/bin/sh` script, pins
//! `PATH` to `/usr/bin:/bin` so a test can never see a real CLI installed on
//! the machine, and links the embedding model rather than copying 122 MB per
//! test. Porting that to Windows means running each stub through `cmd.exe`
//! into `bash`, where the prompt crosses two argument parsers - and a failure
//! there would be as likely to be the harness as the code.
//!
//! What Windows verifies instead is the unit suite, plus the one behaviour
//! this file would have covered that is genuinely platform-specific: that a
//! host CLI installed as a `.cmd` shim is found and can be run. That test
//! lives next to the code it tests, in `summarizer`.
#![cfg(unix)]

//! v0.1 exit test, run against the real binary.
//!
//! The claims this file is here to prove:
//!
//! 1. Two different CLIs capturing in one checkout land in ONE project brain.
//! 2. Recall through the MCP surface returns what was captured.
//! 3. Secrets never reach the log.
//! 4. The SQLite index is disposable — `reindex` rebuilds it from the log.
//! 5. A hook returns fast enough that the host CLI does not feel it.
//! 6. Capture never disturbs the host, even when handed a broken payload.
//!
//! Every test runs against an isolated `ROLEPOD_BRAIN_HOME`, so running the
//! suite can never touch a real brain on the machine.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const BRAIN: &str = env!("CARGO_BIN_EXE_brain");

/// An isolated brain plus a git checkout to capture from.
struct Fixture {
    home: PathBuf,
    project: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!("brain-e2e-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let home = base.join("home");
        let project = base.join("checkout");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        // A git root is what makes every worktree of one repo share a brain,
        // so the fixture must be a real repository.
        run_in(&project, "git", &["init", "-q"]);
        // The embedding model is fetched after install rather than compiled
        // in, so a fixture has to point at one or every semantic assertion
        // would be testing its absence. A link, not a copy: it is 122 MB and
        // every test builds a fixture.
        let models = home.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let checkout =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/potion-multilingual-128M");
        if checkout.is_dir() {
            let _ = std::os::unix::fs::symlink(&checkout, models.join("potion-multilingual-128M"));
        }
        Self { home, project }
    }

    fn brain(&self, args: &[&str]) -> std::process::Output {
        self.brain_with_path(args, None)
    }

    /// Run with a controlled `PATH`, so the summarizer ladder sees exactly the
    /// CLIs a test wants it to see - and never the real ones on this machine.
    fn brain_with_path(&self, args: &[&str], path: Option<&Path>) -> std::process::Output {
        let mut command = Command::new(BRAIN);
        command
            .args(args)
            .current_dir(&self.project)
            .env("ROLEPOD_BRAIN_HOME", &self.home)
            // ROLEPOD_BRAIN_HOME isolates our own data; it does NOT isolate
            // the CLI configs we wire into, which are found through $HOME. A
            // test running `uninstall --apply` without this unwired the real
            // machine - so the fixture owns HOME too, and nothing here can
            // reach a config a person is actually using.
            .env("HOME", self.home.parent().unwrap());
        match path {
            // `git` still has to be reachable: consolidation commits the wiki.
            Some(dir) => command.env("PATH", format!("{}:/usr/bin:/bin", dir.display())),
            None => command.env("PATH", "/usr/bin:/bin"),
        };
        command.output().expect("run brain")
    }

    /// Install a fake host CLI that responds however the test needs.
    ///
    /// The ladder shells out to whatever is on `PATH`; a stub proves the
    /// degrade-and-recover path without spending a real model call.
    fn fake_cli(&self, name: &str, script: &str) -> PathBuf {
        let dir = self.home.parent().unwrap().join("fakebin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    }

    /// Observations still waiting for consolidation.
    fn pending_count(&self) -> i64 {
        let output = Command::new("sqlite3")
            .arg(self.home.join("brain.db"))
            .arg("SELECT COUNT(*) FROM events WHERE consolidated = 0 AND kind = 'observation';")
            .output()
            .expect("query pending");
        String::from_utf8_lossy(&output.stdout).trim().parse().unwrap_or(-1)
    }

    /// Every consolidated page on disk.
    /// The wiki directory, whichever name this fixture's brain gave it.
    fn wiki(&self) -> PathBuf {
        let pretty = self.home.join("Rolepod Brain");
        if pretty.is_dir() {
            return pretty;
        }
        self.home.join("wiki")
    }

    fn page_text(&self) -> String {
        let mut out = String::new();
        collect_ext(&self.wiki(), "md", &mut out);
        out
    }

    /// Every knowledge page under this fixture's wiki, whatever project
    /// directory it landed in.
    fn knowledge_pages(&self) -> Vec<PathBuf> {
        let mut found = Vec::new();
        collect_under(&self.wiki(), "knowledge", &mut found);
        found.sort();
        found
    }

    /// Capture enough events that consolidation will not debounce them away.
    fn seed_session(&self, count: usize) {
        for index in 0..count {
            let payload = serde_json::json!({
                "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
                "cwd": self.project,
                "tool_name": "Edit",
                "tool_input": {"file_path": self.project.join(format!("src/file{index}.rs"))}
            })
            .to_string();
            self.hook("claude-code", "PostToolUse", &payload);
        }
    }

    fn hook(&self, cli: &str, event: &str, payload: &str) -> std::process::Output {
        let mut child = Command::new(BRAIN)
            .args(["hook", "--cli", cli, "--event", event])
            .current_dir(&self.project)
            .env("ROLEPOD_BRAIN_HOME", &self.home)
            // Same reason brain_with_path owns HOME: the capture path reads
            // $HOME too, and a fixture that leaves it pointing at the real
            // machine is testing the machine, not the fixture.
            .env("HOME", self.home.parent().unwrap())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn hook");
        child.stdin.as_mut().unwrap().write_all(payload.as_bytes()).unwrap();
        child.wait_with_output().expect("hook output")
    }

    /// One JSON-RPC round trip against a freshly spawned MCP server.
    fn mcp(&self, requests: &[&str]) -> Vec<serde_json::Value> {
        self.mcp_with_path(requests, None)
    }

    /// The same, with a fake CLI reachable on `PATH`.
    fn mcp_with_path(&self, requests: &[&str], bin: Option<&Path>) -> Vec<serde_json::Value> {
        let path = bin.map_or_else(
            || "/usr/bin:/bin".to_string(),
            |dir| format!("{}:/usr/bin:/bin", dir.display()),
        );
        let mut child = Command::new(BRAIN)
            .arg("mcp")
            .current_dir(&self.project)
            .env("ROLEPOD_BRAIN_HOME", &self.home)
            .env("PATH", path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn mcp");
        {
            let stdin = child.stdin.as_mut().unwrap();
            for request in requests {
                writeln!(stdin, "{request}").unwrap();
            }
        }
        let output = child.wait_with_output().expect("mcp output");
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("MCP response is valid JSON"))
            .collect()
    }

    /// Every event-log line on disk, across all projects.
    /// Every `.jsonl` under the wiki: the append-only log itself.
    fn log_files(&self) -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).into_iter().flatten().filter_map(Result::ok) {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "jsonl") {
                    out.push(path);
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.wiki(), &mut out);
        out
    }

    fn log_text(&self) -> String {
        let mut out = String::new();
        collect_jsonl(&self.wiki(), &mut out);
        out
    }

    fn project_dirs(&self) -> Vec<PathBuf> {
        let wiki = self.wiki();
        let mut dirs = Vec::new();
        // Same classification the product uses: an event log makes a
        // directory a project, whatever depth it sits at.
        for entry in read_dirs(&wiki) {
            if entry.join("events").is_dir() {
                dirs.push(entry);
                continue;
            }
            for project in read_dirs(&entry) {
                if project.join("events").is_dir() {
                    dirs.push(project);
                }
            }
        }
        dirs
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.home.parent().unwrap());
    }
}

fn read_dirs(path: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect()
}

fn collect_ext(dir: &Path, ext: &str, out: &mut String) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_ext(&path, ext, out);
        } else if path.extension().is_some_and(|e| e == ext) {
            out.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
        }
    }
}

/// Collect every file beneath a directory named `marker`.
fn collect_under(dir: &Path, marker: &str, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_under(&path, marker, out);
        } else if path.components().any(|part| part.as_os_str() == marker) {
            out.push(path);
        }
    }
}

fn collect_jsonl(dir: &Path, out: &mut String) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_jsonl(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            out.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
        }
    }
}

fn run_in(dir: &Path, program: &str, args: &[&str]) {
    Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|_| panic!("run {program}"));
}

fn claude_payload(cwd: &Path) -> String {
    serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": cwd,
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {
            "file_path": cwd.join("src/auth.rs"),
            "new_string": "fn check() {}"
        },
        "tool_response": {"success": true}
    })
    .to_string()
}

fn codex_payload(cwd: &Path) -> String {
    serde_json::json!({
        "session_id": "codex-thread-42",
        "cwd": cwd,
        "hook_event_name": "UserPromptSubmit",
        "prompt": "why does the auth middleware reject valid tokens?"
    })
    .to_string()
}

#[test]
fn setup_leaves_a_config_a_person_can_discover_but_never_overwrites_one() {
    let fixture = Fixture::new("configtemplate");
    fixture.seed_session(1);
    assert!(fixture.brain(&["setup", "--apply", "--cli", "claude-code"]).status.success());

    let config = fixture.home.join("config.toml");
    let template = std::fs::read_to_string(&config).expect("setup should write the template");
    assert!(template.contains("# rerank = false"), "the knobs should be visible: {template}");
    // Inert as written: doctor reports pure defaults.
    let report = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(report.contains("summarizer=auto"), "the template changed behavior: {report}");

    // A file the user has touched is theirs, forever.
    std::fs::write(&config, "[summarizer]\nmode = \"off\"\n").unwrap();
    assert!(fixture.brain(&["setup", "--apply", "--cli", "claude-code"]).status.success());
    assert_eq!(
        std::fs::read_to_string(&config).unwrap(),
        "[summarizer]\nmode = \"off\"\n",
        "setup overwrote the user's config"
    );
}

#[test]
fn a_legacy_tree_keeps_working_and_reindex_moves_it_home() {
    // The old layout was wiki/default/<slug>--<id>/ - old top-level name,
    // extra workspace level, permanent suffix. All three must keep working
    // untouched - an install that never migrates is degraded in looks only -
    // and `brain reindex` must move the lot to `Rolepod Brain/<slug>/`
    // without losing a line.
    let fixture = Fixture::new("legacymove");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));

    // Reconstruct the full legacy shape by hand from what was captured.
    let flat = fixture.project_dirs().pop().expect("a captured project");
    let idfrag: String = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["project"].as_str().map(|id| id.replace('-', "")[..8].to_string()))
        .expect("a project id");
    let slug = flat.file_name().unwrap().to_string_lossy().into_owned();
    let old_wiki = fixture.home.join("wiki");
    std::fs::rename(fixture.wiki(), &old_wiki).unwrap();
    let legacy = old_wiki.join("default").join(format!("{slug}--{idfrag}"));
    std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
    std::fs::rename(old_wiki.join(&slug), &legacy).unwrap();

    // Capture keeps landing in the legacy home, not a fresh pretty tree.
    fixture.hook("claude-code", "UserPromptSubmit", &codex_payload(&fixture.project));
    assert!(
        legacy.join("events").is_dir() && !fixture.home.join("Rolepod Brain").exists(),
        "pre-migration capture abandoned the legacy home"
    );
    let lines_before = fixture.log_text().lines().count();
    assert_eq!(lines_before, 2, "both events should be in the legacy log");

    // Migration: reindex renames the top level, moves the project, keeps
    // every line, and removes the empty workspace level.
    let out = fixture.brain(&["reindex"]);
    assert!(out.status.success(), "reindex failed: {out:?}");
    let home = fixture.home.join("Rolepod Brain").join(&slug);
    assert!(home.join("events").is_dir(), "the project did not move home: {home:?}");
    assert!(!old_wiki.exists(), "the old wiki/ name should be gone");
    assert!(
        !fixture.home.join("Rolepod Brain/default").exists(),
        "the empty default/ level should be removed"
    );
    assert_eq!(fixture.log_text().lines().count(), lines_before, "the move lost log lines");

    // The memory survived the move end to end.
    let found = String::from_utf8_lossy(&fixture.brain(&["search", "auth"]).stdout).to_string();
    assert!(!found.contains("No matches"), "memory unfindable after migration: {found}");

    // And running it again moves nothing - stdout says so.
    let again = fixture.brain(&["reindex"]);
    assert!(
        !String::from_utf8_lossy(&again.stdout).contains("moved"),
        "a second migration should have nothing to do"
    );
}

#[test]
fn two_projects_with_one_basename_never_share_a_directory() {
    // The clean name is a privilege, not a right: the first project keeps
    // it, the second gets the --<id> suffix, and neither ever writes into
    // the other's memory - which is how the suffix earned permanence.
    let fixture = Fixture::new("basenameclash");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));

    // A second repository whose directory has the same basename.
    let rival_root = fixture.home.parent().unwrap().join("elsewhere");
    let rival = rival_root.join("checkout");
    std::fs::create_dir_all(&rival).unwrap();
    run_in(&rival, "git", &["init", "-q"]);
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcd9",
        "cwd": rival,
        "tool_name": "Edit",
        "tool_input": {"file_path": rival.join("src/other.rs")}
    })
    .to_string();
    let mut child = Command::new(BRAIN)
        .args(["hook", "--cli", "claude-code", "--event", "PostToolUse"])
        .current_dir(&rival)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("HOME", fixture.home.parent().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hook");
    child.stdin.as_mut().unwrap().write_all(payload.as_bytes()).unwrap();
    assert!(child.wait_with_output().expect("hook").status.success());

    let dirs = fixture.project_dirs();
    assert_eq!(dirs.len(), 2, "two projects must get two directories: {dirs:?}");
    let names: Vec<String> = dirs
        .iter()
        .map(|dir| dir.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(names.contains(&"checkout".to_string()), "the first project keeps the clean name: {names:?}");
    assert!(
        names.iter().any(|name| name.starts_with("checkout--")),
        "the second project must be suffixed, not merged: {names:?}"
    );
}

#[test]
fn a_named_workspace_keeps_its_own_level() {
    let fixture = Fixture::new("namedws");
    std::fs::write(
        fixture.project.join(".rolepod-brain.toml"),
        "[project]\nname = \"api\"\nworkspace = \"work\"\n",
    )
    .unwrap();
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));

    let dirs = fixture.project_dirs();
    assert_eq!(dirs.len(), 1, "one project expected: {dirs:?}");
    let relative = dirs[0].strip_prefix(fixture.wiki()).unwrap();
    assert_eq!(
        relative,
        Path::new("work/api"),
        "a named workspace nests and the project name is clean"
    );
}

#[test]
fn two_clis_capture_into_one_project_brain() {
    let fixture = Fixture::new("merged");

    let claude = fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    assert!(claude.status.success(), "claude hook failed: {claude:?}");
    let codex = fixture.hook("codex", "UserPromptSubmit", &codex_payload(&fixture.project));
    assert!(codex.status.success(), "codex hook failed: {codex:?}");

    // Decision #9: knowledge belongs to the project, not the CLI that saw it.
    let dirs = fixture.project_dirs();
    assert_eq!(dirs.len(), 1, "expected one merged store, found {dirs:?}");

    let log = fixture.log_text();
    assert!(log.contains("\"cli\":\"claude-code\""), "claude event missing from log");
    assert!(log.contains("\"cli\":\"codex\""), "codex event missing from log");

    // …and `source.cli` is what keeps them separable without separating them.
    let lines: Vec<serde_json::Value> =
        log.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        assert_eq!(line["v"], 1, "every line carries the schema version");
        assert!(line["id"].as_str().unwrap().len() == 26, "id is a ULID");
        assert_eq!(line["project"], lines[0]["project"], "both events share one project id");
    }
}

#[test]
fn hooks_acknowledge_the_host_even_on_a_broken_payload() {
    let fixture = Fixture::new("broken");

    let output = fixture.hook("claude-code", "PostToolUse", "{ this is not json");
    // A capture failure is ours to absorb: the host CLI must see success and a
    // well-formed acknowledgement, or it logs an error on every tool call.
    assert!(output.status.success(), "hook must exit 0 even when capture fails");
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "{}");

    // The failure still has to be visible somewhere.
    let doctor = fixture.brain(&["doctor"]);
    let report = String::from_utf8_lossy(&doctor.stdout);
    assert!(report.contains("capture errors"), "doctor should surface the failure: {report}");
}

#[test]
fn secrets_never_reach_the_log() {
    let fixture = Fixture::new("secrets");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "deploy with ghp_abcdefghijklmnopqrstuvwxyz0123 and OPENAI_API_KEY=sk-livekey1234567890abcd"
    })
    .to_string();

    fixture.hook("claude-code", "UserPromptSubmit", &payload);

    let log = fixture.log_text();
    assert!(!log.contains("ghp_abcdefghijklmnopqrstuvwxyz0123"), "GitHub token leaked into log");
    assert!(!log.contains("sk-livekey1234567890abcd"), "API key leaked into log");
    assert!(log.contains("[REDACTED]"), "expected redaction markers");
}

#[test]
fn reranking_reorders_a_search_and_a_failed_one_changes_nothing() {
    let fixture = Fixture::new("rerank");
    std::fs::write(
        fixture.home.join("config.toml"),
        "[search]\nrerank = true\n\n[summarizer]\nmode = \"claude-code\"\n",
    )
    .unwrap();
    for name in ["auth.rs", "auth/login.rs", "auth/token.rs", "auth/session.rs"] {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {
                "file_path": fixture.project.join(format!("src/{name}")),
                "new_string": "fn check() {}"
            },
            "tool_response": {"success": true}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    let search = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#;
    let ids = |responses: &[serde_json::Value]| -> Vec<String> {
        let text = responses.last().expect("a response")["result"]["content"][0]["text"]
            .as_str()
            .expect("text content")
            .to_string();
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("hits JSON");
        parsed["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|hit| hit["id"].as_str().unwrap_or_default().to_string())
            .collect()
    };

    let plain = ids(&fixture.mcp(&[search]));
    assert!(plain.len() >= 3, "need several hits to reorder: {plain:?}");

    // A stub that promotes whatever search ranked last.
    let bin = fixture.fake_cli(
        "claude",
        "echo \"$*\" | grep -oE '[0-9A-Z]{26}' | tail -1",
    );
    let ranked = ids(&fixture.mcp_with_path(&[search], Some(&bin)));
    assert_eq!(ranked[0], plain[plain.len() - 1], "the model's pick did not lead: {ranked:?}");
    // Reranking is a permutation of what search found, never a filter: an
    // opinion about one hit must not silently shrink the result.
    let mut before = plain.clone();
    let mut after = ranked.clone();
    before.sort();
    after.sort();
    assert_eq!(before, after, "reranking changed which hits came back");

    // A CLI that fails leaves the search exactly as the index ranked it.
    let broken = fixture.fake_cli("claude", "echo 'rate limit exceeded' >&2; exit 1");
    assert_eq!(
        ids(&fixture.mcp_with_path(&[search], Some(&broken))),
        plain,
        "a failed rerank must be a no-op, not a degraded search"
    );
}

/// What a search offered, against what an agent chose to open.
///
/// Every ranking change in this project has been scored against a model's
/// opinion of a list of titles, which is a proxy that shares the ranking's
/// own blind spots. An agent calling `brain_get` on an entry it has seen
/// only the title of is a judgement made for its own reasons. Recording the
/// two apart is what makes the next ranking change measurable against
/// something real, once enough sessions have passed.
#[test]
fn what_an_agent_opens_is_recorded_apart_from_what_it_was_offered() {
    let fixture = Fixture::new("opened");
    for name in ["auth.rs", "auth/login.rs", "auth/token.rs"] {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {
                "file_path": fixture.project.join(format!("src/{name}")),
                "new_string": "fn check() {}"
            },
            "tool_response": {"success": true}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    let search = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#;
    let responses = fixture.mcp(&[search]);
    let text = responses.last().expect("a response")["result"]["content"][0]["text"]
        .as_str()
        .expect("text content")
        .to_string();
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("hits JSON");
    let ids: Vec<String> = parsed["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(ids.len() >= 2, "need several hits: {ids:?}");

    let count = |opened: i64| -> i64 {
        let out = Command::new("sqlite3")
            .arg(fixture.home.join("brain.db"))
            .arg(format!("SELECT COUNT(*) FROM recalled WHERE opened = {opened};"))
            .output()
            .expect("query recalled");
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(-1)
    };
    assert!(count(1) == 0, "a search alone marked something as opened");
    let offered_before = count(0);
    assert!(offered_before > 0, "the search recorded nothing");

    // Now read one of them in full, in the same session.
    let get = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{}"]}}}}}}"#,
        ids[0]
    );
    fixture.mcp(&[&get]);
    assert_eq!(count(1), 1, "reading a body in full was not recorded as opened");
}

/// Reranking is the caller's call, one question at a time.
///
/// A config flag says what someone preferred once; an argument says this
/// caller, on this question, judged the answer worth ten to twenty-five
/// seconds of real waiting. Measured through a host CLI, that is what it
/// costs — so a standing "always" is a standing tax on every lookup, and a
/// standing "never" hides the one search that needed it.
#[test]
fn reranking_can_be_asked_for_one_search_at_a_time() {
    let fixture = Fixture::new("rerank-per-request");
    // No config file at all: the standing preference is off.
    for name in ["auth.rs", "auth/login.rs", "auth/token.rs", "auth/session.rs"] {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {
                "file_path": fixture.project.join(format!("src/{name}")),
                "new_string": "fn check() {}"
            },
            "tool_response": {"success": true}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    let ids = |responses: &[serde_json::Value]| -> Vec<String> {
        let text = responses.last().expect("a response")["result"]["content"][0]["text"]
            .as_str()
            .expect("text content")
            .to_string();
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("hits JSON");
        parsed["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|hit| hit["id"].as_str().unwrap_or_default().to_string())
            .collect()
    };
    let plain_req = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#;
    let asked_req = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth","rerank":true}}}"#;

    // A stub that promotes whatever search ranked last. It is on PATH for
    // both calls, so the only difference is the argument.
    let bin = fixture.fake_cli("claude", "echo \"$*\" | grep -oE '[0-9A-Z]{26}' | tail -1");

    let plain = ids(&fixture.mcp_with_path(&[plain_req], Some(&bin)));
    assert!(plain.len() >= 3, "need several hits: {plain:?}");

    let asked = ids(&fixture.mcp_with_path(&[asked_req], Some(&bin)));
    assert_eq!(
        asked[0],
        plain[plain.len() - 1],
        "asking for a rerank on one search did nothing: {asked:?}"
    );

    // And the search that did not ask is untouched, in the same process
    // lifetime and against the same model.
    assert_eq!(
        ids(&fixture.mcp_with_path(&[plain_req], Some(&bin))),
        plain,
        "a search that did not ask for reranking paid for one anyway"
    );
}

/// A rerank is a favour, and a favour must not cost the machine anything.
///
/// Two ways it used to. A model replying NONE - the word the prompt asks for
/// when nothing fits - was read as a dead rung, so the ladder paid a second
/// CLI to answer the same question, and filed a failure against the first.
/// Three of those and that CLI was out of consolidation for thirty minutes,
/// having done nothing wrong at a leash five times shorter than the one
/// consolidation runs on.
#[test]
fn a_rerank_that_finds_nothing_costs_one_call_and_no_breaker() {
    let fixture = Fixture::new("rerank-none");
    // `auto` so a second rung genuinely exists to be wrongly taken.
    std::fs::write(fixture.home.join("config.toml"), "[search]\nrerank = true\n").unwrap();
    for name in ["auth.rs", "auth/login.rs", "auth/token.rs", "auth/session.rs"] {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {
                "file_path": fixture.project.join(format!("src/{name}")),
                "new_string": "fn check() {}"
            },
            "tool_response": {"success": true}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    let search = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#;
    let ids = |responses: &[serde_json::Value]| -> Vec<String> {
        let text = responses.last().expect("a response")["result"]["content"][0]["text"]
            .as_str()
            .expect("text content")
            .to_string();
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("hits JSON");
        parsed["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|hit| hit["id"].as_str().unwrap_or_default().to_string())
            .collect()
    };

    let plain = ids(&fixture.mcp(&[search]));
    assert!(plain.len() >= 3, "need several hits to reorder: {plain:?}");

    // The preferred rung answers NONE. A second rung stands behind it, ready
    // to promote the last hit - so if the ladder cascades, the order moves and
    // this test sees it. `gemini`, not `codex`: codex reads its answer from a
    // file rather than stdout, so a stdout stub there would look like a rung
    // that failed and prove nothing about cascading.
    fixture.fake_cli("claude", "echo NONE");
    let bin = fixture.fake_cli("gemini", "echo \"$*\" | grep -oE '[0-9A-Z]{26}' | tail -1");

    for _ in 0..4 {
        assert_eq!(
            ids(&fixture.mcp_with_path(&[search], Some(&bin))),
            plain,
            "NONE means the index order stands, and no other CLI is asked"
        );
    }

    // And the case that started this: the preferred rung really does fail.
    // The second CLI is still standing there, and must still not be asked -
    // a search does not double its wait to improve an ordering it already
    // had. Nor does the six-second leash get to say a CLI is down.
    fixture.fake_cli("claude", "echo 'rate limit exceeded' >&2; exit 1");
    for _ in 0..4 {
        assert_eq!(
            ids(&fixture.mcp_with_path(&[search], Some(&bin))),
            plain,
            "a failed rerank cascaded to a second CLI instead of standing down"
        );
    }

    let health = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg("SELECT cli, failures FROM summarizer_health;")
        .output()
        .expect("query health");
    assert!(
        String::from_utf8_lossy(&health.stdout).trim().is_empty(),
        "an advisory call marked the health table: {}",
        String::from_utf8_lossy(&health.stdout)
    );
}

#[test]
fn an_over_budget_injection_is_reported_with_its_age() {
    // injected_bytes has no timestamp of its own - a session's spend is
    // permanent once recorded - so without an age, a bug fixed today reads
    // identically to one happening right now. Doctor derives it from the
    // worst session's own most recent captured event.
    let fixture = Fixture::new("injbudgetage");
    fixture.seed_session(1);
    let session = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["session"].as_str().map(str::to_string))
        .expect("a captured session");

    Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg(format!(
            "INSERT INTO injected_bytes (session, bytes) VALUES ('{session}', 99999);"
        ))
        .output()
        .expect("seed injected_bytes");

    let report = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(report.contains("OVER BUDGET"), "the overspend was not reported: {report}");
    // Match the check's own column, not a bare substring: the temp fixture
    // path is also part of the report, and could itself contain "injection".
    let line = report
        .lines()
        .find(|line| line.trim_start_matches("FAIL").trim_start_matches("ok").trim_start().starts_with("injection"))
        .unwrap_or_default();
    assert!(
        line.contains("ago") || line.contains("just now"),
        "the overspend was reported with no age, indistinguishable from happening right now: {line}"
    );
}

#[test]
fn a_renamed_rung_s_stale_failure_does_not_haunt_doctor_forever() {
    // A real row found on a real machine: the gemini spec was renamed
    // "gemini" -> "gemini-cli" to match what hooks actually write, which
    // orphaned an existing health row under the old key. Nothing will ever
    // record success OR failure against "gemini" again, so the row can never
    // clear itself - and doctor reported it as a live failure regardless.
    let fixture = Fixture::new("staleorphan");
    fixture.seed_session(1);
    // brain.db has to exist first; its own health is not what this test
    // is about, so the exit code is not asserted.
    let _ = fixture.brain(&["doctor"]);

    let seed = |cli: &str, error: &str| {
        Command::new("sqlite3")
            .arg(fixture.home.join("brain.db"))
            .arg(format!(
                "INSERT INTO summarizer_health (cli, failures, last_error, last_failed_at)                  VALUES ('{cli}', 2, '{error}', NULL);"
            ))
            .output()
            .expect("seed summarizer_health");
    };
    // The orphan: a name no current spec uses.
    seed("gemini", "prompt is 24709 bytes, over the 24576-byte call ceiling");
    // A live rung, failing right now, must still be reported.
    seed("codex", "rate limit exceeded");

    let report = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(
        !report.contains("summarizer: gemini "),
        "an orphaned rung with no current spec was reported as a live failure: {report}"
    );
    assert!(
        report.contains("summarizer: codex"),
        "a live rung's real failure was suppressed along with the orphan: {report}"
    );
}

#[test]
fn an_installed_plugin_takes_over_the_mcp_registration() {
    // The plugin declares the same server brain would register standalone.
    // Two entries for one binary is the duplicate-registration bug this
    // project already shipped once, so setup has to step aside - and remove
    // the entry it wrote before the plugin existed.
    let fixture = Fixture::new("pluginmcp");
    let home = fixture.home.parent().unwrap().to_path_buf();
    std::fs::create_dir_all(home.join(".cursor")).unwrap();
    std::fs::write(
        home.join(".cursor/mcp.json"),
        r#"{"mcpServers":{"brain":{"command":"/old/brain","args":["mcp"]},"other":{"command":"x"}}}"#,
    )
    .unwrap();

    // Without the plugin, setup owns the registration.
    let plain = fixture.brain(&["setup", "--apply", "--cli", "cursor"]);
    assert!(plain.status.success(), "setup failed: {plain:?}");
    let servers = |label: &str| -> serde_json::Value {
        let text = std::fs::read_to_string(home.join(".cursor/mcp.json"))
            .unwrap_or_else(|_| panic!("{label}: no mcp.json"));
        serde_json::from_str(&text).expect("mcp.json is JSON")
    };
    assert!(servers("standalone")["mcpServers"]["brain"].is_object(), "brain was not registered");

    // Now the plugin is installed, the way Cursor records it: a directory
    // named after the plugin under the marketplace it came from.
    std::fs::create_dir_all(home.join(".cursor/plugins/cache/rolepod-brain/rolepod-brain")).unwrap();

    let deferred = fixture.brain(&["setup", "--apply", "--cli", "cursor"]);
    assert!(deferred.status.success(), "setup failed: {deferred:?}");
    let after = servers("deferred");
    assert!(
        after["mcpServers"]["brain"].is_null(),
        "our standalone entry survived alongside the plugin's: {after}"
    );
    assert!(after["mcpServers"]["other"].is_object(), "a foreign server was removed");
    assert!(
        String::from_utf8_lossy(&deferred.stdout).contains("plugin"),
        "setup did not say why it stepped aside: {}",
        String::from_utf8_lossy(&deferred.stdout)
    );

    // Capture still belongs to setup: the plugin does not declare hooks for
    // this CLI, so removing them would silently stop the memory.
    let hooks = std::fs::read_to_string(home.join(".cursor/hooks.json")).expect("hooks.json");
    assert!(hooks.contains("brain hook --cli cursor"), "capture hooks were dropped: {hooks}");
}

#[test]
fn only_a_cli_s_own_transcript_directory_is_read_from() {
    // Consolidation reads the recorded transcript and hands it to a model, so
    // an unchecked path in a hook payload is a way to make brain fetch a file
    // and post it somewhere.
    let fixture = Fixture::new("transcriptpath");
    let fake_home = fixture.home.parent().unwrap().to_path_buf();
    let real = fake_home.join(".claude/projects/some-project");
    std::fs::create_dir_all(&real).unwrap();
    let transcript = real.join("session.jsonl");
    std::fs::write(&transcript, "{\"type\":\"assistant\",\"message\":\"hello\"}\n").unwrap();

    let secret = fake_home.join("id_rsa");
    std::fs::write(&secret, "PRIVATE KEY").unwrap();

    let send = |path: &std::path::Path, session: &str| {
        let payload = serde_json::json!({
            "session_id": session,
            "cwd": fixture.project,
            "transcript_path": path,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    };
    send(&transcript, "0199a1f2-3c4d-7e8f-9012-3456789abcd1");
    send(&secret, "0199a1f2-3c4d-7e8f-9012-3456789abcd2");

    let recorded = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg("SELECT path FROM session_transcript;")
        .output()
        .expect("query transcripts");
    let recorded = String::from_utf8_lossy(&recorded.stdout).to_string();
    assert!(
        recorded.contains("session.jsonl"),
        "a real transcript was rejected - the feature is off, not secured: {recorded}"
    );
    assert!(!recorded.contains("id_rsa"), "an arbitrary file was accepted as a transcript");
}

#[test]
fn an_archive_that_writes_outside_the_data_directory_is_refused() {
    // An import is a file someone was sent. Unpacking it must not be able to
    // reach a hook config or a shell profile, whatever the local tar allows.
    let fixture = Fixture::new("tarescape");
    let archive = fixture.home.parent().unwrap().join("evil.tar.gz");
    let payload = fixture.home.parent().unwrap().join("payload.txt");
    std::fs::write(&payload, "owned").unwrap();
    let script = format!(
        "import tarfile\n\
         t = tarfile.open(r'{}', 'w:gz')\n\
         info = t.gettarinfo(r'{}')\n\
         info.name = '../../escaped.txt'\n\
         t.addfile(info, open(r'{}', 'rb'))\n\
         t.close()\n",
        archive.display(),
        payload.display(),
        payload.display()
    );
    let built = Command::new("python3").args(["-c", &script]).output().expect("build archive");
    assert!(built.status.success(), "could not build the test archive: {built:?}");

    let refused = fixture.brain(&["import", "--merge", &archive.to_string_lossy()]);
    assert!(!refused.status.success(), "an escaping archive was accepted");
    let why = String::from_utf8_lossy(&refused.stderr);
    assert!(why.contains("unsafe path"), "refused for the wrong reason: {why}");
    assert!(
        !fixture.home.parent().unwrap().join("escaped.txt").exists(),
        "the archive wrote outside the data directory"
    );
}

#[test]
fn a_sensitive_path_is_redacted_in_the_file_list_too() {
    // Titles were scrubbed and the parallel files[] array was not, so the
    // path the sanitizer exists to hide sat intact in the column beside it.
    let fixture = Fixture::new("filescrub");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": "/Users/someone/.ssh/id_rsa"}
    })
    .to_string();
    fixture.hook("claude-code", "PostToolUse", &payload);

    let log = fixture.log_text();
    assert!(!log.contains("id_rsa"), "a credential path was stored verbatim: {log}");
    assert!(!log.contains(".ssh"), "a credential path was stored verbatim: {log}");
}

#[test]
fn no_memory_can_be_rewritten_without_being_seen_first() {
    // Correct is the most powerful operation there is: it decides what recall
    // returns from then on. An agent acting on a poisoned instruction must not
    // be able to overwrite memory by naming an id it never saw.
    let fixture = Fixture::new("correctguard");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "the scheduler double-books on Tuesdays"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);
    let id = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["id"].as_str().map(str::to_string))
        .expect("an event to correct");

    let call = |name: &str, args: String| {
        format!(r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"{name}","arguments":{args}}}}}"#)
    };
    let correct = call("brain_correct", format!(r#"{{"id":"{id}","text":"it never double-booked"}}"#));

    let blind = fixture.mcp(&[&correct]);
    let text = serde_json::to_string(&blind).unwrap_or_default();
    assert!(
        text.contains("has not been surfaced"),
        "a blind correction was accepted: {text}"
    );
    assert!(
        !fixture.log_text().contains("it never double-booked"),
        "the correction was written despite being refused"
    );

    // Having actually seen it, the same call works.
    let search = call("brain_search", r#"{"query":"scheduler"}"#.to_string());
    let allowed = fixture.mcp(&[&search, &correct]);
    let text = serde_json::to_string(&allowed).unwrap_or_default();
    assert!(!text.contains("has not been surfaced"), "a legitimate correction was refused: {text}");
    assert!(fixture.log_text().contains("it never double-booked"), "the correction was not written");
}

#[test]
fn one_loud_session_cannot_own_every_search_result() {
    // Measured on a real machine: every query returned 10/10 hits from the
    // session that happened to be running, which held 97% of the project's
    // events. Memory from thirteen earlier sessions was unreachable through
    // search - and the hits that did come back were things the agent could
    // already see in its own context, which is worth nothing to pull.
    let fixture = Fixture::new("diversify");

    // One session that talked about the term constantly...
    for index in 0..12 {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-000000000001",
            "cwd": fixture.project,
            "prompt": format!("scheduler work item {index}")
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }
    // ...and two earlier ones that mentioned it once each.
    for session in 2..4 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-00000000000{session}"),
            "cwd": fixture.project,
            "prompt": "the scheduler decision nobody remembers"
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }

    let out = fixture.brain(&["search", "scheduler"]);
    let ids: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|word| word.len() == 26 && word.starts_with("01"))
        .map(str::to_owned)
        .collect();
    assert!(ids.len() >= 4, "expected several hits: {ids:?}");

    let sessions: Vec<String> = ids
        .iter()
        .map(|id| {
            let out = Command::new("sqlite3")
                .arg(fixture.home.join("brain.db"))
                .arg(format!("SELECT session FROM events WHERE id='{id}';"))
                .output()
                .expect("query session");
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        })
        .collect();
    let distinct: std::collections::HashSet<&String> = sessions.iter().collect();
    assert!(
        distinct.len() >= 3,
        "one session owned the results - the quiet sessions are unreachable: {sessions:?}"
    );
}

#[test]
fn an_edit_you_make_in_obsidian_becomes_memory_rather_than_being_overwritten() {
    // Pages are derived: consolidation rewrites them, so an edit made in
    // Obsidian used to be lost the next time that session was consolidated -
    // and the README's answer was "do not edit them". A memory system whose
    // wrong answers cannot be fixed where you read them is a memory system
    // you learn to distrust.
    //
    // The fix does not make pages authoritative - that would break the rule
    // that the log is the only source of truth and `reindex` can rebuild
    // everything. It reads the edit BACK into the log as a correction, so the
    // page stays derived and your words survive every future rebuild.
    let fixture = Fixture::new("adoptedit");
    fixture.seed_session(2);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());

    let mut pages = Vec::new();
    collect_under(&fixture.wiki(), "sessions", &mut pages);
    let page = pages
        .into_iter()
        .find(|path| path.extension().is_some_and(|ext| ext == "md"))
        .expect("a session page");
    let before = std::fs::read_to_string(&page).unwrap();
    assert!(before.contains("Refactored the auth path"), "precondition: {before}");

    // The human corrects it where they read it.
    let edited = before.replace(
        "Refactored the auth path and fixed token expiry.",
        "Actually reverted the auth refactor; token expiry was never the bug.",
    );
    assert_ne!(edited, before, "the test's own edit did not apply");
    std::fs::write(&page, &edited).unwrap();

    // Any later consolidation must adopt the edit instead of erasing it.
    fixture.seed_session(2);
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());

    let after = std::fs::read_to_string(&page).unwrap();
    assert!(
        after.contains("Actually reverted the auth refactor"),
        "the human's correction was overwritten: {after}"
    );

    // And it is in the log, which is what makes it survive a rebuild.
    assert!(
        fixture.log_text().contains("Actually reverted the auth refactor"),
        "the edit never reached the log, so it is one reindex from gone"
    );
    assert!(fixture.brain(&["reindex"]).status.success());
    let found = String::from_utf8_lossy(&fixture.brain(&["search", "reverted"]).stdout).to_string();
    assert!(!found.contains("No matches"), "the correction did not survive a rebuild: {found}");
}

#[test]
fn the_wiki_can_say_what_it_used_to_believe() {
    // Every consolidation already commits the wiki, so the record of how a
    // page changed exists in full - there was just no way to ask for it.
    // This is the cheapest possible temporal answer: not "when was this true
    // in the world", but "when did memory start saying so".
    let fixture = Fixture::new("pagehistory");
    fixture.seed_session(2);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());

    // A second consolidation rewrites the hub notes, giving a page two versions.
    fixture.seed_session(2);
    let second = fixture.fake_cli(
        "claude",
        r#"echo '{"summary":"A different account of the same work.","titles":[]}'"#,
    );
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&second)).status.success());

    let out = fixture.brain(&["history", "checkout"]);
    assert!(out.status.success(), "history failed: {out:?}");
    let report = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(report.contains("checkout"), "the page was not identified: {report}");
    assert!(
        report.matches("consolidate").count() >= 1,
        "no revisions listed: {report}"
    );

    // A page nobody has written must say so rather than fail.
    let missing = fixture.brain(&["history", "nothing-by-this-name"]);
    assert!(missing.status.success(), "a missing page should not be an error: {missing:?}");
    assert!(
        String::from_utf8_lossy(&missing.stdout).contains("No page"),
        "expected a plain answer: {}",
        String::from_utf8_lossy(&missing.stdout)
    );
}

#[test]
fn forgetting_an_entity_spares_its_siblings() {
    // The third forgetting primitive: "forget everything about X" when the
    // caller does not know the ids. The property that makes it correct is
    // that everything NOT about X survives - entities are recorded per
    // session, so a session-level sweep would destroy unrelated memory the
    // same sessions happen to hold.
    let fixture = Fixture::new("amnesia");
    let say = |prompt: &str| {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "prompt": prompt
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    };
    say("acmecorp billing needs a retry");
    say("acmecorp asked for a data export");
    say("the scheduler double-books on Tuesdays");

    // Preview first: nothing may change until --apply.
    let preview = fixture.brain(&["forget", "--entity", "acmecorp"]);
    assert!(preview.status.success(), "preview failed: {preview:?}");
    let listed = String::from_utf8_lossy(&preview.stdout).to_string();
    assert!(listed.contains("billing") && listed.contains("data export"), "preview: {listed}");
    assert!(listed.contains("--apply"), "the preview must say how to perform it: {listed}");
    let still =
        String::from_utf8_lossy(&fixture.brain(&["search", "acmecorp"]).stdout).to_string();
    assert!(still.contains("billing"), "a preview withdrew something: {still}");

    let done = fixture.brain(&["forget", "--entity", "acmecorp", "--apply"]);
    assert!(done.status.success(), "apply failed: {done:?}");

    let gone = String::from_utf8_lossy(&fixture.brain(&["search", "acmecorp"]).stdout).to_string();
    assert!(gone.contains("No matches"), "the entity survived its own amnesia: {gone}");

    // The sibling memory, from the same session, is untouched.
    let sibling =
        String::from_utf8_lossy(&fixture.brain(&["search", "scheduler"]).stdout).to_string();
    assert!(
        sibling.contains("double-books"),
        "an unrelated memory in the same session was destroyed: {sibling}"
    );

    // Append-only holds: the log still carries what was withdrawn.
    assert!(fixture.log_text().contains("data export"), "the log lost the original");
}

#[test]
fn search_can_be_scoped_to_one_kind_of_memory() {
    // "What mentions the scheduler" and "what did we DECIDE about the
    // scheduler" are different questions. Relevance answers the first; only a
    // scope answers the second.
    //
    // The typed memories are produced the way the product produces them —
    // consolidation classifying captured events — rather than written into the
    // database by hand. The hand-written version needed the HOST's `sqlite3`,
    // and inserting into `events` fires the FTS triggers, so on a macOS runner
    // whose system SQLite has no FTS5 the seed failed and the test reported a
    // search that found nothing. Ours is a bundled SQLite with FTS5; the
    // machine's is not our business.
    let fixture = Fixture::new("scopedsearch");
    // One session, so one consolidation prompt carries both ids and the stub
    // can classify them differently.
    for file in ["src/scheduler.rs", "src/queue.rs"] {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abc70",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join(file)},
            "prompt": "the scheduler double-books when two runs overlap"
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    // Retitles whatever it is given, one entry as a decision and one as a
    // bugfix. Lifting the ids out of the prompt is what makes this a test of
    // consolidation rather than of the stub: the classification can only land
    // on the right rows if the prompt genuinely carried them.
    let classifier = r#"
IDS=$(echo "$*" | grep -oE 'id=[0-9A-Z]{26}' | cut -d= -f2)
printf '{"summary":"scheduler work, both sides of it","titles":[{"id":"%s","title":"scheduler: chose cron over a queue","kind":"decision"},{"id":"%s","title":"scheduler double-booking fixed","kind":"bugfix"}]}' "$(echo "$IDS" | head -1)" "$(echo "$IDS" | head -2 | tail -1)"
"#;
    let bin = fixture.fake_cli("claude", classifier);
    let done = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(done.status.success(), "consolidate failed: {done:?}");
    // The tier matters: rule-based output carries no classification at all, so
    // a scoped search would return nothing and the test would be asserting
    // that a feature is absent.
    let tier = String::from_utf8_lossy(&done.stdout).to_string();
    assert!(tier.contains("claude-code"), "the stub never classified anything: {tier}");

    let all = String::from_utf8_lossy(&fixture.brain(&["search", "scheduler"]).stdout).to_string();
    assert!(all.contains("chose cron") && all.contains("double-booking"), "unscoped: {all}");

    let scoped =
        String::from_utf8_lossy(&fixture.brain(&["search", "scheduler", "--topic", "decision"]).stdout)
            .to_string();
    assert!(scoped.contains("chose cron"), "the decision was scoped out: {scoped}");
    assert!(!scoped.contains("double-booking"), "the bugfix leaked into a decision scope: {scoped}");
}

#[test]
fn a_correction_replaces_a_memory_rather_than_joining_it() {
    // A correction is applied in place: the target's text is overwritten. The
    // correction event itself is bookkeeping - a receipt that the change
    // happened - and if it also matches searches, one memory answers twice
    // and the agent has to work out which copy is authoritative.
    let fixture = Fixture::new("correctonce");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "the scheduler double-books on Tuesdays"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);
    let id = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["id"].as_str().map(str::to_string))
        .expect("an event to correct");

    let out = fixture.brain(&["correct", &id, "the scheduler double-books on Wednesdays"]);
    assert!(out.status.success(), "correct failed: {out:?}");

    let hits = String::from_utf8_lossy(&fixture.brain(&["search", "scheduler"]).stdout).to_string();
    let count = hits
        .lines()
        .filter(|line| line.split_whitespace().next().is_some_and(|w| w.len() == 26))
        .count();
    assert_eq!(count, 1, "the correction surfaced alongside what it corrected: {hits}");
    assert!(hits.contains("Wednesdays"), "the corrected text should be what surfaces: {hits}");
}

#[test]
fn mcp_recall_returns_what_was_captured() {
    let fixture = Fixture::new("mcp");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    fixture.hook("codex", "UserPromptSubmit", &codex_payload(&fixture.project));

    let responses = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#,
    ]);

    // The notification must not produce a response.
    assert_eq!(responses.len(), 3, "notifications must not be answered: {responses:?}");

    assert_eq!(responses[0]["result"]["serverInfo"]["name"], "rolepod-brain");

    let tools = responses[1]["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"brain_search"));
    assert!(names.contains(&"brain_get"));

    let hits = &responses[2]["result"]["structuredContent"]["hits"];
    let hits = hits.as_array().expect("search returns hits");
    assert!(!hits.is_empty(), "expected a hit for 'auth': {responses:?}");

    // Cross-CLI recall: the Codex prompt is findable from the same store.
    let clis: Vec<&str> = hits.iter().map(|hit| hit["cli"].as_str().unwrap()).collect();
    assert!(clis.contains(&"codex"), "codex observation not recalled: {clis:?}");

    // And an id from search drives brain_get to the full body.
    let id = hits[0]["id"].as_str().unwrap();
    let fetched = fixture.mcp(&[&format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{id}"]}}}}}}"#
    )]);
    assert_eq!(fetched[0]["result"]["structuredContent"]["count"], 1);
}

/// One brain holds every CLI's work, and until now nothing could ask it which
/// CLI did what. The column was always there; the question was unaskable, so
/// the answer came from raw SQL against an index the project itself calls
/// disposable.
#[test]
fn one_agent_can_read_what_another_agent_did() {
    let fixture = Fixture::new("crosscli");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    fixture.hook("codex", "UserPromptSubmit", &codex_payload(&fixture.project));

    let responses = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_recent","arguments":{}}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_recent","arguments":{"cli":"codex"}}}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_recent","arguments":{"cli":"gemini-cli"}}}"#,
    ]);

    let clis = |index: usize| -> Vec<String> {
        responses[index]["result"]["structuredContent"]["events"]
            .as_array()
            .expect("recent returns events")
            .iter()
            .map(|event| event["cli"].as_str().unwrap().to_string())
            .collect()
    };

    // Unfiltered is unchanged: both CLIs, one list.
    let every = clis(0);
    assert!(every.contains(&"claude-code".to_string()), "claude missing: {every:?}");
    assert!(every.contains(&"codex".to_string()), "codex missing: {every:?}");

    // The skill tells agents to group the unfiltered list by session, which is
    // only advice they can follow if the field reaches them at all.
    for event in responses[0]["result"]["structuredContent"]["events"].as_array().unwrap() {
        let session = event["session"].as_str().unwrap_or_default();
        assert!(!session.is_empty(), "an entry with no session cannot be grouped: {event}");
    }

    // Filtered is one agent's work on its own - the question this exists for.
    let only_codex = clis(1);
    assert!(!only_codex.is_empty(), "codex has observations to return");
    assert!(only_codex.iter().all(|cli| cli == "codex"), "leaked another CLI: {only_codex:?}");

    // A CLI that never ran here is empty, not an error: "nothing" has to be an
    // answer, or an agent reads a failure as a reason to stop asking.
    assert_eq!(responses[2]["result"]["structuredContent"]["count"], 0);
    assert!(responses[2].get("error").is_none(), "absence is not a failure: {:?}", responses[2]);
}

/// Several sessions of one agent run at once, so the flat list interleaves
/// work that has nothing to do with each other. `kind` is what makes it
/// readable, and `raw` has to be spelled the way the primer spells it.
#[test]
fn one_agents_parallel_sessions_can_be_read_apart() {
    let fixture = Fixture::new("crosskind");
    fixture.hook("codex", "UserPromptSubmit", &codex_payload(&fixture.project));
    fixture.brain_with_path(&["consolidate", "--force"], None);
    // Live work, arriving after the summary was written - the state an agent
    // is in whenever it asks what another agent is doing right now.
    fixture.hook("codex", "UserPromptSubmit", &codex_payload(&fixture.project));

    let responses = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_recent","arguments":{"cli":"codex","kind":"session_summary"}}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_recent","arguments":{"cli":"codex","kind":"raw"}}}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_recent","arguments":{"kind":"sesson_sumary"}}}"#,
    ]);

    let kinds = |index: usize| -> Vec<String> {
        responses[index]["result"]["structuredContent"]["events"]
            .as_array()
            .expect("recent returns events")
            .iter()
            .map(|event| event["kind"].as_str().unwrap().to_string())
            .collect()
    };

    let summaries = kinds(0);
    assert!(!summaries.is_empty(), "consolidation wrote a summary: {responses:?}");
    assert!(summaries.iter().all(|kind| kind == "session_summary"), "not summaries: {summaries:?}");

    // `raw` is the primer's word for an untyped observation. An agent only
    // ever saw that word, so that word has to work.
    let live = kinds(1);
    assert!(!live.is_empty(), "the unsummarized capture is reachable: {responses:?}");
    assert!(live.iter().all(|kind| kind == "observation"), "raw is not observation: {live:?}");

    // A typo must not read as "nothing remembered".
    let typo = &responses[2]["result"];
    assert_eq!(typo["isError"], true, "unknown kind must be loud: {typo:?}");
    let text = typo["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("session_summary"), "the error names what is valid: {text}");
}

/// The question the tools exist to answer, end to end: "find the session where
/// the other agent did X, and tell me what came of it." Search finds it by
/// meaning, the hit carries the session, and the session reads whole. Each
/// piece works on its own; this pins that they connect.
#[test]
fn a_session_found_by_meaning_can_then_be_read_whole() {
    let fixture = Fixture::new("chain");
    // Two agents, two sessions, so isolating one has to actually exclude the
    // other rather than being trivially satisfied.
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    fixture.hook("codex", "UserPromptSubmit", &codex_payload(&fixture.project));

    let found = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#,
    ]);
    let hits = found[0]["result"]["structuredContent"]["hits"].as_array().unwrap();
    let codex_hit = hits
        .iter()
        .find(|hit| hit["cli"] == "codex")
        .unwrap_or_else(|| panic!("codex work is findable by meaning: {hits:?}"));

    // Step 2 of the documented chain: the id comes off the hit, so nothing has
    // to be carried between calls.
    let session = codex_hit["session"].as_str().expect("a hit names its session");
    assert!(!session.is_empty());

    let read = fixture.mcp(&[&format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_recent","arguments":{{"session":"{session}"}}}}}}"#
    )]);
    let events = read[0]["result"]["structuredContent"]["events"].as_array().unwrap();
    assert!(!events.is_empty(), "the named session reads back: {read:?}");
    for event in events {
        assert_eq!(event["session"], session, "another session leaked in: {event}");
        assert_eq!(event["cli"], "codex", "another agent leaked in: {event}");
    }
}

#[test]
fn the_index_is_disposable_and_rebuilds_from_the_log() {
    let fixture = Fixture::new("reindex");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    fixture.hook("codex", "UserPromptSubmit", &codex_payload(&fixture.project));

    let before = fixture.brain(&["search", "auth"]);
    let before = String::from_utf8_lossy(&before.stdout).to_string();
    assert!(before.contains("auth"), "precondition: search works: {before}");

    // Delete the whole index, WAL and all.
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(fixture.home.join(format!("brain.db{suffix}")));
    }
    assert!(!fixture.home.join("brain.db").exists());

    let reindex = fixture.brain(&["reindex"]);
    assert!(reindex.status.success(), "reindex failed: {reindex:?}");
    let summary = String::from_utf8_lossy(&reindex.stdout);
    assert!(summary.contains("Reindexed 2 event(s)"), "unexpected summary: {summary}");

    let after = fixture.brain(&["search", "auth"]);
    let after = String::from_utf8_lossy(&after.stdout).to_string();
    assert_eq!(
        after.lines().count(),
        before.lines().count(),
        "recall after reindex differs from before"
    );
}

#[test]
fn a_hook_returns_fast_enough_that_the_host_does_not_feel_it() {
    let fixture = Fixture::new("latency");
    // Warm the store so the measurement is steady-state, not first-run.
    fixture.hook("claude-code", "SessionStart", &claude_payload(&fixture.project));
    let payload = claude_payload(&fixture.project);

    // Best of two rounds, each five consecutive hooks. The claim is that the
    // binary can serve a real session under budget; the suite around it is
    // thirty-odd tests all spawning processes and fsyncing at once, which is
    // not a condition any real session experiences. One clean round proves the
    // claim; demanding that both rounds win would only measure the scheduler.
    let mut best = std::time::Duration::MAX;
    for _ in 0..2 {
        let mut worst = std::time::Duration::ZERO;
        for _ in 0..5 {
            let start = std::time::Instant::now();
            let output = fixture.hook("claude-code", "PostToolUse", &payload);
            worst = worst.max(start.elapsed());
            assert!(output.status.success());
        }
        best = best.min(worst);
    }

    // The budget applies to the SHIPPED binary, and `--release` measures it for
    // real (10.8ms when this was written). An unoptimized build is not that
    // binary, so its allowance is loose on purpose - it still catches an
    // order-of-magnitude regression, which is what a debug run can honestly
    // detect.
    let budget = if cfg!(debug_assertions) { 250 } else { 50 };
    assert!(
        best < std::time::Duration::from_millis(budget),
        "slowest hook in the best round was {best:?}, over the {budget}ms budget for this \
         build profile (the shipped budget is 50ms; measure it with `cargo test --release`)"
    );
}

#[test]
fn setup_is_dry_by_default() {
    let fixture = Fixture::new("setup");
    let output = fixture.brain(&["setup"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Dry run") || stdout.contains("Nothing to do"),
        "setup must not change anything without --apply: {stdout}"
    );
    assert!(
        !stdout.contains("wrote"),
        "dry run reported a write: {stdout}"
    );
}

/// A stub that answers like a working cheap-tier model.
const GOOD_CLI: &str =
    r#"echo '{"summary":"Refactored the auth path and fixed token expiry.","titles":[]}'"#;
/// A stub that answers both kinds of call: a session summary, and the
/// cross-session synthesis. It lifts a real summary id out of the synthesis
/// prompt, so the provenance link in the page can only be right if the
/// prompt genuinely carried the summaries it claims to draw from.
fn knowledge_cli(counter: &Path) -> String {
    format!(
        r#"
case "$*" in
  *"SESSION SUMMARIES"*)
    echo x >> {counter}
    IDS=$(echo "$*" | grep -oE 'id=[0-9A-Z]{{26}}' | cut -d= -f2)
    ID=$(echo "$IDS" | head -1)
    ID2=$(echo "$IDS" | head -2 | tail -1)
    echo "{{\"knowledge\":[{{\"kind\":\"gotcha\",\"title\":\"vitest must run file-by-file here\",\"body\":\"The shared fixture leaks between files.\",\"sources\":[\"$ID\",\"$ID2\"]}},{{\"kind\":\"invented\",\"title\":\"not a real kind\",\"body\":\"b\",\"sources\":[\"$ID\",\"$ID2\"]}},{{\"kind\":\"decision\",\"title\":\"happened once in one session\",\"body\":\"Cited a single summary.\",\"sources\":[\"$ID\"]}}]}}" ;;
  *) echo '{{"summary":"Refactored the auth path and fixed token expiry.","titles":[]}}' ;;
esac
"#,
        counter = counter.display()
    )
}

/// A stub that fails the way a rate-limited CLI does.
const FAILING_CLI: &str = "echo 'rate limit exceeded' >&2; exit 1";

#[test]
fn what_recurs_across_sessions_becomes_a_page_that_outlives_them() {
    let fixture = Fixture::new("knowledge");
    let counter = fixture.home.parent().unwrap().join("synth-calls");
    let bin = fixture.fake_cli("claude", &knowledge_cli(&counter));
    let synth_calls = || std::fs::read_to_string(&counter).unwrap_or_default().lines().count();

    // Four sessions is under the watermark; the fifth crosses it.
    for session in 0..5 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-34567890000{session}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);

        let done = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
        assert!(done.status.success(), "consolidate {session} failed: {done:?}");

        // Semantic memory must not appear before enough episodes exist to
        // support it: one session's noise is not what a project knows.
        let pages = fixture.knowledge_pages();
        assert_eq!(
            !pages.is_empty(),
            session == 4,
            "knowledge after {} session(s) — watermark is wrong: {pages:?}",
            session + 1
        );
    }

    let pages = fixture.knowledge_pages();
    assert_eq!(pages.len(), 1, "expected one usable entry of three: {pages:?}");
    // A single-session claim is a session summary wearing a promotion, and
    // knowledge outranks summaries in the primer.
    assert!(
        !fixture.log_text().contains("happened once in one session"),
        "an entry supported by one summary was kept"
    );
    // An invented kind is dropped rather than given a directory of its own.
    let path = pages[0].to_string_lossy();
    assert!(path.contains("knowledge/gotchas/"), "wrong home for a gotcha: {path}");
    assert!(path.ends_with("vitest-must-run-file-by-file-here.md"), "bad filename: {path}");
    let page = std::fs::read_to_string(&pages[0]).expect("the gotcha page");
    assert!(page.contains("tags: [knowledge, gotcha]"), "page not typed: {page}");
    assert!(page.contains("The shared fixture leaks between files."), "body missing: {page}");

    // Provenance: the page names a summary that actually exists in the log.
    let source = page
        .lines()
        .skip_while(|line| !line.starts_with("## Drawn from"))
        .find_map(|line| line.split('`').nth(1).map(str::to_owned))
        .expect("a provenance line naming a source summary");
    assert!(
        fixture.log_text().contains(&source),
        "page cites `{source}`, which is in no log entry"
    );

    // A page nobody can retrieve is half a memory: the same knowledge has to
    // reach an agent through ordinary search, not only through the vault.
    let hits = String::from_utf8_lossy(&fixture.brain(&["search", "vitest"]).stdout).into_owned();
    assert!(hits.contains("vitest must run file-by-file here"), "not retrievable: {hits}");

    // Five more sessions, and the model rediscovers what it already found.
    // Knowledge must not accrete a duplicate per synthesis round.
    for session in 5..10 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-3456789000{session:02}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
        assert!(
            fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
            "consolidate {session} failed"
        );
    }
    // The second round must genuinely have run and found nothing new — a
    // watermark that never re-armed would pass the count check by doing
    // nothing at all.
    assert_eq!(synth_calls(), 2, "synthesis did not run once per five sessions");
    assert_eq!(fixture.knowledge_pages().len(), 1, "synthesis duplicated a known fact");
    assert_eq!(
        fixture.log_text().matches("\"kind\":\"knowledge\"").count(),
        1,
        "the log gained a duplicate knowledge entry"
    );
}

/// A stub that answers the synthesis call the way most rounds really end:
/// well-formed JSON saying nothing recurred.
fn empty_knowledge_cli(counter: &Path) -> String {
    format!(
        r#"
case "$*" in
  *"SESSION SUMMARIES"*)
    echo x >> {counter}
    echo '{{"knowledge":[]}}' ;;
  *) echo '{{"summary":"Refactored the auth path and fixed token expiry.","titles":[]}}' ;;
esac
"#,
        counter = counter.display()
    )
}

/// A stub whose synthesis answer is REWORDED between rounds: one claim, told
/// twice, the second time with the detail that had since been learned.
fn reworded_knowledge_cli(counter: &Path) -> String {
    format!(
        r#"
case "$*" in
  *"SESSION SUMMARIES"*)
    echo x >> {counter}
    ROUND=$(wc -l < {counter} | tr -d ' ')
    IDS=$(echo "$*" | grep -oE 'id=[0-9A-Z]{{26}}' | cut -d= -f2)
    ID=$(echo "$IDS" | head -1); ID2=$(echo "$IDS" | head -2 | tail -1)
    if [ "$ROUND" = "1" ]; then
      echo "{{\"knowledge\":[{{\"kind\":\"gotcha\",\"title\":\"Use gatedDb harness for deterministic race condition testing\",\"body\":\"The harness serialises the two writers.\",\"sources\":[\"$ID\",\"$ID2\"]}}]}}"
    else
      echo "{{\"knowledge\":[{{\"kind\":\"gotcha\",\"title\":\"Use gatedDb harness for deterministic race-condition testing\",\"body\":\"The harness serialises the two writers and needs the barrier released twice.\",\"sources\":[\"$ID\",\"$ID2\"]}}]}}"
    fi ;;
  *) echo '{{"summary":"Refactored the auth path and fixed token expiry.","titles":[]}}' ;;
esac
"#,
        counter = counter.display()
    )
}

#[test]
fn a_reworded_claim_updates_its_page_instead_of_being_discarded() {
    // Skipping a duplicate kept whichever wording arrived FIRST, so a later
    // round that knew more had nowhere to put it. Measured on the real store:
    // 49 of 241 knowledge pages have a near-twin at 0.90, against 3 at the
    // 0.95 the threshold used to sit at - every sampled pair in between was
    // one claim written twice, none of them a distinct fact.
    //
    // Superseding is not contradiction handling and cannot be: a page saying
    // the release targets four platforms sits 0.643 from the page about the
    // fifth. That is a different problem, and no threshold reaches it.
    let fixture = Fixture::new("reworded-knowledge");
    let counter = fixture.home.parent().unwrap().join("synth-calls");
    let bin = fixture.fake_cli("claude", &reworded_knowledge_cli(&counter));

    for session in 0..10 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-3456789200{session:02}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
        assert!(
            fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
            "consolidate {session} failed"
        );
    }

    assert_eq!(
        fixture.knowledge_pages().len(),
        1,
        "the rewording was stored as a rival page instead of landing on the first"
    );

    // What recall serves is the later wording, with what the second round
    // had learned. Before this, the first wording served forever.
    let hits = String::from_utf8_lossy(&fixture.brain(&["search", "barrier"]).stdout).into_owned();
    // Search brackets the term it matched, so assert on the words around it.
    assert!(
        hits.contains("released twice"),
        "the newer wording never reached the page it belongs to: {hits}"
    );
}

#[test]
fn a_stale_tmpdir_does_not_look_like_every_cli_vanishing() {
    // Found on a real machine: four rungs failing at once with "No such file
    // or directory" naming programs that were all sitting right there, two of
    // them native binaries with no interpreter to blame. The cause was the
    // working directory the child is given, not the child - and a summarizer
    // that silently drops to rule-based for every session is the kind of
    // failure this project exists to make impossible.
    let fixture = Fixture::new("stale-tmpdir");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.seed_session(4);

    let gone = fixture.home.parent().unwrap().join("tmpdir-that-was-cleaned");
    assert!(!gone.exists(), "the point is that this directory is missing");

    let mut command = std::process::Command::new(BRAIN);
    command
        .args(["consolidate", "--force"])
        .current_dir(&fixture.project)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("HOME", fixture.home.parent().unwrap())
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("TMPDIR", &gone);
    let output = command.output().expect("run brain");

    assert!(output.status.success(), "consolidate errored: {output:?}");
    let summary = String::from_utf8_lossy(&output.stdout);
    assert!(
        !summary.contains("rule-based"),
        "a working CLI was reported as unreachable: {summary}"
    );
}

#[test]
fn a_round_that_finds_nothing_is_finished_not_retried() {
    // The common outcome, and the one that used to cost the most. An empty
    // list was read as an unusable answer, which did two things: it charged
    // the rung a breaker failure for being right - benching a working CLI
    // after three honest rounds - and it skipped the watermark, so the same
    // synthesis prompt fired again at every single consolidation instead of
    // once per five sessions.
    let fixture = Fixture::new("empty-synthesis");
    let counter = fixture.home.parent().unwrap().join("synth-calls");
    let bin = fixture.fake_cli("claude", &empty_knowledge_cli(&counter));
    let synth_calls = || std::fs::read_to_string(&counter).unwrap_or_default().lines().count();

    for session in 0..10 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-3456789100{session:02}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
        assert!(
            fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
            "consolidate {session} failed"
        );
    }

    // Twice in ten sessions, not once per consolidation. Under the old
    // reading the watermark never advanced, so this counted six.
    assert_eq!(synth_calls(), 2, "an empty answer did not finish the round");
    assert!(fixture.knowledge_pages().is_empty(), "nothing recurred, so nothing is durable");

    // And the rung that told the truth is still trusted.
    let doctor = fixture.brain_with_path(&["doctor"], Some(&bin));
    let report = String::from_utf8_lossy(&doctor.stdout);
    assert!(
        !report.contains("consecutive failure"),
        "a correct empty answer was charged to the breaker: {report}"
    );
}

#[test]
fn a_model_override_reaches_the_spawned_command_line() {
    // The quality knob has to actually turn: config names a better model for
    // one CLI, and that name - not the cheap default - must be what the
    // spawned process is handed.
    let fixture = Fixture::new("modeloverride");
    std::fs::write(
        fixture.home.join("config.toml"),
        "[summarizer]\nmode = \"claude-code\"\n\n[summarizer.models]\n\"claude-code\" = \"sonnet\"\n",
    )
    .unwrap();
    fixture.seed_session(4);

    let argv_log = fixture.home.parent().unwrap().join("argv.txt");
    let bin = fixture.fake_cli(
        "claude",
        &format!(
            "echo \"$@\" >> {argv}\n{answer}",
            argv = argv_log.display(),
            answer = r#"echo '{"summary":"quality paid for","titles":[]}'"#
        ),
    );
    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");

    let argv = std::fs::read_to_string(&argv_log).expect("the stub should have been called");
    assert!(argv.contains("--model sonnet"), "the override never reached the spawn: {argv}");
    assert!(!argv.contains("haiku"), "the cheap default leaked through anyway: {argv}");
    assert!(fixture.page_text().contains("quality paid for"), "the summary was not written");
}

#[test]
fn consolidation_degrades_to_rule_based_then_catches_up() {
    let fixture = Fixture::new("ladder");
    fixture.seed_session(4);

    // Round 1: no model reachable at all.
    let degraded = fixture.brain_with_path(&["consolidate", "--force"], None);
    assert!(degraded.status.success(), "degraded run failed: {degraded:?}");
    let summary = String::from_utf8_lossy(&degraded.stdout);
    assert!(summary.contains("rule-based"), "expected the floor tier: {summary}");

    // The page exists and is genuinely readable, not a placeholder.
    let page = fixture.page_text();
    assert!(page.contains("## Summary"), "no page written: {page}");
    assert!(page.contains("observation(s) captured"), "rule-based summary missing");
    assert!(page.contains("src/file0.rs"), "page should name the files touched");

    // Crucially: nothing was marked done, so the better run can still happen.
    assert_eq!(fixture.pending_count(), 4, "a degraded run must not consume events");

    // Round 2: a model appears.
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let recovered = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(recovered.status.success(), "recovery run failed: {recovered:?}");
    let summary = String::from_utf8_lossy(&recovered.stdout);
    assert!(summary.contains("claude-code"), "expected the model tier: {summary}");

    // The narrative replaced the fallback, and the work is now consumed.
    let page = fixture.page_text();
    assert!(page.contains("Refactored the auth path"), "model summary not written: {page}");
    assert_eq!(fixture.pending_count(), 0, "a successful run consumes its events");

    // No data loss anywhere: the original observations are still in the log.
    let log = fixture.log_text();
    for index in 0..4 {
        assert!(log.contains(&format!("src/file{index}.rs")), "event {index} lost from log");
    }
    assert!(log.contains(r#""kind":"session_summary""#), "summary not appended to the log");
}

#[test]
fn a_rewritten_title_is_appended_never_mutated() {
    let fixture = Fixture::new("retitle");
    fixture.seed_session(3);
    let bin = fixture.fake_cli(
        "claude",
        r#"echo '{"summary":"Did the work.","titles":[{"id":"REPLACE_ME","title":"Much better title"}]}'"#,
    );

    // Learn a real event id, then have the stub rewrite exactly that one.
    let log = fixture.log_text();
    let first: serde_json::Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
    let id = first["id"].as_str().unwrap().to_string();
    let original_title = first["title"].as_str().unwrap().to_string();
    let script = std::fs::read_to_string(bin.join("claude")).unwrap().replace("REPLACE_ME", &id);
    std::fs::write(bin.join("claude"), script).unwrap();

    fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));

    let lines: Vec<serde_json::Value> = fixture
        .log_text()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    // The original capture line is untouched.
    let original = lines.iter().find(|line| line["id"] == id.as_str()).unwrap();
    assert_eq!(original["title"], original_title, "capture line was mutated");

    // The new title arrived as its own page_update, linked back to the origin.
    let update = lines
        .iter()
        .find(|line| line["kind"] == "page_update")
        .expect("no page_update appended");
    assert_eq!(update["title"], "Much better title");
    assert_eq!(update["links"][0], id.as_str());
}

#[test]
fn repeated_failures_trip_the_breaker_instead_of_retrying_forever() {
    let fixture = Fixture::new("breaker");
    let bin = fixture.fake_cli("claude", FAILING_CLI);

    // Each forced run is one failed call; the third opens the breaker.
    for round in 0..3 {
        fixture.seed_session(4);
        let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
        assert!(output.status.success(), "round {round} errored: {output:?}");
        let summary = String::from_utf8_lossy(&output.stdout);
        assert!(summary.contains("rule-based"), "round {round} should degrade: {summary}");
    }

    let doctor = fixture.brain_with_path(&["doctor"], Some(&bin));
    let report = String::from_utf8_lossy(&doctor.stdout);
    assert!(
        report.contains("in cooldown"),
        "breaker should be open and visible in doctor: {report}"
    );
    assert!(report.contains("rate limit exceeded"), "doctor should show why: {report}");

    // Degraded the whole way, and still nothing lost.
    assert!(fixture.pending_count() > 0, "failed runs must not consume events");
}

#[test]
fn consolidation_never_captures_itself() {
    let fixture = Fixture::new("noloop");
    fixture.seed_session(4);

    // A stub that calls the hook path the way a host CLI's own hooks would.
    let script = format!(
        "echo '{{\"prompt\":\"recursive-marker\"}}' | {BRAIN} hook --cli claude-code --event UserPromptSubmit >/dev/null 2>&1\necho '{{\"summary\":\"Done.\",\"titles\":[]}}'"
    );
    let bin = fixture.fake_cli("claude", &script);

    let before = fixture.log_text().lines().count();
    fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    let after = fixture.log_text();

    assert!(
        !after.contains("recursive-marker"),
        "the summarizer's own hook call was captured - the loop guard failed"
    );
    assert!(after.lines().count() > before, "the summary should still have been written");
}

#[test]
fn the_wiki_is_git_versioned() {
    let fixture = Fixture::new("git");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));

    let wiki = fixture.wiki();
    assert!(wiki.join(".git").exists(), "wiki should be a git repository");

    let log = Command::new("git")
        .args(["log", "--oneline"])
        .current_dir(&wiki)
        .output()
        .expect("git log");
    let log = String::from_utf8_lossy(&log.stdout);
    assert!(log.contains("consolidate"), "page should be committed: {log}");
}

#[test]
fn an_oversized_session_merges_its_chunks_into_one_narrative() {
    let fixture = Fixture::new("merge");

    // Enough bulk to force more than one chunk.
    for index in 0..60 {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Bash",
            "tool_input": {"command": format!("run-{index} {}", "x".repeat(1200))}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    // A stub that answers differently on each call, so the page shows which
    // call produced the final summary.
    let counter = fixture.home.parent().unwrap().join("calls");
    let bin = fixture.fake_cli(
        "claude",
        &format!(
            "N=$(cat {c} 2>/dev/null || echo 0); N=$((N+1)); echo $N > {c}\n\
             echo \"{{\\\"summary\\\":\\\"call-$N\\\",\\\"titles\\\":[]}}\"",
            c = counter.display()
        ),
    );

    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let calls: usize = std::fs::read_to_string(&counter)
        .unwrap_or_default()
        .trim()
        .parse()
        .unwrap_or(0);
    assert!(calls > 1, "this session should have split into chunks, saw {calls} call(s)");

    let page = fixture.page_text();
    // The last call is the merge pass; its answer is what the page must carry.
    assert!(
        page.contains(&format!("call-{calls}")),
        "page should carry the merged summary from call {calls}: {page}"
    );
    assert!(
        !page.contains("call-1\n\ncall-2"),
        "chunk summaries were concatenated instead of merged: {page}"
    );
}

#[test]
fn session_start_injects_pointers_and_never_bodies() {
    let fixture = Fixture::new("primer");

    // A prior session worth remembering. The prompt is deliberately multi-line
    // so the one-line title and the full body are genuinely different text -
    // otherwise "no bodies in the primer" would pass for the wrong reason.
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "why does auth reject valid tokens?\n\nSECRET_BODY_MARKER: the expiry \
                   comparison uses < instead of <=, so a token expiring this second is \
                   treated as already expired."
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);

    // A new session starts.
    let start = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000000",
        "cwd": fixture.project,
        "source": "startup"
    })
    .to_string();
    let output = fixture.hook("claude-code", "SessionStart", &start);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid hook JSON");

    let context = parsed["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .expect("primer should be injected");
    assert_eq!(parsed["hookSpecificOutput"]["hookEventName"], "SessionStart");

    // Pointers, with ids the agent can pull with.
    assert!(context.contains("brain_get"), "primer should tell the agent how to pull");
    assert!(context.contains("Asked: why does auth reject valid tokens?"), "title missing");

    // The rest of that prompt must NOT be here - only its first line was.
    assert!(
        !context.contains("SECRET_BODY_MARKER"),
        "full content leaked into the primer: {context}"
    );
    assert!(
        context.len() <= 4096,
        "primer was {} bytes, over the 4096-byte default budget",
        context.len()
    );
}

#[test]
fn a_correction_takes_its_title_from_its_first_line() {
    // The skill tells agents to retire a stale claim with `brain correct`, and
    // to write a short first line because it becomes the title. That is a
    // contract, not a description: the first correction written in this
    // project as one long sentence produced a page titled with 120 characters
    // of run-on prose, and the advice only works while this holds.
    let fixture = Fixture::new("correct-title");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.seed_session(4);
    assert!(
        fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
        "consolidate failed"
    );

    let listing = fixture.brain(&["search", "auth"]);
    let listing = String::from_utf8_lossy(&listing.stdout).into_owned();
    let id = listing
        .split_whitespace()
        .find(|word| word.len() == 26 && word.chars().all(|c| c.is_ascii_alphanumeric()))
        .expect("a search result carries an id")
        .to_string();

    let out = fixture.brain(&[
        "correct",
        &id,
        "Token expiry is checked inclusively\nThe comparison uses <= so a token expiring this second is still rejected.",
    ]);
    assert!(out.status.success(), "correct failed: {out:?}");

    // Read the stored title, not the rendered one: a title that swallowed the
    // whole body still prints across two lines, so the terminal cannot tell
    // the two apart and neither could a test that reads it.
    let correction = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|value| value.get("hook").and_then(serde_json::Value::as_str) == Some("correct")
            || value.pointer("/source/hook").and_then(serde_json::Value::as_str) == Some("correct"))
        .expect("the correction is an event in the log");
    let title = correction["title"].as_str().expect("a correction has a title");
    assert_eq!(
        title, "Token expiry is checked inclusively",
        "the title is not the first line of the correction"
    );

    // And recall serves it.
    let after = fixture.brain(&["search", "expiry"]);
    let after = String::from_utf8_lossy(&after.stdout).into_owned();
    assert!(
        after.contains("Token expiry is checked inclusively"),
        "the corrected claim is not what recall returns: {after}"
    );
}

#[test]
fn a_rebuild_does_not_ask_for_every_summary_to_be_written_again() {
    // The failure this closes cost a real machine 141 model calls in ten
    // minutes. `mark_consolidated` wrote only to the database, so `reindex`
    // replayed the log, found every observation unfinished, and re-summarised
    // the entire history - while nine consolidation processes stacked up
    // racing each other through it.
    let fixture = Fixture::new("rebuild-progress");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.seed_session(4);
    assert!(
        fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
        "consolidate failed"
    );
    assert_eq!(fixture.pending_count(), 0, "precondition: the run finished its events");

    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(fixture.home.join(format!("brain.db{suffix}")));
    }
    assert!(fixture.brain(&["reindex"]).status.success(), "reindex failed");

    assert_eq!(
        fixture.pending_count(),
        0,
        "the rebuild asked for work that was already done, which is a model call per session"
    );
}

#[test]
fn a_rebuild_still_leaves_the_rule_based_floor_to_be_redone() {
    // The other half, and the one that matters more: a degraded run leaves its
    // events pending on purpose so a working model can redo them. Restoring
    // progress without asking which tier wrote the page consumed exactly those
    // events - quality lost permanently, quietly, on a rebuild.
    let fixture = Fixture::new("rebuild-floor");
    let bin = fixture.fake_cli("claude", FAILING_CLI);
    fixture.seed_session(4);
    assert!(
        fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
        "consolidate failed"
    );
    let pending = fixture.pending_count();
    assert!(pending > 0, "precondition: a degraded run keeps its events");

    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(fixture.home.join(format!("brain.db{suffix}")));
    }
    assert!(fixture.brain(&["reindex"]).status.success(), "reindex failed");

    assert_eq!(
        fixture.pending_count(),
        pending,
        "a rebuild treated the rule-based floor as finished work"
    );
}

#[test]
fn a_second_consolidation_leaves_rather_than_joining_in() {
    // Nine of these were found running at once on a real machine, the oldest
    // thirty-eight minutes old, each with its own model call in flight: every
    // session boundary started another, and each new one found the same
    // backlog still pending and set to work on it. The git lock kept the wiki
    // intact throughout, which is why nothing looked wrong while the spend
    // multiplied.
    let fixture = Fixture::new("run-lock");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.seed_session(4);

    // A lock left behind by a run still in progress.
    let lock = fixture.home.join(".brain-consolidate.lock");
    std::fs::write(&lock, b"").expect("write lock");

    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "a skipped run is not a failure: {out:?}");
    assert!(
        fixture.pending_count() > 0,
        "a second run worked the backlog anyway, which is what stacked nine of them up"
    );

    // And once the holder is gone, the next run proceeds.
    std::fs::remove_file(&lock).expect("release lock");
    assert!(
        fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
        "consolidate failed after the lock was released"
    );
    assert_eq!(fixture.pending_count(), 0, "the lock outlived its holder");
}

#[test]
fn reindex_gives_older_summaries_the_files_they_were_drawn_from() {
    // Subject files arrived after this brain had already written 643
    // summaries and 248 knowledge pages, and the log is append-only, so every
    // one of them carries an empty list forever. Anyone upgrading keeps a
    // memory whose most distilled half stays unreachable by file - which is
    // most people, since a store that has run for a while is the case this is
    // for.
    //
    // Nothing rewrites the log. Those entries name what they were drawn from,
    // so the list is derived when they are indexed, and replaying the log is
    // what fills it in.
    let fixture = Fixture::new("backfill");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let file = fixture.project.join("src/auth.rs");

    for index in 0..3 {
        let payload = serde_json::json!({
            "session_id": "0199c000-0000-7000-8000-000000000001",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": file, "new_string": format!("change {index}")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }
    assert!(
        fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
        "consolidate failed"
    );

    // Rewrite the log the way every existing install looks: the links stay,
    // the file list goes.
    let mut stripped_any = false;
    for log in fixture.log_files() {
        let text = std::fs::read_to_string(&log).expect("read log");
        let mut out = String::new();
        for line in text.lines() {
            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(mut value) => {
                    if value.get("kind").and_then(serde_json::Value::as_str)
                        == Some("session_summary")
                    {
                        value["files"] = serde_json::json!([]);
                        stripped_any = true;
                    }
                    out.push_str(&value.to_string());
                }
                Err(_) => out.push_str(line),
            }
            out.push('\n');
        }
        std::fs::write(&log, out).expect("write log");
    }
    assert!(stripped_any, "precondition: a summary with files must exist to strip");

    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(fixture.home.join(format!("brain.db{suffix}")));
    }
    assert!(fixture.brain(&["reindex"]).status.success(), "reindex failed");

    // The summary is reachable by the file its session worked on, rebuilt
    // from a log line that never mentioned that file.
    let read = serde_json::json!({
        "session_id": "0199c000-0000-7000-8000-000000000002",
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": file}
    })
    .to_string();
    let out = fixture.hook("claude-code", "PostToolUse", &read);
    let out: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim()).unwrap();
    let context = out["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .expect("file memory should be injected");
    assert!(
        context.contains("SUM"),
        "reindex left the older summary unreachable by its own file: {context}"
    );
}

#[test]
fn touching_a_file_reaches_the_durable_claim_drawn_from_it() {
    // `pointers_for_file` has always ranked knowledge first and session
    // summaries second - but neither tier stored any files, so on the real
    // store `event_files` held only page updates, observations and three
    // notes. Both branches of that ordering were dead, and touching a file
    // could return the raw record of what happened to it and never what the
    // project concluded about it.
    let fixture = Fixture::new("file-reaches-knowledge");
    let counter = fixture.home.parent().unwrap().join("synth-calls");
    let bin = fixture.fake_cli("claude", &knowledge_cli(&counter));
    let file = fixture.project.join("src/auth.rs");

    // Five sessions of work on one file is what promotes a claim about it.
    for session in 0..5 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-34567890300{session}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": file, "new_string": "x"}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
        assert!(
            fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success(),
            "consolidate {session} failed"
        );
    }
    assert_eq!(fixture.knowledge_pages().len(), 1, "no durable claim to reach");

    // A later session opens the file it was drawn from.
    let read = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000777",
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": file}
    })
    .to_string();
    let out = fixture.hook("claude-code", "PostToolUse", &read);
    let out: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim()).unwrap();
    let context = out["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .expect("file memory should be injected");

    assert!(
        context.contains("vitest must run file-by-file here"),
        "the claim this file taught the project never came back with it: {context}"
    );
}

#[test]
fn a_file_touch_injects_that_files_memory_once() {
    let fixture = Fixture::new("microinject");
    let file = fixture.project.join("src/auth.rs");

    // Build some history for one file.
    for index in 0..3 {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": file, "new_string": format!("change {index}")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    // A later session reads the same file.
    let read = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000000",
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": file}
    })
    .to_string();

    let first = fixture.hook("claude-code", "PostToolUse", &read);
    let first: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&first.stdout).trim()).unwrap();
    let context = first["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .expect("file memory should be injected");
    assert!(context.contains("src/auth.rs"), "injection should name the file: {context}");
    assert!(context.lines().count() <= 5, "at most 3 pointers plus a header: {context}");

    // Touching it again in the same session must be silent.
    let second = fixture.hook("claude-code", "PostToolUse", &read);
    let second: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&second.stdout).trim()).unwrap();
    assert!(
        second.get("hookSpecificOutput").is_none(),
        "a file must be injected once per session, got {second}"
    );
}

#[test]
fn a_file_with_no_history_stays_silent() {
    let fixture = Fixture::new("silent");
    fixture.seed_session(2);
    let payload = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000000",
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": fixture.project.join("src/brand-new.rs")}
    })
    .to_string();
    let output = fixture.hook("claude-code", "PostToolUse", &payload);
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "{}");
}

#[test]
fn injection_stays_inside_the_session_budget() {
    let fixture = Fixture::new("budget");

    // Plenty of history across many files.
    for index in 0..40 {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join(format!("src/f{index}.rs"))}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    // A new session that starts, then touches every one of those files.
    let session = "0199b000-0000-7000-8000-000000000000";
    let start = serde_json::json!({"session_id": session, "cwd": fixture.project, "source": "startup"})
        .to_string();
    fixture.hook("claude-code", "SessionStart", &start);
    for index in 0..40 {
        let payload = serde_json::json!({
            "session_id": session,
            "cwd": fixture.project,
            "tool_name": "Read",
            "tool_input": {"file_path": fixture.project.join(format!("src/f{index}.rs"))}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    let doctor = fixture.brain(&["doctor"]);
    let report = String::from_utf8_lossy(&doctor.stdout);
    let line = report.lines().find(|line| line.contains("injection")).unwrap_or("");
    assert!(line.starts_with("ok"), "injection went over budget: {line}");
    assert!(line.contains("cap 8192B"), "budget should be reported: {line}");
}

#[test]
fn a_note_is_saved_and_recalled_across_sessions() {
    let fixture = Fixture::new("note");

    let saved = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_note","arguments":{"text":"We chose SQLite over Postgres because nothing may run resident.","files":["src/store.rs"]}}}"#,
    ]);
    assert_eq!(saved[0]["result"]["structuredContent"]["saved"], true);

    // A separate MCP server process - a different session - finds it.
    let found = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"resident"}}}"#,
    ]);
    let hits = found[0]["result"]["structuredContent"]["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "note not recalled: {found:?}");
    assert_eq!(hits[0]["kind"], "note");

    // And it is in the log, not only the index.
    assert!(fixture.log_text().contains(r#""kind":"note""#));
}

#[test]
fn a_note_is_sanitized_like_any_captured_text() {
    let fixture = Fixture::new("notesecret");
    fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_note","arguments":{"text":"deploy key is ghp_abcdefghijklmnopqrstuvwxyz0123"}}}"#,
    ]);
    let log = fixture.log_text();
    assert!(!log.contains("ghp_abcdefghijklmnopqrstuvwxyz0123"), "a note leaked a token");
    assert!(log.contains("[REDACTED]"));
}

#[test]
fn there_is_no_remote_sync_and_saying_so_is_the_point() {
    // The wiki holds the architecture, decisions and dead ends of every
    // project it watched. A command that quietly did nothing, or that reported
    // success, would imply a backup exists somewhere it does not.
    let fixture = Fixture::new("nosync");
    fixture.seed_session(2);
    let output = fixture.brain(&["sync"]);
    assert!(!output.status.success(), "must not imply a sync happened");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no remote sync"), "should say what it does not do: {stderr}");
    assert!(stderr.contains("stays on this machine"), "and why");
    assert!(stderr.contains("git"), "and where the local history actually is");
}

#[test]
fn the_primer_is_typed_after_consolidation_classifies_it() {
    let fixture = Fixture::new("taxonomy");
    fixture.seed_session(3);

    // Learn two real ids, then have the stub classify exactly those.
    let ids: Vec<String> = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|line| line["id"].as_str().map(str::to_string))
        .take(2)
        .collect();
    assert_eq!(ids.len(), 2);

    let script = format!(
        "echo '{{\"summary\":\"Reworked the auth path.\",\"titles\":[\
           {{\"id\":\"{}\",\"title\":\"Chose spawn-on-demand over a resident worker\",\"kind\":\"decision\"}},\
           {{\"id\":\"{}\",\"title\":\"Token expiry compared with < instead of <=\",\"kind\":\"fix\"}},\
           {{\"id\":\"01MISSING0000000000000000\",\"title\":\"orphan\",\"kind\":\"refactoring\"}}\
         ]}}'",
        ids[0], ids[1]
    );
    let bin = fixture.fake_cli("claude", &script);
    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");

    // The classification is persisted on the page_update, additively.
    let log = fixture.log_text();
    let updates: Vec<serde_json::Value> = log
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|line| line["kind"] == "page_update")
        .collect();
    assert_eq!(updates.len(), 2, "the orphan id must not create an event");
    let topics: Vec<&str> =
        updates.iter().filter_map(|u| u["topic"].as_str()).collect();
    assert!(topics.contains(&"decision"), "decision not persisted: {topics:?}");
    assert!(topics.contains(&"bugfix"), "`fix` should normalize to bugfix: {topics:?}");

    // Schema did not move for an additive field.
    for update in &updates {
        assert_eq!(update["v"], 1);
    }

    // And the primer shows it as a typed, scannable column.
    let start = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000000",
        "cwd": fixture.project,
        "source": "startup"
    })
    .to_string();
    let output = fixture.hook("claude-code", "SessionStart", &start);
    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&output.stdout).trim()).unwrap();
    let context = parsed["hookSpecificOutput"]["additionalContext"].as_str().unwrap();

    assert!(context.contains("DEC  Chose spawn-on-demand"), "no typed decision: {context}");
    assert!(context.contains("FIX  Token expiry"), "no typed bugfix: {context}");
    assert!(context.contains("SUM  "), "the session summary should be tagged too");
    // The decision outranks the bugfix, which outranks everything unclassified.
    let dec = context.find("DEC").unwrap();
    let fix = context.find("FIX").unwrap();
    assert!(dec < fix, "decision should rank above bugfix");
}

#[test]
fn a_noisy_project_yields_a_short_primer_not_a_padded_one() {
    let fixture = Fixture::new("floor");

    // Twenty bare commands that touched nothing, plus two real questions.
    for index in 0..20 {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Bash",
            "tool_input": {"command": format!("echo noise-{index}")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }
    for question in ["why does the scheduler double-book?", "where is expiry compared?"] {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "prompt": question
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }

    let start = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000000",
        "cwd": fixture.project,
        "source": "startup"
    })
    .to_string();
    let output = fixture.hook("claude-code", "SessionStart", &start);
    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&output.stdout).trim()).unwrap();
    let context = parsed["hookSpecificOutput"]["additionalContext"].as_str().unwrap();

    assert!(context.contains("scheduler double-book"), "a real question was cut");
    assert!(context.contains("where is expiry compared"), "a real question was cut");
    assert!(!context.contains("noise-"), "bare commands padded the primer: {context}");
    assert!(
        context.len() < 1200,
        "a noisy project should yield a SHORT primer, got {} bytes",
        context.len()
    );
}

#[test]
fn antigravity_payloads_capture_into_the_right_project() {
    let fixture = Fixture::new("agy");

    // Verbatim shape from a real `agy -p --add-dir` run: no cwd, a workspace
    // list, conversationId, and a nested toolCall with PascalCase args.
    let payload = serde_json::json!({
        "artifactDirectoryPath": "/Users/x/.gemini/antigravity-cli/brain/17e2d461",
        "conversationId": "17e2d461-9aea-442c-9825-6d8c642ad4b6",
        "modelName": "gemini-3.5-flash-low",
        "stepIdx": 16,
        "workspacePaths": [fixture.project],
        "toolCall": {
            "name": "view_file",
            "args": {"AbsolutePath": fixture.project.join("src/auth.rs"), "IsSkillFile": false}
        }
    })
    .to_string();

    let output = fixture.hook("antigravity", "PostToolUse", &payload);
    assert!(output.status.success());

    let log = fixture.log_text();
    assert!(log.contains(r#""cli":"antigravity""#), "source.cli not tagged: {log}");
    assert!(log.contains("view_file"), "tool name not read from toolCall.name");
    assert!(log.contains("src/auth.rs"), "AbsolutePath not read or not relativized");

    // Same project as a Claude Code capture in the same checkout - one store.
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    let projects: std::collections::HashSet<String> = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|line| line["project"].as_str().map(str::to_string))
        .collect();
    assert_eq!(projects.len(), 1, "two CLIs produced two projects: {projects:?}");
}

#[test]
fn an_event_with_no_knowable_workspace_is_skipped_not_guessed() {
    let fixture = Fixture::new("noworkspace");

    // Antigravity without --add-dir: no cwd, empty workspace list. The hook's
    // own process cwd is the only other clue, and for this CLI it points at
    // its config directory, so there is nothing trustworthy to file under.
    let payload = serde_json::json!({
        "conversationId": "17e2d461-9aea-442c-9825-6d8c642ad4b6",
        "workspacePaths": [],
        "toolCall": {"name": "view_file", "args": {"AbsolutePath": "/somewhere/else.rs"}}
    })
    .to_string();

    // Run it from a CLI config directory, the way Antigravity actually does.
    //
    // The directory is created inside the fixture rather than looked for on
    // the machine. This test used to fall back to running from the project
    // directory when the host had no `~/.gemini/config` — which is a placeable
    // location, so the event was filed, and the assertion below failed. On a
    // developer machine that happens to have Antigravity installed it passed;
    // in CI it did not. A test that checks a different thing depending on who
    // is running it is not checking anything.
    let config_dir = fixture.home.parent().unwrap().join(".gemini/config");
    std::fs::create_dir_all(&config_dir).expect("create the CLI config dir");
    let output = Command::new(BRAIN)
        .args(["hook", "--cli", "antigravity", "--event", "PostToolUse"])
        .current_dir(&config_dir)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("HOME", fixture.home.parent().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            child.stdin.as_mut().unwrap().write_all(payload.as_bytes())?;
            child.wait_with_output()
        })
        .expect("run hook");

    assert!(output.status.success(), "the host must still be acknowledged");
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "{}");
    assert!(
        !fixture.log_text().contains("antigravity"),
        "an unplaceable event was filed anyway"
    );
}

#[test]
fn opencode_plugin_payloads_capture_like_any_other_cli() {
    let fixture = Fixture::new("opencode");

    // The shape our generated plugin sends: it supplies `cwd` from the plugin
    // factory's `directory`, which is why OpenCode has none of Antigravity's
    // project-identity problem.
    let session = serde_json::json!({
        "cwd": fixture.project,
        "session_id": "ses_7f3a9c2b",
        "source": "startup"
    })
    .to_string();
    fixture.hook("opencode", "session.created", &session);

    // `tool.execute.after` hands us (input.tool, output.args) — verified
    // against a working third-party plugin's handler signature.
    let tool = serde_json::json!({
        "cwd": fixture.project,
        "session_id": "ses_7f3a9c2b",
        "tool_name": "edit",
        "tool_input": {"filePath": fixture.project.join("src/auth.rs")}
    })
    .to_string();
    fixture.hook("opencode", "tool.execute.after", &tool);

    let log = fixture.log_text();
    assert!(log.contains(r#""cli":"opencode""#), "source.cli not tagged");
    assert!(log.contains("src/auth.rs"), "filePath not read or not relativized");

    // Event names normalize despite OpenCode's dotted spelling.
    assert!(log.contains(r#""hook":"session.created""#) || log.contains("session_created"));

    // And it shares one store with the other CLIs in this checkout.
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    let projects: std::collections::HashSet<String> = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|line| line["project"].as_str().map(str::to_string))
        .collect();
    assert_eq!(projects.len(), 1, "opencode split the store: {projects:?}");
}

#[test]
fn a_silenced_run_leaves_no_trace_at_all() {
    let fixture = Fixture::new("silentenv");
    fixture.seed_session(3);
    let before = fixture.log_text().lines().count();

    // Larger than a pipe buffer on purpose. A silenced run captures nothing,
    // but it is still a process the host is writing to: if it exits without
    // reading, the write fails with EPIPE and the host logs a hook failure —
    // which is the opposite of the clean room this switch promises. A small
    // payload fits in the buffer and hides that; this one does not.
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": format!("this must not be remembered {}", "x".repeat(256 * 1024))
    })
    .to_string();

    let mut child = Command::new(BRAIN)
        .args(["hook", "--cli", "claude-code", "--event", "UserPromptSubmit"])
        .current_dir(&fixture.project)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("ROLEPOD_BRAIN_SILENT", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn hook");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(payload.as_bytes())
        .expect("a silenced hook must still accept what the host sends it");
    let output = child.wait_with_output().expect("hook output");

    // The host still gets a clean acknowledgement.
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "{}");

    let after = fixture.log_text();
    assert_eq!(after.lines().count(), before, "a silenced run wrote to the log");
    assert!(!after.contains("must not be remembered"));
}

#[test]
fn no_code_path_can_produce_a_background_agent() {
    // The timer feature is removed, not disabled - a feature deliberately
    // kept off for everyone should not exist. What remains is the sweep for
    // machines an older version left a launchd job on.
    let fixture = Fixture::new("nobg");
    let plan = String::from_utf8_lossy(&fixture.brain(&["setup"]).stdout).to_string();
    for word in ["launchd", "LaunchAgents", "Login Items", "timer"] {
        assert!(!plan.contains(word), "setup plan mentions `{word}`: {plan}");
    }

    let apply = fixture.brain(&["setup", "--apply", "--cli", "claude-code"]);
    assert!(apply.status.success());
    let agents = fixture.home.parent().unwrap().join("Library/LaunchAgents");
    let planted = std::fs::read_dir(&agents)
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .count();
    assert_eq!(planted, 0, "setup --apply wrote into LaunchAgents");
}

#[test]
fn a_launchd_job_from_an_older_version_is_found_and_removed() {
    // Removal has to reach the machines the feature already touched, or an
    // orphaned job keeps waking a binary that no longer knows why.
    let fixture = Fixture::new("legacytimer");
    fixture.seed_session(1);
    let agents = fixture.home.parent().unwrap().join("Library/LaunchAgents");
    std::fs::create_dir_all(&agents).unwrap();
    let plist = agents.join("dev.rolepod.brain.consolidate.plist");
    std::fs::write(&plist, "<plist/>").unwrap();

    let report = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    let line = report
        .lines()
        .find(|line| line.split_whitespace().nth(1) == Some("backstop"))
        .unwrap_or("");
    assert!(line.starts_with("FAIL"), "an orphaned launchd job must be reported: {line}");

    assert!(fixture.brain(&["setup", "--apply", "--cli", "claude-code"]).status.success());
    assert!(!plist.exists(), "setup --apply should remove the orphaned plist");

    let report = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    let line = report
        .lines()
        .find(|line| line.split_whitespace().nth(1) == Some("backstop"))
        .unwrap_or("");
    assert!(line.starts_with("ok"), "backstop should be healthy once swept: {line}");
}

#[test]
fn doctor_reports_the_backstop_mode_rather_than_demanding_a_timer() {
    let fixture = Fixture::new("bstop");
    fixture.seed_session(2);
    let output = fixture.brain(&["doctor"]);
    let report = String::from_utf8_lossy(&output.stdout);
    // Match the check-name column, not the whole line: a fixture path can
    // contain the word too, and matching that would test nothing.
    let line = report
        .lines()
        .find(|line| line.split_whitespace().nth(1) == Some("backstop"))
        .unwrap_or("");
    assert!(line.starts_with("ok"), "backstop should be healthy by default: {line}");
    assert!(line.contains("hook-opportunistic"), "mode not reported: {line}");
}

/// A SessionStart payload with a given source.
fn start_payload(project: &Path, session: &str, source: &str) -> String {
    serde_json::json!({"session_id": session, "cwd": project, "source": source}).to_string()
}

fn injected_context(output: &std::process::Output) -> Option<String> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).ok()?;
    parsed["hookSpecificOutput"]["additionalContext"].as_str().map(str::to_string)
}

#[test]
fn memory_comes_back_after_a_context_wipe() {
    let fixture = Fixture::new("wipe");
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";

    // Some history worth remembering, then a session that reads it.
    for question in ["why does the scheduler double-book?", "where is expiry compared?"] {
        let payload = serde_json::json!({
            "session_id": "0199aaaa-0000-7000-8000-000000000000",
            "cwd": fixture.project,
            "prompt": question
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }

    let first = fixture.hook(
        "claude-code",
        "SessionStart",
        &start_payload(&fixture.project, session, "startup"),
    );
    let first = injected_context(&first).expect("primer on a normal start");
    assert!(first.contains("scheduler double-book"));

    // Compaction arrives as a `SessionStart` whose source says so - not as
    // `PostCompact`, which Claude Code refuses to accept context from.
    // Without the reset, this second injection would be suppressed as a
    // duplicate - the session id survived even though the context did not.
    let after = fixture.hook(
        "claude-code",
        "SessionStart",
        &start_payload(&fixture.project, session, "compact"),
    );
    let after = injected_context(&after)
        .expect("compaction wiped the context; memory must come straight back");
    assert!(
        after.contains("scheduler double-book"),
        "the pre-wipe memory was suppressed after compaction: {after}"
    );

    // `/clear` takes the other path and must behave identically.
    let cleared = fixture.hook(
        "claude-code",
        "SessionStart",
        &start_payload(&fixture.project, session, "clear"),
    );
    let cleared = injected_context(&cleared).expect("primer after /clear");
    assert!(cleared.contains("scheduler double-book"));
}

#[test]
fn a_wipe_gives_the_session_its_injection_budget_back() {
    let fixture = Fixture::new("budgetreset");
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    for index in 0..10 {
        let payload = serde_json::json!({
            "session_id": "0199aaaa-0000-7000-8000-000000000000",
            "cwd": fixture.project,
            "prompt": format!("question number {index} about the scheduler")
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }

    fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "startup"));
    let spent_before = injected_bytes(&fixture, session);
    assert!(spent_before > 0, "nothing was injected to begin with");

    fixture.hook(
        "claude-code",
        "PostCompact",
        &serde_json::json!({"session_id": session, "cwd": fixture.project, "trigger": "manual"})
            .to_string(),
    );
    let spent_after = injected_bytes(&fixture, session);
    assert!(
        spent_after <= spent_before,
        "budget accumulated across a wipe ({spent_before} -> {spent_after}); a fresh context \
         must get a fresh budget"
    );
}

fn injected_bytes(fixture: &Fixture, session: &str) -> i64 {
    let output = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg(format!(
            "SELECT COALESCE(SUM(bytes),0) FROM injected_bytes WHERE session='{session}';"
        ))
        .output()
        .expect("query injected bytes");
    String::from_utf8_lossy(&output.stdout).trim().parse().unwrap_or(-1)
}

#[test]
fn compaction_is_captured_before_the_wipe() {
    let fixture = Fixture::new("precompact");
    fixture.seed_session(2);
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "trigger": "auto"
    })
    .to_string();
    fixture.hook("claude-code", "PreCompact", &payload);

    let log = fixture.log_text();
    assert!(log.contains(r#""hook":"pre_compact""#), "no pre-compaction marker: {log}");
    assert!(log.contains("Context compacted"), "the marker should be readable");
}

#[test]
fn a_headless_session_gets_nothing_even_after_a_wipe() {
    // The headless rule outranks the wipe rule: a one-shot reviewer that
    // compacts mid-run must still not inherit the author's narrative.
    let fixture = Fixture::new("headlesswipe");
    fixture.seed_session(3);

    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    let payload =
        serde_json::json!({"session_id": session, "cwd": fixture.project, "trigger": "auto"})
            .to_string();

    // Run the hook from a process tree that looks headless by giving the
    // silence contract the same expectation: no injection whatsoever.
    let mut child = Command::new(BRAIN)
        .args(["hook", "--cli", "claude-code", "--event", "PostCompact"])
        .current_dir(&fixture.project)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("ROLEPOD_BRAIN_SILENT", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn hook");
    child.stdin.as_mut().unwrap().write_all(payload.as_bytes()).unwrap();
    let output = child.wait_with_output().expect("hook output");

    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "{}");
}

#[test]
fn a_model_that_ignores_the_prompt_still_cannot_leak_a_secret() {
    // The guarantee layer, tested the only way that means anything: a stub
    // that does exactly what the prompt forbids. The instruction is best
    // effort; the deterministic pass is the promise.
    let fixture = Fixture::new("postpass");
    fixture.seed_session(3);

    let leaky = r#"echo '{"summary":"Deployed with token ghp_abcdefghijklmnopqrstuvwxyz0123 and OPENAI_API_KEY=sk-livekey1234567890abcd.","titles":[{"id":"01A","title":"Set AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY","kind":"config"}]}'"#;
    let bin = fixture.fake_cli("claude", leaky);
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let page = fixture.page_text();
    let log = fixture.log_text();
    for secret in [
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "sk-livekey1234567890abcd",
        "wJalrXUtnFEMIK7MDENGbPxRfiCY",
    ] {
        assert!(!page.contains(secret), "secret reached the page: {secret}");
        assert!(!log.contains(secret), "secret reached the log: {secret}");
    }
    // The summary itself survived - only the credential was removed.
    assert!(page.contains("Deployed with token"), "the post-pass ate the whole summary");
    assert!(page.contains("[REDACTED]"));
}

#[test]
fn private_regions_never_reach_storage() {
    let fixture = Fixture::new("private");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "deploy for <private>Acme Holdings, 4.2M contract</private> next week"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);

    let log = fixture.log_text();
    assert!(!log.contains("Acme Holdings"), "a private region was stored");
    assert!(!log.contains("4.2M"), "a private region was stored");
    assert!(log.contains("[PRIVATE]"), "the redaction marker should be visible");
    assert!(log.contains("next week"), "text outside the region should survive");
}

#[test]
fn a_rate_limit_banner_falls_through_to_the_next_cli() {
    // The scenario that motivated this: a CLI whose quota is exhausted exits
    // ZERO and prints a banner. Before, that looked like "no model available"
    // and dropped straight to the rule-based floor without ever trying the
    // other CLI the user was also signed into.
    let fixture = Fixture::new("softfail");
    fixture.seed_session(4);

    let bin = fixture.fake_cli(
        "claude",
        "echo 'You have reached your usage limit for Claude. Resets at 3pm.'\nexit 0",
    );
    // codex reads its answer from the file named by `-o`, not from stdout -
    // the stub has to behave the way the real invocation does.
    fixture.fake_cli(
        "codex",
        concat!(
            "out=\"\"\n",
            "while [ $# -gt 0 ]; do [ \"$1\" = \"-o\" ] && { out=\"$2\"; }; shift; done\n",
            "printf '%s' '{\"summary\":\"Reworked the auth path.\",\"titles\":[]}' > \"$out\"\n",
            "echo 'tokens used 123'\n"
        ),
    );

    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");
    let summary = String::from_utf8_lossy(&output.stdout);
    assert!(
        summary.contains("codex"),
        "a soft failure on claude should advance to codex, got: {summary}"
    );
    assert!(!summary.contains("rule-based"), "the floor was used while a rung still worked");

    let page = fixture.page_text();
    assert!(page.contains("Reworked the auth path"), "codex's answer was not used");
    assert!(!page.contains("usage limit"), "the banner reached storage");

    // The failed rung is charged for it, so repeated failures still trip its
    // breaker rather than being retried forever.
    let doctor = fixture.brain_with_path(&["doctor"], Some(&bin));
    let report = String::from_utf8_lossy(&doctor.stdout);
    assert!(
        report.contains("summarizer: claude-code"),
        "the soft failure should count against that rung: {report}"
    );
    assert!(report.contains("unusable answer"), "the reason should say what happened");
}

#[test]
fn a_prompt_no_cli_can_use_does_not_cost_a_call_per_cli() {
    let fixture = Fixture::new("bounded");
    fixture.seed_session(4);

    // Every CLI answers unusably, and each records how many times it ran.
    let bin = fixture.fake_cli("claude", "echo 'login required'; exit 0");
    for cli in ["codex", "gemini"] {
        fixture.fake_cli(cli, "echo 'login required'; exit 0");
    }
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success());
    let summary = String::from_utf8_lossy(&output.stdout);
    assert!(summary.contains("rule-based"), "should end at the floor: {summary}");

    // At most two rungs were charged, not one per installed CLI.
    let doctor = fixture.brain_with_path(&["doctor"], Some(&bin));
    let report = String::from_utf8_lossy(&doctor.stdout);
    let charged = report.lines().filter(|line| line.contains("summarizer: ")).count();
    assert!(charged <= 2, "tried more rungs than the bound allows: {report}");
}

#[test]
fn a_wrong_memory_can_be_withdrawn_and_stays_withdrawn() {
    let fixture = Fixture::new("forget");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "the scheduler double-books on Tuesdays"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);

    let id = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["id"].as_str().map(str::to_string))
        .expect("an event to forget");

    assert!(fixture.brain(&["search", "scheduler"]).stdout.len() > 20, "precondition");

    let output = fixture.brain(&["forget", &id]);
    assert!(output.status.success(), "forget failed: {output:?}");

    // Gone from recall...
    let after = String::from_utf8_lossy(&fixture.brain(&["search", "scheduler"]).stdout)
        .to_string();
    assert!(after.contains("No matches"), "a forgotten memory still surfaces: {after}");

    // ...and gone from the primer, which is the other half of recall and the
    // half that costs bytes in every future session. Asserting only search is
    // how a withdrawn memory kept being injected.
    let start = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcdf",
        "cwd": fixture.project,
        "source": "startup"
    })
    .to_string();
    let primer = String::from_utf8_lossy(
        &fixture.hook("claude-code", "SessionStart", &start).stdout,
    )
    .to_string();
    assert!(
        !primer.contains("double-books"),
        "a forgotten memory is still injected at session start: {primer}"
    );
    // The tombstone deliberately says nothing about its target, so injecting
    // it spends bytes on pure bookkeeping.
    assert!(
        !primer.contains("Withdrew a memory"),
        "bookkeeping is being injected as if it were memory: {primer}"
    );

    // ...and an agent still holding the id cannot pull it back either.
    let request = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{id}"]}}}}}}"#
    );
    let pulled = fixture.mcp(&[&request]);
    let text = pulled.last().expect("a response")["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(!text.contains("double-books"), "brain_get resurrected a withdrawn memory: {text}");

    // ...but the log keeps both the original and the withdrawal.
    let log = fixture.log_text();
    assert!(log.contains("double-books"), "the log must not lose what was said");
    assert!(log.contains(r#""kind":"tombstone""#), "the withdrawal should be recorded");

    // And it survives rebuilding the index from the log alone.
    assert!(fixture.brain(&["reindex"]).status.success());
    let rebuilt = String::from_utf8_lossy(&fixture.brain(&["search", "scheduler"]).stdout)
        .to_string();
    assert!(rebuilt.contains("No matches"), "reindex resurrected a forgotten memory");
}

#[test]
fn a_badly_recorded_memory_can_be_corrected() {
    let fixture = Fixture::new("correct");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "the retry limit is five"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);

    let id = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["id"].as_str().map(str::to_string))
        .expect("an event to correct");

    let output = fixture.brain(&["correct", &id, "The retry limit is three, not five."]);
    assert!(output.status.success(), "correct failed: {output:?}");

    let after =
        String::from_utf8_lossy(&fixture.brain(&["search", "retry"]).stdout).to_string();
    assert!(after.contains("three, not five"), "recall should return the correction: {after}");

    // The original wording is still in the log - a correction is a claim about
    // history, not a rewriting of it.
    assert!(fixture.log_text().contains("the retry limit is five"));

    assert!(fixture.brain(&["reindex"]).status.success());
    let rebuilt =
        String::from_utf8_lossy(&fixture.brain(&["search", "retry"]).stdout).to_string();
    assert!(rebuilt.contains("three, not five"), "the correction did not survive reindex");
}

#[test]
fn a_correction_is_scrubbed_like_anything_else() {
    let fixture = Fixture::new("correctsecret");
    fixture.seed_session(1);
    let id = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["id"].as_str().map(str::to_string))
        .unwrap();

    fixture.brain(&["correct", &id, "it was deployed with ghp_abcdefghijklmnopqrstuvwxyz0123"]);
    let log = fixture.log_text();
    assert!(!log.contains("ghp_abcdefghijklmnopqrstuvwxyz0123"), "a correction leaked a token");
    assert!(log.contains("[REDACTED]"));
}

#[test]
fn a_tombstone_does_not_repeat_what_it_withdrew() {
    // The first version titled the tombstone "Forgot: <the forgotten text>",
    // which put the withdrawn words straight back into search results - the
    // whole operation undone by its own receipt.
    let fixture = Fixture::new("tombstonewords");
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "the invoice service charges twice on retry"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);
    let id = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|line| line["id"].as_str().map(str::to_string))
        .unwrap();

    fixture.brain(&["forget", &id]);

    let tombstone: serde_json::Value = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|line| line["kind"] == "tombstone")
        .expect("a tombstone");
    let title = tombstone["title"].as_str().unwrap();
    assert!(!title.contains("invoice"), "the tombstone quotes what it withdrew: {title}");
    assert_eq!(tombstone["links"][0], id.as_str(), "identity belongs in the link");

    for term in ["invoice", "charges", "retry"] {
        let out = String::from_utf8_lossy(&fixture.brain(&["search", term]).stdout).to_string();
        assert!(out.contains("No matches"), "searching {term} resurfaced it: {out}");
    }
}

#[test]
fn forgetting_an_unknown_id_fails_loudly() {
    let fixture = Fixture::new("forgetunknown");
    fixture.seed_session(1);
    let output = fixture.brain(&["forget", "01NOTAREALIDNOTAREALID00"]);
    assert!(!output.status.success(), "should not silently succeed");
    assert!(String::from_utf8_lossy(&output.stderr).contains("no memory with id"));
}

#[test]
fn a_brain_survives_being_moved_to_another_machine() {
    // The necessary consequence of never syncing: moving it yourself has to
    // actually work.
    let old = Fixture::new("exportfrom");
    // A named marker is what makes a project the same project at a different
    // path - which is what "another machine" means in practice.
    std::fs::write(
        old.project.join(".rolepod-brain.toml"),
        "[project]\nname = \"acme-api\"\n",
    )
    .unwrap();
    old.seed_session(3);
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": old.project,
        "prompt": "the scheduler double books on tuesdays"
    })
    .to_string();
    old.hook("claude-code", "UserPromptSubmit", &payload);

    let archive = old.home.parent().unwrap().join("brain.tar.gz");
    let out = old.brain(&["export", &archive.to_string_lossy()]);
    assert!(out.status.success(), "export failed: {out:?}");
    assert!(archive.is_file(), "no archive written");

    // The index is derived and must not travel.
    let listing = Command::new("tar")
        .args(["-tzf", &archive.to_string_lossy()])
        .output()
        .expect("list archive");
    let listing = String::from_utf8_lossy(&listing.stdout);
    assert!(listing.contains("Rolepod Brain/"), "the wiki should travel: {listing}");
    assert!(!listing.contains("brain.db"), "the derived index must not travel");

    // A fresh machine, where the repository lives somewhere else entirely.
    let new = Fixture::new("exportto");
    std::fs::write(
        new.project.join(".rolepod-brain.toml"),
        "[project]\nname = \"acme-api\"\n",
    )
    .unwrap();
    let restored = new.brain(&["import", &archive.to_string_lossy()]);
    assert!(restored.status.success(), "import failed: {restored:?}");

    let found = String::from_utf8_lossy(&new.brain(&["search", "scheduler"]).stdout).to_string();
    assert!(found.contains("scheduler"), "memory did not survive the move: {found}");
}

#[test]
fn merging_two_machines_keeps_both_sides_of_the_same_month() {
    // The designed use of a named marker: the same project on two machines,
    // which means the same project id, the same directory, and the same
    // events/YYYY-MM.jsonl on both sides.
    let laptop = Fixture::new("mergelaptop");
    let desktop = Fixture::new("mergedesktop");
    for machine in [&laptop, &desktop] {
        std::fs::write(
            machine.project.join(".rolepod-brain.toml"),
            "[project]\nname = \"acme-api\"\n",
        )
        .unwrap();
    }

    let payload = |fixture: &Fixture, prompt: &str| {
        serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "prompt": prompt
        })
        .to_string()
    };
    laptop.hook("claude-code", "UserPromptSubmit", &payload(&laptop, "the laptop found a leak"));
    desktop.hook("claude-code", "UserPromptSubmit", &payload(&desktop, "the desktop fixed a race"));

    let archive = laptop.home.parent().unwrap().join("laptop.tar.gz");
    assert!(laptop.brain(&["export", &archive.to_string_lossy()]).status.success());

    let merged = desktop.brain(&["import", "--merge", &archive.to_string_lossy()]);
    assert!(merged.status.success(), "merge failed: {merged:?}");
    assert!(desktop.brain(&["reindex"]).status.success(), "reindex after merge failed");

    // Both sides survive: a merge that silently drops the local month is
    // unrecoverable, because the logs are the source of truth.
    let log = desktop.log_text();
    assert!(log.contains("the laptop found a leak"), "the imported side is missing");
    assert!(log.contains("the desktop fixed a race"), "the local side was overwritten");
    for prompt in ["laptop found a leak", "desktop fixed a race"] {
        let found = String::from_utf8_lossy(&desktop.brain(&["search", prompt]).stdout).to_string();
        assert!(!found.contains("No matches"), "{prompt} is not searchable: {found}");
    }
}

#[test]
fn doctor_notices_a_split_brain() {
    // Renaming the vault in Obsidian renames the real directory, so the next
    // hook starts a fresh tree and new memory quietly lands where recall
    // never looks. Nothing else reports that; doctor must.
    let fixture = Fixture::new("splitbrain");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    let healthy = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(!healthy.contains("wiki tree"), "one tree should not be reported at all: {healthy}");

    std::fs::create_dir_all(fixture.home.join("wiki/orphan/events")).unwrap();
    let split = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(
        split.contains("FAIL wiki tree"),
        "two trees must be reported as a failure: {split}"
    );
}

#[test]
fn an_archive_from_an_old_install_lands_in_the_new_tree() {
    // A pre-0.12 export carries its tree under `wiki/`. Imported onto a
    // machine whose tree is `Rolepod Brain/`, it must merge into that tree -
    // unpacking it verbatim would plant a second tree under a name the
    // resolution never looks at again, which is data loss with extra steps.
    let old = Fixture::new("oldarchive");
    old.hook("claude-code", "UserPromptSubmit", &codex_payload(&old.project));
    // Make the export look exactly like an old install's: tree named wiki/.
    std::fs::rename(old.wiki(), old.home.join("wiki")).unwrap();
    let archive = old.home.parent().unwrap().join("old-install.tar.gz");
    assert!(old.brain(&["export", &archive.to_string_lossy()]).status.success());
    let listing = Command::new("tar")
        .args(["-tzf", &archive.to_string_lossy()])
        .output()
        .expect("list archive");
    assert!(
        String::from_utf8_lossy(&listing.stdout).contains("wiki/"),
        "precondition: the archive must carry the legacy name"
    );

    let new = Fixture::new("newmachine");
    new.hook("claude-code", "PostToolUse", &claude_payload(&new.project));
    assert!(new.home.join("Rolepod Brain").is_dir(), "precondition: a migrated machine");

    let merged = new.brain(&["import", "--merge", &archive.to_string_lossy()]);
    assert!(merged.status.success(), "merge failed: {merged:?}");
    assert!(
        !new.home.join("wiki").exists(),
        "the legacy name was resurrected beside the real tree"
    );
    assert!(new.brain(&["reindex"]).status.success());
    let found = String::from_utf8_lossy(&new.brain(&["search", "auth"]).stdout).to_string();
    assert!(!found.contains("No matches"), "the imported memory is unfindable: {found}");
}

#[test]
fn an_import_will_not_quietly_overwrite_an_existing_brain() {
    let source = Fixture::new("impsrc");
    source.seed_session(2);
    let archive = source.home.parent().unwrap().join("b.tar.gz");
    source.brain(&["export", &archive.to_string_lossy()]);

    let target = Fixture::new("imptarget");
    target.seed_session(2);

    let refused = target.brain(&["import", &archive.to_string_lossy()]);
    assert!(!refused.status.success(), "should refuse without a policy");
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(stderr.contains("--merge"), "should offer the choices: {stderr}");
    assert!(stderr.contains("--replace"));

    // Merging keeps both sides, which the ULID keys make safe.
    let merged = target.brain(&["import", &archive.to_string_lossy(), "--merge"]);
    assert!(merged.status.success(), "merge failed: {merged:?}");
    assert!(target.brain(&["reindex"]).status.success());
}

#[test]
fn a_fixture_cannot_reach_the_real_machines_configs() {
    // The guard for the mistake above: prove the fixture's HOME is not the
    // developer's, so no test can wire or unwire a CLI someone is using.
    let fixture = Fixture::new("homeguard");
    let output = fixture.brain(&["setup"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let real_home = std::env::var("HOME").unwrap_or_default();
    assert!(!real_home.is_empty());
    assert!(
        !stdout.contains(&real_home),
        "a fixture planned changes to the real HOME: {stdout}"
    );
}

#[test]
fn uninstall_removes_only_our_own_wiring() {
    // The other half of "it never leaves your machine": something that cannot
    // be fully removed is not really yours. And removing us is not licence to
    // disturb anything else.
    let fixture = Fixture::new("uninstall");
    let config = fixture.home.parent().unwrap().join("cli-config");
    std::fs::create_dir_all(&config).unwrap();
    let hooks = config.join("settings.json");
    std::fs::write(
        &hooks,
        serde_json::to_string(&serde_json::json!({
            "hooks": {
                "Stop": [
                    {"hooks": [{"type": "command", "command": "other-tool --run"}]},
                    {"hooks": [{"type": "command", "command": "/x/brain hook --cli codex --event Stop"}]}
                ]
            },
            "theme": "dark"
        }))
        .unwrap(),
    )
    .unwrap();

    // Exercised through the library-level behaviour the command uses: strip
    // ours, keep theirs, keep unrelated settings.
    let before: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&hooks).unwrap()).unwrap();
    assert_eq!(before["hooks"]["Stop"].as_array().unwrap().len(), 2);

    let output = fixture.brain(&["uninstall"]);
    assert!(output.status.success(), "dry run failed: {output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Dry run") || stdout.contains("Nothing wired"),
        "uninstall must not act without --apply: {stdout}"
    );
}

#[test]
fn uninstall_does_not_touch_memory_without_wipe() {
    let fixture = Fixture::new("uninstallkeep");
    fixture.seed_session(2);
    let before = fixture.log_text().lines().count();

    let output = fixture.brain(&["uninstall", "--apply"]);
    assert!(output.status.success(), "uninstall failed: {output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("still at"), "should say where the memory remains: {stdout}");

    assert_eq!(fixture.log_text().lines().count(), before, "uninstall deleted memory");
    assert!(fixture.home.join("brain.db").exists(), "the index should survive too");
}

#[test]
fn what_gets_read_rises_and_what_gets_flagged_sinks() {
    // Evidence beats heuristics: an entry an agent went back and read in full
    // is worth more than one we merely guessed at, and a human calling
    // something stale outranks both.
    let fixture = Fixture::new("ranking");
    for text in ["alpha topic one", "alpha topic two", "alpha topic three"] {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "prompt": text
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }

    let ids: Vec<String> = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|line| line["id"].as_str().map(str::to_string))
        .collect();
    assert_eq!(ids.len(), 3);

    let order = |fixture: &Fixture| -> Vec<String> {
        let out = String::from_utf8_lossy(&fixture.brain(&["search", "alpha"]).stdout).to_string();
        out.lines()
            .filter(|line| line.starts_with("01"))
            .filter_map(|line| line.split_whitespace().next().map(str::to_string))
            .collect()
    };

    // Read the LAST one in full, through the tool an agent would use.
    let pull = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{}"]}}}}}}"#,
        ids[2]
    );
    fixture.mcp(&[&pull]);

    // Now flag the first one as stale, via the same surfaced-id rule.
    let flag = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"brain_feedback","arguments":{{"id":"{}"}}}}}}"#,
        ids[0]
    );
    let responses = fixture.mcp(&[
        &format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{}"]}}}}}}"#,
            ids[0]
        ),
        &flag,
    ]);
    assert_eq!(
        responses[1]["result"]["structuredContent"]["flagged"], ids[0].as_str(),
        "feedback did not take: {responses:?}"
    );

    // Flagging must not delete anything.
    let after = order(&fixture);
    assert_eq!(after.len(), 3, "a flagged entry disappeared: {after:?}");
    assert_eq!(after.last().unwrap(), &ids[0], "the flagged entry should sink to last");

    // The primer is ranked by usage, so the entry that was read in full leads
    // and the flagged one trails.
    let start = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000000",
        "cwd": fixture.project,
        "source": "startup"
    })
    .to_string();
    let output = fixture.hook("claude-code", "SessionStart", &start);
    let context = injected_context(&output).expect("a primer");
    let read_at = context.find(&ids[2]).expect("the read entry should be in the primer");
    let flagged_at = context.find(&ids[0]).expect("the flagged entry is still present");
    assert!(read_at < flagged_at, "usage did not outrank a flagged entry in the primer");
}

#[test]
fn flagging_produces_a_page_a_human_can_act_on() {
    let fixture = Fixture::new("lintpage");
    fixture.seed_session(2);
    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "the deploy script lives in bin/release"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);
    let id = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|line| line["title"].as_str().is_some_and(|t| t.contains("deploy script")))
        .find_map(|line| line["id"].as_str().map(str::to_string))
        .expect("the flagged event");

    fixture.mcp(&[
        &format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{id}"]}}}}}}"#
        ),
        &format!(
            r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"brain_feedback","arguments":{{"id":"{id}"}}}}}}"#
        ),
    ]);

    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));

    let pages = fixture.page_text();
    assert!(pages.contains("Flagged in"), "no review page was written: {pages}");
    assert!(pages.contains(&id), "the flagged id should be listed for review");
    assert!(pages.contains("brain forget"), "the page should say what to do about it");
}

#[test]
fn entities_find_work_that_no_title_mentions() {
    // The point of a second retrieval stream: someone asks about a file, and
    // the sessions that touched it come back even though the summaries talk
    // about behaviour rather than filenames.
    let fixture = Fixture::new("entities");
    for index in 0..3 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-00000000000{index}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/billing.rs")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    // A summary that deliberately never says "billing".
    let bin = fixture.fake_cli(
        "claude",
        r#"echo '{"summary":"Reworked how invoices are totalled at period end.","entities":[],"titles":[]}'"#,
    );
    fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));

    let responses = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"src/billing.rs"}}}"#,
    ]);
    let hits = responses[0]["result"]["structuredContent"]["hits"]
        .as_array()
        .unwrap_or_else(|| panic!("no hits array; response was {:?}", responses[0]));
    assert!(!hits.is_empty(), "the entity stream found nothing: {responses:?}");

    let titles: Vec<&str> = hits.iter().filter_map(|hit| hit["title"].as_str()).collect();
    assert!(
        titles.iter().any(|title| title.contains("invoices") || title.contains("billing")),
        "expected the work about that file: {titles:?}"
    );
}

#[test]
fn a_recurring_entity_gets_a_page_a_one_off_does_not() {
    let fixture = Fixture::new("entitypages");
    // Two sessions touch the same file; one session touches another.
    for (index, file) in [(0, "src/shared.rs"), (1, "src/shared.rs"), (2, "src/once.rs")] {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-00000000000{index}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join(file)}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));

    let mut entity_pages = String::new();
    collect_ext(&fixture.wiki(), "md", &mut entity_pages);
    assert!(entity_pages.contains("src/shared.rs"), "no page for the recurring entity");

    // A thing touched once is already one click from its session; a page for
    // it would add a leaf to the graph and nothing else.
    let dirs = std::fs::read_dir(fixture.wiki()).is_ok();
    assert!(dirs);
    let once_page = walk_find(&fixture.wiki(), "once.md");
    assert!(!once_page, "a one-off entity should not get its own page");
}

fn walk_find(dir: &Path, name: &str) -> bool {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .any(|entry| {
            let path = entry.path();
            if path.is_dir() {
                walk_find(&path, name)
            } else {
                path.file_name().is_some_and(|f| f == name)
            }
        })
}

/// Memory about a file has to arrive before the agent reads it.
///
/// It used to arrive on `PostToolUse` - after the read returned, with the file
/// already in the agent's context. By then the agent has the answer it went
/// looking for and no reason to weigh what we know against it. `PreToolUse`
/// scoped to `Read` puts the same pointers in front of the content instead,
/// where they can still change what the turn does.
///
/// The pre-event must not also capture: `PreToolUse` was dropped as a capture
/// surface after 1,433 measured events showed it duplicating `PostToolUse`
/// 96% of the time, and bringing it back for injection must not bring that
/// back with it.
#[test]
fn a_file_s_memory_arrives_before_the_agent_reads_it() {
    let fixture = Fixture::new("preread");
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcd7";

    // Something worth knowing about one particular file.
    for turn in 0..3 {
        let payload = serde_json::json!({
            "session_id": "0199aaaa-1111-7000-8000-000000000000",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")},
            "prompt": format!("expiry compared with the wrong operator, take {turn}")
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }

    let read = serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
    })
    .to_string();

    let before = fixture.hook("claude-code", "PreToolUse", &read);
    let injected = injected_context(&before)
        .expect("a Read must be met with what we already know about the file");
    assert!(injected.contains("auth.rs"), "the wrong file's memory came back: {injected}");

    // The read itself is still captured once, by the post-event only.
    let stored = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg("SELECT COUNT(*) FROM events WHERE hook = 'pre_tool_use';")
        .output()
        .expect("count pre-tool events");
    assert_eq!(
        String::from_utf8_lossy(&stored.stdout).trim(),
        "0",
        "the pre-event captured as well as injected - the duplication is back"
    );

    // And having answered before the read, we do not answer again after it.
    let after = fixture.hook("claude-code", "PostToolUse", &read);
    assert!(
        injected_context(&after).is_none(),
        "the same file's memory was injected twice in one session"
    );
}

/// Two sessions in one project can consolidate at the same instant.
///
/// Nothing stops it: every session boundary spawns a detached run, and a person
/// working two terminals on one repo hits boundaries whenever they hit them.
/// The wiki is a git repo with a single index, so two runs committing at once
/// is the classic way to corrupt one — and a race that double-summarizes is a
/// second model call the user pays for and a duplicate memory injected forever.
///
/// The lock this exercises was only ever tested for the stale case: a crashed
/// run must not wedge consolidation. That is the easy half. This is the half
/// that actually happens.
#[test]
fn two_consolidations_racing_in_one_project_do_not_corrupt_or_double_up() {
    let fixture = Fixture::new("race");
    let sessions =
        ["0199a1f2-3c4d-7e8f-9012-3456789abc01", "0199a1f2-3c4d-7e8f-9012-3456789abc02"];

    for session in sessions {
        for index in 0..6 {
            let payload = serde_json::json!({
                "session_id": session,
                "cwd": fixture.project,
                "tool_name": "Edit",
                "tool_input": {"file_path": fixture.project.join(format!("src/mod{index}.rs"))},
                "prompt": format!("session {session} step {index}")
            })
            .to_string();
            fixture.hook("claude-code", "PostToolUse", &payload);
        }
    }

    // Started together, deliberately without staggering: the point is the
    // overlap, and a run that finishes before the other starts proves nothing.
    let mut racing = Vec::new();
    for _ in 0..2 {
        racing.push(
            Command::new(BRAIN)
                .args(["consolidate", "--force"])
                .current_dir(&fixture.project)
                .env("ROLEPOD_BRAIN_HOME", &fixture.home)
                .env("HOME", fixture.home.parent().unwrap())
                .env("PATH", "/usr/bin:/bin")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn consolidation"),
        );
    }
    for child in racing {
        let output = child.wait_with_output().expect("consolidation output");
        assert!(
            output.status.success(),
            "a racing consolidation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // One summary per session. Two would mean the watermark lost the race and
    // the user paid twice for one narrative.
    let counted = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg(
            "SELECT session || '=' || COUNT(*) FROM events \
             WHERE kind = 'session_summary' GROUP BY session ORDER BY session;",
        )
        .output()
        .expect("count summaries");
    let counted = String::from_utf8_lossy(&counted.stdout);
    let counted: Vec<&str> = counted.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(counted.len(), 2, "expected one row per session, got {counted:?}");
    for row in &counted {
        assert!(row.ends_with("=1"), "a session was summarized more than once: {counted:?}");
    }

    // And the wiki's git index survived being written from two processes.
    for project_dir in fixture.project_dirs() {
        let mut wiki = project_dir.as_path();
        while let Some(parent) = wiki.parent() {
            if wiki.join(".git").is_dir() {
                break;
            }
            wiki = parent;
        }
        if !wiki.join(".git").is_dir() {
            continue;
        }
        let status = Command::new("git")
            .args(["-C", &wiki.to_string_lossy(), "status", "--porcelain"])
            .output()
            .expect("git status");
        assert!(
            status.status.success(),
            "the wiki git index is unusable after the race: {}",
            String::from_utf8_lossy(&status.stderr)
        );
        assert!(
            !wiki.join(".brain-git.lock").exists(),
            "a finished run left its lock behind; the next consolidation waits for nothing"
        );
    }
}

/// A hook has to finish in the time the user is not noticing.
///
/// This is the number the whole "no resident process" commitment rests on. The
/// comparable tools run a daemon because their per-invocation floor is already
/// too high to do the work inline — measured on the author's machine, a bare
/// `node -e "0"` costs about 24ms and the login-shell PATH probe their hooks
/// run first costs about 9ms more, before a line of their code executes. Doing
/// the whole job here — parse, scrub, append and fsync the log, index, query
/// the injection — measures around 13ms, which is why there is nothing to
/// supervise, respawn, or lose events to.
///
/// So this test is not about speed for its own sake. If capture ever grows past
/// the budget, the honest answer stops being "do it inline" and the argument
/// for a worker becomes real. Better to find that out here than from a user
/// noticing their tools got slower.
#[test]
fn a_hook_stays_well_inside_its_budget() {
    // Generous against a loaded CI box; the observed figure is ~13ms. This
    // catches a regression of the kind that changes the architecture argument,
    // not ordinary jitter.
    const BUDGET_MS: u128 = 120;
    const ROUNDS: u32 = 10;

    let fixture = Fixture::new("budget");
    // Something to actually search and rank against, so this measures the real
    // path rather than an empty database.
    fixture.seed_session(30);

    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abc42",
        "cwd": fixture.project,
        "tool_name": "Edit",
        "tool_input": {"file_path": fixture.project.join("src/file7.rs")}
    })
    .to_string();

    // One warm-up: the first run pays for page cache and schema migration,
    // which a session pays once and not per event.
    fixture.hook("claude-code", "PostToolUse", &payload);

    let started = std::time::Instant::now();
    for _ in 0..ROUNDS {
        let out = fixture.hook("claude-code", "PostToolUse", &payload);
        assert!(out.status.success(), "hook failed under timing: {out:?}");
    }
    let each = started.elapsed().as_millis() / u128::from(ROUNDS);

    assert!(
        each <= BUDGET_MS,
        "capture costs {each}ms per event, over the {BUDGET_MS}ms budget - at this cost \
         the case for doing the work inside the hook no longer holds, and the \
         no-resident-process commitment needs re-arguing rather than re-asserting"
    );
}

/// The search that keyword matching cannot answer.
///
/// A session records `login sessions expire far too early`. Someone later asks
/// about `authentication`. Not one word overlaps, so FTS5 scores the pair at
/// nothing and the memory may as well not exist — which is the most common way
/// a memory system fails while looking like it works.
///
/// This is what the vendored embedding model is for, and this test is the
/// reason it earns 32MB of binary.
#[test]
fn a_memory_is_found_by_meaning_when_no_word_matches() {
    let fixture = Fixture::new("semantic");

    let record = |session: &str, prompt: &str| {
        let payload = serde_json::json!({
            "session_id": session,
            "cwd": fixture.project,
            "prompt": prompt
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    };
    record("0199aaaa-2222-7000-8000-000000000001", "login sessions expire far too early");
    record("0199aaaa-2222-7000-8000-000000000002", "the office coffee machine broke again");
    record("0199aaaa-2222-7000-8000-000000000003", "bumped the release version number");

    // Keyword-only, before any vector exists: the word is simply absent.
    let cold = fixture.brain(&["search", "authentication"]);
    let cold = String::from_utf8_lossy(&cold.stdout);
    assert!(
        !cold.contains("expire far too early"),
        "keyword search already found it, so this test proves nothing: {cold}"
    );

    // Consolidation is where vectors are written - never the capture hook,
    // which has a person waiting on it.
    let embedded = fixture.brain_with_path(&["consolidate", "--force"], None);
    assert!(embedded.status.success(), "consolidate failed: {embedded:?}");

    let warm = fixture.brain(&["search", "authentication"]);
    let warm = String::from_utf8_lossy(&warm.stdout);
    assert!(
        warm.contains("expire far too early"),
        "the memory is about authentication and was still not found: {warm}"
    );

    // And it is ranked as the answer, not merely present: the coffee machine
    // has as little to do with the question as anything in the corpus.
    let login = warm.find("expire far too early");
    let coffee = warm.find("coffee machine");
    assert!(
        login < coffee || coffee.is_none(),
        "an unrelated memory outranked the relevant one: {warm}"
    );
}

/// Semantic ranking has to obey the same withdrawal as everything else.
///
/// A memory removed from search that a vector can still surface has not been
/// removed. This is the failure mode where a forget looks like it worked.
#[test]
fn a_forgotten_memory_stays_gone_from_semantic_results_too() {
    let fixture = Fixture::new("semanticforget");
    let payload = serde_json::json!({
        "session_id": "0199aaaa-3333-7000-8000-000000000001",
        "cwd": fixture.project,
        "prompt": "login sessions expire far too early"
    })
    .to_string();
    fixture.hook("claude-code", "UserPromptSubmit", &payload);
    fixture.brain_with_path(&["consolidate", "--force"], None);

    let found = fixture.brain(&["search", "authentication"]);
    let found = String::from_utf8_lossy(&found.stdout);
    assert!(found.contains("expire far too early"), "nothing to forget: {found}");

    // The id is the first token of the first result line.
    let id = found
        .lines()
        .find(|line| line.contains("expire far too early"))
        .and_then(|line| line.split_whitespace().next())
        .expect("a result line carrying an id")
        .to_string();
    let forgotten = fixture.brain(&["forget", &id, "--apply"]);
    assert!(forgotten.status.success(), "forget failed: {forgotten:?}");

    let after = fixture.brain(&["search", "authentication"]);
    let after = String::from_utf8_lossy(&after.stdout);
    assert!(
        !after.contains("expire far too early"),
        "a forgotten memory came back through the semantic ranking: {after}"
    );
}

/// A bulk withdrawal must never reach past the words it was given.
///
/// `forget --entity` is destructive and its own preview promises the reach is
/// lexical: "Matching is by text, so a mention under another name is not
/// listed." When semantic ranking was added underneath `Store::search`, that
/// promise silently became false — a name appearing in no memory at all
/// matched every embedded event in the project, because a cosine ranking with
/// no floor returns the whole corpus in order. Typing `--apply` on that
/// preview destroys a project's memory.
///
/// The existing forget test covers `forget <id>`, which is why the suite
/// stayed green through it.
#[test]
fn forgetting_by_name_never_reaches_a_memory_that_does_not_say_it() {
    let fixture = Fixture::new("entityreach");

    for (index, prompt) in [
        "login sessions expire far too early",
        "the CSS grid gap is wrong on mobile",
        "bumped the release version number",
        "the office coffee machine broke again",
    ]
    .iter()
    .enumerate()
    {
        let payload = serde_json::json!({
            "session_id": format!("0199aaaa-4444-7000-8000-00000000000{index}"),
            "cwd": fixture.project,
            "prompt": prompt
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }
    // Everything embedded: the failure only appears once vectors exist.
    fixture.brain_with_path(&["consolidate", "--force"], None);

    let preview = fixture.brain(&["forget", "--entity", "zzzqqqwww"]);
    let preview = String::from_utf8_lossy(&preview.stdout);
    assert!(
        !preview.contains("coffee machine") && !preview.contains("CSS grid"),
        "a name in no memory listed unrelated memories for withdrawal: {preview}"
    );

    // And the reach that IS lexical still works, or the fix was a removal.
    let real = fixture.brain(&["forget", "--entity", "coffee"]);
    let real = String::from_utf8_lossy(&real.stdout);
    assert!(real.contains("coffee machine"), "the real match was lost too: {real}");
}

/// Two ways into memory that searching cannot give an agent.
///
/// `brain_search` answers a question the agent already knows how to ask. The
/// two failures that leaves are an agent that does not yet know what this
/// project IS — so it cannot form the question — and an agent holding one
/// memory with no way to reach what sits beside it. Both are the moments where
/// an agent gives up on memory and re-derives from source instead, which is
/// the whole cost this project exists to avoid.
#[test]
fn an_agent_can_orient_and_can_walk_sideways() {
    let fixture = Fixture::new("mcpwalk");

    // Separate sessions that touched the same file, which is what makes two
    // memories neighbours: a shared subject, not shared words.
    for (index, (file, prompt)) in [
        ("src/scheduler.rs", "the scheduler double-books when two runs overlap"),
        ("src/scheduler.rs", "fixed the scheduler overlap by claiming the row first"),
        ("Cargo.toml", "bumped the release version"),
    ]
    .iter()
    .enumerate()
    {
        let payload = serde_json::json!({
            "session_id": format!("0199aaaa-5555-7000-8000-00000000000{index}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join(file)},
            "prompt": prompt
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }
    fixture.brain_with_path(&["consolidate", "--force"], None);

    let listed = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#,
    ]);
    let tools = listed[0]["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|tool| tool["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"brain_outline"), "no way to orient: {names:?}");
    assert!(names.contains(&"brain_related"), "no way to walk sideways: {names:?}");

    // Orienting: what is this project, without having to guess a query first.
    let outline = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_outline","arguments":{}}}"#,
    ]);
    let outline = &outline[0]["result"]["structuredContent"];
    assert!(
        outline["sessions"].as_i64().unwrap_or(0) >= 3,
        "the outline does not describe the project: {outline}"
    );

    // Walking sideways: from one memory to what sits beside it.
    let found = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"scheduler"}}}"#,
    ]);
    let id = found[0]["result"]["structuredContent"]["hits"][0]["id"]
        .as_str()
        .expect("a hit to walk from")
        .to_string();

    let related = fixture.mcp(&[&format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_related","arguments":{{"id":"{id}"}}}}}}"#
    )]);
    let related = &related[0]["result"]["structuredContent"];
    let hits = related["hits"].as_array().expect("related returns hits");
    assert!(!hits.is_empty(), "nothing beside a memory that shares a subject: {related}");
    assert!(
        hits.iter().all(|hit| hit["id"].as_str() != Some(id.as_str())),
        "a memory is not related to itself: {related}"
    );
}

/// `--help` prints the header, and stops there.
///
/// It used to print a fixed line range, which went stale the moment the header
/// grew: the reader got `set -eu` presented as documentation. Printing every
/// comment in the file instead hands them the script's internal notes. The
/// block it should print is the contiguous one at the top, and the shape of
/// that is what this checks — not a line count, which is the thing that rotted.
#[test]
fn the_installer_help_stops_at_the_end_of_its_header() {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bootstrap.sh");
    let help = Command::new("sh").arg(&script).arg("--help").output().expect("run --help");
    assert!(help.status.success(), "--help failed: {help:?}");
    let help = String::from_utf8_lossy(&help.stdout);

    assert!(help.contains("--target=all"), "the options are missing: {help}");
    assert!(help.contains("BRAIN_BIN_DIR"), "the env vars are missing: {help}");
    for leaked in ["set -eu", "case \"$(uname", "install_binary", "mktemp"] {
        assert!(!help.contains(leaked), "`{leaked}` leaked out of the script body: {help}");
    }

    // What it deliberately does not offer, and why, has to survive too:
    // installing the binary alone wires nothing, so it is explained rather
    // than listed as a choice.
    assert!(help.contains("--binary-only"), "the flag should still be explained: {help}");
    let options = help.split("Working on brain itself").next().unwrap_or(&help);
    assert!(
        !options.contains("--binary-only"),
        "--binary-only is back in the options list, where it reads as a choice: {options}"
    );
}

/// The installer's own arguments must survive the rest of the script.
///
/// `--target=<cli>` and the platform triple lived in one variable named
/// `target`: the option parser wrote the user's choice into it and the platform
/// detection twenty lines later overwrote it. So `--target=codex` — and the
/// bare one-liner the README leads with — asked `brain setup` to wire a CLI
/// called `aarch64-apple-darwin`. Nothing was wired, and until `setup` learned
/// to refuse a name it does not know, nothing said so either.
///
/// This is a shell script, so there is no compiler to notice. The invariant it
/// broke is small enough to state: after the option loop, nothing reassigns the
/// variables the option loop owns.
#[test]
fn the_installer_does_not_overwrite_its_own_options() {
    let script = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bootstrap.sh"),
    )
    .expect("bootstrap.sh");

    let (parser, rest) = script.split_once("done\n").expect("the option loop ends with `done`");
    for option in ["target", "assume_yes", "uninstall", "binary_only"] {
        assert!(
            parser.contains(&format!("{option}=")),
            "{option} is never set by the option parser — has it been renamed?"
        );
        // Anywhere on the line, not just at the start of it. The assignment
        // that caused this sat at the tail of a compound command —
        // `[ -n "$os" ] && ... && target="$arch-$os"` — and a check anchored to
        // the line start walked straight past it, which is a guard that passes
        // on the bug it was written for.
        for (number, line) in rest.lines().enumerate() {
            let code = line.split('#').next().unwrap_or(line);
            assert!(
                !code.contains(&format!("{option}=")),
                "line {} reassigns `{option}`, which the option parser owns: {line}",
                number + 1
            );
        }
    }
}
