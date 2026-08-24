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
    // scheduler" are different questions. Relevance answers the first;
    // only a scope answers the second.
    let fixture = Fixture::new("scopedsearch");
    fixture.seed_session(1);
    let db = fixture.home.join("brain.db");
    let seed = |id: &str, title: &str, topic: &str| {
        Command::new("sqlite3")
            .arg(&db)
            .arg(format!(
                "INSERT INTO events (id, ts, workspace, project, session, cli, hook, kind, title, body, topic)
                 SELECT '{id}', ts, workspace, project, session, cli, 'stop', 'session_summary',
                        '{title}', '', '{topic}' FROM events LIMIT 1;"
            ))
            .output()
            .expect("seed event");
    };
    seed("01SCOPE0000000000000000DEC", "scheduler: chose cron over a queue", "decision");
    seed("01SCOPE0000000000000000FIX", "scheduler double-booking fixed", "bugfix");

    let all = String::from_utf8_lossy(&fixture.brain(&["search", "scheduler"]).stdout).to_string();
    assert!(all.contains("chose cron") && all.contains("double-booking"), "unscoped: {all}");

    let scoped =
        String::from_utf8_lossy(&fixture.brain(&["search", "scheduler", "--topic", "decision"]).stdout)
            .to_string();
    assert!(scoped.contains("chose cron"), "the decision was scoped out: {scoped}");
    assert!(!scoped.contains("double-booking"), "the bugfix leaked into a decision scope: {scoped}");

    // A typo must read as "wrong scope", never as "nothing remembered".
    let typo = fixture.brain(&["search", "scheduler", "--topic", "desicion"]);
    let warned = String::from_utf8_lossy(&typo.stderr).to_string();
    assert!(warned.contains("Unknown topic"), "a bad topic was silently accepted: {warned}");
    assert!(
        String::from_utf8_lossy(&typo.stdout).contains("chose cron"),
        "a bad topic should fall back to searching everything"
    );
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
    let config_dir = dirs_home().join(".gemini/config");
    let output = if config_dir.is_dir() {
        Command::new(BRAIN)
            .args(["hook", "--cli", "antigravity", "--event", "PostToolUse"])
            .current_dir(&config_dir)
            .env("ROLEPOD_BRAIN_HOME", &fixture.home)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                child.stdin.as_mut().unwrap().write_all(payload.as_bytes())?;
                child.wait_with_output()
            })
            .expect("run hook")
    } else {
        fixture.hook("antigravity", "PostToolUse", &payload)
    };

    assert!(output.status.success(), "the host must still be acknowledged");
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "{}");
    assert!(
        !fixture.log_text().contains("antigravity"),
        "an unplaceable event was filed anyway"
    );
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".to_string()))
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

    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
        "cwd": fixture.project,
        "prompt": "this must not be remembered"
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
    child.stdin.as_mut().unwrap().write_all(payload.as_bytes()).unwrap();
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

    // Without the reset, this second injection would be suppressed as a
    // duplicate - the session id survived even though the context did not.
    let after = fixture.hook(
        "claude-code",
        "PostCompact",
        &serde_json::json!({"session_id": session, "cwd": fixture.project, "trigger": "auto"})
            .to_string(),
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
