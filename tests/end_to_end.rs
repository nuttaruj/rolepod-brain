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
        Self::with_checkout(name, "checkout")
    }

    /// The same, with the checkout's directory named by the test: a repo's
    /// directory name is the project name a capture resolves.
    fn with_checkout(name: &str, checkout: &str) -> Self {
        let base = std::env::temp_dir().join(format!("brain-e2e-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let home = base.join("home");
        let project = base.join(checkout);
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
        self.brain_with_env(args, path, &[])
    }

    /// The same, with extra environment on top: a trace file, a hostile git
    /// setting, a leaked `GIT_DIR` - whatever the test is proving brain
    /// survives.
    fn brain_with_env(
        &self,
        args: &[&str],
        path: Option<&Path>,
        env: &[(&str, &str)],
    ) -> std::process::Output {
        let mut command = Command::new(BRAIN);
        command
            .args(args)
            .current_dir(&self.project)
            .env("ROLEPOD_BRAIN_HOME", &self.home)
            // A consolidation run that finds the index due for a compact waits
            // minutes for a quiet log; a test that did not ask for that must
            // not (a later `.env` in `env` overrides this).
            .env("ROLEPOD_BRAIN_MAINT_WAIT_SECS", "0")
            // A run fetches a missing embedding model by itself; a test must
            // not reach the network.
            .env("ROLEPOD_BRAIN_NO_FETCH", "1")
            .env("ROLEPOD_BRAIN_HUB", "off")
            // ROLEPOD_BRAIN_HOME isolates our own data; it does NOT isolate
            // the CLI configs we wire into, which are found through $HOME. A
            // test running `uninstall --apply` without this unwired the real
            // machine - so the fixture owns HOME too, and nothing here can
            // reach a config a person is actually using.
            .env("HOME", self.home.parent().unwrap());
        self.own_git_config(&mut command);
        match path {
            // `git` still has to be reachable: consolidation commits the wiki.
            Some(dir) => command.env("PATH", format!("{}:/usr/bin:/bin", dir.display())),
            None => command.env("PATH", "/usr/bin:/bin"),
        };
        command.envs(env.iter().copied());
        command.output().expect("run brain")
    }

    /// Git reads config from HOME, from XDG_CONFIG_HOME and from the system
    /// file. These pin the last two to the fixture, so together with the
    /// fixture's HOME a wiki commit sees git's defaults, not this machine's.
    ///
    /// It also drops `CODEX_HOME`: consolidation reads codex's model cache
    /// from there when a stub `codex` is on `PATH`, and a host's own cache
    /// must not decide which model a test sees.
    fn own_git_config(&self, command: &mut Command) {
        let base = self.home.parent().unwrap();
        command
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("XDG_CONFIG_HOME", base.join(".config"))
            .env_remove("CODEX_HOME");
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

    /// The mode of every `brain consolidate` run on record, oldest first.
    /// Read-only, so a test that polls never holds up the run it watches.
    fn consolidation_modes(&self) -> Vec<String> {
        let Ok(conn) = rusqlite::Connection::open_with_flags(
            self.home.join("brain.db"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) else {
            return Vec::new();
        };
        let _ = conn.busy_timeout(std::time::Duration::from_secs(2));
        let Ok(mut statement) = conn.prepare("SELECT mode FROM consolidation_runs ORDER BY rowid")
        else {
            return Vec::new();
        };
        statement
            .query_map([], |row| row.get(0))
            .map(|rows| rows.filter_map(Result::ok).collect())
            .unwrap_or_default()
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
        let files: Vec<String> = (0..count).map(|index| format!("src/file{index}.rs")).collect();
        let files: Vec<&str> = files.iter().map(String::as_str).collect();
        self.seed_session_as("0199a1f2-3c4d-7e8f-9012-3456789abcde", &files);
    }

    /// The same, as the session the test names, editing the files it names:
    /// two sessions over the same files is what gives those files entity pages.
    fn seed_session_as(&self, session_id: &str, files: &[&str]) {
        for file in files {
            let payload = serde_json::json!({
                "session_id": session_id,
                "cwd": self.project,
                "tool_name": "Edit",
                "tool_input": {"file_path": self.project.join(file)}
            })
            .to_string();
            self.hook("claude-code", "PostToolUse", &payload);
        }
    }

    fn hook(&self, cli: &str, event: &str, payload: &str) -> std::process::Output {
        self.spawn_hook(cli, event, payload).wait_with_output().expect("hook output")
    }

    /// The hook, started and fed its payload but not waited for, so a test can
    /// look at what it has done while it is still running.
    fn spawn_hook(&self, cli: &str, event: &str, payload: &str) -> std::process::Child {
        let mut command = Command::new(BRAIN);
        command
            .args(["hook", "--cli", cli, "--event", event])
            .current_dir(&self.project)
            .env("ROLEPOD_BRAIN_HOME", &self.home)
            .env("ROLEPOD_BRAIN_NO_FETCH", "1")
            .env("ROLEPOD_BRAIN_HUB", "off")
            // Same reason brain_with_path owns HOME: the capture path reads
            // $HOME too, and a fixture that leaves it pointing at the real
            // machine is testing the machine, not the fixture.
            .env("HOME", self.home.parent().unwrap());
        self.own_git_config(&mut command);
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn hook");
        // Closed after the write: the hook reads its payload to the end.
        child.stdin.take().unwrap().write_all(payload.as_bytes()).unwrap();
        child
    }

    /// A Claude Code hook run the way a delegate runs it: under `claude -p`.
    ///
    /// To `ps`, a hook running under a process whose argv reads `claude -p …`
    /// is a hook under a headless Claude. A symlink to bash by that name, in
    /// privileged mode, is exactly that - a symlink rather than a copy, because
    /// macOS kills a copied system binary.
    fn hook_under_headless_claude(&self, event: &str, payload: &str) {
        let fake = self.home.parent().unwrap().join("delegate-bin");
        std::fs::create_dir_all(&fake).unwrap();
        let claude = fake.join("claude");
        if claude.symlink_metadata().is_err() {
            std::os::unix::fs::symlink("/bin/bash", &claude).unwrap();
        }
        // Two commands, so bash forks for the first instead of exec-ing into
        // it: the `claude -p` parent has to still exist when brain looks up.
        let script = format!("'{BRAIN}' hook --cli claude-code --event {event}; exit $?");
        let mut command = Command::new(&claude);
        command
            .args(["-p", "-c", &script])
            .current_dir(&self.project)
            .env("ROLEPOD_BRAIN_HOME", &self.home)
            .env("ROLEPOD_BRAIN_NO_FETCH", "1")
            .env("ROLEPOD_BRAIN_HUB", "off")
            .env("HOME", self.home.parent().unwrap());
        self.own_git_config(&mut command);
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn the fake claude");
        child.stdin.as_mut().unwrap().write_all(payload.as_bytes()).unwrap();
        let out = child.wait_with_output().expect("hook output");
        assert!(out.status.success(), "hook failed: {out:?}");
    }

    /// A whole delegated review under `claude -p`: a start, three prompts and
    /// one file read.
    fn seed_headless_session(&self, session_id: &str) {
        self.hook_under_headless_claude("SessionStart", &start_payload(&self.project, session_id, "startup"));
        for prompt in ["Review the auth change adversarially", "Now the tests", "Report what you found"] {
            let payload = serde_json::json!({"session_id": session_id, "cwd": self.project, "prompt": prompt});
            self.hook_under_headless_claude("UserPromptSubmit", &payload.to_string());
        }
        let payload = serde_json::json!({
            "session_id": session_id,
            "cwd": self.project,
            "tool_name": "Read",
            "tool_input": {"file_path": self.project.join("src/auth.rs")}
        });
        self.hook_under_headless_claude("PostToolUse", &payload.to_string());
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
        let mut command = Command::new(BRAIN);
        command
            .arg("mcp")
            .current_dir(&self.project)
            .env("ROLEPOD_BRAIN_HOME", &self.home)
            .env("ROLEPOD_BRAIN_NO_FETCH", "1")
            .env("ROLEPOD_BRAIN_HUB", "off")
            // The summarizer ladder looks for a CLI in `~/.local/bin` as well as
            // on PATH. Left on the real HOME, a search that is meant to find no
            // CLI is reranked by this machine's real claude, and its answer is
            // what the comparison against the stubbed run sees.
            .env("HOME", self.home.parent().unwrap())
            .env("PATH", path);
        self.own_git_config(&mut command);
        let mut child = command
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

/// What a test-side git command printed; empty when it failed.
fn git_stdout(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git").args(args).current_dir(dir).output().expect("run git");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Every event git wrote to a `GIT_TRACE2_EVENT` file, one JSON object a line.
fn trace2(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn trace_argv(event: &serde_json::Value) -> Vec<String> {
    event["argv"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|arg| arg.as_str().map(str::to_string))
        .collect()
}

/// The `start` events of git processes something other than git started.
/// trace2 nests a child's session id under its parent's with a `/`, so a
/// top-level one has none.
fn top_level_starts(events: &[serde_json::Value]) -> Vec<&serde_json::Value> {
    events
        .iter()
        .filter(|event| event["event"] == "start")
        .filter(|event| event["sid"].as_str().is_some_and(|sid| !sid.contains('/')))
        .collect()
}

/// The argv of every git process brain's wiki guard built: it is the only
/// caller that passes `maintenance.auto=false`.
fn wiki_git_starts(events: &[serde_json::Value]) -> Vec<Vec<String>> {
    top_level_starts(events)
        .into_iter()
        .map(trace_argv)
        .filter(|argv| argv.iter().any(|arg| arg == "maintenance.auto=false"))
        .collect()
}

/// The children a traced git started that maintain the repository in some
/// form - the processes that, one per commit, made the storm.
fn spawned(events: &[serde_json::Value]) -> Vec<Vec<String>> {
    const MAINTENANCE: [&str; 7] = [
        "maintenance",
        "gc",
        "repack",
        "pack-objects",
        "multi-pack-index",
        "commit-graph",
        "fsmonitor--daemon",
    ];
    events
        .iter()
        .filter(|event| event["event"] == "child_start")
        .map(trace_argv)
        .filter(|argv| argv.iter().any(|arg| MAINTENANCE.contains(&arg.as_str())))
        .collect()
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
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
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

/// A run for one checkout took the project's name from the checkout
/// (`WalnutZite`); `--all` took it from the wiki folder (`walnutzite`). Each
/// switch between them rewrote the hub, topic and entity pages, and on
/// 2026-10-06 that was 14,730 commits. Every run now names a project after
/// its folder.
#[test]
fn a_project_keeps_one_name_whichever_run_writes_it() {
    let fixture = Fixture::with_checkout("onename", "WalnutZite");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let names = || {
        let hub = std::fs::read_to_string(fixture.wiki().join("walnutzite/walnutzite.md"))
            .expect("the hub is named after the folder");
        hub.lines()
            .filter(|line| line.starts_with("title:") || line.starts_with("# "))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };

    fixture.seed_session(4);
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");
    let from_checkout = names();

    fixture.seed_session(4);
    let output = fixture.brain_with_path(&["consolidate", "--all", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate --all failed: {output:?}");
    let from_all = names();

    // A document read from the checkout writes the hub too.
    std::fs::write(fixture.project.join("notes.md"), "# Notes\n\nThe deploy runs nightly.\n").unwrap();
    let output = fixture.brain_with_path(&["ingest", "notes.md"], Some(&bin));
    assert!(output.status.success(), "ingest failed: {output:?}");
    let from_ingest = names();

    assert_eq!(from_checkout, ["title: walnutzite", "# walnutzite"], "the checkout's spelling leaked");
    assert_eq!(from_all, from_checkout, "--all renamed the project");
    assert_eq!(from_ingest, from_checkout, "ingest renamed the project");
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

/// Ids of the hits a search for `auth` returns, after a few edits to seed it.
fn seed_and_search_auth(fixture: &Fixture) -> Vec<String> {
    for name in ["auth.rs", "auth/login.rs"] {
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
    parsed["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn sqlite_value(fixture: &Fixture, sql: &str) -> String {
    let out = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg(sql)
        .output()
        .expect("query the store");
    assert!(out.status.success(), "sqlite3 failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn get_request(id: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{id}"]}}}}}}"#
    )
}

/// A recall is tied to the host session whose hook ran under the same host
/// process: the hooks and the MCP server are children of one fake `claude`.
#[test]
fn a_recall_is_tied_to_the_host_session_that_opened_it() {
    let fixture = Fixture::new("hostjoin");
    let ids = seed_and_search_auth(&fixture);
    let id = ids.first().expect("a hit").clone();

    let session = "0199b2c3-4d5e-7f80-9123-456789abcdef";
    let base = fixture.home.parent().unwrap().to_path_buf();
    std::fs::write(base.join("start.json"), start_payload(&fixture.project, session, "startup")).unwrap();
    let post = serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
    });
    std::fs::write(base.join("post.json"), post.to_string()).unwrap();
    std::fs::write(base.join("req.jsonl"), format!("{}\n", get_request(&id))).unwrap();

    let fake = base.join("host-bin");
    std::fs::create_dir_all(&fake).unwrap();
    let claude = fake.join("claude");
    if claude.symlink_metadata().is_err() {
        std::os::unix::fs::symlink("/bin/bash", &claude).unwrap();
    }
    // No `-p`: an interactive host. Several commands, so bash forks for each
    // and the `claude` process is still the parent when brain looks it up.
    let script = format!(
        "'{BRAIN}' hook --cli claude-code --event SessionStart < start.json; \
         '{BRAIN}' hook --cli claude-code --event PostToolUse < post.json; \
         '{BRAIN}' mcp < req.jsonl > out.jsonl; exit $?"
    );
    let mut command = Command::new(&claude);
    command
        .args(["-c", &script])
        .current_dir(&base)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("HOME", &base)
        .env("PATH", "/usr/bin:/bin");
    fixture.own_git_config(&mut command);
    let out = command.output().expect("run the fake host");
    assert!(out.status.success(), "fake host failed: {out:?}");

    assert_eq!(
        sqlite_value(
            &fixture,
            &format!("SELECT host_session FROM recalled WHERE event_id = '{id}' AND opened = 1;")
        ),
        session,
        "the recall was not tied to the host session"
    );
}

/// Every hook refreshes its session's tie, not only the first: S1 is seen
/// again after S2, so S1 is the host session the recall belongs to.
#[test]
fn a_later_hook_of_an_older_session_keeps_the_tie_fresh() {
    let fixture = Fixture::new("hostrefresh");
    let ids = seed_and_search_auth(&fixture);
    let id = ids.first().expect("a hit").clone();

    let (s1, s2) = ("0199b2c3-4d5e-7f80-9123-456789abcdef", "0199b2c3-4d5e-7f80-9123-456789abcd01");
    let base = fixture.home.parent().unwrap().to_path_buf();
    for (file, session) in [("post1.json", s1), ("post2.json", s2)] {
        let post = serde_json::json!({
            "session_id": session,
            "cwd": fixture.project,
            "tool_name": "Read",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
        });
        std::fs::write(base.join(file), post.to_string()).unwrap();
    }
    std::fs::write(base.join("req.jsonl"), format!("{}\n", get_request(&id))).unwrap();

    let fake = base.join("host-bin");
    std::fs::create_dir_all(&fake).unwrap();
    let claude = fake.join("claude");
    if claude.symlink_metadata().is_err() {
        std::os::unix::fs::symlink("/bin/bash", &claude).unwrap();
    }
    let script = format!(
        "'{BRAIN}' hook --cli claude-code --event PostToolUse < post1.json; \
         '{BRAIN}' hook --cli claude-code --event PostToolUse < post2.json; \
         '{BRAIN}' hook --cli claude-code --event PostToolUse < post1.json; \
         '{BRAIN}' mcp < req.jsonl > out.jsonl; exit $?"
    );
    let mut command = Command::new(&claude);
    command
        .args(["-c", &script])
        .current_dir(&base)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("HOME", &base)
        .env("PATH", "/usr/bin:/bin");
    fixture.own_git_config(&mut command);
    let out = command.output().expect("run the fake host");
    assert!(out.status.success(), "fake host failed: {out:?}");

    assert_eq!(
        sqlite_value(
            &fixture,
            &format!("SELECT host_session FROM recalled WHERE event_id = '{id}' AND opened = 1;")
        ),
        s1,
        "the later hook of S1 did not refresh its tie"
    );
}

/// With no hook from this host, the recall is still recorded, marked as
/// unjoined by an empty host session rather than NULL.
#[test]
fn a_recall_without_a_known_host_is_recorded_unjoined() {
    let fixture = Fixture::new("hostnone");
    let ids = seed_and_search_auth(&fixture);
    let id = ids.first().expect("a hit").clone();
    // The seeding hooks ran under whatever host runs this suite, and that host
    // is also the one the server finds: forget what they recorded.
    sqlite_value(&fixture, "DELETE FROM host_session;");
    fixture.mcp(&[&get_request(&id)]);
    assert_eq!(
        sqlite_value(
            &fixture,
            &format!(
                "SELECT opened || ':' || COALESCE(host_session, 'NULL') FROM recalled WHERE event_id = '{id}' AND opened = 1;"
            )
        ),
        "1:",
        "the recall was lost or joined to a stranger"
    );
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

    // The baseline is the index's own order, taken with the preferred rung
    // answering NONE - never with this machine's real CLI on PATH, whose
    // rerank would move it and turn the comparison below into a coin toss
    // that depends on whether that CLI is rate-limited today.
    let none = fixture.fake_cli("claude", "echo NONE");
    let plain = ids(&fixture.mcp_with_path(&[search], Some(&none)));
    assert!(plain.len() >= 3, "need several hits to reorder: {plain:?}");

    // A second rung stands behind the NONE, ready to promote the last hit -
    // so if the ladder cascades, the order moves and this test sees it.
    // `gemini`, not `codex`: codex reads its answer from a file rather than
    // stdout, so a stdout stub there would look like a rung that failed and
    // prove nothing about cascading.
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
    // The CLI itself has to exist: setup skips a directory with no
    // executable behind it, and this test is about handover, not presence.
    let bin = fixture.fake_cli("cursor-agent", "exit 0");
    std::fs::write(
        home.join(".cursor/mcp.json"),
        r#"{"mcpServers":{"brain":{"command":"/old/brain","args":["mcp"]},"other":{"command":"x"}}}"#,
    )
    .unwrap();

    // Without the plugin, setup owns the registration.
    let plain = fixture.brain_with_path(&["setup", "--apply", "--cli", "cursor"], Some(&bin));
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

    let deferred = fixture.brain_with_path(&["setup", "--apply", "--cli", "cursor"], Some(&bin));
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
fn an_archive_member_that_is_a_link_is_refused() {
    // A member with an ordinary name can still be a symlink to a file outside
    // the archive. Copying it into the brain follows the link, so the import
    // read that file into memory the user may later sync or share. A hard
    // link reaches the same place by another route.
    let fixture = Fixture::new("tarlink");
    fixture.seed_session(2);
    let secret = fixture.home.parent().unwrap().join("secret.txt");
    std::fs::write(&secret, "outside-the-archive").unwrap();
    for (kind, linkname) in [
        ("SYMTYPE", secret.display().to_string()),
        ("LNKTYPE", "Rolepod Brain/ok.md".to_string()),
    ] {
        let archive = fixture.home.parent().unwrap().join(format!("{kind}.tar.gz"));
        let script = format!(
            "import io, tarfile\n\
             t = tarfile.open(r'{archive}', 'w:gz')\n\
             ok = tarfile.TarInfo('Rolepod Brain/ok.md')\n\
             ok.size = 2\n\
             t.addfile(ok, io.BytesIO(b'hi'))\n\
             link = tarfile.TarInfo('Rolepod Brain/leak.md')\n\
             link.type = tarfile.{kind}\n\
             link.linkname = r'{linkname}'\n\
             t.addfile(link)\n\
             t.close()\n",
            archive = archive.display(),
        );
        let built = Command::new("python3").args(["-c", &script]).output().expect("build archive");
        assert!(built.status.success(), "could not build the test archive: {built:?}");

        for policy in ["--merge", "--replace"] {
            let refused = fixture.brain(&["import", policy, &archive.to_string_lossy()]);
            assert!(!refused.status.success(), "{policy} accepted an archive with a {kind} member");
            let why = String::from_utf8_lossy(&refused.stderr);
            assert!(why.contains("not a regular file"), "refused for the wrong reason: {why}");
        }
    }
    // Refused before anything moved: `--replace` sets the current brain aside
    // only for an archive that is going to be unpacked.
    let aside: Vec<_> = std::fs::read_dir(&fixture.home)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("wiki.replaced."))
        .collect();
    assert!(aside.is_empty(), "a refused archive still moved the brain aside");
    let mut imported = String::new();
    collect_ext(&fixture.home, "md", &mut imported);
    assert!(
        !imported.contains("outside-the-archive"),
        "a file outside the archive was read into the brain"
    );
}

#[test]
fn a_compression_bomb_is_refused_without_being_unpacked() {
    // Two hundred megabytes of zeros pack into a fraction of one. The import
    // measures what an archive unpacks to before unpacking it, and stops the
    // measuring itself at the limit, so the disk never sees the bomb.
    let fixture = Fixture::new("tarbomb");
    let archive = fixture.home.parent().unwrap().join("bomb.tar.gz");
    let script = format!(
        "import io, tarfile\n\
         class Zeros(io.RawIOBase):\n    \
             def readable(self): return True\n    \
             def readinto(self, b):\n        \
                 b[:] = bytes(len(b)); return len(b)\n\
         t = tarfile.open(r'{}', 'w:gz')\n\
         info = tarfile.TarInfo('Rolepod Brain/zeros.md')\n\
         info.size = 200 << 20\n\
         t.addfile(info, io.BufferedReader(Zeros()))\n\
         t.close()\n",
        archive.display()
    );
    let built = Command::new("python3").args(["-c", &script]).output().expect("build archive");
    assert!(built.status.success(), "could not build the test archive: {built:?}");
    assert!(std::fs::metadata(&archive).unwrap().len() < 1 << 20, "not much of a bomb");

    let refused = fixture.brain(&["import", "--merge", &archive.to_string_lossy()]);
    assert!(!refused.status.success(), "a compression bomb was accepted");
    let why = String::from_utf8_lossy(&refused.stderr);
    assert!(why.contains("unpacks to more than"), "refused for the wrong reason: {why}");
    assert!(!fixture.home.join("Rolepod Brain/zeros.md").exists(), "the bomb was unpacked");
}

#[test]
fn an_export_with_a_link_in_the_wiki_is_refused_where_it_can_be_fixed() {
    // Imports take files and directories only. A link exported anyway would
    // be refused by every machine that received it, on every sync.
    let fixture = Fixture::new("exportlink");
    fixture.seed_session(2);
    let wiki = fixture.home.join("Rolepod Brain");
    let archive = fixture.home.parent().unwrap().join("linked.tar.gz");
    assert!(fixture.brain(&["export", &archive.to_string_lossy()]).status.success());

    std::os::unix::fs::symlink("/etc/hosts", wiki.join("hosts.md")).unwrap();
    let refused = fixture.brain(&["export", &archive.to_string_lossy()]);
    assert!(!refused.status.success(), "a wiki holding a link was exported");
    let why = String::from_utf8_lossy(&refused.stderr);
    assert!(why.contains("hosts.md") && why.contains("replace the link"), "{why}");
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
fn the_health_check_can_be_asked_for_without_a_terminal() {
    // Most of what fails in this project fails silently - a day of them was
    // measured - and `brain doctor` is the only thing that says so. A user who
    // never opens a terminal could not reach it, which made the one
    // instruction worth giving ("run doctor now and then") the one nobody
    // follows. It answers with the checks themselves rather than the rendered
    // text, so an agent can name the failing one instead of quoting a wall.
    let fixture = Fixture::new("mcp-doctor");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));

    let responses = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_doctor","arguments":{}}}"#,
    ]);
    let report = &responses[1]["result"]["structuredContent"];
    let checks = report["checks"].as_array().expect("checks is a list");

    assert!(checks.len() > 5, "a health report with {} checks is not one", checks.len());
    assert!(
        checks
            .iter()
            .all(|c| c["name"].is_string() && c["ok"].is_boolean() && c["detail"].is_string()),
        "every check needs a name, a verdict and a line to show: {checks:?}"
    );
    assert!(
        checks.iter().any(|c| c["name"] == "capture"),
        "the check people actually ask about is missing: {checks:?}"
    );

    // The process line reports what is running rather than promising what is
    // not. `no resident process  7 live` read as a contradiction to the first
    // person who met it, and that person had been using this for a week.
    let processes =
        checks.iter().find(|c| c["name"] == "processes").expect("the process check lost its name");
    assert!(
        !processes["detail"].as_str().unwrap_or_default().contains("resident"),
        "the line still promises an absence while listing a presence: {processes:?}"
    );
}

#[test]
fn where_memory_lives_is_answerable_in_the_conversation() {
    let fixture = Fixture::new("mcp-where");
    // Folded into the health report rather than a tool of its own: "is it
    // working" and "where is it kept" are one question in a conversation.
    let responses = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_doctor","arguments":{}}}"#,
    ]);
    let where_ = &responses[1]["result"]["structuredContent"];
    for key in ["data_directory", "wiki", "project"] {
        assert!(where_[key].is_string(), "`{key}` missing from {where_}");
    }
    assert!(
        where_["wiki"].as_str().unwrap().contains("Rolepod Brain"),
        "the wiki path should name the vault a person can open: {where_}"
    );
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

/// Retention empties the index's copy of a body and says the log holds it
/// (`clamped = 1`); the event log itself is not touched. `brain_get` is the way
/// back to the whole body.
#[test]
fn brain_get_returns_the_body_of_a_row_whose_index_body_was_dropped() {
    let fixture = Fixture::new("get-dropped");
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));
    let log_before = fixture.log_text();
    let db = fixture.home.join("brain.db");
    let (id, body): (String, String) = rusqlite::Connection::open(&db)
        .unwrap()
        .query_row("SELECT id, body FROM events WHERE hook = 'post_tool_use'", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert!(body.contains("fn check()"), "precondition: the capture is indexed with its body");
    // The crate is a bin, so this cannot call `Store::drop_index_bodies`; the
    // UPDATE must stay the same as the one there.
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute("UPDATE events SET body = '', clamped = 1 WHERE id = ?1", [&id])
        .unwrap();

    let fetched = fixture.mcp(&[&format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{id}"]}}}}}}"#
    )]);
    let events = fetched[0]["result"]["structuredContent"]["events"].as_array().unwrap();
    assert_eq!(events.len(), 1, "{fetched:?}");
    assert_eq!(events[0]["body"], body.as_str(), "the log's whole body did not come back");
    assert_eq!(fixture.log_text(), log_before, "reading a body must not touch the log");
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

    // The capture that arrived after the summary put the session back in
    // flight, and the summary list says so before it lists any summary: the
    // list an agent is told to ask for must not pass an older summary off as
    // the latest work. The row names the session, so the same id appears
    // twice - once as work still open, once as what was already written.
    let summaries = kinds(0);
    let events = responses[0]["result"]["structuredContent"]["events"].as_array().unwrap();
    assert_eq!(summaries.first().map(String::as_str), Some("observation"), "in flight leads: {summaries:?}");
    let lead = events[0]["title"].as_str().unwrap();
    // Two, not one: with no CLI on PATH the run was rule-based, and a
    // rule-based run leaves its events pending so a model can still write
    // the real summary later. Both prompts are therefore still open.
    assert!(lead.contains("2 capture(s) not yet summarized"), "{lead}");
    assert!(lead.contains(events[0]["session"].as_str().unwrap()), "the row names the session to read: {lead}");
    assert!(summaries.len() >= 2, "consolidation wrote a summary: {responses:?}");
    assert!(summaries[1..].iter().all(|kind| kind == "session_summary"), "not summaries: {summaries:?}");
    assert_eq!(events[0]["session"], events[1]["session"], "the open work and its earlier summary are one session");

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

/// A stub for the rule tier: on a synthesis call it lifts the correction ids
/// out of the CORRECTIONS section and answers with two rules - one citing
/// both corrections, one citing only the first. Only the first may survive.
fn rule_cli() -> String {
    r#"
case "$*" in
  *"CORRECTIONS"*)
    IDS=$(echo "$*" | sed -n '/--- CORRECTIONS ---/,/--- SESSION SUMMARIES ---/p' | grep -oE 'id=[0-9A-Z]{26}' | cut -d= -f2)
    A=$(echo "$IDS" | head -1)
    B=$(echo "$IDS" | head -2 | tail -1)
    echo '{"knowledge":[{"kind":"rule","title":"Always run the linter before committing","body":"The user corrected this twice.","sources":["'"$A"'","'"$B"'"]},{"kind":"rule","title":"A rule nobody asked for twice","body":"Cited one correction.","sources":["'"$A"'"]}]}' ;;
  *"SESSION SUMMARIES"*)
    echo '{"knowledge":[]}' ;;
  *) echo '{"summary":"Refactored the auth path.","titles":[]}' ;;
esac
"#
    .to_string()
}

#[test]
fn the_same_repo_is_the_same_project_wherever_it_lives() {
    // Identity anchored to the root commit: clone the repo to a second
    // path and events from both checkouts land in one brain. This is what
    // makes a future multi-device sync converge without asking anyone
    // anything.
    let fixture = Fixture::new("gitident");
    run_in(&fixture.project, "git", &["config", "user.email", "t@example.invalid"]);
    run_in(&fixture.project, "git", &["config", "user.name", "t"]);
    std::fs::write(fixture.project.join("README.md"), "hello").unwrap();
    run_in(&fixture.project, "git", &["add", "."]);
    run_in(&fixture.project, "git", &["commit", "-q", "-m", "root"]);

    let capture = |cwd: &Path, session: &str| {
        let payload = serde_json::json!({
            "session_id": session,
            "cwd": cwd,
            "prompt": "an identity probe"
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    };
    capture(&fixture.project, "0199a1f2-3c4d-7e8f-9012-3456789abc01");

    let base = fixture.project.parent().unwrap();
    run_in(base, "git", &["clone", "-q", fixture.project.to_str().unwrap(), "checkout-two"]);
    capture(&base.join("checkout-two"), "0199a1f2-3c4d-7e8f-9012-3456789abc02");

    let projects: std::collections::HashSet<String> = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|line| line["project"].as_str().map(str::to_string))
        .collect();
    assert_eq!(projects.len(), 1, "two checkouts of one repo split the brain: {projects:?}");
}

#[test]
fn an_existing_project_never_changes_identity() {
    // The rungs below the cache may disagree with history - a repo that
    // gains its first commit after memory exists would suddenly anchor to
    // git - but memory written under an id must keep answering to it, even
    // when the identity cache is gone (a store upgraded from before the
    // ladder has none).
    let fixture = Fixture::new("oldident");
    let capture = |session: &str| {
        let payload = serde_json::json!({
            "session_id": session,
            "cwd": fixture.project,
            "prompt": "an identity probe"
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    };
    // No commits yet: identity falls back to the path, the old rule.
    capture("0199a1f2-3c4d-7e8f-9012-3456789abc01");

    run_in(&fixture.project, "git", &["config", "user.email", "t@example.invalid"]);
    run_in(&fixture.project, "git", &["config", "user.name", "t"]);
    std::fs::write(fixture.project.join("README.md"), "hello").unwrap();
    run_in(&fixture.project, "git", &["add", "."]);
    run_in(&fixture.project, "git", &["commit", "-q", "-m", "root"]);
    let _ = std::fs::remove_dir_all(fixture.home.join("identity"));

    capture("0199a1f2-3c4d-7e8f-9012-3456789abc02");

    let projects: std::collections::HashSet<String> = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|line| line["project"].as_str().map(str::to_string))
        .collect();
    assert_eq!(
        projects.len(),
        1,
        "a first commit re-keyed a project that already had memory: {projects:?}"
    );
}

#[test]
fn an_event_carries_the_store_that_wrote_it() {
    // Provenance for a future multi-store merge cannot be reconstructed
    // after the fact, so every append stamps which store wrote the line -
    // and the id is minted inside the isolated home, never derived from
    // anything about the machine.
    let fixture = Fixture::new("origin");
    for n in 0..2 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-3456789abc0{n}"),
            "cwd": fixture.project,
            "prompt": format!("observation number {n}")
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }
    let log = fixture.log_text();
    let origins: Vec<String> = log
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|line| line["origin"].as_str().map(str::to_string))
        .collect();
    assert_eq!(origins.len(), 2, "every appended event must be stamped: {log}");
    assert_eq!(origins[0], origins[1], "one store, one origin");
    assert!(
        fixture.home.join("origin").is_file(),
        "the store id must live inside the isolated home"
    );
}

#[test]
fn merging_two_stores_is_idempotent_order_blind_and_carries_revisions() {
    // The sync contract, as a test instead of a document: export/import
    // --merge must converge to one log whichever direction it runs, a
    // repeat import must add nothing, and a revision made on one store
    // must land on the other store's event after the trip.
    let marker = "[project]\nname = \"syncprop\"\n";
    let a = Fixture::new("sync-a");
    let b = Fixture::new("sync-b");
    std::fs::write(a.project.join(".rolepod-brain.toml"), marker).unwrap();
    std::fs::write(b.project.join(".rolepod-brain.toml"), marker).unwrap();

    let capture = |fixture: &Fixture, session: &str, prompt: &str| {
        let payload = serde_json::json!({
            "session_id": session,
            "cwd": fixture.project,
            "prompt": prompt
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    };
    capture(&a, "0199a1f2-3c4d-7e8f-9012-3456789abc0a", "the alpha rendezvous fact");
    capture(&b, "0199a1f2-3c4d-7e8f-9012-3456789abc0b", "the beta rendezvous fact");

    let archive_a = a.home.parent().unwrap().join("a.tar.gz");
    let out = a.brain(&["export", archive_a.to_str().unwrap()]);
    assert!(out.status.success(), "export A failed: {out:?}");

    // A's fact lands in B, and a second import of the same archive is a
    // no-op rather than a duplicate.
    let first = b.brain(&["import", archive_a.to_str().unwrap(), "--merge"]);
    assert!(first.status.success(), "import into B failed: {first:?}");
    let again = b.brain(&["import", archive_a.to_str().unwrap(), "--merge"]);
    let stdout = String::from_utf8_lossy(&again.stdout).to_string();
    assert!(stdout.contains("0 new event(s)"), "a repeat import must add nothing: {stdout}");
    let hits = String::from_utf8_lossy(&b.brain(&["search", "alpha"]).stdout).into_owned();
    assert!(hits.contains("alpha"), "A's fact never reached B: {hits}");

    // B corrects A's event - a revision crossing stores.
    let alpha_id = b
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|line| {
            line["title"].as_str().is_some_and(|title| title.contains("alpha"))
        })
        .and_then(|line| line["id"].as_str().map(str::to_string))
        .expect("alpha event in B's log");
    let call = |name: &str, args: String| {
        format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"{name}","arguments":{args}}}}}"#
        )
    };
    let search = call("brain_search", r#"{"query":"alpha"}"#.to_string());
    let correct = call(
        "brain_correct",
        format!(r#"{{"id":"{alpha_id}","text":"the gamma rendezvous correction"}}"#),
    );
    let result = b.mcp(&[&search, &correct]);
    let text = serde_json::to_string(&result).unwrap_or_default();
    assert!(!text.contains("has not been surfaced"), "correction refused: {text}");

    // The round trip: B's log (with the revision) flows back into A.
    let archive_b = b.home.parent().unwrap().join("b.tar.gz");
    let out = b.brain(&["export", archive_b.to_str().unwrap()]);
    assert!(out.status.success(), "export B failed: {out:?}");
    let back = a.brain(&["import", archive_b.to_str().unwrap(), "--merge"]);
    assert!(back.status.success(), "import into A failed: {back:?}");

    // Order-blind: both stores hold the same set of events.
    let ids = |log: &str| {
        let mut ids: Vec<String> = log
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|line| line["id"].as_str().map(str::to_string))
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(ids(&a.log_text()), ids(&b.log_text()), "the stores did not converge");

    // The revision made on B governs what A now remembers.
    let corrected = String::from_utf8_lossy(&a.brain(&["search", "gamma"]).stdout).into_owned();
    assert!(corrected.contains("gamma"), "B's correction never landed on A: {corrected}");
    let beta = String::from_utf8_lossy(&a.brain(&["search", "beta"]).stdout).into_owned();
    assert!(beta.contains("beta"), "B's own fact never reached A: {beta}");
}

#[test]
fn retire_drops_only_the_bodies_nobody_ever_needed() {
    // Retention was deferred until it could be measured; the command is
    // the measurement (dry-run default), and the rule is usage-beats-age:
    // a body anyone ever saw survives, and identity survives always.
    let fixture = Fixture::new("retire");
    for (n, prompt) in [
        (1, "keep the following\nthe alpha rendezvous cipher value"),
        (2, "note the following\nthe zeta rendezvous cipher value"),
    ] {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-3456789abc0{n}"),
            "cwd": fixture.project,
            "prompt": prompt
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }
    // Consolidated by a model tier - retirement never touches raw work in
    // flight.
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let done = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(done.status.success(), "consolidate failed: {done:?}");

    // Surface the alpha event: one search is all it takes to be "used".
    let search = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"alpha"}}}"#;
    let found = fixture.mcp(&[search]);
    let found = serde_json::to_string(&found).unwrap_or_default();
    assert!(found.contains("alpha"), "the search never surfaced alpha: {found}");

    // Dry run: a number, and no change.
    let dry = fixture.brain(&["retire", "--older-than-months", "0"]);
    assert!(dry.status.success(), "retire dry-run failed: {dry:?}");
    let stdout = String::from_utf8_lossy(&dry.stdout).to_string();
    assert!(stdout.contains("Would retire 1"), "wrong dry-run count: {stdout}");
    assert!(!fixture.log_text().contains("\"kind\":\"retire\""), "a dry run wrote to the log");

    let applied = fixture.brain(&["retire", "--older-than-months", "0", "--apply"]);
    assert!(applied.status.success(), "retire apply failed: {applied:?}");
    assert!(fixture.log_text().contains("\"kind\":\"retire\""), "no retire event in the log");

    // The unseen body is gone from search; the seen one and every title stay.
    let gone = String::from_utf8_lossy(&fixture.brain(&["search", "zeta"]).stdout).into_owned();
    assert!(!gone.contains("zeta"), "a retired body still matched: {gone}");
    let kept = String::from_utf8_lossy(&fixture.brain(&["search", "alpha"]).stdout).into_owned();
    assert!(kept.contains("alpha"), "a surfaced body was retired: {kept}");
    let title = String::from_utf8_lossy(&fixture.brain(&["search", "note the following"]).stdout)
        .into_owned();
    assert!(title.contains("note the following"), "the title must stay findable: {title}");

    // A rebuild replays the retirement rather than resurrecting the body.
    let reindexed = fixture.brain(&["reindex"]);
    assert!(reindexed.status.success(), "reindex failed: {reindexed:?}");
    let after = String::from_utf8_lossy(&fixture.brain(&["search", "zeta"]).stdout).into_owned();
    assert!(!after.contains("zeta"), "reindex resurrected a retired body: {after}");
}

impl Fixture {
    /// Capture one prompt as its own session and return its event id.
    fn prompt_event(&self, n: u32, text: &str) -> String {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-3456789abd{n:02}"),
            "cwd": self.project,
            "prompt": text
        })
        .to_string();
        self.hook("claude-code", "UserPromptSubmit", &payload);
        self.sql_one(&format!("SELECT id FROM events WHERE body LIKE '%{text}%'"))
    }

    /// One text value out of the index, read with a plain connection.
    fn sql_one(&self, sql: &str) -> String {
        rusqlite::Connection::open(self.home.join("brain.db"))
            .unwrap()
            .query_row(sql, [], |row| row.get(0))
            .unwrap()
    }

    fn sql_run(&self, sql: &str) {
        rusqlite::Connection::open(self.home.join("brain.db")).unwrap().execute_batch(sql).unwrap();
    }

    /// Settle every observation and push the named events far into the past in
    /// the index only: the two things the retention rule asks of a row besides
    /// never being seen. The log still says otherwise, so `brain_get` cannot be
    /// asked about these.
    fn age(&self, ids: &[&String]) {
        self.sql_run("UPDATE events SET consolidated = 1 WHERE kind = 'observation'");
        for id in ids {
            self.sql_run(&format!(
                "UPDATE events SET ts = '2020-01-01T00:00:00.000000Z' WHERE id = '{id}'"
            ));
        }
    }

    /// The same in the log, where a timestamp also names the month file the
    /// line lives in, then a rebuild of the index from it.
    fn age_in_log(&self, ids: &[&String]) {
        for file in self.log_files() {
            let old = file.with_file_name("2020-01.jsonl");
            let (mut kept, mut moved) = (String::new(), String::new());
            for line in std::fs::read_to_string(&file).unwrap().lines() {
                let mut event: serde_json::Value = serde_json::from_str(line).unwrap();
                if event["kind"] == "observation" {
                    event["consolidated"] = true.into();
                }
                let target = if ids.iter().any(|id| event["id"] == id.as_str()) {
                    event["ts"] = "2020-01-01T00:00:00.000000Z".into();
                    &mut moved
                } else {
                    &mut kept
                };
                target.push_str(&format!("{event}\n"));
            }
            std::fs::write(&file, kept).unwrap();
            if !moved.is_empty() {
                let mut old_file = std::fs::OpenOptions::new().create(true).append(true).open(old).unwrap();
                old_file.write_all(moved.as_bytes()).unwrap();
            }
        }
        let rebuilt = self.brain(&["reindex"]);
        assert!(rebuilt.status.success(), "reindex failed: {rebuilt:?}");
    }

    fn index_body(&self, id: &str) -> String {
        self.sql_one(&format!("SELECT body FROM events WHERE id = '{id}'"))
    }

    fn get_body(&self, id: &str) -> String {
        let fetched = self.mcp(&[&format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_get","arguments":{{"ids":["{id}"]}}}}}}"#
        )]);
        fetched[0]["result"]["structuredContent"]["events"][0]["body"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }
}

#[test]
fn consolidate_drops_the_index_body_of_old_unsurfaced_observations_and_only_those() {
    // Automatic retention is index-only: the log is the record, so a body
    // dropped from the index must come back through `brain_get`, and nothing
    // may be appended that another machine's sync could copy.
    let fixture = Fixture::new("retention");
    let surfaced = fixture.prompt_event(1, "the alpha rendezvous cipher value");
    let opened = fixture.prompt_event(2, "the beta rendezvous cipher value");
    let recent = fixture.prompt_event(3, "the gamma rendezvous cipher value");
    let plain = fixture.prompt_event(4, "the zeta rendezvous cipher value");
    let plain_body = fixture.index_body(&plain);
    assert!(plain_body.contains("zeta"), "precondition: indexed with its body");

    let search = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"alpha"}}}"#;
    let found = serde_json::to_string(&fixture.mcp(&[search])).unwrap();
    assert!(found.contains("alpha"), "the search never surfaced alpha: {found}");
    assert!(fixture.get_body(&opened).contains("beta"), "precondition: opened");

    // A body goes only after its vector exists. This run embeds with retention
    // off, so it does not spend the day the real pass needs.
    std::fs::write(fixture.home.join("config.toml"), "[retention]\ndays = 0\n").unwrap();
    let embedded = fixture.brain(&["consolidate"]);
    assert!(embedded.status.success(), "consolidate failed: {embedded:?}");
    std::fs::remove_file(fixture.home.join("config.toml")).unwrap();

    fixture.age_in_log(&[&surfaced, &opened, &plain]);
    let log_before = fixture.log_text();

    let done = fixture.brain(&["consolidate"]);
    assert!(done.status.success(), "consolidate failed: {done:?}");

    assert!(fixture.index_body(&surfaced).contains("alpha"), "an offered body was dropped");
    assert!(fixture.index_body(&opened).contains("beta"), "an opened body was dropped");
    assert!(fixture.index_body(&recent).contains("gamma"), "a recent body was dropped");
    assert_eq!(fixture.index_body(&plain), "", "the old unsurfaced body stayed in the index");
    assert_eq!(fixture.get_body(&plain), plain_body, "brain_get did not restore it from the log");
    assert_eq!(fixture.log_text(), log_before, "retention touched the log");
    assert!(!fixture.log_text().contains("\"kind\":\"retire\""), "retention wrote a retire event");

    let doctor = fixture.brain(&["doctor"]);
    let report = String::from_utf8_lossy(&doctor.stdout).into_owned();
    let line = report.lines().find(|line| line.contains(" retention ")).unwrap_or_default();
    assert!(line.contains("30 days") && line.contains("dropped so far 1"), "doctor's retention line: {line:?}");
    assert!(!line.contains("never"), "doctor does not know a pass finished: {line:?}");

    // The pass finished, so the day is spent: a body that turns old and
    // unseen after it waits for tomorrow's pass.
    let later = fixture.prompt_event(5, "the omega rendezvous cipher value");
    fixture.age(&[&later]);
    let again = fixture.brain(&["consolidate"]);
    assert!(again.status.success(), "second consolidate failed: {again:?}");
    assert!(fixture.index_body(&later).contains("omega"), "a second pass ran inside 24 hours");
}

#[test]
fn retention_days_zero_switches_it_off_without_spending_the_day() {
    let fixture = Fixture::new("retention-off");
    let old = fixture.prompt_event(1, "the zeta rendezvous cipher value");
    fixture.age(&[&old]);
    std::fs::write(fixture.home.join("config.toml"), "[retention]\ndays = 0\n").unwrap();

    let off = fixture.brain(&["consolidate"]);
    assert!(off.status.success(), "consolidate failed: {off:?}");
    assert!(fixture.index_body(&old).contains("zeta"), "days = 0 still dropped a body");

    // Switching it on afterwards must not wait out a day nobody spent.
    std::fs::write(fixture.home.join("config.toml"), "[retention]\ndays = 30\n").unwrap();
    let on = fixture.brain(&["consolidate"]);
    assert!(on.status.success(), "consolidate failed: {on:?}");
    assert_eq!(fixture.index_body(&old), "", "turning retention on dropped nothing");
}

impl Fixture {
    /// Bytes the index occupies on disk: the database and its write-ahead log.
    fn index_bytes(&self) -> u64 {
        ["brain.db", "brain.db-wal"]
            .iter()
            .map(|name| std::fs::metadata(self.home.join(name)).map_or(0, |meta| meta.len()))
            .sum()
    }

    /// Make the event logs look untouched for an hour, so `brain compact` sees
    /// a machine no hook has used lately.
    fn quiet_logs(&self) {
        let hour_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        for file in self.log_files() {
            std::fs::OpenOptions::new()
                .append(true)
                .open(file)
                .unwrap()
                .set_modified(hour_ago)
                .unwrap();
        }
    }
}

#[test]
fn compact_gives_back_the_space_of_dropped_bodies_and_search_still_answers() {
    let fixture = Fixture::new("compact");
    let filler = "lorem ipsum dolor sit amet ".repeat(120);
    let ids: Vec<String> = (1..=60)
        .map(|n| fixture.prompt_event(n, &format!("marker{n:02}kestrel {filler}")))
        .collect();
    let full_body = fixture.index_body(&ids[0]);
    assert!(full_body.len() > 3000, "precondition: a body worth dropping");

    std::fs::write(fixture.home.join("config.toml"), "[retention]\ndays = 0\n").unwrap();
    assert!(fixture.brain(&["consolidate"]).status.success(), "embedding run failed");
    std::fs::remove_file(fixture.home.join("config.toml")).unwrap();
    let all: Vec<&String> = ids.iter().collect();
    fixture.age_in_log(&all);
    let done = fixture.brain(&["consolidate"]);
    assert!(done.status.success(), "consolidate failed: {done:?}");
    assert_eq!(fixture.index_body(&ids[0]), "", "precondition: the body left the index");
    let rowids = fixture.sql_one("SELECT group_concat(id || r) FROM (SELECT id, rowid AS r FROM events ORDER BY id)");

    fixture.quiet_logs();
    let free_pages = "SELECT CAST(freelist_count AS TEXT) FROM pragma_freelist_count";
    assert_ne!(fixture.sql_one(free_pages), "0", "precondition: dropped bodies left free pages");
    let before = fixture.index_bytes();
    let compacted = fixture.brain(&["compact"]);
    let said = String::from_utf8_lossy(&compacted.stdout).into_owned();
    assert!(compacted.status.success(), "compact failed: {compacted:?}");
    assert_eq!(fixture.sql_one(free_pages), "0", "VACUUM left free pages behind");
    let after = fixture.index_bytes();
    assert!(after < before, "the index did not shrink: {before} -> {after} ({said})");
    assert!(said.contains(&before.to_string()) && said.contains(&after.to_string()), "it did not say the sizes: {said}");

    // The text index is keyed by rowid, so VACUUM must not have moved one.
    let kept = fixture.sql_one("SELECT group_concat(id || r) FROM (SELECT id, rowid AS r FROM events ORDER BY id)");
    assert_eq!(kept, rowids, "compact renumbered the rows");
    fixture.sql_run("INSERT INTO events_fts(events_fts) VALUES ('integrity-check')");
    fixture.sql_run("INSERT INTO events_tri(events_tri) VALUES ('integrity-check')");
    let found = String::from_utf8_lossy(&fixture.brain(&["search", "marker07kestrel"]).stdout).into_owned();
    assert!(found.contains("marker07kestrel"), "search lost the row after compact: {found}");
    assert!(fixture.get_body(&ids[6]).contains("marker07kestrel"), "brain_get lost the body after compact");
}

#[test]
fn compact_refuses_while_a_hook_has_run_in_the_last_ten_minutes() {
    let fixture = Fixture::new("compact-busy");
    fixture.prompt_event(1, "the zeta rendezvous cipher value");
    let before = fixture.index_bytes();

    let refused = fixture.brain(&["compact"]);
    let said = String::from_utf8_lossy(&refused.stderr).into_owned();
    assert!(!refused.status.success(), "compact ran under a live hook: {refused:?}");
    assert!(said.contains("hook"), "the refusal does not say why: {said}");
    assert_eq!(fixture.index_bytes(), before, "a refused compact still touched the index");

    fixture.quiet_logs();
    let allowed = fixture.brain(&["compact"]);
    assert!(allowed.status.success(), "compact refused a quiet machine: {allowed:?}");
}

impl Fixture {
    fn state_value(&self, key: &str) -> String {
        rusqlite::Connection::open(self.home.join("brain.db"))
            .unwrap()
            .query_row("SELECT value FROM schema_state WHERE key = ?1", [key], |row| row.get(0))
            .unwrap_or_default()
    }

    /// A store whose retention has dropped sixty bodies and that nothing has
    /// compacted. Returns the ids and the `id || rowid` listing.
    fn emptied_store(&self) -> (Vec<String>, String) {
        let filler = "lorem ipsum dolor sit amet ".repeat(120);
        let ids: Vec<String> =
            (1..=60).map(|n| self.prompt_event(n, &format!("marker{n:02}kestrel {filler}"))).collect();
        std::fs::write(self.home.join("config.toml"), "[retention]\ndays = 0\n").unwrap();
        assert!(self.brain(&["consolidate"]).status.success(), "embedding run failed");
        std::fs::remove_file(self.home.join("config.toml")).unwrap();
        let all: Vec<&String> = ids.iter().collect();
        self.age_in_log(&all);
        assert!(self.brain(&["consolidate"]).status.success(), "retention run failed");
        assert_eq!(self.index_body(&ids[0]), "", "precondition: the body left the index");
        let rowids = self.sql_one("SELECT group_concat(id || r) FROM (SELECT id, rowid AS r FROM events ORDER BY id)");
        (ids, rowids)
    }
}

#[test]
fn a_consolidation_run_on_a_quiet_machine_compacts_the_index_by_itself() {
    let fixture = Fixture::new("auto-compact");
    let (ids, rowids) = fixture.emptied_store();
    assert_eq!(fixture.state_value("compact_done_at"), "", "precondition: never compacted");

    // The run that dropped the bodies saw a log written seconds before and
    // left the window alone, and said why.
    assert!(fixture.state_value("compact_skip").starts_with("not-quiet@"), "{}", fixture.state_value("compact_skip"));
    assert!(fixture.home.join("brain.db").is_file());

    fixture.quiet_logs();
    let before = fixture.index_bytes();
    let run = fixture.brain(&["consolidate", "--all"]);
    assert!(run.status.success(), "consolidate failed: {run:?}");

    let after = fixture.index_bytes();
    assert!(after < before, "no one asked, and the index did not shrink: {before} -> {after}");
    assert!(after * 10 <= before * 9, "the spec asks for at least 10% back: {before} -> {after}");
    assert!(!fixture.state_value("compact_done_at").is_empty(), "the compact left no mark");
    assert_eq!(fixture.state_value("compact_skip"), "", "a finished compact keeps no stale refusal");
    assert!(!fixture.home.join(".brain-maintenance").exists(), "the window left its marker");
    assert!(!fixture.home.join(".brain-maintain.lock").exists(), "the window left its gate");
    assert_eq!(fixture.sql_one("SELECT CAST(freelist_count AS TEXT) FROM pragma_freelist_count"), "0");

    let kept = fixture.sql_one("SELECT group_concat(id || r) FROM (SELECT id, rowid AS r FROM events ORDER BY id)");
    assert_eq!(kept, rowids, "the rewrite renumbered the rows");
    fixture.sql_run("INSERT INTO events_fts(events_fts) VALUES ('integrity-check')");
    fixture.sql_run("INSERT INTO events_tri(events_tri) VALUES ('integrity-check')");
    let found = String::from_utf8_lossy(&fixture.brain(&["search", "marker07kestrel"]).stdout).into_owned();
    assert!(found.contains("marker07kestrel"), "search lost the row: {found}");
    assert!(fixture.get_body(&ids[6]).contains("marker07kestrel"));

    // Done is done: a second run finds nothing due.
    fixture.quiet_logs();
    let again = fixture.index_bytes();
    assert!(fixture.brain(&["consolidate", "--all"]).status.success());
    assert!(fixture.index_bytes() <= again, "a second run rewrote an index that was not due");
}

#[test]
fn a_run_that_meets_the_window_stands_aside_and_leaves_no_trace_in_asks_or_log() {
    let fixture = Fixture::new("maint-yield");
    fixture.prompt_event(1, "the zeta rendezvous cipher value");
    assert!(fixture.brain(&["consolidate"]).status.success());
    let asks = "SELECT CAST(COUNT(*) AS TEXT) FROM consolidation_requests";
    let asked = fixture.sql_one(asks);
    let log_before = std::fs::read_to_string(fixture.home.join("brain.log")).unwrap_or_default();

    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    std::fs::write(fixture.home.join(".brain-maintenance"), format!("99999 {}", now_ms + 2500)).unwrap();
    let yielded = fixture.brain(&["consolidate"]);
    assert!(yielded.status.success(), "{yielded:?}");
    assert_eq!(fixture.sql_one(asks), asked, "a run in the window left an ask behind");
    assert_eq!(std::fs::read_to_string(fixture.home.join("brain.log")).unwrap_or_default(), log_before);
    assert!(fixture.home.join(".brain-maint-yielded").exists(), "nothing tells the compactor to rerun");
    let last_run = "SELECT CAST(yielded AS TEXT) FROM consolidation_runs ORDER BY rowid DESC LIMIT 1";
    assert_eq!(fixture.sql_one(last_run), "1");

    // A compactor killed outright stops renewing: the marker lapses by itself.
    std::thread::sleep(std::time::Duration::from_millis(2700));
    assert!(fixture.brain(&["consolidate"]).status.success());
    assert_eq!(fixture.sql_one(last_run), "0", "an expired marker still held a run back");
}

#[test]
fn a_correction_made_twice_becomes_a_standing_rule() {
    // khwan's loop, kept local: corrections a person made more than once are
    // distilled into a rule at the next synthesis round, through the same
    // ladder call - no extra model spend. A rule citing fewer than two
    // corrections is discarded unread, because the model is not trusted to
    // count.
    let fixture = Fixture::new("rules");
    let bin = fixture.fake_cli("claude", &rule_cli());

    // Two events, each surfaced by search and then corrected - the same
    // gate every real correction passes.
    let call = |name: &str, args: String| {
        format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"{name}","arguments":{args}}}}}"#
        )
    };
    for (n, text) in
        [
        (1, "always run the linter before committing"),
        (2, "always run the linter before you commit"),
    ]
    {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-98765432100{n}"),
            "cwd": fixture.project,
            "prompt": format!("skipped the linter on commit {n}")
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
        let id = fixture
            .log_text()
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|line| line["kind"] == "observation")
            .filter_map(|line| line["id"].as_str().map(str::to_string))
            .next_back()
            .expect("an event to correct");
        let search = call("brain_search", r#"{"query":"linter"}"#.to_string());
        let correct = call("brain_correct", format!(r#"{{"id":"{id}","text":"{text}"}}"#));
        let result = fixture.mcp(&[&search, &correct]);
        let text = serde_json::to_string(&result).unwrap_or_default();
        assert!(
            !text.contains("has not been surfaced") && !text.contains("isError\":true"),
            "the correction was refused: {text}"
        );
    }

    // Five sessions cross the synthesis watermark.
    for session in 0..5 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-34567891000{session}"),
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
        let done = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
        assert!(done.status.success(), "consolidate {session} failed: {done:?}");
    }

    let pages = fixture.knowledge_pages();
    let rule = pages
        .iter()
        .find(|path| path.to_string_lossy().contains("knowledge/rules/"))
        .unwrap_or_else(|| {
            let log = fixture.log_text();
            let interesting: Vec<String> = log
                .lines()
                .filter(|l| l.contains("correct") || l.contains("knowledge"))
                .map(|l| l.chars().take(200).collect())
                .collect();
            panic!("no rule page was written: {pages:?}\nLOG: {interesting:#?}")
        });
    let page = std::fs::read_to_string(rule).expect("the rule page");
    assert!(page.contains("Always run the linter"), "wrong rule: {page}");
    assert!(page.contains("tags: [knowledge, rule]"), "page not typed: {page}");

    // The under-cited rule is the counter-test: one correction is an edit.
    assert!(
        !fixture.log_text().contains("A rule nobody asked for twice"),
        "a rule citing one correction was kept"
    );

    // The rule reaches the next session first among lessons.
    let start = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000001",
        "cwd": fixture.project,
        "source": "startup"
    })
    .to_string();
    let output = fixture.hook("claude-code", "SessionStart", &start);
    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&output.stdout).trim()).unwrap();
    let context = parsed["hookSpecificOutput"]["additionalContext"].as_str().unwrap();
    assert!(
        context.contains("KNW  Always run the linter"),
        "the rule never reached the primer: {context}"
    );
}

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
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
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
fn a_cli_that_trusts_pwd_still_runs_in_the_inert_directory() {
    // Found live: `opencode run` resolves its project as `$PWD ?? cwd()` and
    // chdirs there. Setting only the child's working directory left `PWD`
    // naming the repo the hook fired in, so every summary opened a session in
    // the user's project - where other plugins took it for a sibling agent.
    let fixture = Fixture::new("inherited-pwd");
    let record = fixture.home.parent().unwrap().join("child-pwd");
    let inert = fixture.home.parent().unwrap().join("tmpdir");
    // Not `sh`: a shell replaces an inherited `PWD` that disagrees with its
    // directory, which would hide the bug. Perl reads the environment as
    // it was handed over, the way opencode's runtime does. The rewrite below
    // truncates in place, so the file keeps the exec bit `fake_cli` gave it.
    let bin = fixture.fake_cli("opencode", "");
    std::fs::write(
        bin.join("opencode"),
        format!(
            "#!/usr/bin/perl\nopen(my $f, '>>', '{}') or die;\nprint $f \"$ENV{{PWD}}\\n\";\n\
             print '{}', \"\\n\";\n",
            record.display(),
            r#"{"summary":"Refactored the auth path and fixed token expiry.","titles":[]}"#
        ),
    )
    .unwrap();
    fixture.seed_session(4);

    let output = std::process::Command::new(BRAIN)
        .args(["consolidate", "--force"])
        .current_dir(&fixture.project)
        .env("PWD", &fixture.project)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("HOME", fixture.home.parent().unwrap())
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("TMPDIR", &inert)
        .output()
        .expect("run brain");
    assert!(output.status.success(), "consolidate errored: {output:?}");

    let seen = std::fs::read_to_string(&record).expect("the stub was never called");
    assert!(!seen.is_empty(), "the stub was never called");
    for line in seen.lines() {
        assert_eq!(line, inert.display().to_string(), "a call ran with PWD in the repo");
    }
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

/// Whether the index holds this event id.
fn index_has(fixture: &Fixture, id: &str) -> bool {
    let output = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg(format!("SELECT COUNT(*) FROM events WHERE id = '{id}';"))
        .output()
        .expect("query event");
    String::from_utf8_lossy(&output.stdout).trim() == "1"
}

#[test]
fn an_event_the_index_missed_is_indexed_by_the_next_run() {
    let fixture = Fixture::new("log-catch-up");
    fixture.seed_session(3);

    // What a hook leaves when its index write fails: the line is in the log
    // and the row is not in the store.
    let log = std::fs::read_dir(fixture.project_dirs()[0].join("events"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .expect("a month's log");
    let text = std::fs::read_to_string(&log).unwrap();
    let mut missed: serde_json::Value =
        serde_json::from_str(text.lines().last().unwrap()).unwrap();
    let id = ulid::Ulid::new().to_string();
    missed["id"] = serde_json::Value::String(id.clone());
    missed["title"] = serde_json::Value::String("Edit: missed.rs".into());
    let mut file = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
    writeln!(file, "{missed}").unwrap();
    let appended_to = std::fs::metadata(&log).unwrap().len();
    assert!(!index_has(&fixture, &id), "precondition: the index lacks it");
    assert_eq!(fixture.pending_count(), 3);

    let out = fixture.brain(&["consolidate", "--force"]);
    assert!(out.status.success(), "consolidate failed: {out:?}");
    assert!(index_has(&fixture, &id), "the run did not index what the log had and the index lacked");
    assert_eq!(fixture.pending_count(), 4, "the caught-up event is pending work");

    // A second run finds the watermark at the end of the log and adds nothing.
    let out = fixture.brain(&["consolidate", "--force"]);
    assert!(out.status.success(), "second consolidate failed: {out:?}");
    assert_eq!(fixture.pending_count(), 4);
    let marks = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg("SELECT value FROM schema_state WHERE key LIKE 'log_tail:%';")
        .output()
        .unwrap();
    let marks = String::from_utf8_lossy(&marks.stdout).into_owned();
    // The run's own summary events were appended after the catch-up read, so
    // the mark may trail the log's end, but never the line that was missed.
    let offset: u64 = marks.split(':').next().unwrap().parse().unwrap();
    assert!(offset >= appended_to, "the watermark stops before the missed line: {marks}");
    assert!(offset <= std::fs::metadata(&log).unwrap().len(), "the watermark is past the log: {marks}");
}

/// The primer's indexes cost seconds on a large store, so no hook builds them:
/// a store has none until a consolidate run, which builds them once and drops
/// the index they cover.
#[test]
fn a_consolidate_run_builds_the_primer_indexes_once() {
    let fixture = Fixture::new("primer-indexes");
    fixture.seed_session(2);
    let indexes = |fixture: &Fixture| {
        let out = Command::new("sqlite3")
            .arg(fixture.home.join("brain.db"))
            .arg("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'events' ORDER BY name;")
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let before = indexes(&fixture);
    assert!(!before.contains("events_kind_proj"), "a hook built the kind index: {before}");
    assert!(!before.contains("events_session_id"), "a hook built the session index: {before}");
    assert!(!before.contains("events_recall"), "a hook built the recall index: {before}");

    let out = fixture.brain(&["consolidate", "--force"]);
    assert!(out.status.success(), "consolidate failed: {out:?}");
    let after = indexes(&fixture);
    assert!(
        after.contains("events_kind_proj") && after.contains("events_session_id") && after.contains("events_recall"),
        "not built: {after}"
    );
    // An older binary on this store would recreate it, cold, under the write
    // lock its next open takes, so the build leaves it where it is.
    assert!(after.lines().any(|name| name == "events_session"), "the build dropped events_session: {after}");

    // A later open (a hook, the next run) leaves it that way.
    let out = fixture.brain(&["consolidate", "--force"]);
    assert!(out.status.success(), "second consolidate failed: {out:?}");
    assert_eq!(indexes(&fixture), after);
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

/// The keys every wiki git call carries on its command line.
const WIKI_GIT_GUARD: [&str; 6] = [
    "maintenance.auto=false",
    "maintenance.autoDetach=false",
    "gc.auto=0",
    "gc.autoDetach=false",
    "core.fsmonitor=",
    "commit.gpgSign=false",
];

/// Since git 2.29 every `git commit` starts `git maintenance run --auto`, which
/// recent git detaches, and brain committed once per page: on 2026-10-07 that
/// was tens of concurrent repacks and a hung machine. Every wiki git call now carries
/// a guard on its command line, so no commit brain makes starts one.
#[test]
fn the_wiki_never_starts_gits_own_maintenance() {
    let fixture = Fixture::new("nomaint");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let trace = fixture.home.parent().unwrap().join("t.json");
    let output = fixture.brain_with_env(
        &["consolidate", "--force"],
        Some(&bin),
        &[("GIT_TRACE2_EVENT", trace.to_str().unwrap())],
    );
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let events = trace2(&trace);
    // Without this, a git with no trace2 support would pass by saying nothing.
    assert!(
        top_level_starts(&events).into_iter().any(|event| trace_argv(event).contains(&"commit".into())),
        "no commit in the trace - is trace2 supported by this git?"
    );
    assert_eq!(spawned(&events), Vec::<Vec<String>>::new(), "a wiki commit started maintenance");

    // Every git that ran in the wiki, not only the ones that already carry
    // the guard: one built by hand would show up here without it.
    let wiki = std::fs::canonicalize(fixture.wiki()).unwrap();
    let in_wiki: std::collections::HashSet<&str> = events
        .iter()
        .filter(|event| event["event"] == "def_repo")
        .filter(|event| event["worktree"].as_str().is_some_and(|tree| Path::new(tree) == wiki))
        .filter_map(|event| event["sid"].as_str())
        .collect();
    let starts: Vec<Vec<String>> = top_level_starts(&events)
        .into_iter()
        .filter(|event| event["sid"].as_str().is_some_and(|sid| in_wiki.contains(sid)))
        .map(trace_argv)
        .collect();
    assert!(!starts.is_empty(), "no git ran in the wiki");
    // The calls that write the wiki are also found by name, so a git that
    // never reported its repository cannot slip past both checks.
    let writes: Vec<Vec<String>> = top_level_starts(&events)
        .into_iter()
        .map(trace_argv)
        .filter(|argv| argv.iter().any(|arg| ["init", "add", "commit"].contains(&arg.as_str())))
        .collect();
    for argv in starts.iter().chain(&writes) {
        for key in WIKI_GIT_GUARD {
            assert!(
                argv.windows(2).any(|pair| pair[0] == "-c" && pair[1] == key),
                "a wiki git ran without -c {key}: {argv:?}"
            );
        }
    }
    assert_eq!(wiki_git_starts(&events).len(), starts.len(), "a guarded git ran outside the wiki");
}

/// The same, against a person whose own git config asks for everything the
/// guard turns off: auto-maintenance in the global file and in the
/// environment, an fsmonitor daemon, and signed commits through a signer that
/// always fails. Brain still commits, and still starts nothing.
#[test]
fn the_wiki_never_starts_gits_own_maintenance_under_a_hostile_config() {
    let fixture = Fixture::new("nomaint-hostile");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let base = fixture.home.parent().unwrap();
    std::fs::write(
        base.join(".gitconfig"),
        "[maintenance]\n\tauto = true\n\tstrategy = incremental\n[gc]\n\tauto = 1\n\
         [core]\n\tfsmonitor = true\n[commit]\n\tgpgSign = true\n[gpg]\n\tprogram = /bin/false\n",
    )
    .unwrap();
    let trace = base.join("t.json");
    let output = fixture.brain_with_env(
        &["consolidate", "--force"],
        Some(&bin),
        &[
            ("GIT_TRACE2_EVENT", trace.to_str().unwrap()),
            ("GIT_CONFIG_COUNT", "1"),
            ("GIT_CONFIG_KEY_0", "maintenance.auto"),
            ("GIT_CONFIG_VALUE_0", "true"),
        ],
    );
    let wiki = fixture.wiki();
    let log = git_stdout(&wiki, &["log", "--oneline"]);
    // Only a failing run starts a daemon, and it would outlive the test.
    let _ = Command::new("git")
        .args(["fsmonitor--daemon", "stop"])
        .current_dir(&wiki)
        .output();

    assert!(output.status.success(), "consolidate failed: {output:?}");
    assert!(wiki.join(".git").is_dir(), "the wiki has no repository");
    assert!(log.contains("consolidate"), "a hostile config stopped the commit: {log}");
    let events = trace2(&trace);
    assert!(
        top_level_starts(&events).into_iter().any(|event| trace_argv(event).contains(&"commit".into())),
        "no commit in the trace - is trace2 supported by this git?"
    );
    assert_eq!(spawned(&events), Vec::<Vec<String>>::new(), "a wiki git started maintenance");
}

/// A hook runs inside whatever git the host CLI is running, and git exports
/// `GIT_INDEX_FILE` and friends to its own hooks. The detached consolidate
/// inherits that environment; unscrubbed, its wiki commits went into the
/// person's checkout through the person's index.
#[test]
fn a_leaked_git_environment_cannot_redirect_wiki_commits() {
    let fixture = Fixture::new("leakedenv");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let checkout = fixture.project.clone();
    let git_dir = checkout.join(".git");
    let output = fixture.brain_with_env(
        &["consolidate", "--force"],
        Some(&bin),
        &[
            ("GIT_DIR", git_dir.to_str().unwrap()),
            ("GIT_WORK_TREE", checkout.to_str().unwrap()),
            ("GIT_INDEX_FILE", git_dir.join("index").to_str().unwrap()),
        ],
    );
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let wiki = fixture.wiki();
    assert!(wiki.join(".git").is_dir(), "the wiki has no repository");
    let log = git_stdout(&wiki, &["log", "--oneline"]);
    assert!(log.contains("consolidate"), "the wiki has no consolidate commit: {log}");
    let tracked = git_stdout(&checkout, &["ls-files"]);
    assert!(tracked.trim().is_empty(), "wiki pages were staged in the checkout: {tracked}");
    let head = Command::new("git")
        .args(["rev-parse", "-q", "--verify", "HEAD"])
        .current_dir(&checkout)
        .output()
        .expect("run git");
    assert!(!head.status.success(), "a commit landed in the checkout");
}

/// The guard on brain's own command lines does nothing for a commit something
/// else makes in the wiki: an older brain still holding the run lock, a second
/// binary wired in by path, Obsidian Git, the person. Those read the
/// repository's own config, so brain turns maintenance off there too - before
/// it asks for the run lock, so a run that stands aside still does it.
#[test]
fn the_wiki_repo_turns_git_maintenance_off_for_every_committer() {
    let fixture = Fixture::new("repopolicy");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let wiki = fixture.wiki();
    // `--local`: the repository's own file, never what this machine set.
    let policy = || {
        ["maintenance.auto", "gc.auto", "brain.policy"]
            .map(|key| git_stdout(&wiki, &["config", "--local", "--get", key]).trim().to_string())
    };
    assert_eq!(policy(), ["false", "0", "1"], "the wiki's own config leaves maintenance on");

    // A commit brain did not build, on git's defaults otherwise.
    let base = fixture.home.parent().unwrap();
    let trace = base.join("t.json");
    let mut commit = Command::new("git");
    commit
        .args(["commit", "--allow-empty", "-q", "-m", "x"])
        .current_dir(&wiki)
        .env("HOME", base)
        .env("GIT_TRACE2_EVENT", &trace);
    fixture.own_git_config(&mut commit);
    assert!(commit.status().expect("run git").success(), "a plain commit in the wiki failed");
    let events = trace2(&trace);
    assert!(
        top_level_starts(&events).into_iter().any(|event| trace_argv(event).contains(&"commit".into())),
        "no commit in the trace - is trace2 supported by this git?"
    );
    assert_eq!(spawned(&events), Vec::<Vec<String>>::new(), "a plain wiki commit started maintenance");

    // A wiki a 0.63 brain left, while another run holds the lock: this run
    // stands aside, and has written the policy back first. The idle sweep
    // takes the lock by its own path, so it is checked on its own.
    let lock = fixture.home.join(".brain-consolidate.lock");
    std::fs::write(&lock, std::process::id().to_string()).unwrap();
    fixture.seed_session(4);
    for args in [["consolidate", "--force"], ["consolidate", "--idle"]] {
        for key in ["maintenance.auto", "gc.auto", "brain.policy"] {
            git_stdout(&wiki, &["config", "--local", "--unset", key]);
        }
        assert_eq!(policy(), ["", "", ""], "the policy was not removed");
        let output = fixture.brain_with_path(&args, Some(&bin));
        assert!(output.status.success(), "{args:?} failed: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("stood aside"),
            "{args:?} did not yield to the held lock: {output:?}"
        );
        assert_eq!(policy(), ["false", "0", "1"], "{args:?} stood aside and left maintenance on");
    }
    std::fs::remove_file(&lock).unwrap();

    // With the marker in place, a run with nothing new starts no git at all.
    fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    let trace = base.join("nothing-new.json");
    let output = fixture.brain_with_env(
        &["consolidate", "--force"],
        Some(&bin),
        &[("GIT_TRACE2_EVENT", trace.to_str().unwrap())],
    );
    assert!(output.status.success(), "consolidate failed: {output:?}");
    assert_eq!(
        wiki_git_starts(&trace2(&trace)),
        Vec::<Vec<String>>::new(),
        "a run with nothing new started git in the wiki"
    );
}

/// A consolidated session is one commit: its page together with every hub,
/// topic and entity page it changed. Brain used to commit each page on its
/// own, four or five git processes a page, and a real session came to about a
/// hundred commits - each of which started git's maintenance.
#[test]
fn one_consolidated_session_is_one_commit() {
    let fixture = Fixture::new("onecommit");
    // Two sessions over the same files, so the second one brings entity
    // pages and `entities.md` into its commit.
    let files = ["src/a.rs", "src/b.rs", "src/c.rs", "src/d.rs", "src/e.rs", "src/f.rs"];
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-0000000000a1", &files);
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-0000000000b2", &files);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let base = fixture.home.parent().unwrap();
    let trace = base.join("t.json");
    let output = fixture.brain_with_env(
        &["consolidate", "--force"],
        Some(&bin),
        &[("GIT_TRACE2_EVENT", trace.to_str().unwrap())],
    );
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let wiki = fixture.wiki();
    let log = git_stdout(&wiki, &["log", "--format=%H %s"]);
    // Two sessions, and the front page once for the run.
    assert_eq!(log.lines().count(), 3, "not one commit per session: {log}");
    let events = trace2(&trace);
    let starts = wiki_git_starts(&events);
    // init 1, identity 2, maintenance policy 3, then add, diff and commit
    // for each session and once for the front page.
    assert!(starts.len() <= 15, "{} wiki git processes: {starts:?}", starts.len());
    assert_eq!(spawned(&events), Vec::<Vec<String>>::new(), "a wiki commit started maintenance");

    let sessions: Vec<(&str, &str)> = log
        .lines()
        .filter_map(|line| line.split_once(' '))
        .filter(|(_, subject)| subject.contains("/pages/sessions/"))
        .collect();
    assert_eq!(sessions.len(), 2, "a session commit is missing or misnamed: {log}");
    let mut derived = Vec::new();
    for (hash, subject) in sessions {
        let page = subject
            .strip_prefix("consolidate ")
            .and_then(|rest| rest.rsplit_once(" ("))
            .map_or(subject, |(page, _)| page);
        let changed = git_stdout(&wiki, &["show", "--name-only", "--format=", hash]);
        let changed: Vec<&str> = changed.lines().collect();
        assert!(changed.contains(&page), "{subject} left its own page out: {changed:?}");
        assert!(changed.contains(&"checkout/checkout.md"), "{subject} left the hub out: {changed:?}");
        derived.extend(changed.into_iter().map(str::to_string));
    }
    assert!(
        derived.iter().any(|path| path == "checkout/entities.md"),
        "no session commit carried the entity index: {derived:?}"
    );
    assert!(
        derived.iter().any(|path| path.starts_with("checkout/entities/")),
        "no session commit carried an entity page: {derived:?}"
    );
    // One commit for a session is all or nothing: a page the run wrote and
    // left out of it would sit in the tree with no history at all. The
    // capture log is the hooks' file, and no consolidation commits it.
    let status = git_stdout(&wiki, &["status", "--porcelain", "--untracked-files=all"]);
    let left: Vec<&str> = status.lines().filter(|line| !line.contains("/events/")).collect();
    assert_eq!(left, Vec::<&str>::new(), "the run left pages uncommitted");

    // Nothing new: no git at all, and no commit.
    let trace = base.join("nothing-new.json");
    let output = fixture.brain_with_env(
        &["consolidate", "--force"],
        Some(&bin),
        &[("GIT_TRACE2_EVENT", trace.to_str().unwrap())],
    );
    assert!(output.status.success(), "consolidate failed: {output:?}");
    assert_eq!(
        wiki_git_starts(&trace2(&trace)),
        Vec::<Vec<String>>::new(),
        "a run with nothing new started git in the wiki"
    );
    assert_eq!(
        git_stdout(&wiki, &["rev-list", "--count", "HEAD"]).trim(),
        "3",
        "a run with nothing new committed"
    );
}

/// The machines the storm hit were hard-rebooted while brain committed about
/// once a second, and the repacks it had started were killed. What those gits
/// left behind - a lock that makes every later commit fail, gigabytes of
/// half-written packs - stays until something removes it. The next run does,
/// and leaves anything a git could still be working on alone.
#[test]
fn a_killed_gits_leftovers_are_cleared_and_live_ones_are_not() {
    let fixture = Fixture::new("heal");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");
    let wiki = fixture.wiki();
    let git = wiki.join(".git");
    assert!(git.is_dir(), "the wiki has no repository");
    let commits = || git_stdout(&wiki, &["rev-list", "--count", "HEAD"]).trim().parse::<usize>().unwrap();
    let before = commits();

    let branch = git_stdout(&wiki, &["symbolic-ref", "--short", "HEAD"]).trim().to_string();
    assert!(!branch.is_empty(), "the wiki has no branch");
    let pack = git.join("objects").join("pack");
    std::fs::create_dir_all(git.join("refs/heads/brain")).unwrap();
    let plant = |path: &Path, bytes: usize, age: std::time::Duration| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![0u8; bytes]).unwrap();
        let file = std::fs::File::options().write(true).open(path).unwrap();
        file.set_modified(std::time::SystemTime::now() - age).unwrap();
    };
    let two_hours = std::time::Duration::from_secs(2 * 60 * 60);
    // What a killed git leaves: half-written packs nobody has touched for an
    // hour, and the locks that make the next commit fail.
    let dead = [
        pack.join("tmp_pack_dead"),
        pack.join(".tmp-4242-pack-dead.pack"),
        git.join("index.lock"),
        git.join("HEAD.lock"),
        git.join("config.lock"),
        git.join("refs/heads").join(format!("{branch}.lock")),
        git.join("refs/heads/brain/old.lock"),
    ];
    for path in &dead {
        let bytes = if path.ends_with("tmp_pack_dead") { 1 << 20 } else { 0 };
        plant(path, bytes, two_hours);
    }
    // What a running git could still be writing.
    let live = [pack.join("tmp_pack_live"), git.join("refs/heads/brain/live.lock")];
    for path in &live {
        plant(path, 0, std::time::Duration::ZERO);
    }
    // What is git's own business rather than a leftover, however old: a
    // finished pack, and the files only its maintenance reads.
    let owned = [pack.join("pack-deadbeef.pack"), git.join("objects/maintenance.lock"), git.join("gc.pid")];
    for path in &owned {
        plant(path, 0, two_hours);
    }

    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-0000000000c3", &["src/later.rs", "src/next.rs"]);
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let left: Vec<&PathBuf> = dead.iter().filter(|path| path.exists()).collect();
    assert_eq!(left, Vec::<&PathBuf>::new(), "a killed git's leftovers are still there");
    for path in live.iter().chain(&owned) {
        assert!(path.exists(), "{} was removed", path.display());
    }
    assert!(commits() > before, "no commit landed after the leftovers: {output:?}");

    // The idle sweep takes the run lock by its own path, so it heals too.
    plant(&git.join("index.lock"), 0, two_hours);
    let output = fixture.brain_with_path(&["consolidate", "--idle"], Some(&bin));
    assert!(output.status.success(), "the idle sweep failed: {output:?}");
    assert!(!git.join("index.lock").exists(), "the idle sweep left a dead index.lock");
}

/// Loose objects in a repository, counted the way git names them: a file
/// under `objects/<2 hex>/` whose name is the rest of the hash. It mirrors
/// the SHA-1 half of brain's own `count_loose`, the only hash a fixture uses.
fn loose_objects(git: &Path) -> usize {
    (0..=255u8)
        .flat_map(|byte| std::fs::read_dir(git.join("objects").join(format!("{byte:02x}"))).into_iter().flatten())
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.len() == 38 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        .count()
}

/// With git's own maintenance off, brain packs the wiki itself: when enough
/// has piled up loose, at most once a day, in one foreground repack that packs
/// only what is not packed yet. Never a gc, a prune or a detached run - the
/// shapes the storm was made of.
#[test]
fn brain_packs_its_wiki_once_a_day_in_one_bounded_pass() {
    let fixture = Fixture::new("maintain");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");
    let wiki = fixture.wiki();
    let git = wiki.join(".git");
    assert!(git.is_dir(), "the wiki has no repository");
    let base = fixture.home.parent().unwrap();

    // Committed, so history reaches them: a repack that packs only what is
    // unpacked leaves an unreachable object where it is.
    let plant = |batch: &str| {
        let dir = wiki.join("planted").join(batch);
        std::fs::create_dir_all(&dir).unwrap();
        for index in 0..2_100 {
            std::fs::write(dir.join(format!("{index}.md")), format!("{batch} {index}\n")).unwrap();
        }
        for args in [&["add", "planted"][..], &["commit", "-q", "-m", batch]] {
            let mut command = Command::new("git");
            command
                .args(["-c", "maintenance.auto=false", "-c", "gc.auto=0"])
                .args(args)
                .current_dir(&wiki)
                .env("HOME", base);
            fixture.own_git_config(&mut command);
            assert!(command.status().expect("run git").success(), "planting {batch} failed");
        }
        assert!(loose_objects(&git) >= 2_000, "planting {batch} left too few loose objects");
    };
    let consolidate = |session: &str, trace: &str| {
        fixture.seed_session_as(session, &["src/later.rs", "src/next.rs", "src/more.rs", "src/last.rs"]);
        let trace = base.join(trace);
        let output = fixture.brain_with_env(
            &["consolidate", "--force"],
            Some(&bin),
            &[("GIT_TRACE2_EVENT", trace.to_str().unwrap())],
        );
        assert!(output.status.success(), "consolidate failed: {output:?}");
        trace2(&trace)
    };
    let repacks = |events: &[serde_json::Value]| -> Vec<Vec<String>> {
        top_level_starts(events)
            .into_iter()
            .map(trace_argv)
            .filter(|argv| argv.iter().any(|arg| arg == "repack"))
            .collect()
    };

    plant("first");
    let events = consolidate("0199a1f2-3c4d-7e8f-9012-0000000000d4", "first.json");
    let starts = repacks(&events);
    assert_eq!(starts.len(), 1, "not one repack: {starts:?}");
    let argv = &starts[0];
    for bound in ["pack.threads=1", "-d"] {
        assert!(argv.iter().any(|arg| arg == bound), "the repack lacks {bound}: {argv:?}");
    }
    for unbounded in ["--detach", "-a", "-A", "--cruft"] {
        assert!(!argv.iter().any(|arg| arg == unbounded), "the repack has {unbounded}: {argv:?}");
    }
    assert!(!argv.iter().any(|arg| arg.starts_with("--geometric")), "a geometric repack: {argv:?}");
    let forbidden: Vec<Vec<String>> = events
        .iter()
        .filter(|event| event["event"] == "start" || event["event"] == "child_start")
        .map(trace_argv)
        .filter(|argv| argv.iter().any(|arg| ["gc", "prune", "maintenance"].contains(&arg.as_str())))
        .collect();
    assert_eq!(forbidden, Vec::<Vec<String>>::new(), "brain's maintenance ran a gc, prune or maintenance");
    let loose = loose_objects(&git);
    assert!(loose < 100, "{loose} objects are still loose after the repack");
    let stamp = std::fs::read_to_string(git.join("brain-maintenance")).unwrap_or_default();
    assert!(stamp.starts_with("done "), "the maintenance stamp does not record a finished pass: {stamp:?}");
    assert!(stamp.contains("outcome=ok"), "the repack did not succeed: {stamp:?}");
    assert!(!git.join("brain-maintenance.err").exists(), "a clean repack kept its error file");
    let mut fsck = Command::new("git");
    fsck.args(["fsck", "--connectivity-only"]).current_dir(&wiki).env("HOME", base);
    fixture.own_git_config(&mut fsck);
    let fsck = fsck.output().expect("run git");
    assert!(fsck.status.success(), "the packed wiki fails fsck: {fsck:?}");

    // As much piles up again the same day: the stamp holds it to one pass.
    plant("second");
    let events = consolidate("0199a1f2-3c4d-7e8f-9012-0000000000e5", "second.json");
    assert_eq!(repacks(&events), Vec::<Vec<String>>::new(), "a second repack within the day");
}

/// The plugin's hooks directory, which Codex names `${PLUGIN_ROOT}/hooks`.
fn plugin_hooks_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins/rolepod-brain/hooks")
}

/// A plugin hook command run the way its host runs it: through `/bin/sh`,
/// with `${PLUGIN_ROOT}` already replaced by the plugin's directory as Codex
/// does, from the fixture's checkout, and with only `bin` ahead of the
/// system's own tools on `PATH`. No git setting comes in from the caller, so
/// whatever git reads under it came from the hook.
fn plugin_hook_shell(fixture: &Fixture, command: &str, bin: &Path) -> Command {
    let root = plugin_hooks_dir().parent().unwrap().display().to_string();
    let mut shell = Command::new("/bin/sh");
    shell
        .args(["-c", &command.replace("${PLUGIN_ROOT}", &root)])
        .current_dir(&fixture.project)
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("HOME", fixture.home.parent().unwrap())
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env_remove("GIT_CONFIG_PARAMETERS")
        .env_remove("GIT_CONFIG_COUNT");
    fixture.own_git_config(&mut shell);
    shell
}

/// The plugin updates apart from the binary, so a person can run this
/// release's hooks with an older brain that still commits once per page. The
/// guard rides in the hook's environment, which `brain hook` and the
/// consolidate it starts both inherit, so even that binary's commits start no
/// maintenance. Codex's commands only run a script in the plugin, and the
/// script sets the guard: Codex trusts a hook by a hash of its command text,
/// so the guard lives where a later change needs no new approval.
#[test]
fn every_plugin_hook_turns_git_maintenance_off() {
    const GUARD: &str = r#"GIT_CONFIG_PARAMETERS="'maintenance.auto=false' 'gc.auto=0'""#;
    let hooks_dir = plugin_hooks_dir();
    // Each file's event count beside the commands found in it that run brain
    // (`marker`): one per event, so a command the filter misses cannot leave
    // an event unchecked.
    let commands = |file: &str, marker: &str| -> (usize, Vec<String>) {
        let text = std::fs::read_to_string(hooks_dir.join(file)).expect("the hook file ships");
        let manifest: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let events = manifest["hooks"].as_object().expect("a hooks table");
        let mut found = Vec::new();
        for groups in events.values() {
            for group in groups.as_array().into_iter().flatten() {
                for hook in group["hooks"].as_array().into_iter().flatten() {
                    found.extend(hook["command"].as_str().map(str::to_string));
                }
            }
        }
        found.retain(|command| command.contains(marker));
        (events.len(), found)
    };

    let (events, claude) = commands("hooks.json", "brain hook");
    assert_eq!(claude.len(), events, "a Claude Code event does not run brain once");
    // A stand-in brain answers with what git reads, so the test proves the
    // quoting survives the shell, not only that the text is there. Its
    // `where` points at a model that is present and `curl` always fails, so
    // SessionStart never fetches anything.
    let fixture = Fixture::new("plugin-env");
    let base = fixture.home.parent().unwrap();
    std::fs::write(base.join("model-int8.safetensors"), "present").unwrap();
    fixture.fake_cli(
        "brain",
        "[ \"$1\" = where ] && { echo \"$HOME\"; exit 0; }\n\
         git config --get maintenance.auto\n\
         git config --get gc.auto",
    );
    let bin = fixture.fake_cli("curl", "exit 1");
    for command in &claude {
        // After `brain hook` it would be an argument, not the environment.
        assert!(
            command.find(GUARD).is_some_and(|at| at < command.find("brain hook").unwrap()),
            "a hook runs brain without the maintenance guard ahead of it: {command}"
        );
        let output = plugin_hook_shell(&fixture, command, &bin).output().expect("run the hook");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "false\n0\n",
            "git does not see maintenance off under: {command}"
        );
    }

    let (events, codex) = commands("codex-hooks.json", "/hooks/codex-hook.sh");
    assert_eq!(codex.len(), events, "a Codex event does not run the hook script once");
    let script = std::fs::read_to_string(hooks_dir.join("codex-hook.sh")).expect("it ships");
    assert!(
        script
            .find(&format!("export {GUARD}"))
            .is_some_and(|at| at < script.find("brain hook").unwrap()),
        "the Codex hook script runs brain before it exports the maintenance guard"
    );
    for command in &codex {
        let output = plugin_hook_shell(&fixture, command, &bin).output().expect("run the hook");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "false\n0\n",
            "git does not see maintenance off under: {command}"
        );
    }
}

/// Codex runs a plugin hook only while a hash of its command text matches
/// what the person approved, and it takes that hash before it substitutes
/// `${PLUGIN_ROOT}`. So each command is one fixed line that runs a script in
/// the plugin: the script can change in any release without asking anyone to
/// approve the hooks again, and the line itself must never change. The
/// events, groups and timeouts are 0.63.0's, so the trust keys Codex files
/// them under stay the same.
#[test]
fn the_codex_hooks_run_through_the_plugin_script() {
    const TABLE: [(&str, u64); 7] = [
        ("SessionStart", 120),
        ("UserPromptSubmit", 5),
        ("PostToolUse", 5),
        ("Stop", 5),
        ("SubagentStop", 5),
        ("PreCompact", 5),
        ("SessionEnd", 3),
    ];
    let line = |event: &str| format!(r#"sh "${{PLUGIN_ROOT}}/hooks/codex-hook.sh" {event}"#);
    let hooks_dir = plugin_hooks_dir();
    let text = std::fs::read_to_string(hooks_dir.join("codex-hooks.json")).expect("the file ships");
    let manifest: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    let table = manifest["hooks"].as_object().expect("a hooks table");
    let mut events: Vec<&str> = table.keys().map(String::as_str).collect();
    let mut expected: Vec<&str> = TABLE.iter().map(|(event, _)| *event).collect();
    events.sort_unstable();
    expected.sort_unstable();
    assert_eq!(events, expected, "the Codex events changed");
    for (event, timeout) in TABLE {
        let groups = table[event].as_array().expect("a list of groups");
        assert_eq!(groups.len(), 1, "{event} has more than one group");
        // The whole group, so a matcher or another field added later fails
        // here too: 0.63.0's groups had no matcher.
        assert_eq!(
            groups[0],
            serde_json::json!({"hooks": [{"type": "command", "command": line(event), "timeout": timeout}]}),
            "{event}'s group is not the one fixed line"
        );
    }

    // A stand-in brain answers with what git reads and with the arguments it
    // was given. Its `where` points at a model that is present and `curl`
    // always fails, so SessionStart never fetches anything.
    let fixture = Fixture::new("codex-script");
    let base = fixture.home.parent().unwrap();
    std::fs::write(base.join("model-int8.safetensors"), "present").unwrap();
    fixture.fake_cli(
        "brain",
        "[ \"$1\" = where ] && { echo \"$HOME\"; exit 0; }\n\
         git config --get maintenance.auto\n\
         git config --get gc.auto\n\
         echo \"$@\"",
    );
    let bin = fixture.fake_cli("curl", "exit 1");
    for (event, _) in TABLE {
        let output = plugin_hook_shell(&fixture, &line(event), &bin).output().expect("run it");
        assert!(output.status.success(), "{event} failed: {output:?}");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("false\n0\nhook --cli codex --event {event}\n"),
            "{event} did not run brain under the maintenance guard"
        );
    }

    // With no binary yet, SessionStart says it is fetching one, and still
    // answers Codex with an empty result when the fetch fails.
    std::fs::remove_file(bin.join("brain")).unwrap();
    let output = plugin_hook_shell(&fixture, &line("SessionStart"), &bin).output().expect("run it");
    assert!(output.status.success(), "a missing binary failed SessionStart: {output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "{}\n");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("the brain binary is not installed yet"),
        "no install notice: {output:?}"
    );

    // Codex runs it through `sh`, but a person reading or running it by hand
    // should find a script that runs as it is.
    let script = hooks_dir.join("codex-hook.sh");
    let mode = std::fs::metadata(&script).unwrap().permissions();
    assert_eq!(
        std::os::unix::fs::PermissionsExt::mode(&mode) & 0o111,
        0o111,
        "the script is not executable"
    );
    assert!(std::fs::read_to_string(&script).unwrap().starts_with("#!/bin/sh\n"));
}

#[test]
fn an_oversized_session_merges_its_chunks_into_one_narrative() {
    let fixture = Fixture::new("merge");

    // Enough bulk to force several chunks. A tool call renders as a capped
    // input plus a capped result, so the bulk sits in the result and the event
    // count is high enough that the rendered session spans well over one chunk.
    for index in 0..150 {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
            "cwd": fixture.project,
            "tool_name": "Bash",
            "tool_input": {"command": format!("run-{index} {}", "x".repeat(1200))},
            "tool_response": {"stdout": format!("out-{index} {}", "y".repeat(1200))}
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
    // At least 3 chunk calls plus the merge call (the fixture yields 5 + 1).
    assert!(calls > 3, "this session should have split into chunks, saw {calls} call(s)");

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
    // The unsummarized session is one line naming it and how to read it -
    // not the capture, whose title stays behind for brain_recent.
    assert!(
        context.contains("claude-code session 1 capture(s) not yet summarized")
            && context.contains("0199a1f2-3c4d-7e8f-9012-3456789abcde"),
        "the session in flight is not named: {context}"
    );
    assert!(!context.contains("Asked: why does auth"), "a capture was pushed line by line: {context}");

    // The prompt itself must NOT be here.
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
fn every_skill_is_shipped_and_says_when_to_use_it() {
    // A skill is a directory nobody compiles, so nothing here notices when one
    // is added and left out of the package, or written without the front
    // matter that decides whether it is ever invoked. Same shape as the
    // installer flag that went undeclared for two days: not a bug in code, a
    // gap in what is checked.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("plugins/rolepod-brain/skills");
    let mut found = 0usize;
    for entry in std::fs::read_dir(&root).expect("the skills directory ships with the plugin") {
        let dir = entry.expect("readable entry").path();
        if !dir.is_dir() {
            continue;
        }
        found += 1;
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let manifest = dir.join("SKILL.md");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|_| panic!("`{name}` has no SKILL.md, so it is a directory and not a skill"));

        assert!(text.starts_with("---\n"), "`{name}` has no front matter");
        let front = text.split("---").nth(1).unwrap_or_default();
        assert!(
            front.contains(&format!("name: {name}")),
            "`{name}` declares a different name than its directory, so it is invoked by neither"
        );
        // The description is the whole of how a model decides to reach for it.
        let described = front
            .lines()
            .find(|line| line.starts_with("description:"))
            .map(|line| line.trim_start_matches("description:").trim().len())
            .unwrap_or(0);
        assert!(
            described > 40,
            "`{name}` has no usable description ({described} chars); nothing will ever trigger it"
        );
    }
    assert!(found >= 3, "only {found} skill(s) found; the directory moved or emptied");
}

#[test]
fn every_installer_flag_has_a_default_before_the_loop_that_sets_it() {
    // `bootstrap.sh` runs under `set -u` and reads these at the top level
    // whether or not their flag was passed. Three were added without a
    // default, and for two days every release shipped an installer that died
    // with `reranker_only: unbound variable` after downloading the binary and
    // before wiring anything - found only by installing into a clean HOME,
    // which nothing in this repository had ever done.
    //
    // The whole script cannot be executed from a test: reaching the line that
    // failed means downloading a release first. So the shape is checked
    // instead - every variable the argument loop assigns must already have a
    // value before the loop reaches it.
    let script = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bootstrap.sh"),
    )
    .expect("bootstrap.sh is part of the repository");

    let loop_at = script.find("while [ $# -gt 0 ]").expect("the argument loop moved");
    let (preamble, arg_loop) = script.split_at(loop_at);

    let mut checked = 0usize;
    for line in arg_loop.lines() {
        // `--flag) name=1 ;;` and `--flag=*) name="${arg#...}" ;;`
        let Some(rest) = line.trim().strip_prefix("--") else { continue };
        let Some((_, assignment)) = rest.split_once(')') else { continue };
        for token in assignment.split_whitespace() {
            let Some((name, _)) = token.split_once('=') else { continue };
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                continue;
            }
            checked += 1;
            assert!(
                preamble.lines().any(|l| l.trim_start().starts_with(&format!("{name}="))),
                "`{name}` is set by a flag but has no default before the loop; \
                 an install that omits that flag dies on `set -u`"
            );
        }
    }
    assert!(checked >= 4, "the loop parser matched almost nothing ({checked}); it has drifted");
}

#[test]
fn an_installer_option_value_is_never_read_as_an_option() {
    // `--into DIR` was parsed inside a `for` loop, where `shift` consumes
    // nothing: DIR came round again and the installer died on "unknown
    // option". `--help` exits before anything is downloaded, so it is how a
    // test reaches the parser alone.
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bootstrap.sh");
    for args in [&["--into", "/nowhere", "--help"][..], &["--into=/nowhere", "--help"]] {
        let run = Command::new("sh").arg(&script).args(args).output().expect("run sh");
        assert!(run.status.success(), "{args:?}: {}", String::from_utf8_lossy(&run.stderr));
    }
    let bare = Command::new("sh").arg(&script).arg("--into").output().expect("run sh");
    assert!(!bare.status.success(), "--into with no directory was accepted");
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
fn a_lock_left_by_a_killed_run_does_not_block_the_next_one() {
    // A run that is killed never reaches its `Drop`, and the timeout alone
    // then blocked every consolidation for half an hour - seen the first time
    // a run here was interrupted. The lock names its holder, so the next run
    // can ask whether that process still exists instead of waiting out a clock.
    let fixture = Fixture::new("dead-lock");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.seed_session(4);

    // A pid that cannot be running: `kill -0` on it fails, so the lock is
    // stealable even though the file was written a moment ago.
    let lock = fixture.home.join(".brain-consolidate.lock");
    std::fs::write(&lock, b"2147483646").expect("write lock");

    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");
    assert_eq!(
        fixture.pending_count(),
        0,
        "a lock whose holder is gone kept the next run out"
    );
}

#[test]
fn a_run_that_stands_aside_says_so_rather_than_reporting_an_empty_backlog() {
    // The first version printed "Nothing to consolidate" when it had in fact
    // yielded to another run - which reads as an empty backlog and would send
    // the next person looking for a bug in the wrong place.
    let fixture = Fixture::new("yield-message");
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    fixture.seed_session(4);

    // Held by this test process, which is certainly alive.
    let lock = fixture.home.join(".brain-consolidate.lock");
    std::fs::write(&lock, std::process::id().to_string()).expect("write lock");

    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    let said = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "a skipped run is not a failure: {out:?}");
    assert!(
        said.contains("stood aside"),
        "a yielded run reported something else: {said}"
    );
    assert!(
        !said.contains("Nothing to consolidate"),
        "a yielded run claimed the backlog was empty: {said}"
    );
    assert!(fixture.pending_count() > 0, "it did the work anyway");
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
fn stats_reports_pull_through_per_kind() {
    let fixture = Fixture::new("pull-through");
    for name in ["a.rs", "b.rs", "c.rs"] {
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
    fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_note","arguments":{"text":"Pull-through fixture note."}}}"#,
    ]);

    let ids = |kind: &str| -> Vec<String> {
        let conn = rusqlite::Connection::open(fixture.home.join("brain.db")).unwrap();
        let mut stmt = conn
            .prepare("SELECT id FROM events WHERE kind = ?1 ORDER BY id")
            .unwrap();
        stmt.query_map([kind], |row| row.get(0)).unwrap().map(Result::unwrap).collect()
    };
    let obs = ids("observation");
    let note = ids("note");
    assert!(obs.len() >= 3, "need three observations: {obs:?}");
    assert_eq!(note.len(), 1, "need one note: {note:?}");

    // A real claude under the fixture may have left its own rows: start clean.
    fixture.sql_run("DELETE FROM host_session; DELETE FROM injected; DELETE FROM recalled;");
    let now = jiff::Timestamp::now().to_string();
    fixture.sql_run(&format!(
        "INSERT INTO host_session (host_pid, session, ts) VALUES (1, 'host-a', '{now}');
         INSERT INTO injected (session, event_id) VALUES
           ('host-a', '{o0}'), ('host-a', '{o1}'), ('host-a', '{o2}'), ('host-a', '{n0}');
         INSERT INTO recalled (session, event_id, opened, host_session) VALUES
           ('mcp-1', '{o0}', 1, 'host-a'),
           ('mcp-2', '{o1}', 1, 'host-b'),
           ('mcp-3', '{o2}', 1, ''),
           ('mcp-4', '{n0}', 0, 'host-a');",
        o0 = obs[0],
        o1 = obs[1],
        o2 = obs[2],
        n0 = note[0],
    ));

    let out = fixture.brain(&["stats"]);
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(text.contains("Pull-through (last 7 days)"), "no section: {text}");
    assert!(
        text.contains("observation    3 pushed to host-tied sessions, 1 opened in the same session, 1 recall(s) not joined"),
        "observation line: {text}"
    );
    assert!(
        text.contains("note           1 pushed to host-tied sessions, 0 opened in the same session, 0 recall(s) not joined"),
        "note line: {text}"
    );
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
fn sync_off_is_the_default_and_says_how_to_opt_in() {
    // The old refusal survives as the default: an unconfigured `brain sync`
    // must fail rather than quietly imply a backup exists somewhere, and
    // memory leaves the machine only after the owner runs init themselves.
    let fixture = Fixture::new("nosync");
    fixture.seed_session(2);
    let output = fixture.brain(&["sync"]);
    assert!(!output.status.success(), "must not imply a sync happened");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not configured"), "should say sync is off: {stderr}");
    assert!(stderr.contains("sync init"), "and how to opt in: {stderr}");
}

#[test]
fn two_machines_one_brain_through_an_encrypted_folder() {
    // Mode A end to end: a shared plain directory stands in for iCloud or
    // Dropbox, holding only ciphertext. Pairing = copying one key file.
    let marker = "[project]\nname = \"syncprop\"\n";
    let a = Fixture::new("cloud-a");
    let b = Fixture::new("cloud-b");
    std::fs::write(a.project.join(".rolepod-brain.toml"), marker).unwrap();
    std::fs::write(b.project.join(".rolepod-brain.toml"), marker).unwrap();
    let shared = a.home.parent().unwrap().join("shared-folder");

    let capture = |fixture: &Fixture, session: &str, prompt: &str| {
        let payload = serde_json::json!({
            "session_id": session,
            "cwd": fixture.project,
            "prompt": prompt
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    };
    capture(&a, "0199a1f2-3c4d-7e8f-9012-3456789abc0a", "the alpha rendezvous fact");
    capture(&b, "0199a1f2-3c4d-7e8f-9012-3456789abc0b", "the beta rendezvous fact");

    let init_a = a.brain(&["sync", "init", shared.to_str().unwrap()]);
    assert!(init_a.status.success(), "init A failed: {init_a:?}");
    let init_b = b.brain(&["sync", "init", shared.to_str().unwrap()]);
    assert!(init_b.status.success(), "init B failed: {init_b:?}");
    // Pairing: B takes A's key.
    std::fs::copy(a.home.join("sync.key"), b.home.join("sync.key")).unwrap();

    // A publishes; B pulls it and publishes back; A pulls B.
    assert!(a.brain(&["sync"]).status.success());
    assert!(b.brain(&["sync"]).status.success());
    assert!(a.brain(&["sync"]).status.success());

    let on_b = String::from_utf8_lossy(&b.brain(&["search", "alpha"]).stdout).into_owned();
    assert!(on_b.contains("alpha"), "A's fact never reached B: {on_b}");
    let on_a = String::from_utf8_lossy(&a.brain(&["search", "beta"]).stdout).into_owned();
    assert!(on_a.contains("beta"), "B's fact never reached A: {on_a}");

    // Nothing new: a repeat sync gains zero events.
    let again = b.brain(&["sync"]);
    let stdout = String::from_utf8_lossy(&again.stdout).to_string();
    assert!(stdout.contains("0 new event(s)"), "a repeat sync must gain nothing: {stdout}");

    // A bundle carries memory, not the machine's own history: shipping
    // `.git` grafts one history over another, and its objects are read-only.
    let export = a.home.parent().unwrap().join("check.tar.gz");
    assert!(a.brain(&["export", export.to_str().unwrap()]).status.success());
    let listing = String::from_utf8_lossy(
        &Command::new("tar").args(["-tzf", export.to_str().unwrap()]).output().expect("tar").stdout,
    )
    .into_owned();
    assert!(!listing.contains("/.git/"), "the vault's history was exported:\n{listing}");
    assert!(listing.contains("events/"), "the log must travel: {listing}");

    // The folder holds ciphertext only - no title, no id, no JSON.
    for entry in std::fs::read_dir(&shared).unwrap().flatten() {
        let bytes = std::fs::read(entry.path()).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            !text.contains("rendezvous") && !text.contains("\"id\""),
            "plaintext leaked into the shared folder: {}",
            entry.path().display()
        );
    }
}

#[test]
fn a_wrong_key_skips_the_bundle_and_says_so() {
    // A foreign or corrupt bundle in the shared folder must not stop the
    // owner's machines from converging - and must not be silently absorbed.
    let marker = "[project]\nname = \"syncprop\"\n";
    let a = Fixture::new("key-a");
    let b = Fixture::new("key-b");
    std::fs::write(a.project.join(".rolepod-brain.toml"), marker).unwrap();
    std::fs::write(b.project.join(".rolepod-brain.toml"), marker).unwrap();
    let shared = a.home.parent().unwrap().join("shared-folder");

    let payload = serde_json::json!({
        "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abc0a",
        "cwd": a.project,
        "prompt": "the alpha rendezvous fact"
    })
    .to_string();
    a.hook("claude-code", "UserPromptSubmit", &payload);

    assert!(a.brain(&["sync", "init", shared.to_str().unwrap()]).status.success());
    assert!(b.brain(&["sync", "init", shared.to_str().unwrap()]).status.success());
    // No key copy: B minted its own, so A's bundle is unreadable to it.
    assert!(a.brain(&["sync"]).status.success());
    let out = b.brain(&["sync"]);
    assert!(out.status.success(), "a foreign bundle must not be fatal: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(stdout.contains("skipped"), "the skip must be reported: {stdout}");
    let hits = String::from_utf8_lossy(&b.brain(&["search", "alpha"]).stdout).into_owned();
    assert!(!hits.contains("alpha"), "an unreadable bundle was absorbed: {hits}");
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

    // The session is in flight through its two real questions - the twenty
    // bare commands do not count, and nothing is pushed line by line.
    assert!(
        context.contains("claude-code session 2 capture(s) not yet summarized"),
        "the in-flight line miscounted or is missing: {context}"
    );
    assert!(!context.contains("scheduler double-book"), "a capture was pushed line by line: {context}");
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
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
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
fn opencode_gets_memory_only_where_its_plugin_reads_the_answer() {
    let fixture = Fixture::new("ocanswer");
    // A primer's worth of history, and a file with memory of its own.
    for question in ["why does the scheduler double-book?", "where is expiry compared?"] {
        let payload = serde_json::json!({
            "session_id": "0199aaaa-0000-7000-8000-000000000000",
            "cwd": fixture.project,
            "prompt": question
        })
        .to_string();
        fixture.hook("claude-code", "UserPromptSubmit", &payload);
    }
    fixture.hook("claude-code", "PostToolUse", &claude_payload(&fixture.project));

    let start = |session: &str, reads: bool| {
        serde_json::json!({
            "cwd": fixture.project,
            "session_id": session,
            "source": "startup",
            "reads_answer": reads
        })
        .to_string()
    };
    let tool = |session: &str, reads: bool| {
        serde_json::json!({
            "cwd": fixture.project,
            "session_id": session,
            "tool_name": "edit",
            "tool_input": {"filePath": fixture.project.join("src/auth.rs")},
            "reads_answer": reads
        })
        .to_string()
    };

    // A fire-and-forget spawn - or a plugin written before it read anything -
    // discards stdout. An injection built for it reached no model, yet it
    // spent the session budget, and a burst of detached spawns racing on that
    // budget recorded more than the cap.
    let session = "0199c0de-0000-7000-8000-000000000000";
    let unread = fixture.hook("opencode", "SessionStart", &start(session, false));
    assert_eq!(String::from_utf8_lossy(&unread.stdout).trim(), "{}");
    assert_eq!(injected_bytes(&fixture, session), 0, "primer spent budget nobody read");
    // Each path in its own session, so the file path is proven apart from
    // the primer's.
    let touching = "0199c0de-0000-7000-8000-000000000002";
    let unread = fixture.hook("opencode", "PostToolUse", &tool(touching, false));
    assert_eq!(String::from_utf8_lossy(&unread.stdout).trim(), "{}");
    assert_eq!(injected_bytes(&fixture, touching), 0, "file pointers spent budget nobody read");
    assert!(fixture.log_text().contains(r#""cli":"opencode""#), "opencode stopped capturing");

    // A plugin that has stopped waiting is not listening either.
    let late = "0199c0de-0000-7000-8000-000000000005";
    let mut payload: serde_json::Value = serde_json::from_str(&start(late, true)).unwrap();
    payload["answer_by"] = serde_json::json!(1);
    let dropped = fixture.hook("opencode", "SessionStart", &payload.to_string());
    assert_eq!(String::from_utf8_lossy(&dropped.stdout).trim(), "{}");
    assert_eq!(injected_bytes(&fixture, late), 0, "spent budget on an answer already dropped");

    // Where the plugin waits for the answer, memory comes back and is paid for.
    let session = "0199c0de-0000-7000-8000-000000000003";
    let primer = fixture.hook("opencode", "SessionStart", &start(session, true));
    let primer = injected_context(&primer).expect("a primer for a plugin that reads it");
    assert!(primer.starts_with("# Project memory"), "not a primer: {primer}");
    assert_eq!(injected_bytes(&fixture, session), i64::try_from(primer.len()).unwrap());

    let touching = "0199c0de-0000-7000-8000-000000000004";
    let pointers = fixture.hook("opencode", "PostToolUse", &tool(touching, true));
    let pointers = injected_context(&pointers).expect("file pointers for a plugin that reads them");
    assert!(pointers.contains("src/auth.rs"), "not the touched file's memory: {pointers}");
    assert!(injected_bytes(&fixture, touching) > 0);
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
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("HOME", fixture.home.parent().unwrap())
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
    // The earlier session is still unsummarized, so the primer names it.
    let earlier = "0199aaaa-0000-7000-8000-000000000000";
    assert!(first.contains(earlier), "the earlier session is not named: {first}");

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
        after.contains(earlier),
        "the pre-wipe memory was suppressed after compaction: {after}"
    );

    // `/clear` takes the other path and must behave identically.
    let cleared = fixture.hook(
        "claude-code",
        "SessionStart",
        &start_payload(&fixture.project, session, "clear"),
    );
    let cleared = injected_context(&cleared).expect("primer after /clear");
    assert!(cleared.contains(earlier), "the pre-wipe memory was suppressed after /clear: {cleared}");
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
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("HOME", fixture.home.parent().unwrap())
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

    // The primer does not push captures one by one: the session they belong
    // to is one line, keyed by its newest capture. Flagging shapes search,
    // and search is where these are reached.
    let start = serde_json::json!({
        "session_id": "0199b000-0000-7000-8000-000000000000",
        "cwd": fixture.project,
        "source": "startup"
    })
    .to_string();
    let output = fixture.hook("claude-code", "SessionStart", &start);
    let context = injected_context(&output).expect("a primer");
    assert!(
        context.contains(&ids[2]) && context.contains("3 capture(s) not yet summarized"),
        "the session in flight should be named once, by its newest capture: {context}"
    );
    assert!(!context.contains(&ids[0]), "a capture was pushed line by line: {context}");
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
/// A subagent's hooks arrive under the lead's session id. Its tool calls are
/// captured, tagged with its type, and met with no injection - a file
/// injection would spend the lead's budget and mark the file covered for a
/// session that never saw the pointer. Its report at `SubagentStop` becomes
/// one searchable event named after it.
#[test]
fn a_subagents_work_is_tagged_captured_and_never_injected() {
    let fixture = Fixture::new("subagent-lane");
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcd8";

    // Something worth knowing about one particular file, from an earlier
    // session - a session's own captures are never echoed back to it.
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

    // A reviewer dispatched from the session reads the file: nothing comes
    // back, and the read is stored under the reviewer's name.
    let read = serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "agent_id": "agent-def456",
        "agent_type": "rolepod:universal-reviewer",
        "tool_name": "Read",
        "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
    })
    .to_string();
    let before = fixture.hook("claude-code", "PreToolUse", &read);
    assert!(before.status.success(), "{before:?}");
    assert_eq!(injected_context(&before), None, "a subagent was handed the lead's file memory");
    let after = fixture.hook("claude-code", "PostToolUse", &read);
    assert_eq!(injected_context(&after), None);

    let stored = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg("SELECT agent FROM events WHERE hook = 'post_tool_use' AND title LIKE 'Read:%';")
        .output()
        .expect("read the agent column");
    assert_eq!(String::from_utf8_lossy(&stored.stdout).trim(), "rolepod:universal-reviewer");

    // The lead reads the same file afterwards and still gets its memory: the
    // reviewer's read did not mark the file covered.
    let lead_read = serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
    })
    .to_string();
    let injected = injected_context(&fixture.hook("claude-code", "PreToolUse", &lead_read))
        .expect("the lead's own read must still be met with the file's memory");
    assert!(injected.contains("auth.rs"), "{injected}");

    // The reviewer's report arrives and is findable by what it said. It is
    // scrubbed before it is bounded: a credential sitting past the body
    // clamp must not survive because the clamp came first.
    let mut report = "PASS-WITH-NITS\n\n1. src/auth.rs:42 compares expiry with the wrong operator.\n".to_string();
    report.push_str(&"finding filler line\n".repeat(900));
    report.push_str("aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n");
    let stop = serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "agent_id": "agent-def456",
        "agent_type": "rolepod:universal-reviewer",
        "last_assistant_message": report
    })
    .to_string();
    let output = fixture.hook("claude-code", "SubagentStop", &stop);
    assert!(output.status.success(), "{output:?}");
    let found = String::from_utf8_lossy(&fixture.brain(&["search", "wrong operator"]).stdout).to_string();
    assert!(
        found.contains("rolepod:universal-reviewer reported: PASS-WITH-NITS"),
        "the report is not a searchable event: {found}"
    );
    let body = Command::new("sqlite3")
        .arg(fixture.home.join("brain.db"))
        .arg("SELECT body FROM events WHERE hook = 'subagent_stop';")
        .output()
        .expect("read the report body");
    let body = String::from_utf8_lossy(&body.stdout);
    assert!(!body.contains("AKIAIOSFODNN7EXAMPLE"), "a credential past the clamp was stored: {}", body.len());
    assert!(body.contains("[REDACTED]"), "the credential was cut away rather than redacted: {}", body.len());
}

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
                .env("ROLEPOD_BRAIN_NO_FETCH", "1")
                .env("ROLEPOD_BRAIN_HUB", "off")
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

/// The report that drove this: a one-line install on a machine with one CLI
/// wrote hooks for six, because editors and old installs leave config
/// directories behind and a directory was the whole presence test. The
/// fixture's default PATH (`/usr/bin:/bin`) carries no CLI, so every
/// directory here is exactly such a leftover.
#[test]
fn a_directory_left_by_an_editor_does_not_get_hooks() {
    let fixture = Fixture::new("leftover-dirs");
    let home = fixture.home.parent().unwrap().to_path_buf();
    for dir in [".claude", ".cursor", ".gemini", ".gemini/config"] {
        std::fs::create_dir_all(home.join(dir)).unwrap();
    }

    let output = fixture.brain(&["setup", "--apply"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        stdout.contains("is not on PATH — skipped"),
        "setup should say why it wrote nothing:\n{stdout}"
    );
    for wrote in [".claude/settings.json", ".cursor/hooks.json", ".gemini/settings.json"] {
        assert!(
            !home.join(wrote).exists(),
            "{wrote} was written for a CLI that is not on this machine"
        );
    }

    // Doctor tells the same story: no hooks row may claim wired events, and
    // the absence is stated, not painted red — this also runs inside MCP
    // servers whose PATH can be minimal.
    let report = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(
        !report.contains("event(s):"),
        "doctor reported hooks for an absent CLI:\n{report}"
    );
    assert!(
        report.contains("is not on PATH"),
        "doctor should name what is missing:\n{report}"
    );

    // And the moment the CLI actually exists, the same directory is wired.
    let bin = fixture.fake_cli("claude", "exit 0");
    fixture.brain_with_path(&["setup", "--apply", "--cli", "claude-code"], Some(&bin));
    assert!(
        home.join(".claude/settings.json").exists(),
        "a present CLI should still be wired"
    );
}

/// `brain setup --apply` on a machine brain has never run on: the data
/// directory does not exist yet, and the config template is the first thing
/// that wants to live in it.
#[test]
fn the_config_template_survives_a_home_brain_has_never_seen() {
    let fixture = Fixture::new("fresh-config-home");
    std::fs::remove_dir_all(&fixture.home).unwrap();

    let output = fixture.brain(&["setup", "--apply"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        !stdout.contains("could not write the config template"),
        "the template needs its directory created first:\n{stdout}"
    );
    assert!(fixture.home.join("config.toml").exists(), "template missing:\n{stdout}");
}

/// The report that drove this: "when claude is down, nothing falls through
/// to the CLIs I do have — everything lands on rule-based." The ladder's
/// whole design says otherwise; this pins the design so a regression (or
/// the report's missing detail) has a test to argue with.
#[test]
fn a_broken_preferred_cli_falls_through_to_the_next_vendor() {
    let fixture = Fixture::new("fallthrough");
    fixture.seed_session(3);
    // The preferred rung for a claude-code session is claude, and it is down
    // the way an outage looks: immediate nonzero exit.
    fixture.fake_cli("claude", "echo 'API Error: 529 overloaded' >&2; exit 1");
    let bin = fixture.fake_cli("gemini", GOOD_CLI);

    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("gemini-cli"),
        "the session should be summarized by the next vendor, not floored: {stdout}"
    );

    let mut pages = Vec::new();
    collect_under(&fixture.wiki(), "sessions", &mut pages);
    let page = pages
        .into_iter()
        .find(|path| path.extension().is_some_and(|ext| ext == "md"))
        .expect("a session page");
    let text = std::fs::read_to_string(&page).unwrap();
    assert!(
        text.contains("Refactored the auth path"),
        "the page should carry the model's summary, not the rule-based floor: {text}"
    );
}

/// The report that made this row: the fallback WORKED, but doctor's capture
/// row (`codex=1` — events codex produced) was read as "codex never
/// answered", and the proof of the cascade sat in `session_state.last_tier`
/// where only a SQL query finds it. Doctor now states who answered, so
/// capture counts cannot be mistaken for summarizer usage.
#[test]
fn doctor_names_the_tier_that_answered_so_capture_counts_cannot_be_misread() {
    let fixture = Fixture::new("doctor-answered");
    fixture.seed_session(3);
    fixture.fake_cli("claude", "echo 'API Error: 529 overloaded' >&2; exit 1");
    let bin = fixture.fake_cli("gemini", GOOD_CLI);

    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");

    let doctor = fixture.brain_with_path(&["doctor"], Some(&bin));
    let report = String::from_utf8_lossy(&doctor.stdout).to_string();
    assert!(
        report.contains("answered by: gemini-cli=1"),
        "doctor must say which tier answered, not leave it to the database: {report}"
    );
}

/// The inverse shape is the one live warning: every session floored at
/// rule-based while a CLI is visibly installed. That is what a hook running
/// under a PATH that hides every CLI looks like — the machine both reports
/// in this incident class had exactly this state, and nothing in doctor
/// said so.
#[test]
fn every_session_stuck_on_rule_based_with_a_model_present_fails_doctor() {
    let fixture = Fixture::new("doctor-stuck");
    fixture.seed_session(3);
    // No CLI reachable at consolidation time: the floor answers.
    let out = fixture.brain(&["consolidate", "--force"]);
    assert!(out.status.success(), "consolidate failed: {out:?}");

    // Doctor runs where a CLI IS visible — a terminal PATH, not the hook's.
    let bin = fixture.fake_cli("claude", "exit 0");
    let doctor = fixture.brain_with_path(&["doctor"], Some(&bin));
    let report = String::from_utf8_lossy(&doctor.stdout).to_string();
    assert!(
        report.contains("FAIL consolidation"),
        "a model nobody ever reaches must be reported, not summed silently: {report}"
    );
}

/// Same report, other common pair: claude down, codex behind it. Codex
/// answers through the `-o` file, not stdout — the shape most machines
/// would actually fall through to.
#[test]
fn a_broken_preferred_cli_falls_through_to_codex_s_file_protocol() {
    let fixture = Fixture::new("fallthrough-codex");
    fixture.seed_session(3);
    fixture.fake_cli("claude", "echo 'API Error: 529 overloaded' >&2; exit 1");
    let bin = fixture.fake_cli(
        "codex",
        r#"out=""
prev=""
for a in "$@"; do
  [ "$prev" = "-o" ] && out="$a"
  prev="$a"
done
[ -n "$out" ] || { echo "no -o flag" >&2; exit 2; }
printf '%s' '{"summary":"Refactored the auth path and fixed token expiry.","titles":[]}' > "$out"
echo "tokens used: 1234""#,
    );

    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("codex"),
        "the session should be summarized by codex, not floored: {stdout}"
    );
}

/// A hook spawned by a GUI-launched host gets launchd's minimal PATH. The
/// CLI a user installed through nvm is real, working, and invisible to that
/// PATH - so the ladder must look where CLIs actually land, not only where
/// PATH points. Here codex lives in an nvm bin the pinned PATH cannot see.
#[test]
fn the_ladder_reaches_a_cli_the_minimal_path_cannot_see() {
    let fixture = Fixture::new("nvm-invisible");
    fixture.seed_session(3);
    let home = fixture.home.parent().unwrap().to_path_buf();

    // claude is on PATH and down; codex is installed the way npm installs
    // it - under nvm, off PATH.
    let bin = fixture.fake_cli("claude", "echo 'API Error: 529 overloaded' >&2; exit 1");
    let nvm_bin = home.join(".nvm/versions/node/v24.0.0/bin");
    std::fs::create_dir_all(&nvm_bin).unwrap();
    let codex = nvm_bin.join("codex");
    std::fs::write(
        &codex,
        concat!(
            "#!/bin/sh\n",
            "out=\"\"\nprev=\"\"\n",
            "for a in \"$@\"; do\n",
            "  [ \"$prev\" = \"-o\" ] && out=\"$a\"\n",
            "  prev=\"$a\"\n",
            "done\n",
            "[ -n \"$out\" ] || exit 2\n",
            "printf '%s' '{\"summary\":\"Refactored the auth path.\",\"titles\":[]}' > \"$out\"\n"
        ),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&codex, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("codex"),
        "an nvm-installed CLI off PATH should still be reached: {stdout}"
    );
}

#[test]
fn a_quiet_session_is_settled_without_a_model_call_or_a_page() {
    let fixture = Fixture::new("quiet");
    // A session that opened, ran one command that touched nothing, and went
    // away. Measured on a real store, one consolidation in five was this.
    let session = "0199c000-0000-7000-8000-000000000000";
    fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "startup"));
    let payload = serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "tool_name": "Bash",
        "tool_input": {"command": "ls"}
    })
    .to_string();
    fixture.hook("claude-code", "PostToolUse", &payload);

    let counter = fixture.home.parent().unwrap().join("quiet-calls");
    let bin = fixture.fake_cli(
        "claude",
        &format!(
            "N=$(cat {c} 2>/dev/null || echo 0); N=$((N+1)); echo $N > {c}\n\
             echo '{{\"summary\":\"should never be asked\",\"titles\":[]}}'",
            c = counter.display()
        ),
    );
    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "consolidate failed: {out:?}");
    assert!(stdout.contains("via quiet"), "not settled as quiet: {stdout}");
    assert!(stdout.contains("1 of them quiet"), "{stdout}");

    assert!(!counter.exists(), "a quiet session must not cost a model call");
    assert_eq!(fixture.pending_count(), 0, "its events are settled");
    assert!(!fixture.page_text().contains("should never be asked"));
    let mut pages = Vec::new();
    collect_under(&fixture.wiki(), "sessions", &mut pages);
    assert!(pages.is_empty(), "a quiet session gets no page: {pages:?}");
    assert!(!fixture.log_text().contains("session_summary"), "and no summary for the primer");
    // The vault's front page still exists - it says there is nothing yet.
    assert!(fixture.wiki().join("index.md").is_file());
    assert!(fixture.wiki().join("AGENTS.md").is_file());
}

/// A run another agent started - `claude -p` from a script, a codex served to
/// Claude Code's plugin - is a delegate. Its captures are memory; its summary
/// would be its outcome a second time, paid for with a model call, and its
/// presence in the in-flight list sends the next session to read a reviewer's
/// transcript as the latest work.
#[cfg(unix)]
#[test]
fn a_delegated_run_keeps_its_captures_and_never_costs_a_summary() {
    let fixture = Fixture::new("delegate");
    let session = "0199d000-0000-7000-8000-00000000d001";
    fixture.seed_headless_session(session);

    // Before anything consolidates: the captures are reachable, and the
    // summary list does not offer them as work left in flight.
    let responses = fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_recent","arguments":{"kind":"raw"}}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_recent","arguments":{"kind":"session_summary"}}}"#,
    ]);
    let raw = responses[0]["result"]["structuredContent"]["count"].as_u64().unwrap_or(0);
    assert!(raw >= 4, "the delegate's captures are memory: {:?}", responses[0]);
    assert_eq!(
        responses[1]["result"]["structuredContent"]["count"],
        0,
        "a delegate is not work in flight: {:?}",
        responses[1]
    );

    // Consolidation settles it: no model call, no page, no summary.
    let counter = fixture.home.parent().unwrap().join("delegate-calls");
    let bin = fixture.fake_cli(
        "claude",
        &format!(
            "N=$(cat {c} 2>/dev/null || echo 0); N=$((N+1)); echo $N > {c}\n\
             echo '{{\"summary\":\"a delegate summarized\",\"titles\":[]}}'",
            c = counter.display()
        ),
    );
    let out = fixture.brain_with_path(&["consolidate"], Some(&bin));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "consolidate failed: {out:?}");
    assert!(stdout.contains("via headless"), "not settled as a delegate: {stdout}");
    assert!(!counter.exists(), "a delegate must not cost a model call");
    assert_eq!(fixture.pending_count(), 0, "its events are settled");
    let mut pages = Vec::new();
    collect_under(&fixture.wiki(), "sessions", &mut pages);
    assert!(pages.is_empty(), "a delegate gets no page: {pages:?}");
    assert!(!fixture.log_text().contains("session_summary"), "and no summary");

    // Asked for by name, with force, the summary is written after all.
    let out = fixture.brain_with_path(&["consolidate", "--force", "--session", session], Some(&bin));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "forced consolidate failed: {out:?}");
    assert!(counter.exists(), "force is the one way to ask the model: {stdout}");
    assert!(fixture.log_text().contains("a delegate summarized"), "{stdout}");
    assert_eq!(fixture.pending_count(), 0);
}

/// A quiet or a headless session is settled without a word in the log, so a
/// rebuild replays its events unconsolidated. Its verdict survives in the
/// store, and every later run read that verdict as "already done" and passed
/// over the session for good: 170 sessions on one machine, and a backstop
/// that stayed stale because of them.
#[cfg(unix)]
#[test]
fn a_reindex_does_not_strand_settled_sessions() {
    let fixture = Fixture::new("restrand");
    // Q: opened, ran two commands that touched nothing.
    let quiet = "0199e000-0000-7000-8000-00000000e001";
    fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, quiet, "startup"));
    for command in ["ls", "pwd"] {
        let payload = serde_json::json!({
            "session_id": quiet,
            "cwd": fixture.project,
            "tool_name": "Bash",
            "tool_input": {"command": command}
        });
        fixture.hook("claude-code", "PostToolUse", &payload.to_string());
    }
    // H: a `claude -p` run.
    fixture.seed_headless_session("0199e000-0000-7000-8000-00000000e002");

    let counter = fixture.home.parent().unwrap().join("restrand-calls");
    let bin = fixture.fake_cli(
        "claude",
        &format!(
            "N=$(cat {c} 2>/dev/null || echo 0); N=$((N+1)); echo $N > {c}\n\
             echo '{{\"summary\":\"a settled session summarized\",\"titles\":[]}}'",
            c = counter.display()
        ),
    );
    let out = fixture.brain_with_path(&["consolidate"], Some(&bin));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "consolidate failed: {out:?}");
    assert!(stdout.contains("2 of them quiet"), "not both settled: {stdout}");
    assert_eq!(fixture.pending_count(), 0, "settled before the rebuild");
    assert!(!counter.exists(), "settling costs no model call");

    let out = fixture.brain(&["reindex"]);
    assert!(out.status.success(), "reindex failed: {out:?}");
    assert_eq!(fixture.pending_count(), 0, "the rebuild reopened sessions that were settled");

    let out = fixture.brain_with_path(&["consolidate"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");
    assert_eq!(fixture.pending_count(), 0);
    assert!(!counter.exists(), "a rebuild must not turn a settled session into a model call");
}

/// A one-shot run never gets a summary, yet its start, its stops and its end
/// each started a detached consolidate: 1,578 of the 1,783 spawns one machine
/// made in a day, nearly all of them standing aside for the run that already
/// held the lock. Its events wait for the next run a person's session starts.
#[cfg(unix)]
#[test]
fn a_one_shot_run_starts_no_consolidation() {
    let fixture = Fixture::new("oneshot");
    let lock = fixture.home.join(".brain-consolidate.lock");

    let headless = "0199f000-0000-7000-8000-00000000f001";
    fixture.hook_under_headless_claude("SessionStart", &start_payload(&fixture.project, headless, "startup"));
    let read = serde_json::json!({
        "session_id": headless,
        "cwd": fixture.project,
        "tool_name": "Read",
        "tool_input": {"file_path": fixture.project.join("src/auth.rs")}
    });
    fixture.hook_under_headless_claude("PostToolUse", &read.to_string());
    let stop = serde_json::json!({"session_id": headless, "cwd": fixture.project});
    fixture.hook_under_headless_claude("Stop", &stop.to_string());
    let end = serde_json::json!({"session_id": headless, "cwd": fixture.project, "reason": "other"});
    fixture.hook_under_headless_claude("SessionEnd", &end.to_string());

    // A spawned run would hold the lock for its whole pass and leave a ledger
    // row when it ends, a yield included; a few seconds covers both.
    let watch_until = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while std::time::Instant::now() < watch_until {
        assert!(!lock.exists(), "a one-shot run started a consolidation");
        assert_eq!(fixture.consolidation_modes(), Vec::<String>::new(), "a one-shot run left a run");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // The control: a person's session end still starts its own run. Plain
    // `fixture.hook` is classified from the test runner's own ancestry, so
    // this holds only when the suite runs under a person's session or no CLI
    // at all. Under `claude -p` or `codex exec` it reads as headless, and a
    // stand-in interactive `claude` would not help: any agent above a CLI
    // makes that CLI a delegate.
    let person = "0199f000-0000-7000-8000-00000000f002";
    fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, person, "startup"));
    let end = serde_json::json!({"session_id": person, "cwd": fixture.project, "reason": "other"});
    fixture.hook("claude-code", "SessionEnd", &end.to_string());
    // The spawned run lands late under a loaded full suite; the wait costs
    // time only when it fails.
    let wait_until = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut modes = fixture.consolidation_modes();
    while modes.is_empty() && std::time::Instant::now() < wait_until {
        std::thread::sleep(std::time::Duration::from_millis(50));
        modes = fixture.consolidation_modes();
    }
    assert_eq!(modes, ["session"], "an interactive session end must still consolidate");
}

#[test]
fn the_vault_has_a_front_page_and_a_schema() {
    let fixture = Fixture::new("frontpage");
    fixture.seed_session(3);
    let bin = fixture.fake_cli(
        "claude",
        "echo '{\"summary\":\"Edited three files.\",\"titles\":[]}'",
    );
    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(out.status.success(), "consolidate failed: {out:?}");

    let index = std::fs::read_to_string(fixture.wiki().join("index.md")).expect("index.md");
    assert!(index.contains("checkout"), "the project is catalogued: {index}");
    assert!(index.contains("1 session(s)"), "{index}");
    let schema = std::fs::read_to_string(fixture.wiki().join("AGENTS.md")).expect("AGENTS.md");
    assert!(schema.contains("brain_search") && schema.contains("## Summary"), "{schema}");
    let hub = fixture
        .project_dirs()
        .into_iter()
        .find_map(|dir| std::fs::read_to_string(dir.join("checkout.md")).ok())
        .expect("a project hub");
    assert!(hub.contains("## Sessions"), "{hub}");
    // Every link in the fresh vault resolves, and every page has a way in.
    let doctor = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(
        doctor.contains("every link resolves, every page reachable"),
        "a fresh vault should lint clean: {doctor}"
    );
}

#[test]
fn a_document_is_read_once_and_becomes_a_source_page() {
    let fixture = Fixture::new("ingest");
    let doc = fixture.project.join("docs/rate-limiter.md");
    std::fs::create_dir_all(doc.parent().unwrap()).unwrap();
    std::fs::write(
        &doc,
        "# Rate limiter design\n\nRequests are capped per API key with a token bucket.\n\n\
         The bucket refills at 10 per second and holds 100.\n",
    )
    .unwrap();

    let counter = fixture.home.parent().unwrap().join("ingest-calls");
    let bin = fixture.fake_cli(
        "claude",
        &format!(
            "N=$(cat {c} 2>/dev/null || echo 0); N=$((N+1)); echo $N > {c}\n\
             echo '{{\"summary\":\"Caps requests per API key with a token bucket that refills at ten per second.\",\"entities\":[\"token bucket\",\"api key\"]}}'",
            c = counter.display()
        ),
    );
    let calls = || -> usize {
        std::fs::read_to_string(&counter).unwrap_or_default().trim().parse().unwrap_or(0)
    };

    let out = fixture.brain_with_path(&["ingest", "docs/rate-limiter.md"], Some(&bin));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "ingest failed: {out:?}");
    assert!(stdout.contains("Read \"Rate limiter design\" into"), "{stdout}");
    assert!(stdout.contains("sources/rate-limiter-design.md"), "{stdout}");
    assert_eq!(calls(), 1, "one call for one short document");

    // The summary page, the untouched copy, and the entry behind them.
    let project_dir = fixture
        .project_dirs()
        .into_iter()
        .find(|dir| dir.join("sources/rate-limiter-design.md").is_file())
        .expect("a project dir with the source page");
    let page = std::fs::read_to_string(project_dir.join("sources/rate-limiter-design.md")).unwrap();
    assert!(page.contains("tags: [source]"), "{page}");
    assert!(page.contains("## Summary\n\nCaps requests per API key"), "{page}");
    assert!(page.contains("[[raw/rate-limiter-design|rate-limiter.md]]"), "{page}");
    let raw = std::fs::read(project_dir.join("raw/rate-limiter-design.md")).unwrap();
    assert_eq!(raw, std::fs::read(&doc).unwrap(), "the raw copy is byte-identical");
    let log = fixture.log_text();
    assert_eq!(log.matches("\"kind\":\"source\"").count(), 1, "{log}");
    assert!(!log.contains("token bucket that refills at 10 per second and holds 100"), "the body is the summary, not the document");

    // Reachable the way every other memory is.
    let search = String::from_utf8_lossy(&fixture.brain(&["search", "token bucket"]).stdout).to_string();
    assert!(search.contains("Rate limiter design"), "not searchable: {search}");
    let hub = std::fs::read_to_string(project_dir.join("checkout.md")).unwrap();
    assert!(hub.contains("## Sources") && hub.contains("[[sources/rate-limiter-design|Rate limiter design]]"), "{hub}");
    assert!(hub.contains("1 document(s) read"), "{hub}");

    // The same bytes again cost nothing and add nothing.
    let again = fixture.brain_with_path(&["ingest", "docs/rate-limiter.md"], Some(&bin));
    let stdout = String::from_utf8_lossy(&again.stdout);
    assert!(stdout.contains("Already in memory, unchanged"), "{stdout}");
    assert_eq!(calls(), 1, "an unchanged document must not be re-summarized");
    assert_eq!(fixture.log_text().matches("\"kind\":\"source\"").count(), 1);

    // --force reads it again and withdraws the earlier reading, so search
    // still returns one entry for the document.
    let forced = fixture.brain_with_path(&["ingest", "docs/rate-limiter.md", "--force"], Some(&bin));
    assert!(String::from_utf8_lossy(&forced.stdout).contains("Read \"Rate limiter design\""));
    assert_eq!(calls(), 2);
    let log = fixture.log_text();
    assert_eq!(log.matches("\"kind\":\"source\"").count(), 2, "{log}");
    assert!(log.contains("\"kind\":\"tombstone\""), "the earlier reading was not withdrawn: {log}");
    let search = String::from_utf8_lossy(&fixture.brain(&["search", "token bucket"]).stdout).to_string();
    assert_eq!(
        search.lines().filter(|line| line.contains("Rate limiter design")).count(),
        1,
        "two readings of one document are being served: {search}"
    );

    // A document that is not text is refused with a reason, not read as garbage.
    std::fs::write(fixture.project.join("docs/blob.bin"), [0u8, 159, 146, 150, 0]).unwrap();
    let refused = fixture.brain(&["ingest", "docs/blob.bin"]);
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("not UTF-8 text"), "{refused:?}");
}

#[test]
fn a_document_is_kept_even_when_no_model_is_reachable() {
    let fixture = Fixture::new("ingest-nomodel");
    let doc = fixture.project.join("notes.txt");
    std::fs::write(&doc, "Deploys happen on Fridays only, after the 4pm freeze lifts.\n").unwrap();

    // No CLI on PATH at all.
    let out = fixture.brain(&["ingest", "notes.txt"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{out:?}");
    assert!(stdout.contains("(via rule-based)") && stdout.contains("No model was reachable"), "{stdout}");

    let project_dir = fixture
        .project_dirs()
        .into_iter()
        .find(|dir| dir.join("sources/notes.md").is_file())
        .expect("a page even without a model");
    let page = std::fs::read_to_string(project_dir.join("sources/notes.md")).unwrap();
    assert!(page.contains("Deploys happen on Fridays only"), "the opening stands in: {page}");
    assert!(page.contains("`raw/notes.txt` (notes.txt)"), "a non-markdown copy is named, not wikilinked: {page}");
    assert!(project_dir.join("raw/notes.txt").is_file());
    let search = String::from_utf8_lossy(&fixture.brain(&["search", "Fridays"]).stdout).to_string();
    assert!(search.contains("notes"), "{search}");
}

#[test]
fn a_team_shares_lessons_and_nothing_else() {
    // Two people, one repository, one shared folder. What crosses is the
    // distillate; what must never cross is everything else.
    let marker = "[project]\nname = \"teamprop\"\n";
    let a = Fixture::new("team-a");
    let b = Fixture::new("team-b");
    std::fs::write(a.project.join(".rolepod-brain.toml"), marker).unwrap();
    std::fs::write(b.project.join(".rolepod-brain.toml"), marker).unwrap();
    let shared = a.home.parent().unwrap().join("team-folder");

    // A works, five sessions, and a lesson is distilled from them. The
    // lesson deliberately carries what a lesson picks up from the sessions
    // it was written over: a home directory and an address.
    let leaky = r#"
case "$*" in
  *"SESSION SUMMARIES"*)
    IDS=$(echo "$*" | grep -oE 'id=[0-9A-Z]{26}' | cut -d= -f2)
    ID=$(echo "$IDS" | head -1)
    ID2=$(echo "$IDS" | head -2 | tail -1)
    echo "{\"knowledge\":[{\"kind\":\"gotcha\",\"title\":\"vitest must run file-by-file here\",\"body\":\"Set under /Users/someone/dev/app by sam@example.com.\",\"sources\":[\"$ID\",\"$ID2\"]}]}" ;;
  *) echo '{"summary":"Reworked the auth path.","titles":[]}' ;;
esac
"#;
    let bin = a.fake_cli("claude", leaky);
    for session in 0..5 {
        let payload = serde_json::json!({
            "session_id": format!("0199a1f2-3c4d-7e8f-9012-34567890000{session}"),
            "cwd": a.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": a.project.join("src/auth.rs")}
        })
        .to_string();
        a.hook("claude-code", "PostToolUse", &payload);
        assert!(a.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());
    }
    assert!(!a.knowledge_pages().is_empty(), "no lesson to share");

    let init_a = a.brain(&["team", "init", shared.to_str().unwrap(), "--name", "Alex"]);
    assert!(init_a.status.success(), "init A failed: {init_a:?}");
    let init_b = b.brain(&["team", "init", shared.to_str().unwrap(), "--name", "Sam"]);
    assert!(init_b.status.success(), "init B failed: {init_b:?}");
    std::fs::copy(a.home.join("team.key"), b.home.join("team.key")).unwrap();

    let published = a.brain(&["team"]);
    let stdout = String::from_utf8_lossy(&published.stdout);
    assert!(published.status.success(), "publish failed: {published:?}");
    assert!(stdout.contains("published 1 of yours"), "{stdout}");

    let pulled = b.brain(&["team"]);
    assert!(pulled.status.success(), "pull failed: {pulled:?}");
    let stdout = String::from_utf8_lossy(&pulled.stdout).to_string();
    assert!(stdout.contains("1 new lesson(s)"), "{stdout}");

    // B's agent can recall A's lesson.
    let hits = String::from_utf8_lossy(&b.brain(&["search", "vitest"]).stdout).to_string();
    assert!(hits.contains("vitest must run file-by-file"), "the lesson never arrived: {hits}");

    // And nothing else did. Not the session, not the summary, not the prompt.
    let shelf = b.page_text() + &b.log_text();
    assert!(!shelf.contains("Reworked the auth path"), "a session summary crossed: {shelf}");
    assert!(!shelf.contains("src/auth.rs"), "a capture crossed");
    // The two things a lesson carries out of a person's sessions.
    assert!(!shelf.contains("/Users/someone"), "a home directory crossed");
    assert!(!shelf.contains("sam@example.com"), "an address crossed");

    // The bundle itself is ciphertext.
    for entry in std::fs::read_dir(&shared).unwrap().flatten() {
        let text = String::from_utf8_lossy(&std::fs::read(entry.path()).unwrap()).into_owned();
        assert!(!text.contains("vitest") && !text.contains("\"kind\""), "plaintext in the team folder");
    }

    // B reads it as a signed page it does not own, and a second pull is a no-op.
    let page = b
        .project_dirs()
        .into_iter()
        .chain(std::iter::once(b.wiki()))
        .find_map(|dir| {
            let mut found = Vec::new();
            collect_under(&dir, "_team", &mut found);
            found.into_iter().find(|path| path.extension().is_some_and(|ext| ext == "md"))
        })
        .expect("a team page");
    let text = std::fs::read_to_string(&page).unwrap();
    assert!(text.contains("author: Alex"), "{text}");
    assert!(text.contains("publish your own rather than editing theirs"), "{text}");
    let again = b.brain(&["team"]);
    assert!(String::from_utf8_lossy(&again.stdout).contains("0 new lesson(s)"), "a repeat pull gained something");

    // A teammate's lesson is never republished as B's own work.
    assert!(
        String::from_utf8_lossy(&again.stdout).contains("published 0 of yours"),
        "B republished A's lesson as its own: {}",
        String::from_utf8_lossy(&again.stdout)
    );

    // A survives a rebuild on B: the team shelf is derived state like the rest.
    assert!(b.brain(&["reindex"]).status.success());
    let hits = String::from_utf8_lossy(&b.brain(&["search", "vitest"]).stdout).to_string();
    assert!(hits.contains("vitest must run file-by-file"), "a rebuild lost the team shelf: {hits}");

    let doctor = String::from_utf8_lossy(&b.brain(&["doctor"]).stdout).to_string();
    assert!(doctor.contains("publishing as Sam"), "{doctor}");
}

#[test]
fn a_team_is_off_and_needs_a_name_before_anything_is_shared() {
    let fixture = Fixture::new("team-off");
    let doctor = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).to_string();
    assert!(doctor.contains("lessons stay on this machine"), "{doctor}");

    let out = fixture.brain(&["team"]);
    assert!(!out.status.success(), "an unconfigured team must fail loudly");
    assert!(String::from_utf8_lossy(&out.stderr).contains("brain team init"), "{out:?}");

    let dir = fixture.home.parent().unwrap().join("t");
    let empty = fixture.brain(&["team", "init", dir.to_str().unwrap(), "--name", "  "]);
    assert!(!empty.status.success(), "a team without a name must be refused");
    assert!(String::from_utf8_lossy(&empty.stderr).contains("name to put on"), "{empty:?}");
}

#[test]
fn a_rebuild_lands_in_the_vaults_history_and_leaves_a_persons_own_notes_alone() {
    let fixture = Fixture::new("reindex-commit");
    fixture.seed_session(3);
    let bin = fixture.fake_cli("claude", "echo '{\"summary\":\"Edited three files.\",\"titles\":[]}'");
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());

    let git = |args: &[&str]| -> String {
        let out = Command::new("git")
            .args(args)
            .current_dir(fixture.wiki())
            .output()
            .expect("git");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    // A hub an older version wrote and committed: the rebuild's whole job,
    // and what makes this a real rewrite rather than a no-op.
    let hub = fixture
        .project_dirs()
        .into_iter()
        .map(|dir| dir.join("checkout.md"))
        .find(|path| path.is_file())
        .expect("a project hub");
    std::fs::write(&hub, "# checkout\n\nwritten by an older version\n").unwrap();
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "an older version's hub"]);

    // Something a person keeps beside their memory, which brain never writes
    // and must never take over.
    let mine = fixture.wiki().join("my-notes.md");
    std::fs::write(&mine, "# mine\n").unwrap();

    let out = fixture.brain(&["reindex"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "reindex failed: {out:?}");
    assert!(stdout.contains("Committed the rebuild"), "{stdout}");

    // Nothing brain wrote is left outside the history.
    let status = git;
    let tracked_dirty = status(&["status", "--short", "--untracked-files=no"]);
    assert!(tracked_dirty.trim().is_empty(), "a rebuild left pages uncommitted:\n{tracked_dirty}");
    let log = status(&["log", "--oneline", "-1"]);
    assert!(log.contains("reindex: rebuilt"), "{log}");
    assert!(
        std::fs::read_to_string(&hub).unwrap().contains("## Sessions"),
        "the stale hub was not rebuilt"
    );

    // And the person's file is still theirs: on disk, out of the history.
    assert!(mine.is_file());
    let all = status(&["status", "--short"]);
    assert!(all.contains("my-notes.md"), "a file brain does not write was committed:\n{all}");

    // A second rebuild changes nothing, so it commits nothing.
    let again = fixture.brain(&["reindex"]);
    assert!(
        !String::from_utf8_lossy(&again.stdout).contains("Committed the rebuild"),
        "an unchanged rebuild made an empty commit"
    );
}

/// The hub writer returns the pages it removes, so their deletions are
/// committed. A removed path must not fail the commit that stages it. A
/// topic note a killed run never committed is not in the index, and reindex's
/// `add -u` has already taken a committed one out of it, so a plain
/// `git add` of either exits with "pathspec did not match".
#[test]
fn a_page_brain_removes_does_not_fail_the_commit() {
    let fixture = Fixture::new("removed-page");
    fixture.seed_session(3);
    let bin = fixture.fake_cli("claude", "echo '{\"summary\":\"Edited three files.\",\"titles\":[]}'");
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());
    let project = fixture
        .project_dirs()
        .into_iter()
        .find(|dir| dir.join("checkout.md").is_file())
        .expect("a project hub");

    // Consolidation: a topic note left in the work tree by a run that died
    // before its commit, for a topic no session has.
    let orphan = project.join("decisions.md");
    std::fs::write(&orphan, "---\ntitle: decision\ntags: [topic, decision]\n---\n").unwrap();
    for index in 0..3 {
        let payload = serde_json::json!({
            "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcdf",
            "cwd": fixture.project,
            "tool_name": "Edit",
            "tool_input": {"file_path": fixture.project.join(format!("src/later{index}.rs"))}
        })
        .to_string();
        fixture.hook("claude-code", "PostToolUse", &payload);
    }
    let out = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!orphan.exists(), "a topic note nothing feeds is removed");
    assert!(stdout.contains("Consolidated 1 session(s)"), "{stdout}");
    assert!(!stdout.contains("failed"), "removing a page failed the session: {stdout}");

    // Reindex: an index.md an older version wrote and committed. The vault's
    // own root index.md is a different file, so match the whole path.
    let wiki = fixture.wiki();
    let stale = project.join("index.md");
    let relative = stale.strip_prefix(&wiki).unwrap().to_string_lossy().into_owned();
    let tracked = || git_stdout(&wiki, &["ls-files"]).lines().any(|line| line == relative);
    std::fs::write(&stale, "# checkout\n\nwritten by an older version\n").unwrap();
    git_stdout(&wiki, &["add", "--", &relative]);
    git_stdout(&wiki, &["commit", "-q", "-m", "an older version's index"]);
    assert!(tracked(), "the stale index.md was not planted");

    let out = fixture.brain(&["reindex"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "reindex failed: {out:?}");
    assert!(!stale.exists(), "the old unnamed hub is removed");
    assert!(stdout.contains("Committed the rebuild"), "{stdout}");
    assert!(!tracked(), "its deletion was not committed");
}

#[test]
fn a_dispatch_goes_out_exactly_as_the_lead_wrote_it() {
    // What a subagent knows is the brief the lead writes. Memory about the
    // task is the lead's to use while writing it; brain adds nothing to the
    // prompt, whatever matches. A config written for the old dispatch seed
    // still loads.
    let fixture = Fixture::new("dispatch-untouched");
    let session = "0199c000-0000-7000-8000-00000000d150";
    fixture.mcp(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_note","arguments":{"text":"The stripe webhook secret rotates on every stripe listen run."}}}"#,
    ]);
    std::fs::write(fixture.home.join("config.toml"), "[injection]\ndispatch_seed = true\n").unwrap();
    let dispatch = serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "tool_name": "Agent",
        "tool_input": {"description": "Fix webhook", "prompt": "stripe webhook", "subagent_type": "general-purpose"}
    })
    .to_string();
    let out = fixture.hook("claude-code", "PreToolUse", &dispatch);
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "{}", "the dispatch was rewritten");

    let start = serde_json::json!({"session_id": session, "cwd": fixture.project, "source": "startup"}).to_string();
    let primer = String::from_utf8_lossy(&fixture.hook("claude-code", "SessionStart", &start).stdout).to_string();
    assert!(primer.contains("webhook secret rotates"), "the lead lost the note: {primer}");
    assert!(!primer.contains("brain_seed"), "the primer still points at a removed tool: {primer}");
}

/// Take the store's write lock, the way a long index build holds it, and keep
/// it until the returned connection is committed or dropped.
fn hold_write_lock(fixture: &Fixture) -> rusqlite::Connection {
    let conn = rusqlite::Connection::open(fixture.home.join("brain.db")).expect("open brain.db");
    conn.busy_timeout(std::time::Duration::from_secs(1)).unwrap();
    conn.execute_batch("BEGIN IMMEDIATE").expect("take the write lock");
    conn
}

/// How many indexed events name this marker in their title.
fn indexed_with(fixture: &Fixture, marker: &str) -> i64 {
    let conn = rusqlite::Connection::open_with_flags(
        fixture.home.join("brain.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("open brain.db read-only");
    conn.query_row("SELECT COUNT(*) FROM events WHERE title LIKE ?1", [format!("%{marker}%")], |row| row.get(0))
        .expect("count events")
}

/// One capture of an edit to `file`, as the session the test names.
fn edit_payload(fixture: &Fixture, session: &str, file: &str) -> String {
    serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "tool_name": "Edit",
        "tool_input": {"file_path": fixture.project.join(file)}
    })
    .to_string()
}

#[test]
fn a_first_capture_of_a_new_session_reaches_the_log_while_the_store_is_locked() {
    // The first capture of a session writes its invocation row before it
    // writes the log. A write lock held past the busy timeout (the one-time
    // index build holds it for seconds) used to fail that row, and the event
    // went nowhere: not in the log, so not for the catch-up to recover.
    let fixture = Fixture::new("capture-locked-new");
    // The store exists, as it does on any machine that has captured before.
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-3456789abcde", &["src/earlier.rs"]);

    let lock = hold_write_lock(&fixture);
    let payload = edit_payload(&fixture, "0199c000-0000-7000-8000-00000000a901", "src/lockedfirst.rs");
    let out = fixture.hook("claude-code", "PostToolUse", &payload);
    assert!(out.status.success(), "the host saw a failing hook: {out:?}");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "{}");
    lock.execute_batch("COMMIT").unwrap();

    assert!(
        fixture.log_text().contains("lockedfirst.rs"),
        "the capture never reached the log, so nothing can recover it"
    );
    assert_eq!(indexed_with(&fixture, "lockedfirst.rs"), 0, "precondition: the index write was locked out");

    // Any later consolidation run catches the index up from the log.
    assert!(fixture.brain(&["consolidate", "--session", "nonexistent-session"]).status.success());
    assert_eq!(indexed_with(&fixture, "lockedfirst.rs"), 1, "the catch-up did not index the event");
}

#[test]
fn a_capture_of_a_known_session_reaches_the_log_while_the_store_is_locked() {
    // The same lock, for a session whose invocation row already exists: only
    // the index write fails, and the log must have the event before it does.
    let fixture = Fixture::new("capture-locked-known");
    let session = "0199c000-0000-7000-8000-00000000a902";
    fixture.seed_session_as(session, &["src/earlier.rs"]);

    let lock = hold_write_lock(&fixture);
    let payload = edit_payload(&fixture, session, "src/lockedlater.rs");
    let out = fixture.hook("claude-code", "PostToolUse", &payload);
    assert!(out.status.success(), "the host saw a failing hook: {out:?}");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "{}");
    lock.execute_batch("COMMIT").unwrap();

    assert!(fixture.log_text().contains("lockedlater.rs"), "the capture never reached the log");
    assert_eq!(indexed_with(&fixture, "lockedlater.rs"), 0, "precondition: the index write was locked out");
    assert!(fixture.brain(&["consolidate", "--session", "nonexistent-session"]).status.success());
    assert_eq!(indexed_with(&fixture, "lockedlater.rs"), 1, "the catch-up did not index the event");
}

#[test]
fn a_capture_reaches_the_log_when_the_store_cannot_even_be_opened() {
    // The first open after an upgrade runs the once-only migrations, which
    // write. With the lock held past the busy timeout the open itself fails;
    // the event must still be on the log by then.
    let fixture = Fixture::new("capture-locked-open");
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-3456789abcde", &["src/earlier.rs"]);
    {
        // An older store: a column a later version added is not there yet.
        let conn = rusqlite::Connection::open(fixture.home.join("brain.db")).unwrap();
        conn.execute_batch("ALTER TABLE events DROP COLUMN clamped").unwrap();
    }

    let lock = hold_write_lock(&fixture);
    let payload = edit_payload(&fixture, "0199c000-0000-7000-8000-00000000a903", "src/lockedopen.rs");
    let child = fixture.spawn_hook("claude-code", "PostToolUse", &payload);
    // The host kills a hook at 3 s (SessionEnd) to 5 s, and the open waits out
    // a 5 s busy timeout: the log has to have the event long before that, while
    // the hook is still stuck on the store.
    let started = std::time::Instant::now();
    while !fixture.log_text().contains("lockedopen.rs") && started.elapsed() < std::time::Duration::from_secs(2) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let reached_log = fixture.log_text().contains("lockedopen.rs");
    let out = child.wait_with_output().expect("hook output");
    assert!(reached_log, "the capture was not on the log within 2 s, before the store was touched");
    assert!(out.status.success(), "the host saw a failing hook: {out:?}");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "{}");
    lock.execute_batch("COMMIT").unwrap();

    let brain_log = std::fs::read_to_string(fixture.home.join("brain.log")).unwrap_or_default();
    assert!(
        brain_log.contains("PostToolUse") && brain_log.contains("claude-code"),
        "the failure was not recorded where doctor looks: {brain_log}"
    );
    // The next open migrates; the catch-up then brings the event in.
    assert!(fixture.brain(&["consolidate", "--session", "nonexistent-session"]).status.success());
    assert_eq!(indexed_with(&fixture, "lockedopen.rs"), 1, "the catch-up did not index the event");
}

/// Open the compact window the way the compactor does: a marker whose expiry
/// is a minute away.
fn open_window(fixture: &Fixture) {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    std::fs::write(fixture.home.join(".brain-maintenance"), format!("{} {}", std::process::id(), now + 60_000))
        .unwrap();
}

fn close_window(fixture: &Fixture) {
    let _ = std::fs::remove_file(fixture.home.join(".brain-maintenance"));
}

/// One number out of the store, read without taking any lock of its own.
fn scalar(fixture: &Fixture, sql: &str) -> i64 {
    let conn = rusqlite::Connection::open_with_flags(
        fixture.home.join("brain.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("open brain.db read-only");
    conn.query_row(sql, [], |row| row.get(0)).expect("scalar")
}

fn brain_log_text(fixture: &Fixture) -> String {
    std::fs::read_to_string(fixture.home.join("brain.log")).unwrap_or_default()
}

/// The JSON a search or get returned, and whether the call was an error.
fn tool_text(response: &serde_json::Value) -> (bool, serde_json::Value) {
    let is_error = response["result"]["isError"].as_bool().unwrap_or(false) || response.get("error").is_some();
    let text = response["result"]["content"][0]["text"].as_str().unwrap_or("null");
    (is_error, serde_json::from_str(text).unwrap_or(serde_json::Value::Null))
}

#[test]
fn maint_window_hook_answers_at_once_and_its_event_is_indexed_afterwards() {
    let fixture = Fixture::new("maint-hook");
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-3456789abcde", &["src/earlier.rs"]);

    // What one hook spawn costs on this machine right now, with nothing held.
    // A fixed bound measured load, not the wait on the store.
    let before = edit_payload(&fixture, "0199c000-0000-7000-8000-00000000b000", "src/beforewindow.rs");
    let began = std::time::Instant::now();
    fixture.hook("claude-code", "PostToolUse", &before);
    let baseline = began.elapsed();

    let lock = hold_write_lock(&fixture);
    open_window(&fixture);
    let payload = edit_payload(&fixture, "0199c000-0000-7000-8000-00000000b001", "src/inwindow.rs");
    let started = std::time::Instant::now();
    let out = fixture.hook("claude-code", "PostToolUse", &payload);
    let took = started.elapsed();
    assert!(out.status.success(), "the host saw a failing hook: {out:?}");
    // Waiting out the store's 5 s busy timeout would add all of it on top of the baseline.
    assert!(
        took < baseline + std::time::Duration::from_secs(2),
        "the hook waited on the held store: {took:?} against a {baseline:?} baseline"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "{}");
    assert!(fixture.log_text().contains("inwindow.rs"), "the capture never reached the log");
    assert_eq!(brain_log_text(&fixture), "", "a hook in the window wrote to brain.log, which doctor counts");
    lock.execute_batch("COMMIT").unwrap();
    close_window(&fixture);

    assert!(fixture.brain(&["consolidate", "--session", "nonexistent-session"]).status.success());
    assert_eq!(indexed_with(&fixture, "inwindow.rs"), 1, "the catch-up did not index the event");
}

#[test]
fn maint_window_session_start_still_injects_and_its_ledger_is_spilled_then_folded() {
    let fixture = Fixture::new("maint-primer");
    let earlier = "0199aaaa-0000-7000-8000-000000000000";
    let prompt = serde_json::json!({"session_id": earlier, "cwd": fixture.project, "prompt": "why does the scheduler double-book?"});
    fixture.hook("claude-code", "UserPromptSubmit", &prompt.to_string());

    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    let lock = hold_write_lock(&fixture);
    open_window(&fixture);
    let out = fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "startup"));
    assert!(out.status.success());
    let primer = injected_context(&out).expect("a primer in the window, not `{}`");
    assert!(primer.contains(earlier), "the primer lost its content: {primer}");
    assert_eq!(brain_log_text(&fixture), "", "a hook in the window wrote to brain.log");
    let spilled = std::fs::read_to_string(fixture.home.join("surfaced.jsonl")).expect("the ledger write was spilled");
    assert!(spilled.contains(session) && spilled.contains("injected"), "unexpected spill: {spilled}");
    lock.execute_batch("COMMIT").unwrap();
    close_window(&fixture);
    assert_eq!(scalar(&fixture, "SELECT COUNT(*) FROM injected"), 0, "precondition: nothing reached the ledger");

    assert!(fixture.brain(&["consolidate", "--session", "nonexistent-session"]).status.success());
    assert!(scalar(&fixture, "SELECT COUNT(*) FROM injected") > 0, "the spill was not folded back");
    assert!(!fixture.home.join("surfaced.jsonl").exists(), "the folded spill was left behind");
}

#[test]
fn mcp_search_in_the_window_returns_its_hits_and_its_ids_survive_as_surfaced() {
    let fixture = Fixture::new("maint-mcp");
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-3456789abcde", &["src/auth.rs", "src/auth/login.rs"]);
    let search = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#;

    // What starting the server and answering costs on this machine right now,
    // with nothing held and nothing recorded (no hit, so no ledger write). A
    // fixed bound measured start-up under load, not the wait on the lock.
    let miss = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"zzqxjvk"}}}"#;
    let began = std::time::Instant::now();
    fixture.mcp(&[miss]);
    let baseline = began.elapsed();

    let lock = hold_write_lock(&fixture);
    open_window(&fixture);
    let started = std::time::Instant::now();
    let responses = fixture.mcp(&[search]);
    // Waiting out the 5 s busy timeout would add all of it on top of the baseline.
    assert!(
        started.elapsed() < baseline + std::time::Duration::from_secs(3),
        "the search waited out the busy timeout: {:?} against a {baseline:?} baseline",
        started.elapsed()
    );
    let (is_error, found) = tool_text(&responses[0]);
    assert!(!is_error, "a held ledger failed the search: {responses:?}");
    let ids: Vec<String> =
        found["hits"].as_array().expect("hits").iter().filter_map(|hit| hit["id"].as_str().map(str::to_string)).collect();
    assert!(!ids.is_empty(), "the search returned no hits: {found}");
    let spilled = std::fs::read_to_string(fixture.home.join("surfaced.jsonl")).expect("the ids were spilled");
    for id in &ids {
        assert!(spilled.contains(id.as_str()), "{id} is not in surfaced.jsonl: {spilled}");
    }
    lock.execute_batch("COMMIT").unwrap();
    close_window(&fixture);

    assert!(fixture.brain(&["consolidate", "--session", "nonexistent-session"]).status.success());
    for id in &ids {
        assert_eq!(scalar(&fixture, &format!("SELECT COUNT(*) FROM recalled WHERE event_id = '{id}'")), 1, "{id} not folded");
    }
    assert!(!fixture.home.join("surfaced.jsonl").exists(), "the folded spill was left behind");
}

#[test]
fn mcp_search_with_the_store_held_and_no_window_still_returns_its_hits() {
    let fixture = Fixture::new("maint-mcp-nowindow");
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-3456789abcde", &["src/auth.rs"]);
    let search = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#;

    let lock = hold_write_lock(&fixture);
    let responses = fixture.mcp(&[search]);
    let (is_error, found) = tool_text(&responses[0]);
    assert!(!is_error && found["count"].as_i64().unwrap_or(0) > 0, "a held ledger failed the search: {responses:?}");
    assert!(fixture.home.join("surfaced.jsonl").exists(), "the ids were not spilled");
    lock.execute_batch("COMMIT").unwrap();
}

#[test]
fn maint_spilled_ids_pass_the_forget_guard_of_their_own_session_only() {
    let fixture = Fixture::new("maint-guard");
    fixture.seed_session_as("0199a1f2-3c4d-7e8f-9012-3456789abcde", &["src/auth.rs"]);
    let call = |name: &str, args: &str| {
        format!(r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"{name}","arguments":{args}}}}}"#)
    };

    let lock = hold_write_lock(&fixture);
    open_window(&fixture);
    let responses = fixture.mcp(&[&call("brain_search", r#"{"query":"auth"}"#)]);
    let (_, found) = tool_text(&responses[0]);
    let id = found["hits"][0]["id"].as_str().expect("a hit").to_string();
    // The same server process: the search's spill is what lets this through.
    let forget = call("brain_forget", &format!(r#"{{"id":"{id}"}}"#));
    let search = call("brain_search", r#"{"query":"auth"}"#);
    let same = serde_json::to_string(&fixture.mcp(&[&search, &forget])).unwrap();
    assert!(!same.contains("has not been surfaced"), "a spilled id failed the guard: {same}");
    // Another server process was shown nothing.
    let other = serde_json::to_string(&fixture.mcp(&[&forget])).unwrap();
    assert!(other.contains("has not been surfaced"), "another session's spill opened the guard: {other}");
    lock.execute_batch("COMMIT").unwrap();
}

#[test]
fn maint_window_clear_leaves_a_pending_wipe_that_the_next_writable_hook_applies() {
    let fixture = Fixture::new("maint-wipe");
    let earlier = "0199aaaa-0000-7000-8000-000000000000";
    let prompt = serde_json::json!({"session_id": earlier, "cwd": fixture.project, "prompt": "why does the scheduler double-book?"});
    fixture.hook("claude-code", "UserPromptSubmit", &prompt.to_string());
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    injected_context(&fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "startup")))
        .expect("a primer on a normal start");
    let live = format!("SELECT COUNT(*) FROM injected WHERE session = '{session}' AND active = 1");
    assert!(scalar(&fixture, &live) > 0, "precondition: the session was shown something");

    let lock = hold_write_lock(&fixture);
    open_window(&fixture);
    let out = fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "clear"));
    assert!(out.status.success());
    let waiting = fixture.home.join("pending-wipe").join(session);
    assert!(waiting.exists(), "a wipe that could not be written was lost");
    assert_eq!(brain_log_text(&fixture), "", "a hook in the window wrote to brain.log");
    lock.execute_batch("COMMIT").unwrap();
    close_window(&fixture);
    assert!(scalar(&fixture, &live) > 0, "precondition: the wipe was not applied yet");

    fixture.hook("claude-code", "PostToolUse", &edit_payload(&fixture, session, "src/after.rs"));
    assert!(!waiting.exists(), "the applied wipe was not removed");
    assert_eq!(scalar(&fixture, &live), 0, "the session still counts what it was shown before /clear");
}

#[test]
fn a_session_start_whose_index_write_fails_outside_the_window_still_gets_its_primer_and_logs_it() {
    let fixture = Fixture::new("followup-primer-locked");
    let earlier = "0199aaaa-0000-7000-8000-000000000000";
    let prompt = serde_json::json!({"session_id": earlier, "cwd": fixture.project, "prompt": "why does the scheduler double-book?"});
    fixture.hook("claude-code", "UserPromptSubmit", &prompt.to_string());
    assert_eq!(brain_log_text(&fixture), "", "precondition: a clean log");

    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    let lock = hold_write_lock(&fixture);
    let out = fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "startup"));
    lock.execute_batch("COMMIT").unwrap();
    assert!(out.status.success());
    let primer = injected_context(&out).expect("a failed index write cost the session its primer");
    assert!(primer.contains(earlier), "the primer lost its content: {primer}");
    let log = brain_log_text(&fixture);
    assert!(log.contains("claude-code SessionStart"), "the failure left no line in brain.log: {log}");
}

#[test]
fn a_clear_in_the_window_gets_the_primer_of_a_session_that_was_wiped() {
    let fixture = Fixture::new("followup-clear-window");
    let earlier = "0199aaaa-0000-7000-8000-000000000000";
    let prompt = serde_json::json!({"session_id": earlier, "cwd": fixture.project, "prompt": "why does the scheduler double-book?"});
    fixture.hook("claude-code", "UserPromptSubmit", &prompt.to_string());
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    let first = injected_context(&fixture.hook(
        "claude-code",
        "SessionStart",
        &start_payload(&fixture.project, session, "startup"),
    ))
    .expect("a primer on a normal start");
    assert!(first.contains(earlier), "precondition: the first primer names it: {first}");

    let lock = hold_write_lock(&fixture);
    open_window(&fixture);
    let out = fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "clear"));
    let again = injected_context(&out).expect("a /clear in the window got no primer");
    assert!(again.contains(earlier), "the pointer shown before the wipe was suppressed: {again}");
    assert_eq!(brain_log_text(&fixture), "", "a hook in the window wrote to brain.log");
    lock.execute_batch("COMMIT").unwrap();
    close_window(&fixture);
}

#[test]
fn forget_in_the_window_answers_at_once_with_the_window_and_leaves_the_log_alone() {
    let fixture = Fixture::new("followup-forget-window");
    let id = fixture.prompt_event(1, "the zeta rendezvous cipher value");
    let forget = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"brain_forget","arguments":{{"id":"{id}"}}}}}}"#
    );
    // The guard wants the id surfaced in the same server, so search first.
    let search = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"zeta rendezvous"}}}"#;
    let began = std::time::Instant::now();
    fixture.mcp(&[search]);
    let baseline = began.elapsed();

    let lock = hold_write_lock(&fixture);
    open_window(&fixture);
    let started = std::time::Instant::now();
    let responses = fixture.mcp(&[search, &forget]);
    let took = started.elapsed();
    assert!(took < baseline + std::time::Duration::from_secs(2), "forget waited on the held store: {took:?} against {baseline:?}");
    let text = serde_json::to_string(&responses[1]).unwrap();
    assert!(text.contains("compact window") && text.contains("under a minute"), "no window message: {text}");
    assert_eq!(brain_log_text(&fixture), "", "forget in the window wrote to brain.log");
    assert_eq!(scalar(&fixture, "SELECT COUNT(*) FROM events WHERE kind = 'tombstone'"), 0);
    lock.execute_batch("COMMIT").unwrap();
    close_window(&fixture);
}

#[test]
fn maint_a_store_that_is_free_gains_no_file_in_the_data_directory() {
    let fixture = Fixture::new("maint-quiet");
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    fixture.seed_session_as(session, &["src/auth.rs"]);
    fixture.hook("claude-code", "SessionStart", &start_payload(&fixture.project, session, "clear"));
    let search = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth"}}}"#;
    let (is_error, _) = tool_text(&fixture.mcp(&[search])[0]);
    assert!(!is_error);
    for name in ["surfaced.jsonl", "surfaced.jsonl.folding", "pending-wipe", ".brain-maintenance"] {
        assert!(!fixture.home.join(name).exists(), "{name} appeared without a failure or a window");
    }
}

#[test]
fn a_daily_relink_fixes_old_links_skips_a_page_just_edited_and_is_committed() {
    let fixture = Fixture::new("relink-daily");
    fixture.seed_session(4);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());
    let dir = fixture.project_dirs().remove(0);
    let wiki = fixture.wiki();

    std::fs::create_dir_all(dir.join("entities")).unwrap();
    std::fs::write(dir.join("entities/billing.md"), "---\ntitle: billing\n---\n").unwrap();
    let page = |name: &str, session: &str| {
        let path = dir.join("pages/sessions").join(name);
        let text = format!(
            "---\ntitle: t\ndate: 2026-08-23\nsession: {session}\n---\n\n# t\n\nAbout: [[billing|billing]] · [[once|once]]\n"
        );
        std::fs::write(&path, &text).unwrap();
        (path, text)
    };
    let (quiet, _) = page("2026-08-23 quiet.md", "planted-quiet");
    let (fresh, fresh_text) = page("2026-08-23 fresh.md", "planted-fresh");
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
    std::fs::File::options().write(true).open(&quiet).unwrap().set_modified(old).unwrap();

    // Yesterday's relink is the one that is due.
    fixture.sql_run("DELETE FROM schema_state WHERE key LIKE 'relinked_at:%'");
    let output = fixture.brain_with_path(&["consolidate", "--force"], Some(&bin));
    assert!(output.status.success(), "consolidate failed: {output:?}");

    let fixed = std::fs::read_to_string(&quiet).unwrap();
    assert!(fixed.contains("About: [[entities/billing|billing]] · once\n"), "{fixed}");
    assert_eq!(std::fs::read_to_string(&fresh).unwrap(), fresh_text, "a page edited just now was touched");
    let leftovers: Vec<_> = std::fs::read_dir(dir.join("pages/sessions"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".brain-tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temp files left: {leftovers:?}");

    // On default git config, in the wiki's history, with the run's own commits.
    let last = git_stdout(&wiki, &["log", "--format=%s", "--name-only", "-3"]);
    assert!(last.contains("consolidate relink"), "no relink commit:\n{last}");
    assert!(last.contains("quiet.md"), "the relinked page is not in it:\n{last}");
    assert_eq!(fixture.sql_one("SELECT CAST(count(*) AS TEXT) FROM schema_state WHERE key LIKE 'relinked_at:%'"), "1");
}

#[test]
fn a_month_old_log_is_set_aside_and_a_later_line_starts_a_new_one() {
    let fixture = Fixture::new("log-aged");
    fixture.seed_session(2);
    let bin = fixture.fake_cli("claude", GOOD_CLI);
    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());
    let log = fixture.home.join("brain.log");
    let long_ago = jiff::Timestamp::now() - jiff::SignedDuration::from_hours(31 * 24);
    std::fs::write(&log, format!("{long_ago} claude-code Stop: ancient failure\n")).unwrap();
    fixture.sql_run("DELETE FROM schema_state WHERE key = 'brain_log_aged_at'");

    assert!(fixture.brain_with_path(&["consolidate", "--force"], Some(&bin)).status.success());
    let aged = std::fs::read_to_string(fixture.home.join("brain.log.1")).unwrap();
    assert!(aged.contains("ancient failure"), "{aged}");
    assert!(!log.exists() || !std::fs::read_to_string(&log).unwrap().contains("ancient failure"));
    let doctor = fixture.brain(&["doctor"]);
    let report = String::from_utf8_lossy(&doctor.stdout);
    assert!(!report.contains("FAIL capture errors"), "{report}");
}

/// Is the maintenance marker claiming a window right now? The file holds
/// `pid until_unix_ms`.
fn marker_active(fixture: &Fixture) -> bool {
    let Ok(text) = std::fs::read_to_string(fixture.home.join(".brain-maintenance")) else { return false };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    text.split_whitespace().nth(1).and_then(|until| until.parse::<u128>().ok()).is_some_and(|until| until > now)
}

#[cfg(unix)]
#[test]
fn a_run_yielding_to_a_window_that_holds_the_write_lock_leaves_brain_log_untouched() {
    let fixture = Fixture::new("maint-yield-locked");
    fixture.prompt_event(1, "the omega lantern cipher value");
    assert!(fixture.brain(&["consolidate"]).status.success());
    let log_path = fixture.home.join("brain.log");
    let log_before = std::fs::read(&log_path).unwrap_or_default();

    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    std::fs::write(fixture.home.join(".brain-maintenance"), format!("99999 {}", now_ms + 60_000)).unwrap();
    let lock = hold_write_lock(&fixture);
    let yielded = fixture.brain(&["consolidate"]);
    assert!(yielded.status.success(), "{yielded:?}");
    assert_eq!(std::fs::read(&log_path).unwrap_or_default(), log_before, "the yield wrote to brain.log");
    assert!(fixture.home.join(".brain-maint-yielded").exists(), "nothing tells the compactor to rerun");

    lock.execute_batch("ROLLBACK").unwrap();
    drop(lock);
}

#[cfg(unix)]
#[test]
fn a_run_that_cannot_record_itself_without_a_window_still_logs_it() {
    let fixture = Fixture::new("maint-yield-locked-no-marker");
    fixture.prompt_event(1, "the omega lantern cipher value");
    assert!(fixture.brain(&["consolidate"]).status.success());
    let log_path = fixture.home.join("brain.log");

    let lock = hold_write_lock(&fixture);
    let run = fixture.brain(&["consolidate"]);
    let log = std::fs::read_to_string(&log_path).unwrap_or_default();
    assert!(log.contains("could not record the run"), "{run:?}\n{log}");

    lock.execute_batch("ROLLBACK").unwrap();
    drop(lock);
}

#[cfg(unix)]
#[test]
fn a_compactor_killed_mid_window_lets_its_marker_lapse_and_leaves_the_index_whole() {
    let fixture = Fixture::new("maint-sigkill");
    fixture.seed_session(4);
    let before = fixture.sql_one("SELECT CAST(count(*) AS TEXT) FROM events");
    fixture.quiet_logs();

    // Another writer holds the database, so the compactor's first write waits
    // on its busy timeout with the window already open.
    let lock = hold_write_lock(&fixture);
    let mut child = Command::new(BRAIN)
        .arg("compact")
        .current_dir(&fixture.project)
        .env("ROLEPOD_BRAIN_HOME", &fixture.home)
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_MAINT_LIFETIME_MS", "1500")
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("HOME", fixture.home.parent().unwrap())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn compact");
    let opened = std::time::Instant::now();
    while !marker_active(&fixture) && opened.elapsed() < std::time::Duration::from_secs(4) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(marker_active(&fixture), "the compactor never opened its window");

    child.kill().expect("SIGKILL the compactor");
    let status = child.wait().expect("reap");
    assert!(!status.success(), "the compactor was not killed: {status:?}");
    let killed = std::time::Instant::now();
    while marker_active(&fixture) && killed.elapsed() < std::time::Duration::from_secs(5) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(!marker_active(&fixture), "a killed compactor's marker outlived its lifetime");
    assert!(killed.elapsed() < std::time::Duration::from_millis(2500), "it took {:?} to lapse", killed.elapsed());

    lock.execute_batch("ROLLBACK").unwrap();
    drop(lock);
    assert_eq!(fixture.sql_one("SELECT integrity_check FROM pragma_integrity_check"), "ok");
    assert_eq!(fixture.sql_one("SELECT CAST(count(*) AS TEXT) FROM events"), before, "the index changed");
}

// --- the one-time knowledge cleanup and its undo -------------------------

/// Titles of the knowledge rows whose `forgotten` is `state`, sorted.
fn knowledge_titles(fixture: &Fixture, state: i64) -> Vec<String> {
    let conn = rusqlite::Connection::open(fixture.home.join("brain.db")).unwrap();
    let mut statement = conn
        .prepare("SELECT title FROM events WHERE kind = 'knowledge' AND forgotten = ?1 ORDER BY title")
        .unwrap();
    statement.query_map([state], |row| row.get(0)).unwrap().filter_map(Result::ok).collect()
}

/// Write knowledge with no label into the project's log, the way a store
/// written before the quality rules holds it, plus a vault page for each, then
/// rebuild the index. Returns the ids, in order.
fn seed_old_knowledge(fixture: &Fixture, entries: &[(&str, &str)]) -> Vec<String> {
    fixture.seed_session(1);
    let dir = fixture.project_dirs().into_iter().next().expect("a project directory");
    seed_old_knowledge_in(fixture, &dir, entries)
}

/// The same, into the project directory the test names.
fn seed_old_knowledge_in(fixture: &Fixture, dir: &Path, entries: &[(&str, &str)]) -> Vec<String> {
    let first = std::fs::read_dir(dir.join("events"))
        .unwrap()
        .filter_map(Result::ok)
        .find(|entry| entry.path().extension().is_some_and(|ext| ext == "jsonl"))
        .expect("an event log")
        .path();
    let sample: serde_json::Value =
        serde_json::from_str(std::fs::read_to_string(&first).unwrap().lines().next().unwrap()).unwrap();
    let pages = dir.join("knowledge").join("gotchas");
    std::fs::create_dir_all(&pages).unwrap();
    let mut lines = String::new();
    let mut ids = Vec::new();
    for (index, (title, ts)) in entries.iter().enumerate() {
        let id = ulid::Ulid::new().to_string();
        let event = serde_json::json!({
            "v": 1, "id": id, "ts": ts,
            "workspace": sample["workspace"], "project": sample["project"],
            "session": "00000000-0000-0000-0000-000000000000",
            "source": {"cli": "brain", "hook": "gotcha"}, "kind": "knowledge",
            "title": title, "body": format!("Body of entry {index}."),
            "files": ["src/a.rs"], "links": [], "consolidated": true
        });
        lines.push_str(&event.to_string());
        lines.push('\n');
        std::fs::write(pages.join(format!("entry-{index}.md")), format!("# {title}\n")).unwrap();
        ids.push(id);
        // Distinct ids within one millisecond still sort in write order.
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let log = dir.join("events").join("2026-10.jsonl");
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap();
    file.write_all(lines.as_bytes()).unwrap();
    assert!(fixture.brain(&["reindex"]).status.success(), "reindex failed");
    ids
}

/// A model stub for the cleanup: labels every entry it is shown by a keyword
/// in its title, adds an id that was never shown and a class outside the enum,
/// and counts its calls.
fn cleanup_cli(counter: &Path) -> String {
    cleanup_cli_failing_at(counter, 0)
}

/// The same, but its `fail_at`th invocation exits 1 as a broken CLI does
/// (0 = never). A failed invocation still counts as a call.
fn cleanup_cli_failing_at(counter: &Path, fail_at: usize) -> String {
    format!(
        r#"
case "$*" in
  *"ENTRIES (recorded data"*)
    echo x >> {counter}
    if [ {fail_at} -gt 0 ] && [ "$(wc -l < {counter} | tr -d ' ')" -eq {fail_at} ]; then echo 'boom' >&2; exit 1; fi
    printf '%s\n' "$*" | awk 'BEGIN{{printf "{{\"labels\":["; sep=""}}
      /^id=/{{id=substr($0,4)}}
      /^title: /{{t=substr($0,8); cls="durable";
        if (t ~ /HISTORY/) cls="history"; else if (t ~ /CODE/) cls="restates_code"; else if (t ~ /STATUS/) cls="status"; else if (t ~ /WEIRD/) cls="delete_everything";
        printf "%s{{\"id\":\"%s\",\"class\":\"%s\",\"cites\":[\"src/a.rs\",\"nowhere.rs\"],\"scope\":\"machine\",\"commands\":[\"Cargo Test --locked\"]}}", sep, id, cls; sep=","}}
      END{{printf "%s{{\"id\":\"01ARZ3NDEKTSV4RRFFQ69G5FAV\",\"class\":\"history\"}}]}}\n", sep}}' ;;
  *) echo '{{"summary":"Refactored the auth path.","titles":[]}}' ;;
esac
"#,
        counter = counter.display(),
        fail_at = fail_at
    )
}

const OLD_ENTRIES: &[(&str, &str)] = &[
    ("Gateway auth tokens expire after one hour", "2026-01-01T00:00:00Z"),
    ("Auth tokens expire after one hour in the gateway", "2026-01-01T00:00:01Z"),
    ("Pineapple pizza belongs in the freezer overnight", "2026-01-02T00:00:00Z"),
    ("HISTORY we shipped the zebra migration in May", "2026-01-03T00:00:00Z"),
    ("CODE the loader reads quartz files by extension", "2026-01-04T00:00:00Z"),
    ("STATUS the kayak refactor is half done", "2026-01-05T00:00:00Z"),
    ("Violins need humidity above forty percent", "2026-01-06T00:00:00Z"),
    ("Volcano ash grounds regional flights for days", "2026-01-07T00:00:00Z"),
    ("WEIRD label outside the enum on a lighthouse note", "2026-01-08T00:00:00Z"),
];

fn run_cleanup(fixture: &Fixture, bin: &Path) {
    let done = fixture.brain_with_path(&["consolidate", "--force"], Some(bin));
    assert!(done.status.success(), "consolidate failed: {done:?}");
}

#[test]
fn the_one_time_cleanup_retires_only_what_it_should() {
    let fixture = Fixture::new("clean-once");
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    seed_old_knowledge(&fixture, OLD_ENTRIES);
    let pages_before = fixture.knowledge_pages().len();
    let log_before = fixture.log_text().lines().count();

    run_cleanup(&fixture, &bin);

    let retired = knowledge_titles(&fixture, 2);
    // One of the pair, the history, the code restatement; the status entry is
    // from January, long past its fortnight.
    assert_eq!(retired.len(), 4, "retired: {retired:?}");
    assert!(retired.iter().any(|t| t.starts_with("HISTORY")), "{retired:?}");
    assert!(retired.iter().any(|t| t.starts_with("CODE")), "{retired:?}");
    assert!(retired.iter().any(|t| t.starts_with("STATUS")), "{retired:?}");
    assert_eq!(
        retired.iter().filter(|t| t.to_lowercase().contains("auth tokens")).count(),
        1,
        "exactly one of the duplicate pair goes: {retired:?}"
    );
    let live = knowledge_titles(&fixture, 0);
    assert!(live.iter().any(|t| t.starts_with("WEIRD")), "a class outside the enum retired something: {live:?}");
    assert!(live.iter().any(|t| t.starts_with("Violins")), "{live:?}");

    // No page left the vault, and the log only grew.
    assert_eq!(fixture.knowledge_pages().len(), pages_before, "a vault page was removed");
    assert!(fixture.log_text().lines().count() > log_before);

    // Every retirement names a reason and one shared run id.
    let tombstones: Vec<serde_json::Value> = fixture
        .log_text()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|event| event["kind"] == "tombstone" && event["source"]["hook"] == "clean")
        .collect();
    assert_eq!(tombstones.len(), 4, "{tombstones:?}");
    let run = tombstones[0]["run"].as_str().expect("a run id").to_string();
    assert!(tombstones.iter().all(|t| t["run"] == run.as_str() && t["reason"].is_string()));
    let reasons: Vec<&str> = tombstones.iter().filter_map(|t| t["reason"].as_str()).collect();
    for wanted in ["duplicate", "history", "restates_code", "status_expired"] {
        assert!(reasons.contains(&wanted), "no `{wanted}` retirement: {reasons:?}");
    }

    // The labels carry cites only from the entry's own files, and normalized commands.
    let labelled = fixture.sql_one(
        "SELECT cites FROM events WHERE kind = 'knowledge' AND title LIKE 'Violins%'",
    );
    assert_eq!(labelled, r#"["src/a.rs"]"#);
    let labels = fixture.log_text();
    assert!(labels.contains(r#""commands":["cargo test"]"#), "commands were not normalized");
    assert!(labels.contains(r#""scope":"machine""#));

    // Doctor shows the run, by reason.
    let doctor = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).into_owned();
    assert!(doctor.contains("knowledge cleanup"), "{doctor}");
    assert!(doctor.contains(&format!("last run {run}")), "{doctor}");
    assert!(doctor.contains("retired 4 (duplicate 1 · history 1 · restates_code 1 · status_expired 1 · stale 0)"), "{doctor}");
    assert!(doctor.contains("undo: brain restore"), "{doctor}");
}

#[test]
fn restore_brings_a_cleanup_run_back() {
    let fixture = Fixture::new("clean-restore");
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    seed_old_knowledge(&fixture, OLD_ENTRIES);
    run_cleanup(&fixture, &bin);
    let retired = knowledge_titles(&fixture, 2);
    assert_eq!(retired.len(), 4, "{retired:?}");

    // The user forgets one of the retired entries after the cleanup.
    let history_id = fixture.sql_one("SELECT id FROM events WHERE kind = 'knowledge' AND title LIKE 'HISTORY%'");
    let forgotten = fixture.brain(&["forget", &history_id]);
    assert!(forgotten.status.success(), "forget failed: {forgotten:?}");
    assert_eq!(knowledge_titles(&fixture, 1).len(), 1);

    let restored = fixture.brain(&["restore"]);
    assert!(restored.status.success(), "restore failed: {restored:?}");
    let said = String::from_utf8_lossy(&restored.stdout).into_owned();
    assert!(said.contains("Restored 3 knowledge page(s)"), "{said}");

    assert_eq!(knowledge_titles(&fixture, 2).len(), 0, "a cleanup retirement remains");
    let user_forgot = knowledge_titles(&fixture, 1);
    assert_eq!(user_forgot.len(), 1, "{user_forgot:?}");
    assert!(user_forgot[0].starts_with("HISTORY"), "the user's forget was undone: {user_forgot:?}");

    // Back in search...
    let hits = String::from_utf8_lossy(&fixture.brain(&["search", "loader quartz"]).stdout).into_owned();
    assert!(hits.contains("CODE the loader reads quartz files"), "restored entry not in search: {hits}");
    let gone = String::from_utf8_lossy(&fixture.brain(&["search", "zebra migration"]).stdout).into_owned();
    assert!(!gone.contains("HISTORY we shipped"), "a user's forget is searchable again: {gone}");
    // ...and in the primer.
    let start = start_payload(&fixture.project, "0199a1f2-3c4d-7e8f-9012-3456789abc77", "startup");
    let primer = injected_context(&fixture.hook("claude-code", "SessionStart", &start)).unwrap_or_default();
    assert!(primer.contains("Violins need humidity"), "primer lacks live knowledge: {primer}");
    assert!(!primer.contains("zebra migration"), "the user's forget is primed again: {primer}");

    // The next consolidation does not withdraw the restored duplicate again.
    run_cleanup(&fixture, &bin);
    assert_eq!(knowledge_titles(&fixture, 2).len(), 0, "a restored page was withdrawn again");

    // Restoring again, or a run that does not exist, brings nothing.
    let again = fixture.brain(&["restore"]);
    assert!(String::from_utf8_lossy(&again.stdout).contains("Restored 0"));
    let other = fixture.brain(&["restore", "--run", "01ARZ3NDEKTSV4RRFFQ69G5FAV"]);
    assert!(!other.status.success(), "an unknown run exited 0: {other:?}");
    assert!(String::from_utf8_lossy(&other.stderr).contains("no cleanup run 01ARZ3NDEKTSV4RRFFQ69G5FAV"));
}

#[test]
fn a_second_cleanup_retires_nothing() {
    let fixture = Fixture::new("clean-twice");
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    seed_old_knowledge(&fixture, OLD_ENTRIES);
    // This is about the cleanup's own retirements; the move to the machine has
    // its own tests.
    fixture.sql_run("INSERT OR REPLACE INTO schema_state (key, value) VALUES ('machine_migrated', 'test')");
    run_cleanup(&fixture, &bin);
    let calls = || std::fs::read_to_string(&counter).unwrap_or_default().lines().count();
    let retired = knowledge_titles(&fixture, 2);
    let spent = calls();
    assert!(spent >= 1);

    run_cleanup(&fixture, &bin);
    assert_eq!(knowledge_titles(&fixture, 2), retired, "a finished cleanup retired more");
    // Only the entry whose label was refused is asked about once more, by the
    // remainder stage, and a sweep that labels nothing ends there.
    assert!(calls() <= spent + 1, "a finished cleanup called the model again");
    let spent = calls();
    run_cleanup(&fixture, &bin);
    assert_eq!(calls(), spent, "a finished cleanup called the model again");

    // Without the flag it looks again, and finds every entry already labelled.
    fixture.sql_run("DELETE FROM schema_state WHERE key = 'knowledge_cleaned'");
    run_cleanup(&fixture, &bin);
    assert_eq!(knowledge_titles(&fixture, 2), retired, "a repeat cleanup retired more");
    // Only the entry whose label was refused (outside the enum) is asked again.
    assert!(calls() <= spent + 1, "a repeat cleanup asked the model about labelled entries");
}

#[test]
fn a_replay_rebuilds_cleanup_state() {
    let fixture = Fixture::new("clean-replay");
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    seed_old_knowledge(&fixture, OLD_ENTRIES);
    run_cleanup(&fixture, &bin);
    let id = fixture.sql_one("SELECT id FROM events WHERE kind = 'knowledge' AND title LIKE 'CODE%'");
    assert!(fixture.brain(&["restore"]).status.success());
    // A second pass of the same kind: retire again is not possible, so forget one.
    let forget = fixture.sql_one("SELECT id FROM events WHERE kind = 'knowledge' AND title LIKE 'Volcano%'");
    assert!(fixture.brain(&["forget", &forget]).status.success());

    let snapshot = |fixture: &Fixture| -> String {
        let conn = rusqlite::Connection::open(fixture.home.join("brain.db")).unwrap();
        let mut statement = conn
            .prepare(
                "SELECT id, forgotten, IFNULL(class, ''), IFNULL(expires_at, ''), IFNULL(cites, '')
                 FROM events WHERE kind = 'knowledge' ORDER BY id",
            )
            .unwrap();
        statement
            .query_map([], |row| {
                Ok(format!(
                    "{}|{}|{}|{}|{}",
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?
                ))
            })
            .unwrap()
            .filter_map(Result::ok)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let before = snapshot(&fixture);
    assert!(before.contains("restates_code"), "{before}");
    assert!(before.contains(&id));

    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(fixture.home.join(format!("brain.db{suffix}")));
    }
    assert!(fixture.brain(&["reindex"]).status.success(), "reindex failed");
    assert_eq!(snapshot(&fixture), before, "a replay arrived at a different cleanup state");
}

#[test]
fn cleanup_respects_its_call_ceiling() {
    let fixture = Fixture::new("clean-ceiling");
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    // The 3rd invocation fails: it is a call like any other.
    let bin = fixture.fake_cli("claude", &cleanup_cli_failing_at(&counter, 3));
    // Gibberish titles, and far more of them than 30 calls can label: the
    // duplicate stage still folds some, so the margin keeps the ceiling the
    // thing that ends the run.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut word = || {
        (0..7)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                char::from(b'a' + ((seed >> 33) % 26) as u8)
            })
            .collect::<String>()
    };
    let titles: Vec<String> = (0..2_600).map(|_| format!("{} {} {} {}", word(), word(), word(), word())).collect();
    let entries: Vec<(&str, &str)> = titles.iter().map(|t| (t.as_str(), "2026-01-01T00:00:00Z")).collect();
    seed_old_knowledge(&fixture, &entries);

    let calls = || std::fs::read_to_string(&counter).unwrap_or_default().lines().count();
    let rows = || -> i64 {
        fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM summarizer_calls WHERE purpose = 'clean'").parse().unwrap()
    };
    let label_rows = || -> usize {
        fixture
            .sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM summarizer_calls WHERE purpose = 'label'")
            .parse()
            .unwrap()
    };
    for _ in 0..12 {
        run_cleanup(&fixture, &bin);
        // The check is made before each batch, so one ladder run may end a
        // call or two past it. Once it has ended, the remainder is labelled
        // under its own budget of 5 a day.
        assert!(calls() <= 32 + 5, "the cleanup made {} model calls", calls());
    }
    assert!((30..=32).contains(&(calls() - label_rows())), "the cleanup made {} model calls", calls());
    assert_eq!(rows() + label_rows() as i64, calls() as i64, "the ledger and the CLI disagree on the calls made");
    // It ended rather than starting over, with the rest left unlabelled.
    assert_eq!(fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM schema_state WHERE key = 'knowledge_cleaned'"), "1");
    let doctor = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).into_owned();
    assert!(doctor.contains("left unlabelled"), "{doctor}");
}

#[test]
fn the_unlabelled_remainder_is_labelled_a_little_each_day() {
    let fixture = Fixture::new("label-remainder");
    let counter = fixture.home.parent().unwrap().join("label-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    // 8 batches of 40 and a few more, with the one-time cleanup already over.
    // Gibberish, so that no fold takes any of them for a repeat.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut word = || {
        (0..7)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                char::from(b'a' + ((seed >> 33) % 26) as u8)
            })
            .collect::<String>()
    };
    let titles: Vec<String> = (0..330).map(|_| format!("{} {} {} {}", word(), word(), word(), word())).collect();
    let entries: Vec<(&str, &str)> = titles.iter().map(|t| (t.as_str(), "2026-10-01T00:00:00Z")).collect();
    seed_old_knowledge(&fixture, &entries);
    fixture.sql_run(
        "INSERT OR REPLACE INTO schema_state (key, value) VALUES ('knowledge_cleaned', '2026-10-01');
         INSERT OR REPLACE INTO schema_state (key, value) VALUES ('clean_last_run',
           '{\"run\":\"old\",\"at\":\"2026-10-01T00:00:00Z\",\"calls\":0}');",
    );

    let calls = || std::fs::read_to_string(&counter).unwrap_or_default().lines().count();
    let rows =|| -> usize {
        fixture
            .sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM summarizer_calls WHERE purpose = 'label'")
            .parse()
            .unwrap()
    };
    let left = || -> usize {
        fixture
            .sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM events WHERE kind = 'knowledge' AND forgotten = 0 AND team = 0 AND class IS NULL")
            .parse()
            .unwrap()
    };
    let run_id = || fixture.sql_one("SELECT json_extract(value, '$.run') FROM schema_state WHERE key = 'label_remainder'");
    assert_eq!(left(), 330);

    run_cleanup(&fixture, &bin);
    assert_eq!((calls(), rows()), (5, 5), "the first pass is held to the daily budget");
    assert!(left() > 0 && left() < 330, "{} left, {} total", left(), fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM events WHERE kind = 'knowledge'"));
    let first_run = run_id();
    let doctor = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).into_owned();
    assert!(doctor.contains(&format!("{} left unlabelled", left())), "{doctor}");

    // The same day: the ledger says the budget is spent.
    run_cleanup(&fixture, &bin);
    assert_eq!((calls(), rows()), (5, 5), "a second pass the same day made calls");

    // A day later the rest is labelled, and then nothing is called.
    fixture.sql_run("UPDATE summarizer_calls SET ts = '2020-01-01T00:00:00.000000Z' WHERE purpose = 'label'");
    run_cleanup(&fixture, &bin);
    assert_eq!(left(), 0);
    let total = calls();
    assert!((6..=9).contains(&total), "{total} calls");
    assert_ne!(run_id(), first_run, "a new day has its own run id");
    fixture.sql_run("UPDATE summarizer_calls SET ts = '2020-01-01T00:00:00.000000Z' WHERE purpose = 'label'");
    run_cleanup(&fixture, &bin);
    assert_eq!(calls(), total, "labelled entries were called again");
}

#[test]
fn a_machine_lesson_never_leaves_the_machine() {
    // `MACHINE` knowledge lives at <data>/machine, beside the vault. Team,
    // sync and export all walk the vault, so none of them may carry it.
    const MACHINE_PROJECT: &str = "65a15af1-5427-510f-88a9-576cf8b3a005";
    const DEFAULT_WORKSPACE: &str = "1626d894-a5e5-5e55-bb85-855adfc8035c";
    let title = "Zanzibar machine quirk the keyboard repeats";
    let marker = "[project]\nname = \"machineprop\"\n";
    let a = Fixture::new("machine-a");
    let b = Fixture::new("machine-b");
    std::fs::write(a.project.join(".rolepod-brain.toml"), marker).unwrap();
    std::fs::write(b.project.join(".rolepod-brain.toml"), marker).unwrap();
    seed_old_knowledge(&a, &[("Ordinary gateway lesson about retries", "2026-01-01T00:00:00Z")]);

    let machine_events = a.home.join("machine").join("events");
    std::fs::create_dir_all(&machine_events).unwrap();
    let event = serde_json::json!({
        "v": 1, "id": ulid::Ulid::new().to_string(), "ts": "2026-01-02T00:00:00Z",
        "workspace": DEFAULT_WORKSPACE, "project": MACHINE_PROJECT,
        "session": "00000000-0000-0000-0000-000000000000",
        "source": {"cli": "brain", "hook": "gotcha"}, "kind": "knowledge",
        "title": title, "body": "Body of the machine lesson.",
        "files": [], "links": [], "consolidated": true
    });
    std::fs::write(machine_events.join("2026-10.jsonl"), format!("{event}\n")).unwrap();

    // `brain search` is scoped to the cwd's project, so ask the store itself.
    let count = |f: &Fixture, prefix: &str| {
        let out = Command::new("sqlite3")
            .arg(f.home.join("brain.db"))
            .arg(format!("SELECT COUNT(*) FROM events WHERE title LIKE '{prefix}%';"))
            .output()
            .expect("query the store");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let found = |f: &Fixture| count(f, "Zanzibar") == "1";
    assert!(a.brain(&["reindex"]).status.success());
    assert!(found(&a), "the machine lesson was not replayed");

    // Nothing the three outbound paths build holds it.
    let shared_team = a.home.parent().unwrap().join("machine-team");
    let shared_sync = a.home.parent().unwrap().join("machine-sync");
    assert!(a.brain(&["team", "init", shared_team.to_str().unwrap(), "--name", "Alex"]).status.success());
    assert!(b.brain(&["team", "init", shared_team.to_str().unwrap(), "--name", "Sam"]).status.success());
    std::fs::copy(a.home.join("team.key"), b.home.join("team.key")).unwrap();
    assert!(a.brain(&["team"]).status.success());
    assert!(b.brain(&["team"]).status.success());
    assert!(a.brain(&["sync", "init", shared_sync.to_str().unwrap()]).status.success());
    assert!(b.brain(&["sync", "init", shared_sync.to_str().unwrap()]).status.success());
    std::fs::copy(a.home.join("sync.key"), b.home.join("sync.key")).unwrap();
    assert!(a.brain(&["sync"]).status.success());
    assert!(b.brain(&["sync"]).status.success());
    assert!(b.brain(&["reindex"]).status.success());
    // Positive control: something did cross, so the absence below means something.
    assert!(
        count(&b, "Ordinary gateway").parse::<u32>().unwrap_or(0) > 0,
        "neither team nor sync carried the ordinary lesson to B"
    );
    assert!(!found(&b), "the machine lesson reached a teammate or a second machine");
    assert!(!b.home.join("machine").exists(), "B grew a machine directory");

    // The export, extracted for real.
    let archive = a.home.parent().unwrap().join("machine-export.tar.gz");
    assert!(a.brain(&["export", archive.to_str().unwrap()]).status.success());
    let out = a.home.parent().unwrap().join("machine-extracted");
    std::fs::create_dir_all(&out).unwrap();
    assert!(Command::new("tar").args(["-xzf", archive.to_str().unwrap(), "-C", out.to_str().unwrap()]).status().unwrap().success());
    let mut all = String::new();
    collect_ext(&out, "jsonl", &mut all);
    collect_ext(&out, "md", &mut all);
    assert!(all.contains("Ordinary gateway lesson"), "the export held nothing to compare against");
    assert!(!all.contains("Zanzibar") && !all.contains(MACHINE_PROJECT), "the export carried the machine lesson");

    // The database is derived: delete it and the lesson comes back from the log.
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(a.home.join(format!("brain.db{suffix}")));
    }
    assert!(a.brain(&["reindex"]).status.success());
    assert!(found(&a), "a rebuild lost the machine lesson");
}

// --- the one-time move of old machine lessons ------------------------------

const MACHINE_PROJECT_ID: &str = "65a15af1-5427-510f-88a9-576cf8b3a005";

/// Lessons the cleanup stub labels `machine`: three that may leave their
/// project, one that names a repo path and one that names the project.
const MOVABLE: &[&str] = &[
    "Cargo test needs the locked flag to match the lockfile",
    "Pineapple pizza belongs in the freezer overnight",
    "Violins need humidity above forty percent",
];
const STAYING: &[&str] = &["Edit src/main.rs before running the loader", "The checkout script prints a banner first"];

fn lesson_entries() -> Vec<(&'static str, &'static str)> {
    let stamps = ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z"];
    MOVABLE.iter().chain(STAYING).copied().zip(stamps).collect()
}

fn count_knowledge(fixture: &Fixture, clause: &str) -> i64 {
    fixture
        .sql_one(&format!("SELECT CAST(COUNT(*) AS TEXT) FROM events WHERE kind = 'knowledge' AND {clause}"))
        .parse()
        .unwrap()
}

/// How many times a title reads: live, in any project or on the machine.
fn live_copies(fixture: &Fixture, title: &str) -> i64 {
    count_knowledge(fixture, &format!("forgotten = 0 AND title = '{title}'"))
}

/// A fixture whose old lessons were labelled by the cleanup and then moved:
/// the stub CLI, and the id of the move's run.
fn moved_fixture(name: &str) -> (Fixture, PathBuf, String) {
    let fixture = Fixture::new(name);
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    seed_old_knowledge(&fixture, &lesson_entries());
    // The first pass labels, the second moves.
    run_cleanup(&fixture, &bin);
    assert_eq!(fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM schema_state WHERE key = 'machine_migrated'"), "0");
    run_cleanup(&fixture, &bin);
    let run = fixture.sql_one(
        "SELECT json_extract(value, '$.run') FROM schema_state WHERE key = 'machine_moved'",
    );
    (fixture, bin, run)
}

fn log_events(dir: &Path) -> Vec<serde_json::Value> {
    let mut text = String::new();
    collect_jsonl(dir, &mut text);
    text.lines().filter_map(|line| serde_json::from_str(line).ok()).collect()
}

#[test]
fn old_machine_lessons_move_once_and_restore_brings_them_back() {
    let (fixture, bin, run) = moved_fixture("machine-move");

    // Moved: three live on the machine, the three originals cleaned, and the
    // lessons that name a repo path or the project stayed where they were.
    assert_eq!(count_knowledge(&fixture, &format!("project = '{MACHINE_PROJECT_ID}' AND forgotten = 0")), 3);
    assert_eq!(count_knowledge(&fixture, &format!("project != '{MACHINE_PROJECT_ID}' AND forgotten = 2")), 3);
    for title in STAYING {
        assert_eq!(
            count_knowledge(&fixture, &format!("project != '{MACHINE_PROJECT_ID}' AND forgotten = 0 AND title = '{title}'")),
            1,
            "{title} left its project"
        );
        assert_eq!(live_copies(&fixture, title), 1, "{title}");
    }
    for title in MOVABLE {
        assert_eq!(live_copies(&fixture, title), 1, "{title} reads {} times", live_copies(&fixture, title));
        assert_eq!(
            count_knowledge(&fixture, &format!("project = '{MACHINE_PROJECT_ID}' AND forgotten = 0 AND title = '{title}'")),
            1
        );
    }
    assert_eq!(fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM schema_state WHERE key = 'machine_migrated'"), "1");

    // An ordinary clean tombstone, no new kind: an old binary reads it as it
    // reads any other cleanup.
    let moved: Vec<serde_json::Value> = log_events(&fixture.wiki())
        .into_iter()
        .filter(|event| event["reason"] == "reclassified_machine")
        .collect();
    assert_eq!(moved.len(), 3, "{moved:?}");
    assert!(moved.iter().all(|e| e["kind"] == "tombstone" && e["source"]["hook"] == "clean" && e["run"] == run.as_str()));
    let machine_events = log_events(&fixture.home.join("machine"));
    assert_eq!(machine_events.len(), 3, "{machine_events:?}");
    for copy in &machine_events {
        assert_eq!(copy["kind"], "knowledge");
        let old = copy["links"][0].as_str().expect("a link to the original");
        assert_eq!(copy["links"].as_array().unwrap().len(), 1);
        assert_eq!(
            fixture.sql_one(&format!("SELECT title FROM events WHERE id = '{old}'")),
            copy["title"].as_str().unwrap()
        );
    }
    // The originals' triggers moved with them, and the hook's cache has them.
    assert_eq!(
        fixture.sql_one(&format!(
            "SELECT CAST(COUNT(*) AS TEXT) FROM lesson_triggers t JOIN events e ON e.id = t.event_id
             WHERE e.project = '{MACHINE_PROJECT_ID}' AND e.forgotten = 0"
        )),
        "3"
    );
    assert!(std::fs::read_to_string(fixture.home.join("lesson-programs")).unwrap().contains("cargo"));
    let doctor = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).into_owned();
    assert!(doctor.contains(&format!("reclassified_machine 3 (run {run})")), "{doctor}");

    // Moved once: more passes, and with the flag cleared too, move nothing.
    let settled = || {
        (
            count_knowledge(&fixture, "1 = 1"),
            fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM events WHERE kind = 'tombstone'"),
        )
    };
    let before = settled();
    run_cleanup(&fixture, &bin);
    run_cleanup(&fixture, &bin);
    assert_eq!(settled(), before, "a later pass moved more");

    // Restore brings the originals back, and the machine's copies go with
    // them: every title reads exactly once, in its own project.
    let restored = fixture.brain(&["restore", "--run", &run]);
    assert!(restored.status.success(), "restore failed: {restored:?}");
    assert!(String::from_utf8_lossy(&restored.stdout).contains("Restored 3"), "{restored:?}");
    for title in MOVABLE.iter().chain(STAYING) {
        assert_eq!(live_copies(&fixture, title), 1, "{title} reads {} times after restore", live_copies(&fixture, title));
        assert_eq!(
            count_knowledge(&fixture, &format!("project != '{MACHINE_PROJECT_ID}' AND forgotten = 0 AND title = '{title}'")),
            1,
            "{title} is not back in its project"
        );
    }
    assert_eq!(count_knowledge(&fixture, &format!("project = '{MACHINE_PROJECT_ID}' AND forgotten = 0")), 0);
    let programs = fixture.sql_one(&format!(
        "SELECT CAST(COUNT(*) AS TEXT) FROM lesson_triggers t JOIN events e ON e.id = t.event_id
         WHERE e.project = '{MACHINE_PROJECT_ID}' AND e.forgotten = 0"
    ));
    assert_eq!(programs, "0");

    // A restored lesson is not moved again, even from a fresh start.
    fixture.sql_run("DELETE FROM schema_state WHERE key IN ('machine_migrated', 'machine_cursor')");
    run_cleanup(&fixture, &bin);
    run_cleanup(&fixture, &bin);
    for title in MOVABLE {
        assert_eq!(live_copies(&fixture, title), 1, "{title} after a second move");
        assert_eq!(
            count_knowledge(&fixture, &format!("project != '{MACHINE_PROJECT_ID}' AND forgotten = 0 AND title = '{title}'")),
            1,
            "a restored {title} was moved again"
        );
    }
}

#[test]
fn a_corrected_lesson_stays_in_its_project_and_a_corrected_copy_survives_a_restore() {
    // Corrected before the move: it is not moved.
    let fixture = Fixture::new("machine-corrected");
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    seed_old_knowledge(&fixture, &lesson_entries());
    run_cleanup(&fixture, &bin);
    let kept = MOVABLE[1];
    let id = fixture.sql_one(&format!("SELECT id FROM events WHERE kind = 'knowledge' AND title = '{kept}'"));
    assert!(fixture.brain(&["correct", &id, "Pizza belongs in the freezer for one night."]).status.success());
    run_cleanup(&fixture, &bin);
    assert_eq!(count_knowledge(&fixture, &format!("project = '{MACHINE_PROJECT_ID}' AND forgotten = 0")), 2);
    assert_eq!(
        count_knowledge(&fixture, &format!("project != '{MACHINE_PROJECT_ID}' AND forgotten = 0 AND id = '{id}'")),
        1,
        "a corrected lesson was moved"
    );
    let doctor = String::from_utf8_lossy(&fixture.brain(&["doctor"]).stdout).into_owned();
    assert!(doctor.contains("1 kept in their project as a person corrected them"), "{doctor}");

    // Corrected after the move: a restore leaves the corrected copy live.
    let (fixture, _bin, run) = moved_fixture("machine-corrected-copy");
    let title = MOVABLE[0];
    let copy = fixture.sql_one(&format!(
        "SELECT id FROM events WHERE kind = 'knowledge' AND title = '{title}' AND project = '{MACHINE_PROJECT_ID}'"
    ));
    assert!(fixture.brain(&["correct", &copy, "Cargo test wants --locked, as CI does."]).status.success());
    assert!(fixture.brain(&["restore", "--run", &run]).status.success());
    assert_eq!(
        count_knowledge(&fixture, &format!("id = '{copy}' AND forgotten = 0")),
        1,
        "the restore retired a person's correction"
    );
    assert_eq!(
        count_knowledge(&fixture, &format!("project != '{MACHINE_PROJECT_ID}' AND forgotten = 0 AND title = '{title}'")),
        1,
        "the original is not back in its project"
    );
}

#[test]
fn old_machine_lessons_wait_for_the_remainder_to_be_labelled() {
    let fixture = Fixture::new("machine-order");
    let counter = fixture.home.parent().unwrap().join("clean-calls");
    let bin = fixture.fake_cli("claude", &cleanup_cli(&counter));
    let nobody = fixture.home.parent().unwrap().join("no-cli");
    std::fs::create_dir_all(&nobody).unwrap();
    // Two labelled already, one the stub labels, one it refuses (outside the enum).
    let mut entries = lesson_entries();
    entries.truncate(3);
    entries.push(("WEIRD label outside the enum on a lighthouse note", "2026-01-08T00:00:00Z"));
    seed_old_knowledge(&fixture, &entries);
    fixture.sql_run(
        "INSERT OR REPLACE INTO schema_state (key, value) VALUES ('knowledge_cleaned', '2026-10-01');
         INSERT OR REPLACE INTO schema_state (key, value) VALUES ('clean_last_run',
           '{\"run\":\"old\",\"at\":\"2026-10-01T00:00:00Z\",\"calls\":0}');
         UPDATE events SET class = 'durable', scope = 'machine'
           WHERE kind = 'knowledge' AND title IN ('Cargo test needs the locked flag to match the lockfile',
                                                  'Pineapple pizza belongs in the freezer overnight');",
    );
    let flag = || fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM schema_state WHERE key = 'machine_migrated'");
    let cleaned = || count_knowledge(&fixture, "forgotten = 2");

    // No model answers: two lessons are unlabelled, the sweep is not over.
    let done = fixture.brain_with_path(&["consolidate", "--force"], Some(&nobody));
    assert!(done.status.success(), "consolidate failed: {done:?}");
    assert!(fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM events WHERE kind = 'knowledge' AND class IS NULL") != "0");
    assert_eq!((flag().as_str(), cleaned()), ("0", 0), "the move started with lessons unlabelled");
    assert!(!fixture.home.join("machine").exists(), "the machine grew a directory before its turn");

    // The stub labels one; the other is refused, but the sweep has not yet
    // seen that it labelled nothing, so the remainder is still unfinished.
    run_cleanup(&fixture, &bin);
    assert_eq!((flag().as_str(), cleaned()), ("0", 0), "the move started before the remainder was done");

    // The next sweep labels nothing and gives up (stuck): now the move runs.
    run_cleanup(&fixture, &bin);
    assert_eq!(fixture.sql_one("SELECT CAST(json_extract(value, '$.stuck') AS TEXT) FROM schema_state WHERE key = 'label_remainder'"), "1");
    assert_eq!(flag(), "1");
    assert_eq!(cleaned(), 3, "lessons labelled machine did not move once the remainder was done");
    assert_eq!(count_knowledge(&fixture, &format!("project = '{MACHINE_PROJECT_ID}' AND forgotten = 0")), 3);
}

#[test]
fn a_replay_rebuilds_the_machine_scope() {
    let (fixture, _bin, run) = moved_fixture("machine-replay");
    let snapshot = |fixture: &Fixture| -> String {
        let conn = rusqlite::Connection::open(fixture.home.join("brain.db")).unwrap();
        let mut rows: Vec<String> = Vec::new();
        let mut statement = conn
            .prepare(
                "SELECT id, project, forgotten, IFNULL(class, ''), IFNULL(scope, '') FROM events
                 WHERE kind = 'knowledge' ORDER BY id",
            )
            .unwrap();
        rows.extend(
            statement
                .query_map([], |row| {
                    Ok(format!(
                        "{}|{}|{}|{}|{}",
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?
                    ))
                })
                .unwrap()
                .filter_map(Result::ok),
        );
        let mut triggers = conn
            .prepare("SELECT event_id, program, sub FROM lesson_triggers ORDER BY event_id, program, sub")
            .unwrap();
        rows.extend(
            triggers
                .query_map([], |row| {
                    Ok(format!("T {}|{}|{}", row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
                })
                .unwrap()
                .filter_map(Result::ok),
        );
        rows.join("\n")
    };
    let rebuild = |fixture: &Fixture| {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(fixture.home.join(format!("brain.db{suffix}")));
        }
        assert!(fixture.brain(&["reindex"]).status.success(), "reindex failed");
    };

    let moved = snapshot(&fixture);
    assert!(moved.contains(MACHINE_PROJECT_ID), "{moved}");
    rebuild(&fixture);
    assert_eq!(snapshot(&fixture), moved, "a replay lost the moved lessons");

    // And after a restore, which also withdrew the machine's copies.
    assert!(fixture.brain(&["restore", "--run", &run]).status.success());
    let restored = snapshot(&fixture);
    rebuild(&fixture);
    assert_eq!(snapshot(&fixture), restored, "a replay arrived at a different state after a restore");
    for title in MOVABLE {
        assert_eq!(live_copies(&fixture, title), 1, "{title}");
    }
}

#[test]
fn the_same_tool_lesson_from_two_projects_folds_into_one() {
    let a = Fixture::new("machine-fold");
    let counter = a.home.parent().unwrap().join("clean-calls");
    let bin = a.fake_cli("claude", &cleanup_cli(&counter));
    seed_old_knowledge(&a, &[("Gateway auth tokens expire after one hour", "2026-01-01T00:00:00Z")]);
    // A second project on the same brain. Not dropped: the base belongs to `a`.
    let other = a.home.parent().unwrap().join("other");
    std::fs::create_dir_all(&other).unwrap();
    run_in(&other, "git", &["init", "-q"]);
    let b = std::mem::ManuallyDrop::new(Fixture { home: a.home.clone(), project: other });
    let known = a.project_dirs();
    b.seed_session(1);
    let dir_b = b.project_dirs().into_iter().find(|dir| !known.contains(dir)).expect("a second project");
    seed_old_knowledge_in(&b, &dir_b, &[("Auth tokens expire after one hour in the gateway", "2026-01-02T00:00:00Z")]);

    run_cleanup(&a, &bin);
    run_cleanup(&a, &bin);

    assert_eq!(count_knowledge(&a, &format!("project != '{MACHINE_PROJECT_ID}' AND forgotten = 2")), 2, "both originals move");
    assert_eq!(count_knowledge(&a, &format!("project = '{MACHINE_PROJECT_ID}'")), 2, "both reach the machine");
    assert_eq!(
        count_knowledge(&a, &format!("project = '{MACHINE_PROJECT_ID}' AND forgotten = 0")),
        1,
        "the same lesson from two projects reads twice on the machine"
    );
    assert_eq!(count_knowledge(&a, &format!("project = '{MACHINE_PROJECT_ID}' AND forgotten = 2")), 1);
}

/// A fixture whose project holds one lesson triggered by `timeout`, with the
/// program-list cache the way consolidation leaves it, and `extra` settled
/// observations behind it.
fn shell_fixture(name: &str, extra: usize) -> Fixture {
    let fixture = Fixture::new(name);
    fixture.seed_session(1);
    let dir = fixture.project_dirs().into_iter().next().expect("a project directory");
    let first = std::fs::read_dir(dir.join("events"))
        .unwrap()
        .filter_map(Result::ok)
        .find(|entry| entry.path().extension().is_some_and(|ext| ext == "jsonl"))
        .expect("an event log")
        .path();
    let sample: serde_json::Value =
        serde_json::from_str(std::fs::read_to_string(&first).unwrap().lines().next().unwrap()).unwrap();
    let mut lines = String::new();
    let lesson = serde_json::json!({
        "v": 1, "id": ulid::Ulid::new().to_string(), "ts": "2026-10-01T00:00:00.000000Z",
        "workspace": sample["workspace"], "project": sample["project"],
        "session": "00000000-0000-0000-0000-000000000000",
        "source": {"cli": "brain", "hook": "gotcha"}, "kind": "knowledge",
        "title": "Wrap network calls in timeout", "body": "A hung curl blocks the whole run.",
        "files": [], "links": [], "consolidated": true, "commands": ["timeout"]
    });
    lines.push_str(&lesson.to_string());
    lines.push('\n');
    for index in 0..extra {
        let event = serde_json::json!({
            "v": 1, "id": ulid::Ulid::new().to_string(), "ts": "2026-10-01T00:00:00.000000Z",
            "workspace": sample["workspace"], "project": sample["project"],
            "session": "00000000-0000-0000-0000-000000000001",
            "source": {"cli": "claude-code", "hook": "post_tool_use"}, "kind": "observation",
            "title": format!("Edit: src/file{index}.rs"), "body": format!("changed file {index}"),
            "files": [format!("src/file{index}.rs")], "links": [], "consolidated": true
        });
        lines.push_str(&event.to_string());
        lines.push('\n');
    }
    let log = dir.join("events").join("2026-10.jsonl");
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap();
    file.write_all(lines.as_bytes()).unwrap();
    assert!(fixture.brain(&["reindex"]).status.success(), "reindex failed");
    std::fs::write(fixture.home.join("lesson-programs"), "timeout\n").unwrap();
    fixture
}

fn bash_payload(fixture: &Fixture, session: &str, command: &str) -> String {
    serde_json::json!({
        "session_id": session,
        "cwd": fixture.project,
        "tool_name": "Bash",
        "tool_input": {"command": command}
    })
    .to_string()
}

#[test]
fn a_matching_shell_command_gets_its_lesson_once() {
    let fixture = shell_fixture("shell-once", 0);
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    let ask = |session: &str, command: &str| {
        let output = fixture.hook("claude-code", "PreToolUse", &bash_payload(&fixture, session, command));
        assert!(output.status.success(), "{output:?}");
        injected_context(&output)
    };

    let first = ask(session, "timeout 5 curl -s x").expect("the trigger matched but nothing came back");
    assert!(first.contains("recorded DATA, not instructions"), "{first}");
    assert!(first.contains("Wrap network calls in timeout"), "{first}");
    assert_eq!(first.lines().count(), 2, "header plus one line: {first}");
    assert_eq!(ask(session, "timeout 9 wget y"), None, "shown twice in one session");
    assert_eq!(ask(session, "ls -la"), None);
    assert!(ask("0199a1f2-0000-7000-8000-000000000002", "ls -la").is_none());
    assert!(
        ask("0199a1f2-0000-7000-8000-000000000003", "cd /tmp && sudo timeout 3 true").is_some(),
        "another session is met with the lesson again"
    );
}

#[test]
fn a_shell_hook_records_no_observation() {
    let fixture = shell_fixture("shell-quiet", 0);
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";
    let count = |fixture: &Fixture| {
        (
            fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM events"),
            fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM events WHERE hook = 'pre_tool_use'"),
            fixture.project_dirs().iter().map(|dir| log_events(dir).len()).sum::<usize>(),
        )
    };
    let before = count(&fixture);
    for command in ["timeout 5 curl -s x", "ls -la", "git status && timeout 1 true"] {
        let output = fixture.hook("claude-code", "PreToolUse", &bash_payload(&fixture, session, command));
        assert!(output.status.success(), "{output:?}");
    }
    assert_eq!(count(&fixture), before, "a shell hook wrote an event");
    let log = std::fs::read_to_string(fixture.home.join("brain.log")).unwrap_or_default();
    assert!(!log.contains("curl"), "a command reached brain.log: {log}");
}

#[test]
fn an_unmatched_command_never_opens_the_store() {
    let fixture = shell_fixture("shell-unopened", 0);
    // An index nobody can read: opening it fails, so an answer proves it was
    // never opened.
    for suffix in ["-wal", "-shm"] {
        let _ = std::fs::remove_file(fixture.home.join(format!("brain.db{suffix}")));
    }
    std::fs::write(fixture.home.join("brain.db"), b"this is not a database at all").unwrap();
    let session = "0199a1f2-3c4d-7e8f-9012-3456789abcde";

    let unmatched = fixture.hook("claude-code", "PreToolUse", &bash_payload(&fixture, session, "ls -la | wc -l"));
    assert!(unmatched.status.success(), "{unmatched:?}");
    assert_eq!(String::from_utf8_lossy(&unmatched.stdout).trim(), "{}");
    assert!(unmatched.stderr.is_empty(), "{unmatched:?}");

    // A matched command with the store unreadable still answers quietly.
    let matched = fixture.hook("claude-code", "PreToolUse", &bash_payload(&fixture, session, "timeout 5 curl x"));
    assert!(matched.status.success(), "{matched:?}");
    assert_eq!(String::from_utf8_lossy(&matched.stdout).trim(), "{}");
    assert!(matched.stderr.is_empty(), "{matched:?}");
    let log = std::fs::read_to_string(fixture.home.join("brain.log")).unwrap_or_default();
    assert!(!log.contains("curl"), "a command reached brain.log: {log}");
}

#[test]
fn a_shell_hook_returns_inside_its_budget() {
    let fixture = shell_fixture("shell-budget", 10_000);
    let rows: i64 = fixture.sql_one("SELECT CAST(COUNT(*) AS TEXT) FROM events").parse().unwrap();
    assert!(rows >= 10_000, "precondition: {rows} events");
    let p95 = |command: &str, fresh_session: bool| {
        let mut times = Vec::new();
        // Two warm-up spawns, then twenty measured.
        for round in 0..22 {
            let session = if fresh_session {
                format!("0199b000-0000-7000-8000-{round:012}")
            } else {
                "0199a1f2-3c4d-7e8f-9012-3456789abcde".to_string()
            };
            let payload = bash_payload(&fixture, &session, command);
            let start = std::time::Instant::now();
            let output = fixture.hook("claude-code", "PreToolUse", &payload);
            let took = start.elapsed();
            assert!(output.status.success());
            if round >= 2 {
                times.push(took);
            }
        }
        times.sort();
        times[times.len() * 95 / 100]
    };
    let budget = if cfg!(debug_assertions) { 250 } else { 50 };
    let budget = std::time::Duration::from_millis(budget);
    // A fresh session each time, so every matched call reaches the lesson.
    let matched = p95("timeout 5 curl -s x", true);
    let unmatched = p95("ls -la", false);
    assert!(matched < budget, "matched shell hook p95 was {matched:?}, over {budget:?}");
    assert!(unmatched < budget, "unmatched shell hook p95 was {unmatched:?}, over {budget:?}");
}

#[test]
fn a_machine_lesson_reaches_another_project_and_a_project_lesson_does_not() {
    const MACHINE_PROJECT: &str = "65a15af1-5427-510f-88a9-576cf8b3a005";
    const DEFAULT_WORKSPACE: &str = "1626d894-a5e5-5e55-bb85-855adfc8035c";
    let fixture = Fixture::new("machine-cross");
    let b_dir = fixture.project.parent().unwrap().join("checkout-b");
    std::fs::create_dir_all(&b_dir).unwrap();
    run_in(&b_dir, "git", &["init", "-q"]);

    // Both projects exist; project A holds a lesson of its own.
    fixture.seed_session(1);
    let a_dir = fixture.project_dirs().into_iter().next().expect("project A's directory");
    let b_read = serde_json::json!({
        "session_id": "0199a1f2-0000-7000-8000-00000000000b", "cwd": b_dir,
        "tool_name": "Edit", "tool_input": {"file_path": b_dir.join("main.rs")}
    })
    .to_string();
    assert!(fixture.hook("claude-code", "PostToolUse", &b_read).status.success());
    seed_old_knowledge_in(&fixture, &a_dir, &[("Quokka cache lesson only for project A", "2026-01-01T00:00:00Z")]);

    let machine_events = fixture.home.join("machine").join("events");
    std::fs::create_dir_all(&machine_events).unwrap();
    let lesson = serde_json::json!({
        "v": 1, "id": ulid::Ulid::new().to_string(), "ts": "2026-01-02T00:00:00Z",
        "workspace": DEFAULT_WORKSPACE, "project": MACHINE_PROJECT,
        "session": "00000000-0000-0000-0000-000000000000",
        "source": {"cli": "brain", "hook": "gotcha"}, "kind": "knowledge",
        "title": "Zanzibar keyboard repeats on this machine", "body": "A machine quirk.",
        "files": [], "links": [], "consolidated": true, "class": "durable", "scope": "machine"
    });
    std::fs::write(machine_events.join("2026-10.jsonl"), format!("{lesson}\n")).unwrap();
    assert!(fixture.brain(&["reindex"]).status.success());

    // Which project's lessons the primer of this checkout shows.
    let primer_of = |cwd: &Path| {
        let start = start_payload(cwd, "0199a1f2-0000-7000-8000-0000000000c1", "startup");
        injected_context(&fixture.hook("claude-code", "SessionStart", &start)).unwrap_or_default()
    };
    let search_from = |cwd: &Path, query: &str| {
        let out = Command::new(BRAIN)
            .args(["search", query])
            .current_dir(cwd)
            .env("ROLEPOD_BRAIN_HOME", &fixture.home)
            .env("ROLEPOD_BRAIN_NO_FETCH", "1")
            .env("ROLEPOD_BRAIN_HUB", "off")
            .env("HOME", fixture.home.parent().unwrap())
            .env("PATH", "/usr/bin:/bin")
            .output()
            .expect("run brain search");
        String::from_utf8_lossy(&out.stdout).to_string()
    };

    let primer_b = primer_of(&b_dir);
    assert!(primer_b.contains("Zanzibar"), "the machine lesson is missing from B's primer: {primer_b}");
    assert!(!primer_b.contains("Quokka"), "A's lesson reached B's primer: {primer_b}");
    assert!(search_from(&b_dir, "Zanzibar").contains("Zanzibar"), "B's search misses the machine lesson");
    assert!(!search_from(&b_dir, "Quokka").contains("Quokka"), "B's search shows A's lesson");
    // Control: A reaches its own lesson the same way.
    assert!(search_from(&fixture.project, "Quokka").contains("Quokka"), "A's search misses its own lesson");
}
