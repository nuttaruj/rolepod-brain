//! `brain update` end to end: the real binary copied into a temporary
//! `BRAIN_BIN_DIR`, an isolated HOME and data dir, a `file://` release signed
//! with the fixture test key, and a `curl` stub on PATH that records every
//! request and then runs the real curl. Nothing touches the network.
// The update seams (ROLEPOD_BRAIN_UPDATE_URL, ROLEPOD_BRAIN_UPDATE_PUBKEY) exist only under debug_assertions.
#![cfg(all(unix, debug_assertions))]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

const BRAIN: &str = env!("CARGO_BIN_EXE_brain");
const RUNNING: &str = env!("CARGO_PKG_VERSION");

fn triple() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        other => panic!("no unix release target for {other:?}"),
    }
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/update")
}

struct Env {
    base: PathBuf,
    extra: Vec<(String, String)>,
    key: bool,
}

impl Env {
    fn new() -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let base = PathBuf::from(format!("/tmp/upe-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&base);
        for sub in ["home", "data", "bin", "stub", "rel"] {
            std::fs::create_dir_all(base.join(sub)).unwrap();
        }
        std::fs::set_permissions(base.join("bin"), std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::copy(BRAIN, base.join("bin/brain")).unwrap();
        std::fs::set_permissions(base.join("bin/brain"), std::fs::Permissions::from_mode(0o755)).unwrap();
        let stub = format!(
            "#!/bin/sh\necho \"$@\" >> {log}\n[ -f {sleep} ] && sleep 1\nexec /usr/bin/curl \"$@\"\n",
            log = base.join("stub/log").display(),
            sleep = base.join("stub/sleep").display()
        );
        std::fs::write(base.join("stub/curl"), stub).unwrap();
        std::fs::set_permissions(base.join("stub/curl"), std::fs::Permissions::from_mode(0o755)).unwrap();
        Self { base, extra: Vec::new(), key: true }
    }

    fn with(mut self, k: &str, v: &str) -> Self {
        self.extra.push((k.into(), v.into()));
        self
    }

    fn bin(&self) -> PathBuf {
        self.base.join("bin/brain")
    }

    fn command(&self, exe: &Path, args: &[&str]) -> Command {
        let mut cmd = Command::new(exe);
        cmd.args(args)
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", self.base.join("stub").display()))
            .env("HOME", self.base.join("home"))
            .env("ROLEPOD_BRAIN_HOME", self.base.join("data"))
            .env("BRAIN_BIN_DIR", self.base.join("bin"))
            .env("ROLEPOD_BRAIN_HUB", "off")
            .env("ROLEPOD_BRAIN_NO_FETCH", "1")
            .env("ROLEPOD_BRAIN_UPDATE_URL", self.base.join("rel"))
            // The updater runs tools by absolute path only; this debug seam
            // points it at the stubs instead of /usr/bin.
            .env("ROLEPOD_BRAIN_UPDATE_TOOLS", self.base.join("stub"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if self.key {
            let key = std::fs::read_to_string(fixtures().join("test.pub")).unwrap();
            cmd.env("ROLEPOD_BRAIN_UPDATE_PUBKEY", key.lines().nth(1).unwrap());
        } else {
            // Empty means "no key" whatever the shipped constant holds.
            cmd.env("ROLEPOD_BRAIN_UPDATE_PUBKEY", "");
        }
        for (k, v) in &self.extra {
            cmd.env(k, v);
        }
        cmd
    }

    /// Publish a fake release: `bin_kind` bytes, `sig_kind` signature (None
    /// leaves the `.minisig` out), `ago_hours` old.
    fn release(&self, tag: &str, bin_kind: &str, sig_kind: Option<&str>, ago_hours: i64) {
        let rel = self.base.join("rel");
        let at = jiff::Timestamp::now() - jiff::SignedDuration::from_hours(ago_hours);
        std::fs::write(rel.join("latest.json"), format!(r#"{{"tag_name":"{tag}","published_at":"{at}"}}"#)).unwrap();
        let name = format!("brain-{}", triple());
        std::fs::copy(fixtures().join(format!("bin-{bin_kind}")), rel.join(&name)).unwrap();
        let _ = std::fs::remove_file(rel.join(format!("{name}.minisig")));
        if let Some(kind) = sig_kind {
            std::fs::copy(fixtures().join(format!("{kind}.{}.minisig", triple())), rel.join(format!("{name}.minisig"))).unwrap();
        }
    }

    fn update(&self) -> bool {
        self.command(&self.bin(), &["update"]).status().unwrap().success()
    }

    fn state(&self, key: &str) -> Option<String> {
        let db = self.base.join("data/brain.db");
        let conn = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
        conn.query_row("SELECT value FROM schema_state WHERE key = ?1", [key], |r| r.get(0)).ok()
    }

    fn curl_log(&self) -> Vec<String> {
        std::fs::read_to_string(self.base.join("stub/log")).unwrap_or_default().lines().map(String::from).collect()
    }

    fn binary_requests(&self) -> usize {
        let name = format!("/brain-{}", triple());
        self.curl_log().iter().filter(|l| l.ends_with(&name)).count()
    }

    fn leftovers(&self) -> Vec<String> {
        let mut found = Vec::new();
        for dir in ["bin", "data"] {
            for entry in std::fs::read_dir(self.base.join(dir)).unwrap().flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with(".brain.new-") || name.starts_with("update.meta-") || name.starts_with("update.sig-") {
                    found.push(name);
                }
            }
        }
        found
    }

    fn log_len(&self) -> Option<u64> {
        std::fs::metadata(self.base.join("data/brain.log")).ok().map(|m| m.len())
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn skip_reason(env: &Env) -> String {
    env.state("update_skip").unwrap_or_default().split('@').next().unwrap_or_default().to_string()
}

#[test]
fn signed_newer_release_is_installed() {
    let env = Env::new();
    let original = std::fs::read(env.bin()).unwrap();
    env.release("v9.0.0", "good", Some("good"), 48);
    assert!(env.update());
    assert_eq!(std::fs::read(env.bin()).unwrap(), std::fs::read(fixtures().join("bin-good")).unwrap());
    assert_eq!(std::fs::read(env.base.join("bin/brain.prev")).unwrap(), original, "brain.prev is the old binary");
    assert!(env.state("update_installed").unwrap().starts_with("9.0.0 "));
    assert!(std::fs::read_to_string(env.base.join("data/update.installed")).unwrap().starts_with("9.0.0 "));
    assert_eq!(env.state("update_prev").as_deref(), Some(RUNNING));
    assert!(env.state("update_checked_at").is_some());
    assert!(env.leftovers().is_empty(), "{:?}", env.leftovers());
    // The next hook runs the new one and gets an answer.
    let mut hook = env.command(&env.bin(), &["hook", "--cli", "claude-code", "--event", "Stop"]).spawn().unwrap();
    assert!(hook.wait().unwrap().success());
}

/// Every row: the binary keeps its bytes, the reason is recorded, brain.log
/// is not touched, and only release URLs were asked for.
#[test]
fn rows_that_must_not_install() {
    type Setup = fn(&Env);
    let rows: [(&str, &str, Setup); 9] = [
        ("too-young", "too-young", |e| e.release("v9.0.0", "good", Some("good"), 1)),
        ("not-newer", "not-newer", |e| e.release("v0.0.1", "old", Some("old"), 48)),
        ("unsigned", "unsigned", |e| e.release("v9.0.0", "good", None, 48)),
        ("bad-signature", "bad-signature", |e| e.release("v9.0.0", "good", Some("stfail"), 48)),
        ("signed-target-mismatch", "signed-target-mismatch", |e| e.release("v9.0.0", "good", Some("wrongtarget"), 48)),
        ("signed-version-mismatch", "signed-version-mismatch", |e| e.release("v9.0.0", "good", Some("wrongversion"), 48)),
        ("selftest-version", "selftest-version", |e| e.release("v9.1.0", "badver", Some("badver"), 48)),
        ("published-unknown", "published-unknown", |e| {
            e.release("v9.0.0", "good", Some("good"), 48);
            std::fs::write(e.base.join("rel/latest.json"), r#"{"tag_name":"v9.0.0"}"#).unwrap();
        }),
        ("forged-future-tag", "signed-version-mismatch", |e| e.release("v99.0.0", "good", Some("good"), 48)),
    ];
    for (name, reason, setup) in rows {
        let env = Env::new();
        let original = std::fs::read(env.bin()).unwrap();
        setup(&env);
        let log_before = env.log_len();
        assert!(env.update(), "{name}");
        assert_eq!(std::fs::read(env.bin()).unwrap(), original, "{name}: binary changed");
        assert!(!env.base.join("bin/brain.prev").exists(), "{name}: prev written");
        assert_eq!(skip_reason(&env), reason, "{name}");
        assert!(env.leftovers().is_empty(), "{name}: {:?}", env.leftovers());
        assert_eq!(env.log_len(), log_before, "{name}: brain.log touched");
        let rel = format!("file://{}/", env.base.join("rel").display());
        for line in env.curl_log() {
            assert!(line.split(' ').next_back().unwrap().starts_with(&rel), "{name}: stray request {line}");
        }
    }
    // The mismatching-version row put that version on the bad list.
    let env = Env::new();
    env.release("v9.1.0", "badver", Some("badver"), 48);
    env.update();
    assert!(env.state("update_bad").unwrap().contains("9.1.0"));
}

#[test]
fn no_key_no_binary_request() {
    let mut env = Env::new();
    env.key = false;
    env.release("v9.0.0", "good", Some("good"), 48);
    let original = std::fs::read(env.bin()).unwrap();
    assert!(env.update());
    assert_eq!(env.binary_requests(), 0);
    assert!(env.curl_log().is_empty());
    assert_eq!(skip_reason(&env), "no-key");
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
}

#[test]
fn selftest_failure_not_placed_not_retried() {
    let env = Env::new();
    let original = std::fs::read(env.bin()).unwrap();
    env.release("v9.2.0", "stfail", Some("stfail"), 48);
    assert!(env.update());
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
    assert_eq!(skip_reason(&env), "selftest-failed");
    assert!(env.state("update_bad").unwrap().contains("9.2.0"));
    assert_eq!(env.binary_requests(), 1);
    assert!(env.update());
    assert_eq!(env.binary_requests(), 1, "the bad version is not downloaded again");
    assert_eq!(skip_reason(&env), "marked-bad");
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
}

#[test]
fn opt_out_config_zero_curl_calls() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    std::fs::write(env.base.join("data/config.toml"), "[update]\nauto = false\n").unwrap();
    assert!(env.update());
    assert!(env.curl_log().is_empty());
}

#[test]
fn opt_out_env_zero_curl_calls() {
    let env = Env::new().with("ROLEPOD_BRAIN_NO_UPDATE", "1");
    env.release("v9.0.0", "good", Some("good"), 48);
    assert!(env.update());
    assert!(env.curl_log().is_empty());
}

#[test]
fn two_updates_one_download() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    std::fs::write(env.base.join("stub/sleep"), "").unwrap();
    let mut a = env.command(&env.bin(), &["update"]).spawn().unwrap();
    let mut b = env.command(&env.bin(), &["update"]).spawn().unwrap();
    assert!(a.wait().unwrap().success() && b.wait().unwrap().success());
    assert_eq!(env.binary_requests(), 1);
    assert!(env.state("update_installed").is_some());
}

#[test]
fn oversize_download_aborted_no_partial_left() {
    let env = Env::new().with("ROLEPOD_BRAIN_UPDATE_MAX_BYTES", "10");
    let original = std::fs::read(env.bin()).unwrap();
    env.release("v9.0.0", "good", Some("good"), 48);
    assert!(env.update());
    assert_eq!(skip_reason(&env), "download-failed");
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
    assert!(env.leftovers().is_empty(), "{:?}", env.leftovers());
}

#[test]
fn killed_update_leaves_no_staging_after_next_run() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 1);
    let stale = env.base.join("bin/.brain.new-4000000");
    let file = std::fs::File::create(&stale).unwrap();
    file.set_modified(std::time::SystemTime::now() - Duration::from_secs(7200)).unwrap();
    drop(file);
    assert!(env.update());
    assert!(!stale.exists());
}

#[test]
fn cargo_install_path_not_touched() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    let original = std::fs::read(env.bin()).unwrap();
    // The cargo-built binary is not `brain` in the bootstrap dir.
    assert!(env.command(Path::new(BRAIN), &["update"]).status().unwrap().success());
    assert_eq!(skip_reason(&env), "not-bootstrap-install");
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
    assert!(env.curl_log().is_empty());
}

#[test]
fn hook_loop_during_update_never_fails() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    std::fs::write(env.base.join("stub/sleep"), "").unwrap();
    let mut update = env.command(&env.bin(), &["update"]).spawn().unwrap();
    let mut hooks = 0;
    let started = Instant::now();
    while update.try_wait().unwrap().is_none() && started.elapsed() < Duration::from_secs(60) {
        let mut hook = env
            .command(&env.bin(), &["hook", "--cli", "claude-code", "--event", "Stop"])
            .stdin(Stdio::piped())
            .spawn()
            .expect("the binary path is never empty");
        hook.stdin.take().unwrap().write_all(b"{}").unwrap();
        assert!(hook.wait().unwrap().success());
        hooks += 1;
    }
    assert!(hooks >= 2, "only {hooks} hooks ran");
    assert!(env.state("update_installed").is_some());
}

#[test]
fn mcp_started_before_update_still_answers_search() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    let mut mcp = env.command(&env.bin(), &["mcp"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    let mut stdin = mcp.stdin.take().unwrap();
    let mut out = BufReader::new(mcp.stdout.take().unwrap());
    let mut ask = |line: &str| {
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
        let mut answer = String::new();
        out.read_line(&mut answer).unwrap();
        answer
    };
    assert!(ask(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#).contains("result"));
    assert!(env.update());
    assert!(env.state("update_installed").is_some());
    let search = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"anything"}}}"#;
    let answer = ask(search);
    assert!(answer.contains("\"result\""), "{answer}");
    drop(stdin);
    let _ = mcp.wait();
}

#[test]
fn selftest_does_not_write_store() {
    let env = Env::new();
    // Any run creates the store.
    assert!(env.command(Path::new(BRAIN), &["stats"]).status().unwrap().success());
    let listing = |env: &Env| {
        let mut files: Vec<(String, u64, std::time::SystemTime)> = std::fs::read_dir(env.base.join("data"))
            .unwrap()
            .flatten()
            .map(|e| {
                let m = e.metadata().unwrap();
                (e.file_name().to_string_lossy().into_owned(), m.len(), m.modified().unwrap())
            })
            .collect();
        files.sort();
        files
    };
    let before = listing(&env);
    assert!(before.iter().any(|f| f.0 == "brain.db"));
    assert!(env.command(Path::new(BRAIN), &["self-test"]).status().unwrap().success());
    assert_eq!(listing(&env), before);
}

#[cfg(target_os = "macos")]
#[test]
fn codesign_stub_fails_binary_unchanged() {
    let env = Env::new();
    std::fs::write(env.base.join("stub/codesign"), "#!/bin/sh\nexit 1\n").unwrap();
    std::fs::set_permissions(env.base.join("stub/codesign"), std::fs::Permissions::from_mode(0o755)).unwrap();
    let original = std::fs::read(env.bin()).unwrap();
    env.release("v9.0.0", "good", Some("good"), 48);
    assert!(env.update());
    assert_eq!(skip_reason(&env), "codesign-failed");
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
    assert!(env.leftovers().is_empty());
    assert!(env.state("update_bad").is_none(), "a local fault is not the version's");
}

// ---- daily spawn and rollback (Task 3) -------------------------------------

const PREV_SCRIPT: &str = "#!/bin/sh\n[ \"$1\" = \"--version\" ] && echo \"brain 8.0.0\"\nexit 0\n";
const HOOK: [&str; 5] = ["hook", "--cli", "claude-code", "--event", "Stop"];

fn now_ms() -> i64 {
    jiff::Timestamp::now().as_millisecond()
}

impl Env {
    fn put_state(&self, key: &str, value: &str) {
        let conn = rusqlite::Connection::open(self.base.join("data/brain.db")).unwrap();
        conn.execute("INSERT OR REPLACE INTO schema_state (key, value) VALUES (?1, ?2)", [key, value]).unwrap();
    }

    /// The state of a binary the updater put in place `ago_ms` ago, with a
    /// `brain.prev` that says it is 8.0.0.
    fn fresh(&self, ago_ms: i64) {
        assert!(self.command(&self.bin(), &["stats"]).status().unwrap().success());
        let installed = format!("{RUNNING} {}", now_ms() - ago_ms);
        self.put_state("update_installed", &installed);
        std::fs::write(self.base.join("data/update.installed"), &installed).unwrap();
        self.put_state("update_prev", "8.0.0");
        std::fs::write(self.base.join("data/update.prev"), "8.0.0").unwrap();
        self.write_prev(PREV_SCRIPT);
    }

    fn write_prev(&self, script: &str) {
        let prev = self.base.join("bin/brain.prev");
        std::fs::write(&prev, script).unwrap();
        std::fs::set_permissions(&prev, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn hook_cmd(&self, fault: Option<&str>) -> Command {
        let mut cmd = self.command(&self.bin(), &HOOK);
        if let Some(fault) = fault {
            cmd.env("ROLEPOD_BRAIN_HOOK_FAULT", fault);
        }
        cmd
    }

    fn hook(&self, fault: Option<&str>) -> bool {
        self.hook_cmd(fault).status().unwrap().success()
    }

    fn rolled(&self) -> bool {
        std::fs::read_to_string(self.base.join("bin/brain")).is_ok_and(|s| s == PREV_SCRIPT)
    }

    fn failures(&self) -> Option<String> {
        std::fs::read_to_string(self.base.join("data/update.failures")).ok()
    }
}

#[test]
fn daily_spawns_update_once_and_returns_before_it_ends() {
    let env = Env::new().with("ROLEPOD_BRAIN_UPDATE_SPAWN", "1");
    env.release("v9.0.0", "good", Some("good"), 48);
    std::fs::write(env.base.join("stub/sleep"), "").unwrap();
    let started = Instant::now();
    assert!(env.command(&env.bin(), &["consolidate", "--idle"]).status().unwrap().success());
    let returned = started.elapsed();
    // The stub sleeps a second per request: the update outlasts the run.
    assert!(env.base.join("data/update.running").exists(), "returned only after the update ended ({returned:?})");
    assert!(env.state("update_checked_at").is_some(), "stamped at the spawn");
    let deadline = Instant::now() + Duration::from_secs(60);
    while env.state("update_installed").is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(env.state("update_installed").is_some());
    let deadline = Instant::now() + Duration::from_secs(10);
    while env.base.join("data/update.running").exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!env.base.join("data/update.running").exists(), "the updater removes its marker");
    assert_eq!(env.binary_requests(), 1);
    let requests = env.curl_log().len();
    // Same day: no second spawn.
    assert!(env.command(&env.bin(), &["consolidate", "--idle"]).status().unwrap().success());
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(env.curl_log().len(), requests);
    assert!(!env.base.join("data/update.running").exists());
}

#[test]
fn daily_never_spawns_when_opted_out_or_without_the_seam() {
    // No seam: ROLEPOD_BRAIN_NO_FETCH keeps the daily step off the network.
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    assert!(env.command(&env.bin(), &["consolidate", "--idle"]).status().unwrap().success());
    std::thread::sleep(Duration::from_millis(1000));
    assert!(env.curl_log().is_empty() && !env.base.join("data/update.running").exists());
    // Opted out by config and by env: not even the marker.
    for (cfg, no_update) in [(true, false), (false, true)] {
        let mut env = Env::new().with("ROLEPOD_BRAIN_UPDATE_SPAWN", "1");
        if no_update {
            env = env.with("ROLEPOD_BRAIN_NO_UPDATE", "1");
        }
        env.release("v9.0.0", "good", Some("good"), 48);
        if cfg {
            std::fs::write(env.base.join("data/config.toml"), "[update]\nauto = false\n").unwrap();
        }
        assert!(env.command(&env.bin(), &["consolidate", "--idle"]).status().unwrap().success());
        std::thread::sleep(Duration::from_millis(1000));
        assert!(env.curl_log().is_empty() && !env.base.join("data/update.running").exists());
    }
}

#[test]
fn new_version_store_write_error_triggers_rollback() {
    let env = Env::new();
    env.fresh(0);
    let original = std::fs::read(env.bin()).unwrap();
    for _ in 0..2 {
        assert!(env.hook(Some("1")), "a failing hook still exits 0");
    }
    assert_eq!(std::fs::read(env.bin()).unwrap(), original, "two failures are not enough");
    assert!(env.hook(Some("1")));
    assert!(env.rolled(), "the path is brain.prev");
    assert!(env.state("update_bad").unwrap().split_whitespace().any(|v| v == RUNNING));
    assert!(std::fs::read_to_string(env.base.join("data/update.bad")).unwrap().contains(RUNNING));
    assert!(skip_reason(&env) == "rolled-back");
    assert_eq!(env.state("update_installed").as_deref(), Some("8.0.0 0"), "the window is closed");
    assert!(env.failures().is_none());
    assert!(env.leftovers().is_empty() && !env.base.join("bin/.brain.rollback-1").exists());
    // Hooks now run the restored binary and get an answer.
    assert!(env.hook(None));
}

#[test]
fn store_the_new_build_cannot_open_still_rolls_back() {
    let env = Env::new();
    env.fresh(0);
    let db = env.base.join("data/brain.db");
    std::fs::set_permissions(&db, std::fs::Permissions::from_mode(0o000)).unwrap();
    for _ in 0..3 {
        assert!(env.hook(Some("1")));
    }
    assert!(env.rolled(), "brain.prev is restored without the store");
    let bad = std::fs::read_to_string(env.base.join("data/update.bad")).unwrap();
    assert!(bad.split_whitespace().any(|v| v == RUNNING), "{bad}");
    assert_eq!(std::fs::read_to_string(env.base.join("data/update.installed")).unwrap(), "8.0.0 0");
    assert!(env.failures().is_none());
    std::fs::set_permissions(&db, std::fs::Permissions::from_mode(0o644)).unwrap();
}

#[test]
fn unparsable_config_refuses_a_manual_update() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    std::fs::write(env.base.join("data/config.toml"), "[update\nauto = ").unwrap();
    let original = std::fs::read(env.bin()).unwrap();
    assert!(!env.update(), "refuses with an error, not a silent run");
    assert!(env.curl_log().is_empty());
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
}

#[test]
fn stub_curl_on_path_is_not_run_without_the_seam() {
    let env = Env::new().with("ROLEPOD_BRAIN_UPDATE_TOOLS", "/nonexistent");
    env.release("v9.0.0", "good", Some("good"), 48);
    assert!(env.update());
    assert!(env.curl_log().is_empty(), "the stub first on PATH was not used");
    assert_eq!(skip_reason(&env), "", "the real /usr/bin/curl fetched the release");
    assert_ne!(std::fs::read(env.bin()).unwrap(), std::fs::read(BRAIN).unwrap());
}

#[test]
fn parallel_hooks_one_failure_no_rollback() {
    let env = Env::new();
    env.fresh(0);
    for fault in [Some("1"), Some("1"), None, Some("1"), Some("1")] {
        assert!(env.hook(fault));
    }
    assert!(!env.rolled(), "a success between failures resets the count");
    assert!(env.failures().unwrap().starts_with(&format!("{RUNNING} 2\n")));
}

#[test]
fn three_hooks_in_flight_are_not_failures() {
    let env = Env::new();
    env.fresh(0);
    let mut held: Vec<_> = (0..3).map(|_| env.hook_cmd(None).stdin(Stdio::piped()).spawn().unwrap()).collect();
    let deadline = Instant::now() + Duration::from_secs(30);
    while env.failures().is_none_or(|f| f.matches("pid ").count() < 3) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(env.failures().unwrap().matches("pid ").count() >= 3, "three hooks are waiting on stdin");
    assert!(env.hook(None));
    for child in &mut held {
        drop(child.stdin.take());
        assert!(child.wait().unwrap().success());
    }
    assert!(!env.rolled());
    assert_eq!(env.failures().unwrap(), format!("{RUNNING} 0\n"));
}

#[test]
fn dead_pid_entries_count_as_failures() {
    let env = Env::new();
    env.fresh(0);
    std::fs::write(env.base.join("data/update.failures"), format!("{RUNNING} 0\npid 4000001\npid 4000002\npid 4000003\n")).unwrap();
    assert!(env.hook(None));
    assert!(env.rolled());
}

#[test]
fn aborting_hook_three_times_rolls_back() {
    let env = Env::new();
    env.fresh(0);
    for _ in 0..3 {
        assert!(!env.hook(Some("abort")), "the abort is a real abort");
    }
    assert!(!env.rolled(), "nothing has counted the third yet");
    assert!(env.hook(None));
    assert!(env.rolled());
}

#[test]
fn install_older_than_an_hour_is_never_rolled_back() {
    let env = Env::new();
    env.fresh(2 * 3600 * 1000);
    for _ in 0..4 {
        assert!(env.hook(Some("1")));
    }
    assert!(!env.rolled());
    assert!(env.failures().is_none());
}

#[test]
fn hook_outside_the_first_hour_touches_no_update_file() {
    let env = Env::new();
    env.fresh(0);
    // The executable itself is old: the hook stops after one stat of it.
    let old = std::time::SystemTime::now() - Duration::from_secs(2 * 3600);
    // Read-only open: a write-open of the exe makes macOS SIGKILL its next launch.
    std::fs::File::open(env.bin()).unwrap().set_modified(old).unwrap();
    let seeded = format!("{RUNNING} 2\npid 4000001\n");
    std::fs::write(env.base.join("data/update.failures"), &seeded).unwrap();
    for fault in [None, Some("1"), Some("1"), Some("1")] {
        let out = env.hook_cmd(fault).stderr(Stdio::piped()).output().unwrap();
        assert!(out.status.success(), "{:?} {}", out.status, String::from_utf8_lossy(&out.stderr));
    }
    assert!(!env.rolled());
    assert_eq!(env.failures().unwrap(), seeded);
    assert!(!env.base.join("data/update.lock").exists(), "no lock was taken or created");
    let mut names: Vec<String> = std::fs::read_dir(env.base.join("data"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("update."))
        .collect();
    names.sort();
    assert_eq!(names, ["update.failures", "update.installed", "update.prev"], "only what the test put there");
}

#[test]
fn rollback_without_prev_reports_and_stops() {
    for (name, script) in [("missing", None), ("other-version", Some("#!/bin/sh\necho \"brain 7.7.7\"\n"))] {
        let env = Env::new();
        env.fresh(0);
        match script {
            None => std::fs::remove_file(env.base.join("bin/brain.prev")).unwrap(),
            Some(script) => env.write_prev(script),
        }
        let original = std::fs::read(env.bin()).unwrap();
        for _ in 0..3 {
            assert!(env.hook(Some("1")), "{name}");
        }
        assert_eq!(std::fs::read(env.bin()).unwrap(), original, "{name}: nothing invented");
        assert_eq!(skip_reason(&env), "rollback-unavailable", "{name}");
        assert!(env.failures().is_none(), "{name}: the count is cleared");
        assert!(env.state("update_bad").is_none(), "{name}: not marked bad when it cannot be undone");
    }
}

#[test]
fn rollback_is_idempotent_under_concurrent_hooks() {
    let env = Env::new();
    env.fresh(0);
    std::fs::write(env.base.join("data/update.failures"), format!("{RUNNING} 2\n")).unwrap();
    let mut kids: Vec<_> = (0..6).map(|_| env.hook_cmd(Some("1")).spawn().unwrap()).collect();
    for kid in &mut kids {
        assert!(kid.wait().unwrap().success());
    }
    assert!(env.rolled());
    let bad = env.state("update_bad").unwrap();
    assert_eq!(bad.split_whitespace().filter(|v| *v == RUNNING).count(), 1, "{bad}");
    assert_eq!(env.state("update_installed").as_deref(), Some("8.0.0 0"));
}

#[test]
fn rolled_back_version_not_downloaded_again() {
    let env = Env::new();
    env.fresh(0);
    env.put_state("update_bad", "9.0.0");
    for _ in 0..3 {
        assert!(env.hook(Some("1")));
    }
    assert!(env.rolled());
    let bad = env.state("update_bad").unwrap();
    assert!(bad.split_whitespace().any(|v| v == "9.0.0") && bad.split_whitespace().any(|v| v == RUNNING), "{bad}");
    // The list the rollback wrote is the one the updater reads: 9.0.0 stays refused.
    std::fs::copy(BRAIN, env.bin()).unwrap();
    env.release("v9.0.0", "good", Some("good"), 48);
    assert!(env.update());
    assert_eq!(env.binary_requests(), 0);
    assert_eq!(skip_reason(&env), "marked-bad");
}

impl Env {
    /// `brain doctor` on this environment: the `update` row, and whether the
    /// whole run made a request through the curl stub (it must not).
    fn doctor_update_row(&self) -> String {
        assert!(self.command(&self.bin(), &["stats"]).status().unwrap().success());
        let out = self.command(&self.bin(), &["doctor"]).stdout(Stdio::piped()).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let row = text.lines().find(|l| l.split_whitespace().nth(1) == Some("update")).unwrap_or_default().to_string();
        assert!(row.starts_with("ok "), "the update row is never a FAIL: {row:?}");
        assert!(self.curl_log().is_empty(), "doctor made a request: {:?}", self.curl_log());
        row
    }
}

fn assert_no_command_text(row: &str) {
    for bad in ["run ", "`brain", "brain ", "http", "://", "sudo", "curl"] {
        assert!(!row.contains(bad), "{bad:?} in {row}");
    }
}

#[test]
fn doctor_update_row_has_no_command_text() {
    let ago = |ms: i64| now_ms() - ms;
    let rows: Vec<(&str, String)> = {
        let mut rows = Vec::new();
        let env = Env::new();
        rows.push(("never checked", env.doctor_update_row()));
        assert!(rows[0].1.contains("never checked") && rows[0].1.contains("auto"), "{}", rows[0].1);

        env.put_state("update_installed", &format!("9.0.0 {}", ago(2 * 3600 * 1000)));
        env.put_state("update_prev", RUNNING);
        env.put_state("update_checked_at", &(ago(0) / 1000).to_string());
        rows.push(("installed", env.doctor_update_row()));
        assert!(rows[1].1.contains("installed 9.0.0 2h ago") && rows[1].1.contains(&format!("was {RUNNING}")), "{}", rows[1].1);

        env.put_state("update_skip", &format!("too-young@{}", ago(0) / 1000));
        rows.push(("skip", env.doctor_update_row()));
        assert!(rows[2].1.contains("last skip: too-young") && !rows[2].1.contains("warn:"), "{}", rows[2].1);

        env.put_state("update_installed", "8.0.0 0");
        env.put_state("update_skip", &format!("rolled-back@{}", ago(0) / 1000));
        env.put_state("update_bad", "9.0.0");
        rows.push(("rolled back", env.doctor_update_row()));
        let r = &rows[3].1;
        assert!(r.contains("warn:") && r.contains("after a rollback") && r.contains("marked bad: 9.0.0"), "{r}");

        let env = Env::new().with("ROLEPOD_BRAIN_NO_UPDATE", "1");
        rows.push(("opted out", env.doctor_update_row()));
        assert!(rows[4].1.contains("off (ROLEPOD_BRAIN_NO_UPDATE is set)"), "{}", rows[4].1);

        let mut env = Env::new();
        env.key = false;
        rows.push(("no key", env.doctor_update_row()));
        assert!(rows[5].1.contains("waiting for a signed release"), "{}", rows[5].1);

        let env = Env::new().with("BRAIN_BIN_DIR", "/nonexistent-bin-dir");
        rows.push(("elsewhere", env.doctor_update_row()));
        assert!(rows[6].1.contains("installed another way"), "{}", rows[6].1);
        rows
    };
    for (name, row) in &rows {
        assert_no_command_text(row);
        eprintln!("{name}: {row}");
    }
}

#[test]
fn doctor_counts_sessions_still_on_the_old_build() {
    let env = Env::new();
    assert!(env.command(&env.bin(), &["stats"]).status().unwrap().success());
    let mut mcp = env.command(&env.bin(), &["mcp"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
    let stdin = mcp.stdin.take().unwrap();
    std::thread::sleep(Duration::from_millis(2300));
    env.put_state("update_installed", &format!("9.0.0 {}", now_ms() - 500));
    env.put_state("update_prev", "8.0.0");
    let row = env.doctor_update_row();
    drop(stdin);
    let _ = mcp.wait();
    assert!(row.contains("session(s) still on 8.0.0 until they close"), "{row}");
    assert_no_command_text(&row);
}

// ---- the one line `brain update` prints ------------------------------------

/// What `brain update` prints, and its exit status.
fn said(env: &Env) -> (String, bool) {
    let out = env.command(&env.bin(), &["update"]).stdout(Stdio::piped()).output().unwrap();
    (String::from_utf8(out.stdout).unwrap(), out.status.success())
}

/// One line, ends in a newline, and shows no URL, token or data-dir path.
fn assert_plain_line(env: &Env, name: &str, said: &str) {
    assert_eq!(said.matches('\n').count(), 1, "{name}: {said:?}");
    assert!(said.ends_with('\n'), "{name}: {said:?}");
    for bad in ["://", "token", env.base.to_str().unwrap()] {
        assert!(!said.contains(bad), "{name}: {said:?} shows {bad}");
    }
}

#[test]
fn update_says_what_it_installed() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    let (line, ok) = said(&env);
    assert!(ok);
    assert_eq!(line, format!("installed 9.0.0 (was {RUNNING})\n"));
    assert_plain_line(&env, "installed", &line);
}

#[test]
fn update_says_why_it_did_not() {
    type Setup = fn(&Env);
    let rows: [(&str, &str, Setup); 6] = [
        ("not-newer", "already on ", |e| e.release("v0.0.1", "old", Some("old"), 48)),
        ("too-young", "not updated: the latest release was published 3 h ago; it installs once it is a day old\n", |e| {
            e.release("v9.0.0", "good", Some("good"), 3)
        }),
        ("unsigned", "not updated: the latest release has no signature\n", |e| e.release("v9.0.0", "good", None, 48)),
        ("bad-signature", "not updated: the signature does not match the download\n", |e| {
            e.release("v9.0.0", "good", Some("stfail"), 48)
        }),
        ("selftest-version", "not updated: the new binary reports a different version", |e| {
            e.release("v9.1.0", "badver", Some("badver"), 48)
        }),
        ("published-unknown", "not updated: the latest release has no readable publish time", |e| {
            e.release("v9.0.0", "good", Some("good"), 48);
            std::fs::write(e.base.join("rel/latest.json"), r#"{"tag_name":"v9.0.0"}"#).unwrap();
        }),
    ];
    for (name, expect, setup) in rows {
        let env = Env::new();
        let log_before = env.log_len();
        setup(&env);
        let (line, ok) = said(&env);
        assert!(ok, "{name}: exit code changed");
        assert!(line.starts_with(expect), "{name}: {line:?}");
        assert_plain_line(&env, name, &line);
        assert_eq!(env.log_len(), log_before, "{name}: brain.log touched");
    }
    let env = Env::new();
    env.release("v0.0.1", "old", Some("old"), 48);
    assert_eq!(said(&env).0, format!("already on {RUNNING}, the latest release\n"));
}

#[test]
fn update_with_no_key_says_so() {
    let mut env = Env::new();
    env.key = false;
    env.release("v9.0.0", "good", Some("good"), 48);
    let (line, ok) = said(&env);
    assert!(ok);
    assert_eq!(line, "not updated: this build has no signing key to check releases with\n");
}

#[test]
fn update_says_when_it_is_off() {
    let env = Env::new();
    std::fs::write(env.base.join("data/config.toml"), "[update]\nauto = false\n").unwrap();
    let (line, ok) = said(&env);
    assert!(ok);
    assert_eq!(line, "updates are off: auto = false in config\n");
    assert_plain_line(&env, "config", &line);

    let env = Env::new().with("ROLEPOD_BRAIN_NO_UPDATE", "1");
    let (line, ok) = said(&env);
    assert!(ok);
    assert_eq!(line, "updates are off: ROLEPOD_BRAIN_NO_UPDATE is set\n");
    assert!(env.curl_log().is_empty());
}

#[test]
fn update_says_when_another_is_running() {
    let env = Env::new();
    env.release("v9.0.0", "good", Some("good"), 48);
    // The lock is a flock on a file in the data dir; hold it from here.
    let held = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(env.base.join("data/update.lock")).unwrap();
    held.lock().unwrap();
    let original = std::fs::read(env.bin()).unwrap();
    let (line, ok) = said(&env);
    assert!(ok);
    assert_eq!(line, "another update is running\n");
    assert_eq!(std::fs::read(env.bin()).unwrap(), original);
    assert!(env.curl_log().is_empty());
}

#[test]
fn help_lists_update_but_not_self_test() {
    let env = Env::new();
    let out = env.command(&env.bin(), &["--help"]).stdout(Stdio::piped()).output().unwrap();
    let help = String::from_utf8(out.stdout).unwrap();
    assert!(help.lines().any(|l| l.trim_start().starts_with("update ")), "{help}");
    assert!(!help.contains("self-test"), "{help}");
}
