//! The hub server, end to end: the real `brain hub serve` binary, an isolated
//! HOME and data dir, a socket dir under a test-owned TMPDIR, and the stub
//! reranker (compiled into debug builds only). Unix only.
// The hub's test seams (ROLEPOD_BRAIN_HUB_STUB and friends) exist only under debug_assertions; ci.yml's debug `cargo test` runs this file.
#![cfg(all(unix, debug_assertions))]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const BRAIN: &str = env!("CARGO_BIN_EXE_brain");

struct Env {
    base: PathBuf,
    exe: PathBuf,
    clocks: Vec<(String, String)>,
}

impl Env {
    /// Short paths on purpose: the socket path must stay under `sun_path`.
    fn new() -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let base = PathBuf::from(format!("/tmp/hbe-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&base);
        for sub in ["home", "data", "t"] {
            std::fs::create_dir_all(base.join(sub)).unwrap();
        }
        Self { base, exe: PathBuf::from(BRAIN), clocks: vec![("ROLEPOD_BRAIN_HUB_STUB".into(), "1".into())] }
    }

    fn with(mut self, key: &str, value: &str) -> Self {
        self.clocks.push((format!("ROLEPOD_BRAIN_HUB_{key}"), value.into()));
        self
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new(&self.exe);
        cmd.args(args)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", self.base.join("home"))
            .env("ROLEPOD_BRAIN_HOME", self.base.join("data"))
            .env("TMPDIR", self.base.join("t"))
            .env("XDG_RUNTIME_DIR", self.base.join("t"))
            .env("ROLEPOD_BRAIN_NO_FETCH", "1")
            .env("ROLEPOD_BRAIN_MAINT_WAIT_SECS", "0")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (key, value) in &self.clocks {
            cmd.env(key, value);
        }
        cmd
    }

    fn serve(&self) -> Child {
        self.command(&["hub", "serve"]).spawn().unwrap()
    }

    /// The private dir, once it exists.
    fn sock_dir(&self) -> Option<PathBuf> {
        std::fs::read_dir(self.base.join("t"))
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("rolepod-brain-")))
    }

    fn socket(&self) -> Option<PathBuf> {
        std::fs::read_dir(self.sock_dir()?)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "sock"))
    }

    fn up(&self) -> Child {
        let child = self.serve();
        wait(|| self.socket().is_some_and(|s| UnixStream::connect(s).is_ok()), "hub socket");
        child
    }

    fn raw(&self) -> Raw {
        Raw::connect(&self.socket().expect("socket"), json!({"type":"hello","proto":1,"build":"0.1.0"}))
    }

    /// Kill the hub this env's data dir names, if it is one. A client may have
    /// started it, so no `Child` holds it.
    fn kill_hub(&self) {
        let Some(pid) = std::fs::read_to_string(self.base.join("data/hub.lock")).ok().and_then(|t| t.trim().parse::<u32>().ok()) else {
            return;
        };
        let ps = Command::new("ps").args(["-o", "command=", "-p", &pid.to_string()]).output();
        if ps.is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("hub serve")) {
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
        }
    }

    fn hub_pid(&self) -> u32 {
        std::fs::read_to_string(self.base.join("data/hub.lock")).unwrap().trim().parse().unwrap()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        self.kill_hub();
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn wait(mut done: impl FnMut() -> bool, what: &str) {
    let until = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < until, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Kill and collect a hub, so none is left behind as a zombie.
fn reap(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn exits_within(child: &mut Child, secs: u64) -> Option<std::process::ExitStatus> {
    let until = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < until {
        if let Some(status) = child.try_wait().unwrap() {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

struct Raw {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    welcome: Option<Value>,
}

impl Raw {
    fn connect(socket: &Path, hello: Value) -> Self {
        let stream = UnixStream::connect(socket).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
        let mut raw = Self { reader: BufReader::new(stream.try_clone().unwrap()), writer: stream, welcome: None };
        raw.send(&hello);
        raw.welcome = raw.recv();
        raw
    }

    fn send(&mut self, message: &Value) {
        let _ = self.writer.write_all(format!("{message}\n").as_bytes());
    }

    /// The next reply, or `None` at end of stream or timeout.
    fn recv(&mut self) -> Option<Value> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(n) if n > 0 => serde_json::from_str(&line).ok(),
            _ => None,
        }
    }

    fn rerank(&mut self, id: u64, query: &str, deadline_ms: u64) {
        self.send(&json!({"type":"rerank","id":id,"query":query,"entries":["a","b","c"],"deadline_ms":deadline_ms}));
    }

    fn status(&mut self) -> Value {
        self.send(&json!({"type":"status"}));
        self.recv().expect("status reply")
    }
}

#[test]
fn two_serves_at_once_leave_one() {
    let env = Env::new();
    let mut first = env.up();
    let mut second = env.serve();
    let status = exits_within(&mut second, 5).expect("the loser exits at once");
    assert!(status.success(), "a lost lock is exit 0, not a crash");
    assert!(first.try_wait().unwrap().is_none(), "the holder is still serving");
    let mut raw = env.raw();
    assert_eq!(raw.welcome.as_ref().unwrap()["pid"], first.id());
    assert_eq!(env.hub_pid(), first.id());
    assert_eq!(raw.status()["pid"], first.id());
    reap(first);
}

#[test]
fn idle_exit_after_grace() {
    let env = Env::new().with("IDLE_MS", "600");
    let mut hub = env.up();
    let status = exits_within(&mut hub, 8).expect("an idle hub leaves");
    assert!(status.success());
    assert!(env.socket().is_none(), "the socket is unlinked on the way out");
}

#[test]
fn rerank_answers_in_stub_order_and_status_shows_the_model() {
    let env = Env::new();
    let hub = env.up();
    let mut raw = env.raw();
    assert_eq!(raw.status()["model_loaded"], false);
    raw.rerank(9, "q", 5000);
    let reply = raw.recv().unwrap();
    assert_eq!(reply, json!({"type":"order","id":9,"indices":[2,1,0]}));
    let status = raw.status();
    assert_eq!(status["model_loaded"], true);
    assert_eq!(status["clients"], 1);
    assert!(status["footprint_kb"].as_u64().unwrap_or(1) > 0);
    reap(hub);
}

#[test]
fn exe_swap_hub_retires_after_inflight() {
    let mut env = Env::new().with("EXE_CHECK_MS", "100").with("STUB_MS", "1500");
    env.exe = env.base.join("brain-copy");
    std::fs::copy(BRAIN, &env.exe).unwrap();
    let mut hub = env.up();
    let mut raw = env.raw();
    raw.rerank(1, "q", 10_000);
    std::thread::sleep(Duration::from_millis(300));
    // Read-only: Linux refuses a write open of a running executable (ETXTBSY); futimens needs ownership only.
    let file = std::fs::File::open(&env.exe).unwrap();
    file.set_modified(std::time::SystemTime::now() + Duration::from_secs(60)).unwrap();
    drop(file);
    assert_eq!(raw.recv().unwrap()["type"], "order", "the work already accepted is finished");
    let status = exits_within(&mut hub, 5).expect("then the hub exits");
    assert!(status.success());
}

#[test]
fn exe_swap_gives_the_endpoint_back_before_exiting() {
    let mut env = Env::new().with("EXE_CHECK_MS", "100").with("STUB_MS", "2000");
    env.exe = env.base.join("brain-copy");
    std::fs::copy(BRAIN, &env.exe).unwrap();
    let mut old = env.up();
    let mut busy = env.raw();
    busy.rerank(1, "q", 10_000);
    std::thread::sleep(Duration::from_millis(200));
    std::fs::File::open(&env.exe)
        .unwrap()
        .set_modified(std::time::SystemTime::now() + Duration::from_secs(60))
        .unwrap();
    wait(|| env.socket().is_none(), "the old hub to unlink its socket");
    // A new hub can take over while the old one drains.
    env.exe = PathBuf::from(BRAIN);
    let mut fresh = env.up();
    assert_ne!(fresh.id(), old.id());
    assert_eq!(busy.recv().unwrap()["type"], "order");
    assert!(exits_within(&mut old, 5).is_some());
    assert!(fresh.try_wait().unwrap().is_none());
    reap(fresh);
}

#[test]
fn server_refuses_bind_in_loose_dir() {
    let env = Env::new();
    let dir = env.base.join("t").join(format!("rolepod-brain-{}", std::fs::metadata(env.base.join("data")).unwrap().uid()));
    std::fs::create_dir(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    let status = env.command(&["hub", "serve"]).status().unwrap();
    assert!(!status.success(), "a loose dir is refused with a non-zero exit");
    assert!(env.socket().is_none(), "nothing was bound");
}

#[test]
fn server_refuses_bind_in_symlink_dir() {
    let env = Env::new();
    let real = env.base.join("real");
    std::fs::create_dir(&real).unwrap();
    std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).unwrap();
    let uid = std::fs::metadata(env.base.join("data")).unwrap().uid();
    std::os::unix::fs::symlink(&real, env.base.join("t").join(format!("rolepod-brain-{uid}"))).unwrap();
    let status = env.command(&["hub", "serve"]).status().unwrap();
    assert!(!status.success());
    assert!(std::fs::read_dir(&real).unwrap().next().is_none(), "nothing was bound through the link");
}

#[test]
fn status_and_stop_through_the_cli_and_a_raw_socket() {
    let env = Env::new();
    let mut hub = env.up();
    let out = env.command(&["hub", "status"]).stdout(Stdio::piped()).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success() && text.contains("hub: running") && text.contains(&format!("pid        {}", hub.id())), "{text}");
    let mut raw = env.raw();
    raw.send(&json!({"type":"stop"}));
    assert_eq!(raw.recv().unwrap()["type"], "ack");
    assert!(exits_within(&mut hub, 5).expect("stop ends the hub").success());
    let after = env.command(&["hub", "status"]).stdout(Stdio::piped()).output().unwrap();
    assert!(String::from_utf8_lossy(&after.stdout).contains("not running"));
}

#[test]
fn stop_drains_inflight_then_exits() {
    let env = Env::new().with("STUB_MS", "1200");
    let mut hub = env.up();
    let mut worker = env.raw();
    worker.rerank(1, "q", 10_000);
    std::thread::sleep(Duration::from_millis(200));
    let mut stopper = env.raw();
    stopper.send(&json!({"type":"stop"}));
    assert_eq!(stopper.recv().unwrap()["type"], "ack");
    assert_eq!(worker.recv().unwrap()["type"], "order", "the in-flight request is answered");
    assert!(exits_within(&mut hub, 5).expect("then it exits").success());
}

#[test]
fn oversize_line_closes_only_that_conn() {
    let env = Env::new();
    let hub = env.up();
    let mut bad = env.raw();
    bad.writer.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
    let _ = bad.writer.write_all(&vec![b'x'; (1 << 20) + 1]);
    assert!(bad.recv().is_none(), "the connection is closed, nothing answered");
    let mut good = env.raw();
    assert_eq!(good.status()["type"], "status");
    reap(hub);
}

#[test]
fn stalled_reader_does_not_block_others() {
    let env = Env::new();
    let hub = env.up();
    let mut stalled = env.raw();
    stalled.rerank(1, "q", 5000);
    // Never reads.
    let started = Instant::now();
    let mut other = env.raw();
    other.rerank(2, "q", 5000);
    assert_eq!(other.recv().unwrap()["type"], "order");
    assert!(started.elapsed() < Duration::from_secs(3));
    reap(hub);
}

#[test]
fn silent_connection_dropped_after_handshake_timeout() {
    let env = Env::new().with("HANDSHAKE_MS", "300");
    let hub = env.up();
    let stream = UnixStream::connect(env.socket().unwrap()).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let started = Instant::now();
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    assert_eq!(reader.read_line(&mut line).unwrap_or(0), 0, "closed with nothing said");
    assert!(started.elapsed() < Duration::from_secs(3));
    reap(hub);
}

#[test]
fn new_hub_old_client_refused_clean() {
    let env = Env::new();
    let mut hub = env.up();
    let mut old = Raw::connect(&env.socket().unwrap(), json!({"type":"hello","proto":0,"build":"0.1.0"}));
    assert_eq!(old.welcome.as_ref().unwrap()["type"], "refused");
    assert!(old.recv().is_none(), "then the connection closes");
    let mut newer = Raw::connect(&env.socket().unwrap(), json!({"type":"hello","proto":2,"build":"0.1.0"}));
    assert_eq!(newer.welcome.as_ref().unwrap()["type"], "refused");
    newer.send(&json!({"type":"retire"}));
    assert!(newer.recv().is_none(), "an older build's Retire is not obeyed");
    assert!(hub.try_wait().unwrap().is_none());
    reap(hub);
}

#[test]
fn retire_from_a_strictly_newer_build_is_obeyed() {
    let env = Env::new();
    let mut hub = env.up();
    let mut newer = Raw::connect(&env.socket().unwrap(), json!({"type":"hello","proto":2,"build":"99.0.0"}));
    assert_eq!(newer.welcome.as_ref().unwrap()["type"], "refused");
    newer.send(&json!({"type":"retire"}));
    assert_eq!(newer.recv().unwrap()["type"], "ack");
    assert!(exits_within(&mut hub, 5).expect("the old hub retires").success());
}

#[test]
fn empty_and_huge_entries_do_not_crash_hub() {
    let env = Env::new();
    let mut hub = env.up();
    let mut raw = env.raw();
    let send = |raw: &mut Raw, id: u64, entries: Value, query: &str| {
        raw.send(&json!({"type":"rerank","id":id,"query":query,"entries":entries,"deadline_ms":2000}));
        raw.recv().unwrap()
    };
    assert_eq!(send(&mut raw, 1, json!([]), "q")["type"], "unavailable");
    assert_eq!(send(&mut raw, 2, json!(["only"]), "q")["type"], "unavailable");
    assert_eq!(send(&mut raw, 3, json!(vec!["x"; 100]), "q")["type"], "unavailable");
    assert_eq!(send(&mut raw, 4, json!(["a".repeat(2000), "b"]), "q")["type"], "unavailable");
    assert_eq!(send(&mut raw, 5, json!(["a", "b"]), &"q".repeat(5000))["type"], "unavailable");
    assert_eq!(send(&mut raw, 6, json!(["a", "b"]), "ERROR")["type"], "unavailable");
    assert_eq!(raw.status()["type"], "status");
    assert!(hub.try_wait().unwrap().is_none());
    reap(hub);
}

#[test]
fn stale_socket_after_kill9_is_replaced() {
    let env = Env::new();
    let mut first = env.up();
    first.kill().unwrap();
    first.wait().unwrap();
    assert!(env.socket().is_some(), "kill -9 leaves the socket behind");
    let second = env.up();
    assert_ne!(second.id(), first.id());
    let mut raw = env.raw();
    assert_eq!(raw.welcome.as_ref().unwrap()["pid"], second.id());
    assert_eq!(raw.status()["type"], "status");
    reap(second);
}

#[test]
fn poison_query_crashes_twice_then_refused() {
    let env = Env::new();
    for round in 0..2 {
        let mut hub = env.up();
        let mut raw = env.raw();
        raw.rerank(round, "POISON", 5000);
        assert!(raw.recv().is_none(), "the stub aborts the process");
        hub.wait().unwrap();
    }
    let hub = env.up();
    let mut raw = env.raw();
    raw.rerank(7, "POISON", 5000);
    assert_eq!(raw.recv().unwrap(), json!({"type":"unavailable","id":7}), "refused without being run");
    raw.rerank(8, "fine", 5000);
    assert_eq!(raw.recv().unwrap()["type"], "order");
    reap(hub);
}

#[test]
fn rss_over_limit_releases_model() {
    let env = Env::new().with("RSS_LIMIT_KB", "1").with("RSS_BUSY_MS", "100");
    let mut hub = env.up();
    let mut raw = env.raw();
    raw.rerank(1, "q", 5000);
    assert_eq!(raw.recv().unwrap()["type"], "order");
    let until = Instant::now() + Duration::from_secs(8);
    loop {
        if raw.status()["model_loaded"] == false {
            break;
        }
        assert!(Instant::now() < until, "the model was never released");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(hub.try_wait().unwrap().is_none(), "releasing the model does not drop the hub");
    reap(hub);
}

#[test]
fn status_answers_while_rerank_busy() {
    let env = Env::new().with("STUB_MS", "1500");
    let hub = env.up();
    let mut worker = env.raw();
    worker.rerank(1, "q", 10_000);
    std::thread::sleep(Duration::from_millis(200));
    let mut other = env.raw();
    let started = Instant::now();
    assert_eq!(other.status()["type"], "status");
    assert!(started.elapsed() < Duration::from_millis(900), "status does not wait for the model");
    reap(hub);
}

#[test]
fn a_request_past_its_deadline_gets_busy() {
    let env = Env::new().with("STUB_MS", "1500");
    let hub = env.up();
    let mut first = env.raw();
    first.rerank(1, "q", 10_000);
    std::thread::sleep(Duration::from_millis(200));
    let mut second = env.raw();
    let asked = Instant::now();
    second.rerank(2, "q", 300);
    assert_eq!(second.recv().unwrap(), json!({"type":"busy","id":2}));
    assert!(asked.elapsed() < Duration::from_millis(1200), "Busy comes at the deadline, not after the running job");
    assert_eq!(first.recv().unwrap()["type"], "order");
    reap(hub);
}


// ---------------------------------------------------------------------------
// Task 4: the client, through the real `brain mcp`.

use std::io::Read;
use std::os::unix::net::UnixListener;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

impl Env {
    fn project(&self) -> PathBuf {
        self.base.join("proj")
    }

    /// A git checkout with four captured edits, so `auth` finds several hits.
    fn seeded(self) -> Self {
        let proj = self.project();
        std::fs::create_dir_all(&proj).unwrap();
        assert!(Command::new("git").args(["init", "-q"]).current_dir(&proj).env("HOME", self.base.join("home")).status().unwrap().success());
        for name in ["auth.rs", "auth/login.rs", "auth/token.rs", "auth/session.rs"] {
            let payload = json!({
                "session_id": "0199a1f2-3c4d-7e8f-9012-3456789abcde",
                "cwd": proj,
                "tool_name": "Edit",
                "tool_input": {"file_path": proj.join(format!("src/{name}")), "new_string": "fn check() {}"},
                "tool_response": {"success": true}
            });
            let mut child = self
                .command(&["hook", "--cli", "claude-code", "--event", "PostToolUse"])
                .current_dir(&proj)
                .stdin(Stdio::piped())
                .spawn()
                .unwrap();
            child.stdin.take().unwrap().write_all(payload.to_string().as_bytes()).unwrap();
            assert!(child.wait().unwrap().success());
        }
        std::fs::write(self.base.join("data/config.toml"), "[search]\nrerank = true\n").unwrap();
        self
    }

    fn session(&self) -> Session {
        self.session_with(&[], &[], None)
    }

    fn session_with(&self, extra: &[(&str, &str)], unset: &[&str], path: Option<&Path>) -> Session {
        let mut cmd = self.command(&["mcp"]);
        cmd.current_dir(self.project()).stdin(Stdio::piped()).stdout(Stdio::piped());
        for (key, value) in extra {
            cmd.env(key, value);
        }
        for key in unset {
            cmd.env_remove(key);
        }
        if let Some(dir) = path {
            cmd.env("PATH", format!("{}:/usr/bin:/bin", dir.display()));
        }
        let mut child = cmd.spawn().unwrap();
        let reader = BufReader::new(child.stdout.take().unwrap());
        let stdin = child.stdin.take().unwrap();
        Session { child, stdin, reader, next: 1 }
    }

    /// Run the real binary as the hub's creator, then take its place: the
    /// private dir and socket path exist, the hub is dead, the lock is free.
    fn layout(&self) -> PathBuf {
        let mut hub = self.up();
        let socket = self.socket().unwrap();
        reap_hard(&mut hub);
        socket
    }

    /// Everything a client may have left in the data dir.
    fn hub_files(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.base.join("data"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("hub."))
            .collect();
        names.sort();
        names
    }
}

fn reap_hard(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

struct Session {
    child: Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    next: u64,
}

/// What one search came back with.
struct Found {
    ids: Vec<String>,
    rerank: Option<Value>,
    is_error: bool,
    took: Duration,
}

impl Found {
    fn reason(&self) -> String {
        self.rerank.as_ref().map(|r| r["reason"].as_str().unwrap_or("?").to_owned()).unwrap_or_default()
    }

    fn engine(&self) -> String {
        self.rerank.as_ref().map(|r| r["engine"].as_str().unwrap_or("?").to_owned()).unwrap_or_default()
    }
}

impl Session {
    fn search(&mut self, rerank: bool) -> Found {
        let id = self.next;
        self.next += 1;
        let request = json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":"auth","rerank":rerank}}});
        let started = Instant::now();
        writeln!(self.stdin, "{request}").unwrap();
        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();
        let took = started.elapsed();
        let response: Value = serde_json::from_str(&line).expect("a JSON-RPC response");
        let is_error = response["result"]["isError"].as_bool().unwrap_or(false);
        let text = response["result"]["content"][0]["text"].as_str().unwrap_or("{}");
        let payload: Value = serde_json::from_str(text).unwrap_or(Value::Null);
        let ids = payload["hits"].as_array().map(|hits| hits.iter().map(|h| h["id"].as_str().unwrap_or("").to_owned()).collect()).unwrap_or_default();
        Found { ids, rerank: payload.get("rerank").cloned(), is_error, took }
    }

    fn close(mut self) {
        drop(self.stdin);
        let _ = self.child.wait();
    }
}

/// A stand-in for the hub: it holds `hub.lock` (with the pid it is given),
/// listens on the real socket path, and answers each request line with
/// whatever the handler returns. Every line it receives is kept.
struct Fake {
    seen: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    lock: LockSlot,
    handle: Option<std::thread::JoinHandle<()>>,
}

type LockSlot = Arc<Mutex<Option<std::fs::File>>>;
type Handler = Arc<dyn Fn(&Value) -> Vec<Value> + Send + Sync>;

impl Fake {
    fn start(env: &Env, lock_pid: u32, handler: impl Fn(&Value) -> Vec<Value> + Send + Sync + 'static) -> Self {
        let socket = env.layout();
        let _ = std::fs::remove_file(&socket);
        let listener = UnixListener::bind(&socket).unwrap();
        std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let lock = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(env.base.join("data/hub.lock")).unwrap();
        lock.try_lock().expect("the fake takes the lock");
        std::fs::write(env.base.join("data/hub.lock"), lock_pid.to_string()).unwrap();
        let lock = Arc::new(Mutex::new(Some(lock)));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let handler: Handler = Arc::new(handler);
        let (seen2, stop2) = (seen.clone(), stop.clone());
        let handle = std::thread::spawn(move || {
            while !stop2.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let (seen, handler) = (seen2.clone(), handler.clone());
                        std::thread::spawn(move || {
                            let _ = stream.set_nonblocking(false);
                            let mut writer = stream.try_clone().unwrap();
                            let mut reader = BufReader::new(stream);
                            let mut line = String::new();
                            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                                if let Ok(request) = serde_json::from_str::<Value>(&line) {
                                    seen.lock().unwrap().push(request.clone());
                                    for reply in handler(&request) {
                                        let _ = writeln!(writer, "{reply}");
                                    }
                                }
                                line.clear();
                            }
                        });
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            }
        });
        Self { seen, stop, lock, handle: Some(handle) }
    }

    fn kinds(&self) -> Vec<String> {
        self.seen.lock().unwrap().iter().map(|r| r["type"].as_str().unwrap_or("?").to_owned()).collect()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let _ = self.lock.lock().unwrap().take();
    }
}

fn welcome(pid: u32, build: &str) -> Value {
    json!({"type":"welcome","proto":1,"build":build,"pid":pid})
}

#[test]
fn mcp_search_goes_through_the_hub_and_reports_it() {
    let env = Env::new().with("IDLE_MS", "30000").seeded();
    let mut plain = env.session_with(&[("ROLEPOD_BRAIN_HUB", "off")], &[], None);
    let before = plain.search(false);
    assert!(before.ids.len() >= 3, "{:?}", before.ids);
    assert!(before.rerank.is_none());
    plain.close();
    let mut session = env.session();
    let first = session.search(true);
    assert!(!first.is_error);
    assert_eq!((first.engine().as_str(), first.reason().as_str()), ("hub", ""), "{:?}", first.rerank);
    let mut reversed = before.ids.clone();
    reversed.reverse();
    assert_eq!(first.ids, reversed, "the stub reverses the pool");
    let pid = env.hub_pid();
    let second = session.search(true);
    assert_eq!(second.engine(), "hub");
    assert_eq!(env.hub_pid(), pid, "the second search finds the same hub");
    assert!(session.search(false).rerank.is_none(), "a search that did not ask for a rerank reports none");
    session.close();
}

#[test]
fn cli_search_goes_through_the_hub_and_explain_names_it() {
    let env = Env::new().with("IDLE_MS", "30000").seeded();
    let run = |hub: Option<&str>| {
        let mut cmd = env.command(&["search", "auth", "--rerank", "--explain"]);
        cmd.current_dir(env.project()).stdout(Stdio::piped());
        if let Some(value) = hub {
            cmd.env("ROLEPOD_BRAIN_HUB", value);
        }
        let output = cmd.output().unwrap();
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let off = run(Some("off"));
    assert!(!off.contains("rerank: hub"), "{off}");
    assert!(env.hub_files().is_empty(), "mode off leaves no hub file: {:?}", env.hub_files());
    let on = run(None);
    assert!(on.contains("rerank: hub"), "{on}");
}

#[test]
fn two_sessions_are_counted_by_status_and_doctor_and_an_old_hello_by_none() {
    let env = Env::new().with("IDLE_MS", "60000").seeded();
    let mut first = env.session();
    assert_eq!(first.search(true).engine(), "hub");
    let mut second = env.session();
    assert_eq!(second.search(true).engine(), "hub");
    first.search(true);
    // An old client's Hello has no `client`; it is welcomed and counts for nothing.
    let mut old = env.raw();
    assert_eq!(old.status()["sessions"], 2);
    let out = env.command(&["hub", "status"]).stdout(Stdio::piped()).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("sessions   2 in 10 min"), "{text}");
    let report = doctor(&env);
    assert!(doctor_row(&report, "hub ").contains("2 session(s) in 10 min"), "{report}");
    first.close();
    second.close();
}

#[test]
fn a_new_hello_with_a_client_pid_is_counted_once_and_never_signalled() {
    let env = Env::new().with("IDLE_MS", "60000");
    let hub = env.up();
    let socket = env.socket().unwrap();
    let hello = |pid: u32| json!({"type":"hello","proto":1,"build":"0.1.0","client":pid});
    let mut a = Raw::connect(&socket, hello(4_000_000));
    let _b = Raw::connect(&socket, hello(4_000_000));
    let _c = Raw::connect(&socket, hello(4_000_001));
    assert_eq!(a.status()["sessions"], 2, "the same pid twice is one session");
    // Nothing was signalled or opened by that pid: the hub is still up.
    assert!(Command::new("kill").args(["-0", &hub.id().to_string()]).status().unwrap().success());
    reap(hub);
}

#[test]
fn mode_off_touches_no_hub_files() {
    let env = Env::new().seeded();
    let mut session = env.session_with(&[("ROLEPOD_BRAIN_HUB", "off")], &[], None);
    let found = session.search(true);
    assert!(!found.is_error);
    assert!(found.rerank.is_none(), "mode=off keeps the 0.67.0 result shape");
    session.close();
    assert!(env.hub_files().is_empty(), "{:?}", env.hub_files());
    assert!(env.sock_dir().is_none(), "no private dir was made");
    std::fs::write(env.base.join("data/config.toml"), "[search]\nrerank = true\n[hub]\nmode = \"off\"\n").unwrap();
    let mut session = env.session();
    assert!(session.search(true).rerank.is_none());
    session.close();
    assert!(env.hub_files().is_empty() && env.sock_dir().is_none());
}

#[test]
fn twenty_clients_one_hub_le3_spawns() {
    let env = Env::new().with("IDLE_MS", "30000").seeded();
    let log = env.base.join("spawns.log");
    let script = env.base.join("spawn.sh");
    std::fs::write(&script, format!("#!/bin/sh\necho x >> {}\nexec {} \"$@\"\n", log.display(), BRAIN)).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let script_path = script.to_string_lossy().into_owned();
    let mut sessions: Vec<Session> = (0..20).map(|_| env.session_with(&[("ROLEPOD_BRAIN_HUB_SPAWN_EXE", &script_path)], &[], None)).collect();
    let handles: Vec<_> = sessions
        .drain(..)
        .map(|mut s| {
            std::thread::spawn(move || {
                let found = s.search(true);
                s.close();
                found
            })
        })
        .collect();
    let found: Vec<Found> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(found.iter().all(|f| !f.is_error && f.rerank.is_some()), "every search answers");
    let spawns = std::fs::read_to_string(&log).map(|t| t.lines().count()).unwrap_or(0);
    assert!((1..=3).contains(&spawns), "{spawns} spawns");
    let engines: Vec<String> = found.iter().map(Found::engine).collect();
    assert!(engines.iter().any(|e| e == "hub"), "{engines:?}");
}

#[test]
fn kill9_mid_rerank_returns_index_order() {
    let env = Env::new().with("STUB_MS", "6000").with("IDLE_MS", "30000").seeded();
    let mut hub = env.up();
    let old_pid = hub.id();
    let mut session = env.session();
    let started = std::thread::spawn(move || {
        let found = session.search(true);
        (session, found)
    });
    let mut probe = env.raw();
    wait(|| probe.status()["model_loaded"] == true, "the hub to be scoring");
    reap_hard(&mut hub);
    let (mut session, found) = started.join().unwrap();
    assert!(!found.is_error, "a dead hub is not a failed search");
    assert_eq!(found.reason(), "hub-down", "{:?}", found.rerank);
    assert!(found.ids.len() >= 3);
    let again = session.search(true);
    assert_eq!(again.engine(), "hub", "the next search meets a new hub: {:?}", again.rerank);
    assert_ne!(env.hub_pid(), old_pid);
    assert_eq!(std::fs::read_to_string(env.base.join("data/hub.deaths")).unwrap().lines().count(), 1);
    session.close();
}

#[test]
fn busy_never_calls_the_cli() {
    let env = Env::new().with("STUB_MS", "3000").with("DEADLINE_MS", "400").with("IDLE_MS", "30000").seeded();
    let bin = env.base.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let hit = env.base.join("cli-called");
    let stub = bin.join("claude");
    std::fs::write(&stub, format!("#!/bin/sh\ntouch {}\necho NONE\n", hit.display())).unwrap();
    std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    let hub = env.up();
    let mut worker = env.raw();
    worker.rerank(1, "q", 20_000);
    std::thread::sleep(Duration::from_millis(200));
    let mut session = env.session_with(&[], &[], Some(&bin));
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!((found.engine().as_str(), found.reason().as_str()), ("none", "hub-busy"), "{:?}", found.rerank);
    assert!(found.ids.len() >= 3, "index order, nothing dropped");
    assert!(!hit.exists(), "the CLI was called");
    session.close();
    reap(hub);
}

#[test]
fn hub_unsafe_reason_on_loose_dir() {
    let env = Env::new().seeded();
    let socket = env.layout();
    let dir = socket.parent().unwrap().to_path_buf();
    let _ = std::fs::remove_file(&socket);
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let before = env.hub_files();
    let mut session = env.session();
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!((found.engine().as_str(), found.reason().as_str()), ("none", "hub-unsafe"));
    session.close();
    assert!(listener.accept().is_err(), "the client did not connect to a socket in a loose dir");
    assert_eq!(env.hub_files(), before, "and spawned nothing");
}

#[test]
fn socket_without_lock_holder_rejected() {
    let env = Env::new().seeded();
    let socket = env.layout();
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).unwrap();
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut session = env.session_with(&[("ROLEPOD_BRAIN_HUB_SPAWN_EXE", "/nonexistent/brain")], &[], None);
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.engine(), "none");
    session.close();
    let mut bytes = Vec::new();
    if let Ok((mut stream, _)) = listener.accept() {
        let _ = stream.read_to_end(&mut bytes);
    }
    assert!(bytes.is_empty(), "a socket nobody holds the lock for gets no byte");
}

#[test]
fn welcome_from_the_wrong_pid_gets_hello_and_nothing_else() {
    let env = Env::new().seeded();
    let me = std::process::id();
    let fake = Fake::start(&env, me, move |request| match request["type"].as_str() {
        Some("hello") => vec![welcome(me + 1, env!("CARGO_PKG_VERSION"))],
        _ => vec![],
    });
    let mut session = env.session();
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.reason(), "hub-unsafe");
    assert_eq!(fake.kinds(), vec!["hello"], "no query left the process");
    session.close();
}

#[test]
fn fake_hub_sends_forged_order() {
    let env = Env::new().seeded();
    let me = std::process::id();
    let turn = Arc::new(Mutex::new(0usize));
    let fake = Fake::start(&env, me, move |request| match request["type"].as_str() {
        Some("hello") => vec![welcome(me, env!("CARGO_PKG_VERSION"))],
        Some("rerank") => {
            let id = request["id"].as_u64().unwrap();
            let mut turn = turn.lock().unwrap();
            *turn += 1;
            match *turn {
                1 => vec![json!({"type":"order","id":id,"indices":[0,99]})],
                2 => vec![json!({"type":"order","id":id,"indices":[1,1]})],
                3 => vec![json!({"type":"order","id":id + 1,"indices":[0,1]})],
                _ => vec![json!({"type":"ack"})],
            }
        }
        _ => vec![],
    });
    let mut session = env.session_with(&[], &[], None);
    let first = session.search(false).ids;
    for turn in 1..=4 {
        let found = session.search(true);
        assert!(!found.is_error, "turn {turn}");
        assert_eq!(found.reason(), "hub-unsafe", "turn {turn}: {:?}", found.rerank);
        assert_eq!(found.ids, first, "turn {turn}: a forged order changes nothing");
    }
    session.close();
    drop(fake);
}

#[test]
fn wedged_hub_search_within_deadline() {
    let env = Env::new().with("DEADLINE_MS", "500").seeded();
    let me = std::process::id();
    let fake = Fake::start(&env, me, |_| vec![]);
    let mut session = env.session();
    let _ = session.search(false);
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.reason(), "hub-down");
    assert!(found.took < Duration::from_millis(1500), "{:?}", found.took);
    session.close();
    drop(fake);
}

#[test]
fn a_hub_silent_after_hello_costs_one_search_then_none() {
    let env = Env::new().with("DEADLINE_MS", "1000").seeded();
    let me = std::process::id();
    let fake = Fake::start(&env, me, move |request| match request["type"].as_str() {
        Some("hello") => vec![welcome(me, env!("CARGO_PKG_VERSION"))],
        _ => vec![],
    });
    let mut session = env.session();
    let _ = session.search(false);
    let first = session.search(true);
    assert!(!first.is_error);
    assert_eq!(first.reason(), "hub-down", "{:?}", first.rerank);
    assert!(first.took < Duration::from_secs(14), "{:?}", first.took);
    let mark = std::fs::read_to_string(env.base.join("data/hub.wedged")).expect("the hub was marked wedged");
    assert!(mark.starts_with(&format!("{me} ")), "{mark}");
    let second = session.search(true);
    assert_eq!(second.reason(), "hub-down", "{:?}", second.rerank);
    assert!(second.took < Duration::from_secs(1), "{:?}", second.took);
    assert_eq!(fake.kinds().iter().filter(|k| *k == "rerank").count(), 1, "the second search never asked");
    session.close();
    drop(fake);
}

#[test]
fn an_order_clears_the_wedged_mark() {
    let env = Env::new().seeded();
    let me = std::process::id();
    let fake = Fake::start(&env, me, move |request| match request["type"].as_str() {
        Some("hello") => vec![welcome(me, env!("CARGO_PKG_VERSION"))],
        Some("rerank") => vec![json!({"type":"order","id":request["id"].as_u64().unwrap(),"indices":[1,0]})],
        _ => vec![],
    });
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    // Another pid's mark is a different hub's: it does not hold this one back.
    let wedged = env.base.join("data/hub.wedged");
    std::fs::write(&wedged, format!("{} {now}\n", me + 1)).unwrap();
    let mut session = env.session();
    let _ = session.search(false);
    let found = session.search(true);
    assert_eq!(found.engine(), "hub", "{:?}", found.rerank);
    assert!(!wedged.exists(), "an answer ends the mark");
    session.close();
    drop(fake);
}

#[test]
fn a_scoring_that_starts_just_before_the_deadline_still_answers() {
    // Starts at once (inside its 2 s start-by), scores 3.5 s: past the old
    // 1 s reply slack (3 s), inside the 3 s one (5 s) and the hub's watchdog (15 s floor).
    let env = Env::new().with("DEADLINE_MS", "2000").with("STUB_MS", "3500").seeded();
    let mut session = env.session();
    let _ = session.search(false);
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.engine(), "hub", "{:?}", found.rerank);
    assert_ne!(found.reason(), "hub-down");
    session.close();
}

#[test]
fn old_hub_new_client_retires_old() {
    let env = Env::new().with("IDLE_MS", "30000").seeded();
    let me = std::process::id();
    let socket_path = Arc::new(Mutex::new(PathBuf::new()));
    let lock_slot: Arc<Mutex<Option<LockSlot>>> = Arc::new(Mutex::new(None));
    let (sp, ls) = (socket_path.clone(), lock_slot.clone());
    let fake = Fake::start(&env, me, move |request| match request["type"].as_str() {
        Some("hello") => vec![welcome(me, "0.0.1")],
        Some("retire") => {
            let _ = std::fs::remove_file(&*sp.lock().unwrap());
            if let Some(lock) = ls.lock().unwrap().as_ref() {
                let _ = lock.lock().unwrap().take();
            }
            vec![json!({"type":"ack"})]
        }
        _ => vec![],
    });
    *socket_path.lock().unwrap() = env.socket().unwrap();
    *lock_slot.lock().unwrap() = Some(fake.lock.clone());
    let mut session = env.session();
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.engine(), "hub", "the new client used a new hub: {:?}", found.rerank);
    assert_eq!(fake.kinds().iter().filter(|k| *k == "retire").count(), 1);
    assert!(!fake.kinds().contains(&"rerank".to_owned()), "the old hub was never asked");
    session.close();
}

#[test]
fn a_real_older_hub_is_retired_by_a_newer_client() {
    let env = Env::new().with("IDLE_MS", "30000").seeded();
    // The real server, posing as an older build of the same protocol.
    let mut old = env.command(&["hub", "serve"]);
    let mut old = old.env("ROLEPOD_BRAIN_HUB_BUILD", "0.0.1").spawn().unwrap();
    wait(|| env.socket().is_some_and(|s| UnixStream::connect(s).is_ok()), "old hub socket");
    let old_pid = env.hub_pid();
    let mut session = env.session();
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.engine(), "hub", "{:?}", found.rerank);
    assert_ne!(env.hub_pid(), old_pid, "a new hub took over");
    assert!(exits_within(&mut old, 5).is_some(), "the old hub left");
    session.close();
}

#[test]
fn missing_exe_does_not_lock_out() {
    let env = Env::new().with("IDLE_MS", "30000").seeded();
    for _ in 0..5 {
        let mut session = env.session_with(&[("ROLEPOD_BRAIN_HUB_SPAWN_EXE", "/nonexistent/brain")], &[], None);
        let found = session.search(true);
        assert_eq!(found.reason(), "hub-down");
        session.close();
    }
    assert!(env.hub_files().is_empty(), "{:?}", env.hub_files());
    let mut session = env.session();
    assert_eq!(session.search(true).engine(), "hub", "a good executable gets a hub within one call");
    session.close();
}

#[test]
fn a_crash_lockout_answers_at_once_and_spawns_nothing() {
    let env = Env::new().seeded();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    std::fs::write(env.base.join("data/hub.deaths"), format!("11 {now}\n12 {now}\n13 {now}\n")).unwrap();
    let mut session = env.session();
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.reason(), "hub-down");
    assert!(found.took < Duration::from_millis(1500), "{:?}", found.took);
    session.close();
    assert_eq!(env.hub_files(), vec!["hub.deaths".to_string()]);
    assert!(env.sock_dir().is_none());
}

#[test]
fn no_private_dir_runs_index_order() {
    let env = Env::new().seeded();
    let mut session = env.session_with(&[], &["TMPDIR", "XDG_RUNTIME_DIR"], None);
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.reason(), "hub-unsafe");
    session.close();
    assert!(env.hub_files().is_empty());
}

#[test]
fn long_tmpdir_falls_back_quietly() {
    let env = Env::new().seeded();
    let long = env.base.join("t").join("x".repeat(120));
    std::fs::create_dir_all(&long).unwrap();
    let long = long.to_string_lossy().into_owned();
    let mut session = env.session_with(&[("TMPDIR", &long), ("XDG_RUNTIME_DIR", &long)], &[], None);
    let found = session.search(true);
    assert!(!found.is_error);
    assert_eq!(found.reason(), "hub-unsafe");
    session.close();
}

/// `brain doctor` on this env, as text.
fn doctor(env: &Env) -> String {
    let mut cmd = env.command(&["doctor"]);
    cmd.stdout(Stdio::piped());
    String::from_utf8_lossy(&cmd.output().unwrap().stdout).into_owned()
}

fn doctor_row<'a>(report: &'a str, name: &str) -> &'a str {
    report.lines().find(|l| l.contains(name)).unwrap_or_else(|| panic!("no `{name}` row in:\n{report}"))
}

#[test]
fn doctor_shows_the_live_hub_and_counts_it_apart_from_mcp() {
    let env = Env::new();
    let hub = env.up();
    let _raw = env.raw();
    let report = doctor(&env);
    let row = doctor_row(&report, "hub ");
    assert!(row.contains(&format!("pid {}", env.hub_pid())), "{row}");
    // Neither doctor's own probe nor a Hello with no pid is a session.
    assert!(row.contains("0 session(s) in 10 min"), "{row}");
    assert!(row.contains("model not loaded"), "{row}");
    assert!(row.contains("0 restart(s)/24h"), "{row}");
    let processes = doctor_row(&report, "processes");
    // ps sees the whole machine, so other MCP servers may be listed; the hub is not one.
    let pid = env.hub_pid();
    let (hub_part, mcp_part) = processes.split_once(';').unwrap_or((processes, ""));
    assert!(hub_part.contains(&format!("1 hub (pid {pid})")), "{processes}");
    assert!(!mcp_part.contains(&format!("pid {pid}")), "{processes}");
    reap(hub);
}

#[test]
fn doctor_without_a_hub_says_it_starts_itself() {
    let env = Env::new();
    let report = doctor(&env);
    let row = doctor_row(&report, "hub ");
    assert!(row.contains("starts with the next reranked search"), "{row}");
    assert!(!row.contains("brain hub"), "no command in the row: {row}");
    assert!(env.hub_files().is_empty(), "doctor starts nothing");
}

#[test]
fn doctor_says_when_the_hub_is_off() {
    let env = Env::new();
    let mut cmd = env.command(&["doctor"]);
    cmd.env("ROLEPOD_BRAIN_HUB", "off").stdout(Stdio::piped());
    let report = String::from_utf8_lossy(&cmd.output().unwrap().stdout).into_owned();
    assert!(doctor_row(&report, "hub ").contains("off"), "{report}");
}

#[test]
fn doctor_warns_when_most_recent_reranks_got_no_hub() {
    let env = Env::new();
    let _ = doctor(&env); // creates the store
    let conn = rusqlite::Connection::open(env.base.join("data/brain.db")).unwrap();
    let rows = [("none", "hub-busy"); 14].into_iter().chain([("local", ""); 6]);
    for (engine, reason) in rows {
        conn.execute(
            "INSERT INTO rerank_runs (ts, engine, reason, ms, cold) VALUES ('2026-10-08T00:00:00Z', ?1, ?2, 5, 0)",
            [engine, reason],
        )
        .unwrap();
    }
    drop(conn);
    let report = doctor(&env);
    let row = doctor_row(&report, "hub answers");
    assert!(row.contains("14 of the last 20") && row.contains("hub-busy x14"), "{row}");
    assert!(!row.contains("run "), "no command in the row: {row}");
}

#[test]
fn doctor_is_quiet_when_the_hub_mostly_answers() {
    let env = Env::new();
    let _ = doctor(&env);
    let conn = rusqlite::Connection::open(env.base.join("data/brain.db")).unwrap();
    for reason in ["hub-busy", "", "", "", "", ""] {
        conn.execute(
            "INSERT INTO rerank_runs (ts, engine, reason, ms, cold) VALUES ('2026-10-08T00:00:00Z', 'hub', ?1, 5, 0)",
            [reason],
        )
        .unwrap();
    }
    drop(conn);
    assert!(!doctor(&env).contains("hub answers"));
}
