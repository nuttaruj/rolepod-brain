//! `brain hub serve`: one reranker for every brain process of one data dir.
//!
//! One process per data dir, picked by `hub.lock`. It listens on a socket in
//! the private directory, queues rerank requests first in first out, and
//! scores them one at a time on a single worker. It writes nothing but its own
//! `hub.*` files, opens no store, and starts no helper but `ps`.
//!
//! It leaves by itself: after an idle spell, when its own executable changes
//! on disk, past a day of age, or when its memory stays over the limit. In
//! each case it first stops accepting and gives up the lock and the socket
//! path, so the next hub can take over, and exits when the work already
//! accepted is answered.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufReader, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

use super::endpoint::{self, Endpoint};
use super::proto::{self, LineError, Reply, Request, PROTO};
use crate::config::Paths;

/// Open connections at once. The rest are closed on accept.
const MAX_CONNS: usize = 64;
/// Sessions are the distinct client pids seen this long.
const SESSION_WINDOW: Duration = Duration::from_secs(600);
/// The most pids remembered; the oldest goes first.
const MAX_SESSIONS: usize = 256;
/// The main loop's `tick` runs at least this often.
const TICK_EVERY: Duration = Duration::from_millis(20);
/// A lock lost to a probing client is tried again this many times.
const LOCK_TRIES: u32 = 10;
const LOCK_PAUSE: Duration = Duration::from_millis(10);
/// The longest a request may wait in all: a reply that has not come by then
/// is `Unavailable`.
const REPLY_CEILING: Duration = Duration::from_secs(600);
/// A deadline asked for beyond this is cut to it.
const MAX_DEADLINE: Duration = Duration::from_secs(120);
/// Crash records kept in `hub.crashes`.
const MAX_CRASH_RECORDS: usize = 64;

fn build() -> &'static str {
    // A test hub can pose as another build, to prove the retire path against
    // the real server; release binaries have no such knob.
    #[cfg(debug_assertions)]
    {
        static FAKE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        if let Some(fake) = FAKE.get_or_init(|| std::env::var("ROLEPOD_BRAIN_HUB_BUILD").ok()) {
            return fake.as_str();
        }
    }
    env!("CARGO_PKG_VERSION")
}

/// Every clock and limit of the hub. [`Clocks::live`] is the real one; the
/// environment shortens them for a test, the way `ROLEPOD_BRAIN_MAINT_*` does.
#[derive(Debug, Clone, Copy)]
pub struct Clocks {
    /// No client and no request this long: exit.
    pub idle: Duration,
    pub exe_check: Duration,
    pub max_age: Duration,
    pub rss_busy: Duration,
    pub rss_idle: Duration,
    /// A loaded stub model is let go after this (the real one has its own).
    pub model_idle: Duration,
    pub rss_limit_kb: u64,
    pub handshake: Duration,
    pub read: Duration,
    pub write: Duration,
    /// A worker stuck past `max(2 x its deadline, this)` ends the process.
    pub watchdog_floor: Duration,
}

fn env_ms(name: &str, default: Duration) -> Duration {
    std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok()).map_or(default, Duration::from_millis)
}

impl Clocks {
    #[must_use]
    pub fn live() -> Self {
        Self {
            idle: env_ms("ROLEPOD_BRAIN_HUB_IDLE_MS", Duration::from_secs(600)),
            exe_check: env_ms("ROLEPOD_BRAIN_HUB_EXE_CHECK_MS", Duration::from_secs(30)),
            max_age: env_ms("ROLEPOD_BRAIN_HUB_MAX_AGE_MS", Duration::from_secs(24 * 3600)),
            rss_busy: env_ms("ROLEPOD_BRAIN_HUB_RSS_BUSY_MS", Duration::from_secs(30)),
            rss_idle: env_ms("ROLEPOD_BRAIN_HUB_RSS_IDLE_MS", Duration::from_secs(300)),
            model_idle: env_ms("ROLEPOD_BRAIN_HUB_MODEL_IDLE_MS", Duration::from_secs(90)),
            rss_limit_kb: std::env::var("ROLEPOD_BRAIN_HUB_RSS_LIMIT_KB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_500 * 1024),
            handshake: env_ms("ROLEPOD_BRAIN_HUB_HANDSHAKE_MS", Duration::from_secs(2)),
            read: env_ms("ROLEPOD_BRAIN_HUB_READ_MS", Duration::from_secs(5)),
            write: Duration::from_secs(2),
            watchdog_floor: env_ms("ROLEPOD_BRAIN_HUB_WATCHDOG_MS", Duration::from_secs(15)),
        }
    }
}

// ---------------------------------------------------------------------------
// Policies: small pure functions, so the table of each is a unit test.

/// The identity of an executable file. A different one means a new build was
/// put in place (or the file is gone, mid-deploy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExeId {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime: (i64, i64),
}

impl ExeId {
    fn of(path: &Path) -> Option<Self> {
        let meta = fs::metadata(path).ok()?;
        Some(Self { dev: meta.dev(), ino: meta.ino(), size: meta.size(), mtime: (meta.mtime(), meta.mtime_nsec()) })
    }
}

/// Retire when the file at the exe path is no longer the one this process
/// started from. A start identity that could not be read never retires.
#[must_use]
pub fn exe_changed(started: Option<ExeId>, now: Option<ExeId>) -> bool {
    started.is_some() && started != now
}

/// What the memory check decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rss {
    Keep,
    /// Let go of the model; connections stay.
    Release,
    /// Over the limit with nothing left to release and nothing to do: go.
    Retire,
}

/// A failure to sample is unknown, which keeps things as they are.
#[must_use]
pub fn rss_policy(busy: bool, loaded: bool, rss_kb: Option<u64>, limit_kb: u64) -> Rss {
    match rss_kb {
        Some(rss) if rss > limit_kb && loaded => Rss::Release,
        Some(rss) if rss > limit_kb && !busy => Rss::Retire,
        _ => Rss::Keep,
    }
}

/// Past the maximum age, retire once nothing is going on.
#[must_use]
pub fn age_due(age: Duration, max_age: Duration, busy: bool) -> bool {
    age >= max_age && !busy
}

/// A worker that has run for `max(2 x window, floor)` is stuck.
#[must_use]
pub fn watchdog_due(busy: Option<(Instant, Duration)>, now: Instant, floor: Duration) -> bool {
    busy.is_some_and(|(since, window)| now.saturating_duration_since(since) > (window * 2).max(floor))
}

/// How a Hello is answered.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Welcome,
    /// Refused; the connection closes.
    Refused(&'static str),
    /// Refused because the client is newer than this hub; the connection
    /// stays open for exactly one `Retire`.
    RefusedMayRetire,
}

#[must_use]
pub fn hello_verdict(proto: u32, their_build: &str, our_build: &str) -> Verdict {
    if !proto::build_ok(their_build) {
        return Verdict::Refused("bad-build");
    }
    if proto::accepts(proto, PROTO) {
        return Verdict::Welcome;
    }
    if proto > PROTO && proto::newer_build(their_build, our_build) {
        return Verdict::RefusedMayRetire;
    }
    Verdict::Refused(if proto > PROTO { super::reason::OLDER } else { "proto-too-old" })
}

/// `Retire` is obeyed only from a strictly newer build.
#[must_use]
pub fn retire_allowed(their_build: &str, our_build: &str) -> bool {
    proto::newer_build(their_build, our_build)
}

// ---------------------------------------------------------------------------
// The queue.

struct Job {
    id: u64,
    hash: u64,
    query: String,
    entries: Vec<String>,
    /// The request must start by this.
    deadline: Instant,
    window: Duration,
    /// Set by the worker when it takes the job up.
    started: Arc<AtomicBool>,
    reply: mpsc::Sender<Reply>,
}

enum Taken {
    Run(Job),
    /// Its deadline passed before it could start.
    Expired(Job),
}

/// The head of the queue, first in first out.
fn take_next(queue: &mut VecDeque<Job>, now: Instant) -> Option<Taken> {
    let job = queue.pop_front()?;
    Some(if now > job.deadline { Taken::Expired(job) } else { Taken::Run(job) })
}

/// Open connections, capped.
struct Slots {
    open: AtomicUsize,
    cap: usize,
}

impl Slots {
    fn new(cap: usize) -> Self {
        Self { open: AtomicUsize::new(0), cap }
    }

    fn acquire(&self) -> bool {
        self.open.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < self.cap).then_some(n + 1)).is_ok()
    }

    fn release(&self) {
        self.open.fetch_sub(1, Ordering::SeqCst);
    }

    fn count(&self) -> usize {
        self.open.load(Ordering::SeqCst)
    }
}

/// The client pids that said Hello lately. A pid is data from the wire: it is
/// only counted, never signalled and never opened, and the set is capped so a
/// client cannot grow it.
#[derive(Default)]
struct Sessions {
    seen: Vec<(u32, Instant)>,
}

impl Sessions {
    fn note(&mut self, pid: u32, now: Instant) {
        self.seen.retain(|(_, at)| now.saturating_duration_since(*at) < SESSION_WINDOW);
        if let Some(entry) = self.seen.iter_mut().find(|(seen, _)| *seen == pid) {
            entry.1 = now;
            return;
        }
        if self.seen.len() >= MAX_SESSIONS {
            self.seen.remove(0);
        }
        self.seen.push((pid, now));
    }

    fn count(&self, now: Instant) -> u32 {
        let live = self.seen.iter().filter(|(_, at)| now.saturating_duration_since(*at) < SESSION_WINDOW).count();
        u32::try_from(live).unwrap_or(u32::MAX)
    }
}

/// Take the lock, trying again for a moment: a client probing holds it shared
/// for an instant. `Ok(false)` is a real holder.
fn lock_with_retry(file: &fs::File, tries: u32, pause: Duration) -> std::io::Result<bool> {
    for attempt in 0..tries.max(1) {
        match file.try_lock() {
            Ok(()) => return Ok(true),
            Err(fs::TryLockError::WouldBlock) => {
                if attempt + 1 < tries {
                    std::thread::sleep(pause);
                }
            }
            Err(fs::TryLockError::Error(error)) => return Err(error),
        }
    }
    Ok(false)
}

/// Is the main loop's `tick` due? It runs on a clock of its own, so a steady
/// stream of connections cannot starve the watchdog.
fn tick_due(last: Instant, now: Instant) -> bool {
    now.saturating_duration_since(last) >= TICK_EVERY
}

// ---------------------------------------------------------------------------
// The poison record: `panic = "abort"` means a request that kills the model
// call kills the process, so the request in flight is written down first.

struct Poison {
    inflight: PathBuf,
    crashes: PathBuf,
}

impl Poison {
    /// Open the records in `dir`. A marker left by a process that died mid
    /// request becomes one crash record for that request.
    fn recover(dir: &Path) -> Self {
        let poison = Self { inflight: dir.join("hub.inflight"), crashes: dir.join("hub.crashes") };
        if let Ok(left) = fs::read_to_string(&poison.inflight) {
            let hash = left.trim();
            if !hash.is_empty() {
                let mut lines: Vec<String> =
                    fs::read_to_string(&poison.crashes).unwrap_or_default().lines().map(str::to_owned).collect();
                lines.push(hash.to_owned());
                let from = lines.len().saturating_sub(MAX_CRASH_RECORDS);
                let _ = fs::write(&poison.crashes, lines[from..].join("\n") + "\n");
            }
            let _ = fs::remove_file(&poison.inflight);
        }
        poison
    }

    fn mark(&self, hash: u64) {
        let _ = fs::write(&self.inflight, format!("{hash:016x}"));
    }

    fn clear(&self) {
        let _ = fs::remove_file(&self.inflight);
    }

    /// Has this request killed a hub twice?
    fn blocked(&self, hash: u64) -> bool {
        let want = format!("{hash:016x}");
        fs::read_to_string(&self.crashes).map_or(0, |text| text.lines().filter(|line| *line == want).count()) >= 2
    }
}

fn request_hash(query: &str, entries: &[String]) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(build().as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(query.as_bytes());
    for entry in entries {
        bytes.push(0);
        bytes.extend_from_slice(entry.as_bytes());
    }
    endpoint::fnv1a(&bytes)
}

// ---------------------------------------------------------------------------
// The model. Real under `local-rerank`; a stub for the end-to-end tests in a
// debug build only (`debug_assertions`), never present in a release build.

enum Scored {
    #[cfg_attr(all(not(debug_assertions), not(feature = "local-rerank")), allow(dead_code))]
    Ranked(Vec<usize>),
    #[cfg_attr(not(feature = "local-rerank"), allow(dead_code))]
    Busy,
    Unavailable,
}

#[cfg(debug_assertions)]
mod stub {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    pub static LOADED: AtomicBool = AtomicBool::new(false);
    static LAST_USED: Mutex<Option<Instant>> = Mutex::new(None);

    /// `ROLEPOD_BRAIN_HUB_STUB=1` turns it on; `ROLEPOD_BRAIN_HUB_STUB_MS` is
    /// how long one score takes. It answers in reverse order, and the query
    /// `POISON` aborts the process.
    pub fn on() -> bool {
        std::env::var_os("ROLEPOD_BRAIN_HUB_STUB").is_some()
    }

    pub fn score(query: &str, entries: &[String]) -> Option<Vec<usize>> {
        if entries.len() < 2 {
            return None;
        }
        LOADED.store(true, Ordering::SeqCst);
        *LAST_USED.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
        if query == "POISON" {
            std::process::abort();
        }
        let ms = std::env::var("ROLEPOD_BRAIN_HUB_STUB_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        std::thread::sleep(Duration::from_millis(ms));
        *LAST_USED.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
        (query != "ERROR").then(|| (0..entries.len()).rev().collect())
    }

    pub fn idle_release(window: Duration) {
        let mut last = LAST_USED.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if last.is_some_and(|at| at.elapsed() >= window) {
            LOADED.store(false, Ordering::SeqCst);
            *last = None;
        }
    }
}

fn score(model_dir: &Path, query: &str, entries: &[String], wait: Duration) -> Scored {
    #[cfg(debug_assertions)]
    if stub::on() {
        return stub::score(query, entries).map_or(Scored::Unavailable, Scored::Ranked);
    }
    real_score(model_dir, query, entries, wait)
}

#[cfg(feature = "local-rerank")]
fn real_score(model_dir: &Path, query: &str, entries: &[String], wait: Duration) -> Scored {
    match crate::xencoder::rerank_within(model_dir, query, entries, wait) {
        crate::xencoder::LocalOrder::Ranked(order) => Scored::Ranked(order),
        crate::xencoder::LocalOrder::Busy => Scored::Busy,
        crate::xencoder::LocalOrder::Unavailable => Scored::Unavailable,
    }
}

#[cfg(not(feature = "local-rerank"))]
fn real_score(_: &Path, _: &str, _: &[String], _: Duration) -> Scored {
    Scored::Unavailable
}

fn model_loaded() -> bool {
    #[cfg(debug_assertions)]
    if stub::on() {
        return stub::LOADED.load(Ordering::SeqCst);
    }
    real_loaded()
}

#[cfg(feature = "local-rerank")]
fn real_loaded() -> bool {
    crate::xencoder::is_loaded()
}

#[cfg(not(feature = "local-rerank"))]
fn real_loaded() -> bool {
    false
}

fn release_model() {
    #[cfg(debug_assertions)]
    if stub::on() {
        stub::LOADED.store(false, Ordering::SeqCst);
    }
    #[cfg(feature = "local-rerank")]
    crate::xencoder::release();
}

#[cfg(debug_assertions)]
fn model_idle_tick(window: Duration) {
    if stub::on() {
        stub::idle_release(window);
    }
}

#[cfg(not(debug_assertions))]
fn model_idle_tick(_: Duration) {}

/// Resident memory of `pid` in KiB, from `ps`. This is the one place the hub
/// starts another program. Unknown (`None`) when `ps` cannot answer.
fn ps_rss_kb(pid: u32) -> Option<u64> {
    let out = std::process::Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output().ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

// ---------------------------------------------------------------------------
// The server.

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Shared {
    clocks: Clocks,
    started: Instant,
    pid: u32,
    model_dir: PathBuf,
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
    slots: Slots,
    sessions: Mutex<Sessions>,
    /// Requests accepted and not yet answered, plus a Stop being acknowledged.
    inflight: AtomicUsize,
    last_activity: Mutex<Instant>,
    retiring: AtomicBool,
    /// The lock file, until the endpoint is given up.
    held: Mutex<Option<fs::File>>,
    socket: PathBuf,
    /// When the worker began the job it is on, and that job's window.
    busy: Mutex<Option<(Instant, Duration)>>,
    poison: Poison,
}

/// One unit of accepted work. The hub does not exit while any exists.
struct Work<'a>(&'a Shared);

impl Drop for Work<'_> {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::SeqCst);
        self.0.touch();
    }
}

impl Shared {
    fn touch(&self) {
        *lock(&self.last_activity) = Instant::now();
    }

    fn work(&self) -> Work<'_> {
        self.inflight.fetch_add(1, Ordering::SeqCst);
        self.touch();
        Work(self)
    }

    fn is_retiring(&self) -> bool {
        self.retiring.load(Ordering::SeqCst)
    }

    /// Stop taking work and give the endpoint back, so a new hub can start.
    /// The first call does it; the work already accepted still finishes.
    fn begin_retire(&self) {
        if self.retiring.swap(true, Ordering::SeqCst) {
            return;
        }
        let _ = fs::remove_file(&self.socket);
        lock(&self.held).take();
        self.wake.notify_all();
    }

    fn status(&self) -> Reply {
        Reply::Status {
            pid: self.pid,
            build: build().to_owned(),
            proto: PROTO,
            uptime_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            footprint_kb: ps_rss_kb(self.pid),
            clients: self.slots.count(),
            sessions: lock(&self.sessions).count(Instant::now()),
            queued: lock(&self.queue).len(),
            model_loaded: model_loaded(),
            retiring: self.is_retiring(),
        }
    }
}

/// Run the hub until it retires.
///
/// # Errors
/// An unsafe endpoint, or a socket that cannot be bound. A lost race for the
/// lock is not an error: the loser returns `Ok` at once.
pub fn serve() -> Result<()> {
    let paths = Paths::resolve()?;
    fs::create_dir_all(&paths.data_dir).context("cannot create the data dir")?;
    let clocks = Clocks::live();
    let pid = std::process::id();

    let mut lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(paths.data_dir.join(endpoint::LOCK_FILE))
        .context("cannot open hub.lock")?;
    if !lock_with_retry(&lock_file, LOCK_TRIES, LOCK_PAUSE).context("cannot lock hub.lock")? {
        return Ok(());
    }
    lock_file.set_len(0)?;
    lock_file.write_all(pid.to_string().as_bytes())?;

    let ep: Endpoint = endpoint::resolve(&paths.data_dir).map_err(|e| anyhow!("{e}"))?;
    ep.prepare().map_err(|e| anyhow!("{e}"))?;
    // The lock is ours, so a socket still on disk belongs to a dead hub.
    match endpoint::Meta::of(&ep.socket) {
        Ok(_) => {
            endpoint::check_socket(&ep.socket, ep.euid).map_err(|e| anyhow!("{e}"))?;
            fs::remove_file(&ep.socket).context("cannot remove the stale socket")?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("cannot inspect the socket path"),
    }
    let listener = UnixListener::bind(&ep.socket).context("cannot bind the hub socket")?;
    fs::set_permissions(&ep.socket, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
    if let Err(unsafe_) = endpoint::check_socket(&ep.socket, ep.euid) {
        let _ = fs::remove_file(&ep.socket);
        return Err(anyhow!("{unsafe_}"));
    }
    listener.set_nonblocking(true)?;

    let exe = std::env::current_exe().ok();
    let exe_start = exe.as_deref().and_then(ExeId::of);
    let shared = Arc::new(Shared {
        clocks,
        started: Instant::now(),
        pid,
        model_dir: paths.model_dir_for(crate::rerank::LOCAL_MODEL),
        queue: Mutex::new(VecDeque::new()),
        wake: Condvar::new(),
        slots: Slots::new(MAX_CONNS),
        sessions: Mutex::new(Sessions::default()),
        inflight: AtomicUsize::new(0),
        last_activity: Mutex::new(Instant::now()),
        retiring: AtomicBool::new(false),
        held: Mutex::new(Some(lock_file)),
        socket: ep.socket.clone(),
        busy: Mutex::new(None),
        poison: Poison::recover(&paths.data_dir),
    });
    {
        let shared = Arc::clone(&shared);
        std::thread::Builder::new().name("hub-worker".into()).spawn(move || worker(&shared))?;
    }

    let mut listener = Some(listener);
    let mut last_exe = Instant::now();
    let mut last_rss = Instant::now();
    let mut last_tick = Instant::now();
    loop {
        if shared.is_retiring() {
            listener = None;
            if shared.inflight.load(Ordering::SeqCst) == 0 {
                return Ok(());
            }
        }
        let accepted = listener.as_ref().map(UnixListener::accept);
        match accepted {
            Some(Ok((stream, _))) => {
                if exe_changed(exe_start, exe.as_deref().and_then(ExeId::of)) {
                    shared.begin_retire();
                    continue;
                }
                if shared.slots.acquire() {
                    shared.touch();
                    let shared = Arc::clone(&shared);
                    let spawned = std::thread::Builder::new().name("hub-conn".into()).spawn({
                        let shared = Arc::clone(&shared);
                        move || {
                            serve_conn(&shared, stream);
                            shared.slots.release();
                        }
                    });
                    if spawned.is_err() {
                        shared.slots.release();
                    }
                }
            }
            _ => std::thread::sleep(TICK_EVERY),
        }
        let now = Instant::now();
        if tick_due(last_tick, now) {
            last_tick = now;
            tick(&shared, exe.as_deref(), exe_start, &mut last_exe, &mut last_rss);
        }
    }
}

/// Everything the main loop does between accepts.
fn tick(shared: &Shared, exe: Option<&Path>, exe_start: Option<ExeId>, last_exe: &mut Instant, last_rss: &mut Instant) {
    let clocks = &shared.clocks;
    let now = Instant::now();
    if watchdog_due(*lock(&shared.busy), now, clocks.watchdog_floor) {
        // The marker stays on disk, so the next start counts this request.
        std::process::abort();
    }
    model_idle_tick(clocks.model_idle);
    if shared.is_retiring() {
        return;
    }
    let conns = shared.slots.count();
    let inflight = shared.inflight.load(Ordering::SeqCst);
    let busy = conns > 0 || inflight > 0;
    if now.duration_since(*last_exe) >= clocks.exe_check {
        *last_exe = now;
        if exe_changed(exe_start, exe.and_then(ExeId::of)) {
            return shared.begin_retire();
        }
    }
    let quiet = now.duration_since(*lock(&shared.last_activity)) >= clocks.idle;
    if !busy && quiet {
        return shared.begin_retire();
    }
    if age_due(shared.started.elapsed(), clocks.max_age, busy) {
        return shared.begin_retire();
    }
    let every = if busy { clocks.rss_busy } else { clocks.rss_idle };
    if now.duration_since(*last_rss) >= every {
        *last_rss = now;
        match rss_policy(busy, model_loaded(), ps_rss_kb(shared.pid), clocks.rss_limit_kb) {
            Rss::Keep => {}
            Rss::Release => release_model(),
            Rss::Retire => shared.begin_retire(),
        }
    }
}

fn worker(shared: &Shared) {
    loop {
        let taken = {
            let mut queue = lock(&shared.queue);
            loop {
                if let Some(taken) = take_next(&mut queue, Instant::now()) {
                    break taken;
                }
                queue = shared.wake.wait_timeout(queue, Duration::from_millis(100)).unwrap_or_else(PoisonError::into_inner).0;
            }
        };
        match taken {
            Taken::Expired(job) => {
                let _ = job.reply.send(Reply::Busy { id: job.id });
            }
            Taken::Run(job) => {
                job.started.store(true, Ordering::SeqCst);
                shared.poison.mark(job.hash);
                *lock(&shared.busy) = Some((Instant::now(), job.window));
                let wait = job.deadline.saturating_duration_since(Instant::now());
                let scored = score(&shared.model_dir, &job.query, &job.entries, wait);
                *lock(&shared.busy) = None;
                shared.poison.clear();
                let reply = match scored {
                    Scored::Ranked(indices) => Reply::Order { id: job.id, indices },
                    Scored::Busy => Reply::Busy { id: job.id },
                    Scored::Unavailable => Reply::Unavailable { id: job.id },
                };
                let _ = job.reply.send(reply);
            }
        }
    }
}

/// How a greeting went.
#[derive(Debug, PartialEq, Eq)]
enum Greeted {
    /// A welcomed client, with the build it said it was and the pid it gave.
    Ready(String, Option<u32>),
    /// The client is a newer build speaking a newer protocol: it may send one
    /// `Retire`.
    MayRetire(String),
    Closed,
}

/// Read the Hello and answer it. A silent client, a bad line, or anything but
/// a Hello ends the connection.
fn greet<R: std::io::BufRead, W: Write>(reader: &mut R, writer: &mut W, pid: u32) -> Greeted {
    let Ok(Some(line)) = proto::read_line(reader) else { return Greeted::Closed };
    let Ok(Request::Hello { proto, build: theirs, client }) = serde_json::from_str::<Request>(&line) else {
        return Greeted::Closed;
    };
    match hello_verdict(proto, &theirs, build()) {
        Verdict::Welcome => {
            let welcome = Reply::Welcome { proto: PROTO, build: build().to_owned(), pid };
            if proto::write_line(writer, &welcome).is_ok() { Greeted::Ready(theirs, client) } else { Greeted::Closed }
        }
        Verdict::RefusedMayRetire => {
            let refused = Reply::Refused { reason: super::reason::OLDER.into() };
            if proto::write_line(writer, &refused).is_ok() { Greeted::MayRetire(theirs) } else { Greeted::Closed }
        }
        Verdict::Refused(reason) => {
            let _ = proto::write_line(writer, &Reply::Refused { reason: reason.into() });
            Greeted::Closed
        }
    }
}

fn serve_conn(shared: &Shared, stream: UnixStream) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_write_timeout(Some(shared.clocks.write));
    let _ = stream.set_read_timeout(Some(shared.clocks.handshake));
    let Ok(read_half) = stream.try_clone() else { return };
    let mut reader = BufReader::new(read_half);
    let mut writer = stream;
    let (their_build, one_shot) = match greet(&mut reader, &mut writer, shared.pid) {
        Greeted::Ready(theirs, client) => {
            if let Some(client) = client {
                lock(&shared.sessions).note(client, Instant::now());
            }
            (theirs, false)
        }
        Greeted::MayRetire(theirs) => (theirs, true),
        Greeted::Closed => return,
    };
    let _ = writer.set_read_timeout(Some(shared.clocks.read));

    if one_shot {
        let theirs = &their_build;
        // One line, and only a Retire from a strictly newer build is obeyed.
        if let Ok(Some(line)) = proto::read_line(&mut reader) {
            if matches!(serde_json::from_str::<Request>(&line), Ok(Request::Retire)) && retire_allowed(theirs, build()) {
                let _work = shared.work();
                shared.begin_retire();
                let _ = proto::write_line(&mut writer, &Reply::Ack);
            }
        }
        return;
    }

    loop {
        let request = match proto::read_line(&mut reader) {
            Ok(Some(line)) => match serde_json::from_str::<Request>(&line) {
                Ok(request) => request,
                Err(_) => return,
            },
            Ok(None) | Err(LineError::TooLong | LineError::NotUtf8 | LineError::Io | LineError::TimedOut) => return,
        };
        let reply = match request {
            Request::Hello { .. } => return,
            // A newer client that found this hub older asks it to go; the
            // work in flight finishes first (`begin_retire`).
            Request::Retire => {
                if retire_allowed(&their_build, build()) {
                    let _work = shared.work();
                    shared.begin_retire();
                    let _ = proto::write_line(&mut writer, &Reply::Ack);
                }
                return;
            }
            Request::Status => shared.status(),
            Request::Stop => {
                let _work = shared.work();
                shared.begin_retire();
                let _ = proto::write_line(&mut writer, &Reply::Ack);
                return;
            }
            Request::Rerank { id, query, entries, deadline_ms } => {
                let _work = shared.work();
                let reply = rerank_request(shared, id, query, entries, deadline_ms);
                if proto::write_line(&mut writer, &reply).is_err() {
                    return;
                }
                continue;
            }
        };
        if proto::write_line(&mut writer, &reply).is_err() {
            return;
        }
    }
}

/// Queue one request and wait for its answer. The caller holds the [`Work`].
fn rerank_request(shared: &Shared, id: u64, query: String, entries: Vec<String>, deadline_ms: u64) -> Reply {
    if shared.is_retiring() || !proto::within_caps(&query, &entries) {
        return Reply::Unavailable { id };
    }
    let hash = request_hash(&query, &entries);
    if shared.poison.blocked(hash) {
        return Reply::Unavailable { id };
    }
    let window = Duration::from_millis(deadline_ms).min(MAX_DEADLINE);
    let (reply, answer) = mpsc::channel();
    let started = Arc::new(AtomicBool::new(false));
    let job = Job { id, hash, query, entries, deadline: Instant::now() + window, window, started: Arc::clone(&started), reply };
    lock(&shared.queue).push_back(job);
    shared.wake.notify_one();
    // A request still waiting when its deadline passes is answered Busy now,
    // not when the worker finally reaches it; one already running is waited for.
    match answer.recv_timeout(window) {
        Ok(reply) => reply,
        Err(_) if !started.load(Ordering::SeqCst) => Reply::Busy { id },
        Err(_) => answer.recv_timeout(REPLY_CEILING).unwrap_or(Reply::Unavailable { id }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn id(ino: u64, size: u64, mtime: i64) -> ExeId {
        ExeId { dev: 1, ino, size, mtime: (mtime, 0) }
    }

    #[test]
    fn exe_identity_change_decision() {
        let a = id(1, 10, 100);
        assert!(!exe_changed(Some(a), Some(a)), "same");
        assert!(exe_changed(Some(a), Some(id(1, 11, 100))), "size");
        assert!(exe_changed(Some(a), Some(id(1, 10, 101))), "mtime");
        assert!(exe_changed(Some(a), Some(id(2, 10, 100))), "ino");
        assert!(exe_changed(Some(a), None), "missing");
        assert!(!exe_changed(None, None), "never knew it");
        assert!(!exe_changed(None, Some(a)), "never knew it");
    }

    #[test]
    fn retire_only_from_strictly_newer_build() {
        assert!(retire_allowed("0.99.0", "0.68.0"));
        assert!(!retire_allowed("0.68.0", "0.68.0"));
        assert!(!retire_allowed("0.67.0", "0.68.0"));
        assert!(!retire_allowed("junk", "0.68.0"));
    }

    #[test]
    fn hello_table() {
        assert_eq!(hello_verdict(PROTO, "0.1.0", "0.68.0"), Verdict::Welcome);
        assert_eq!(hello_verdict(0, "0.1.0", "0.68.0"), Verdict::Refused("proto-too-old"));
        assert_eq!(hello_verdict(PROTO + 1, "0.1.0", "0.68.0"), Verdict::Refused("hub-older"));
        assert_eq!(hello_verdict(PROTO + 1, "0.99.0", "0.68.0"), Verdict::RefusedMayRetire);
        assert_eq!(hello_verdict(PROTO, "", "0.68.0"), Verdict::Refused("bad-build"));
    }

    #[test]
    fn rss_policy_table() {
        let limit = 1000;
        assert_eq!(rss_policy(true, true, Some(2000), limit), Rss::Release);
        assert_eq!(rss_policy(false, true, Some(2000), limit), Rss::Release);
        assert_eq!(rss_policy(false, false, Some(2000), limit), Rss::Retire);
        assert_eq!(rss_policy(true, false, Some(2000), limit), Rss::Keep, "busy with nothing to release");
        assert_eq!(rss_policy(false, true, Some(900), limit), Rss::Keep);
        assert_eq!(rss_policy(false, true, None, limit), Rss::Keep, "unknown is not zero and not a crash");
        let day = Duration::from_secs(86_400);
        assert!(age_due(day, day, false));
        assert!(!age_due(day, day, true));
        assert!(!age_due(day / 2, day, false));
    }

    #[test]
    fn watchdog_aborts_after_2x_deadline() {
        let now = Instant::now();
        let since = now.checked_sub(Duration::from_secs(5)).unwrap();
        let window = Duration::from_secs(2);
        assert!(watchdog_due(Some((since, window)), now, Duration::ZERO), "5 s > 2 x 2 s");
        assert!(!watchdog_due(Some((since, window)), now, Duration::from_secs(30)), "the floor holds it");
        assert!(!watchdog_due(Some((since, Duration::from_secs(3))), now, Duration::ZERO), "5 s < 2 x 3 s");
        assert!(!watchdog_due(None, now, Duration::ZERO));
    }

    fn job(id: u64, deadline: Instant) -> Job {
        let (reply, _) = mpsc::channel();
        Job { id, hash: 0, query: String::new(), entries: Vec::new(), deadline, window: Duration::ZERO, started: Arc::default(), reply }
    }

    #[test]
    fn fifo_order_and_busy_after_deadline() {
        let now = Instant::now();
        let mut queue = VecDeque::new();
        queue.push_back(job(1, now + Duration::from_secs(5)));
        queue.push_back(job(2, now - Duration::from_millis(1)));
        queue.push_back(job(3, now + Duration::from_secs(5)));
        assert!(matches!(take_next(&mut queue, now), Some(Taken::Run(j)) if j.id == 1));
        assert!(matches!(take_next(&mut queue, now), Some(Taken::Expired(j)) if j.id == 2));
        assert!(matches!(take_next(&mut queue, now), Some(Taken::Run(j)) if j.id == 3));
        assert!(take_next(&mut queue, now).is_none());
    }

    #[test]
    fn conn_cap_refuses_65th() {
        let slots = Slots::new(MAX_CONNS);
        assert!((0..MAX_CONNS).all(|_| slots.acquire()));
        assert!(!slots.acquire(), "the 65th is refused");
        slots.release();
        assert!(slots.acquire());
    }

    #[test]
    fn sessions_count_distinct_pids_in_the_window_and_stay_capped() {
        let t0 = Instant::now();
        let mut s = Sessions::default();
        s.note(5, t0);
        s.note(5, t0 + Duration::from_secs(1));
        s.note(6, t0 + Duration::from_secs(2));
        assert_eq!(s.count(t0 + Duration::from_secs(3)), 2);
        assert_eq!(s.count(t0 + SESSION_WINDOW + Duration::from_secs(1)), 1, "5 was last seen at +1s");
        assert_eq!(s.count(t0 + SESSION_WINDOW + Duration::from_secs(3)), 0);
        let mut s = Sessions::default();
        for pid in 0..u32::try_from(MAX_SESSIONS * 3).unwrap() {
            s.note(pid, t0);
        }
        assert_eq!(s.count(t0), u32::try_from(MAX_SESSIONS).unwrap());
    }

    #[test]
    fn a_probing_client_does_not_beat_a_starting_hub_to_the_lock() {
        let dir = std::env::temp_dir().join(format!("hub-lockretry-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(endpoint::LOCK_FILE);
        let open = || fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).unwrap();
        let probe = open();
        probe.try_lock_shared().unwrap();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            drop(probe);
        });
        assert!(lock_with_retry(&open(), 20, Duration::from_millis(10)).unwrap(), "won once the probe let go");
        releaser.join().unwrap();
        let holder = open();
        holder.try_lock().unwrap();
        assert!(!lock_with_retry(&open(), 3, Duration::from_millis(1)).unwrap(), "a real holder still wins");
    }

    #[test]
    fn tick_is_due_on_its_own_clock_whatever_accept_does() {
        let t0 = Instant::now();
        assert!(!tick_due(t0, t0 + Duration::from_millis(5)));
        assert!(tick_due(t0, t0 + TICK_EVERY));
        // A connection every 5 ms for a second still ticks 50 times.
        let (mut last, mut ticks) = (t0, 0);
        for i in 1..=200 {
            let now = t0 + Duration::from_millis(5 * i);
            if tick_due(last, now) {
                last = now;
                ticks += 1;
            }
        }
        assert_eq!(ticks, 50);
    }

    #[test]
    fn handshake_timeout_drops_conn() {
        let (server, _silent_client) = UnixStream::pair().unwrap();
        server.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
        let mut reader = BufReader::new(server.try_clone().unwrap());
        let mut writer = server;
        assert_eq!(greet(&mut reader, &mut writer, 1), Greeted::Closed);
    }

    #[test]
    fn greet_answers_hello_and_refuses_the_rest() {
        let hello = |proto: u32| format!("{{\"type\":\"hello\",\"proto\":{proto},\"build\":\"0.1.0\"}}\n");
        let mut out = Vec::new();
        assert_eq!(greet(&mut Cursor::new(hello(PROTO)), &mut out, 7), Greeted::Ready("0.1.0".to_owned(), None));
        let welcome: Reply = serde_json::from_slice(&out).unwrap();
        assert_eq!(welcome, Reply::Welcome { proto: PROTO, build: build().to_owned(), pid: 7 });
        let mut out = Vec::new();
        assert_eq!(greet(&mut Cursor::new(hello(0)), &mut out, 7), Greeted::Closed);
        assert!(matches!(serde_json::from_slice::<Reply>(&out), Ok(Reply::Refused { .. })));
        assert_eq!(greet(&mut Cursor::new(b"{\"type\":\"status\"}\n".to_vec()), &mut Vec::new(), 7), Greeted::Closed);
    }

    #[test]
    fn inflight_marker_blocks_repeat_poison() {
        let dir = std::env::temp_dir().join(format!("hub-poison-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let hash = request_hash("POISON", &["a".into(), "b".into()]);
        let first = Poison::recover(&dir);
        assert!(!first.blocked(hash));
        first.mark(hash);
        // The process died with the marker on disk: one crash on the next start.
        let second = Poison::recover(&dir);
        assert!(!second.blocked(hash), "one crash is not yet a ban");
        second.mark(hash);
        let third = Poison::recover(&dir);
        assert!(third.blocked(hash), "two crashes are");
        assert!(!third.blocked(request_hash("fine", &["a".into(), "b".into()])));
        // A request that finished leaves nothing behind.
        third.mark(1);
        third.clear();
        assert!(!Poison::recover(&dir).blocked(1));
    }

    #[test]
    fn the_os_lock_dies_with_its_holder() {
        let dir = std::env::temp_dir().join(format!("hub-lockloser-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(endpoint::LOCK_FILE);
        let open = || fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).unwrap();
        let holder = open();
        holder.try_lock().unwrap();
        let loser = open();
        assert!(matches!(loser.try_lock(), Err(fs::TryLockError::WouldBlock)));
        drop(holder);
        assert!(loser.try_lock().is_ok(), "the OS drops the lock with its holder");
    }
}
