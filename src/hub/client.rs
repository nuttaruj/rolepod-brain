//! The client of the hub: what a brain process does to get one rerank answered
//! by the shared model.
//!
//! Three promises, each with a test:
//!
//! - Nothing is sent before the endpoint has been judged safe (directory,
//!   socket, lock held, and the pid in `Welcome` the pid in the lock). A hub
//!   that is not there, not safe or not answering costs the caller its
//!   deadline at most, and the answer is [`HubOrder::Down`] - never a model
//!   call from here.
//! - A reply is data. An `Order` is used only when its id is the request's
//!   and every index is in range and unique.
//! - Starting a hub is rare and bounded: a marker file lets one client of
//!   many spawn, and hubs that die after they were ready lock spawning out for
//!   a while.
//!
//! State lives in the data dir, not in a process, because the processes come
//! and go: `hub.spawn-*` (spawn markers), `hub.ready` (the pid of the last hub
//! a client saw answer), `hub.deaths` (one `pid unix_ms` line per hub that
//! died after it was ready) and `hub.wedged` (`pid unix_ms` of a hub that said
//! hello and then did not answer a request: for a minute, searches skip it at
//! once instead of each waiting out the deadline; an answer clears it).

use std::fs;
use std::io::{BufReader, ErrorKind, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::endpoint::{self, Endpoint};
use super::reason;
use super::proto::{self, LineError, Reply, Request, PROTO};
use crate::config::{Config, Paths};

/// A spawn marker is good for this long.
const SPAWN_WINDOW_SECS: u64 = 10;
/// How long a client that did not get an answer yet waits for a hub it knows
/// is starting, at most, and always inside its own deadline.
const WAIT_FOR_HUB: Duration = Duration::from_secs(3);
const POLL: Duration = Duration::from_millis(25);
/// A hub that does not say `Welcome` this fast is wedged.
const HELLO_WAIT: Duration = Duration::from_secs(1);
/// The deadline bounds when the hub starts on a request. Past it the hub
/// answers `Busy`, and a scoring that began just before it still has to
/// finish; this is the room both get to arrive in (11 s + 3 s = 14 s worst case).
const REPLY_SLACK: Duration = Duration::from_secs(3);
/// Abnormal deaths this close together lock spawning out.
const DEATH_WINDOW_MS: u64 = 10 * 60 * 1000;
const DEATHS_TO_LOCK: usize = 3;
const LOCKOUT_BASE_MS: u64 = 2 * 60 * 1000;
const LOCKOUT_CAP_MS: u64 = 10 * 60 * 1000;
const MAX_DEATH_RECORDS: usize = 16;
/// A hub that went silent after hello is skipped for this long.
const WEDGED_MS: u64 = 60_000;
/// Connect threads that may be blocked at once; past it a connect is not tried.
const MAX_CONNECT_THREADS: usize = 4;

pub const READY_FILE: &str = "hub.ready";
pub const DEATHS_FILE: &str = "hub.deaths";
pub const WEDGED_FILE: &str = "hub.wedged";
pub const SPAWN_PREFIX: &str = "hub.spawn-";

/// Is the hub used at all?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubMode {
    On,
    Off,
}

/// What one hub rerank came to. Never a model call from this process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubOrder {
    /// Indexes into the entries, best first. Unique and in range; may be a
    /// subset, the caller keeps the rest in their own order.
    Ranked(Vec<usize>),
    /// The hub is working through other requests; the caller keeps its order.
    Busy,
    /// No answer, and why: one of the [`reason`]s.
    Down(&'static str),
}

/// What a running hub reports about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubStatus {
    pub pid: u32,
    pub build: String,
    pub proto: u32,
    pub uptime_ms: u64,
    pub footprint_kb: Option<u64>,
    pub clients: usize,
    pub sessions: u32,
    pub queued: usize,
    pub model_loaded: bool,
    pub retiring: bool,
}

#[must_use]
pub fn mode(paths: &Paths) -> HubMode {
    mode_of(&Config::load(&paths.config_file()).unwrap_or_default())
}

/// [`mode`] for a caller that has the config already.
#[must_use]
pub fn mode_of(config: &Config) -> HubMode {
    if config.hub.enabled() {
        HubMode::On
    } else {
        HubMode::Off
    }
}

// ---------------------------------------------------------------------------
// Pure judgements: each a table in the tests.

/// Is a hub's order something to apply? Non-empty, at most `n` long, every
/// index below `n`, none twice. Nothing from the wire is used to index
/// before this passes.
#[must_use]
pub fn valid_order(indices: &[usize], n: usize) -> bool {
    if indices.is_empty() || indices.len() > n {
        return false;
    }
    let mut seen = vec![false; n];
    for &index in indices {
        match seen.get_mut(index) {
            Some(slot) if !*slot => *slot = true,
            _ => return false,
        }
    }
    true
}

/// What a reply to the rerank request `id` over `n` entries means. Any reply
/// that is not the one asked for is a protocol violation, and says unsafe.
#[must_use]
pub fn judge(reply: &Reply, id: u64, n: usize) -> HubOrder {
    match reply {
        Reply::Order { id: got, indices } if *got == id => {
            if valid_order(indices, n) {
                HubOrder::Ranked(indices.clone())
            } else {
                HubOrder::Down(reason::UNSAFE)
            }
        }
        Reply::Busy { id: got } if *got == id => HubOrder::Busy,
        Reply::Unavailable { id: got } if *got == id => HubOrder::Down(reason::DOWN),
        _ => HubOrder::Down(reason::UNSAFE),
    }
}

/// Is the pid the hub said the pid that holds the lock? Only then is it the
/// hub, and only then does a query leave this process.
#[must_use]
pub fn welcome_matches_lock(said: u32, holder: u32) -> bool {
    said == holder
}

/// Is `hub_build` strictly older than `ours`? Then, and only then, the hub is
/// asked to retire.
#[must_use]
pub fn should_retire(ours: &str, hub_build: &str) -> bool {
    proto::newer_build(ours, hub_build)
}

/// A hub that was ready and is now gone without a trace of a clean exit: the
/// lock is free and its socket is still on disk. A clean exit unlinks the
/// socket; a lost lock, a missing executable and an exec failure never got
/// ready. Returns the pid to charge.
#[must_use]
pub fn abnormal_death(ready: Option<u32>, holder: Option<u32>, socket_left: bool) -> Option<u32> {
    if holder.is_some() || !socket_left {
        return None;
    }
    ready
}

/// Until when is spawning locked out? Three deaths inside ten minutes lock it
/// for two minutes after the latest; each further death doubles that, to ten
/// minutes at most. `None` when not locked at `now_ms`.
#[must_use]
pub fn lockout_until(deaths_ms: &[u64], now_ms: u64) -> Option<u64> {
    let recent: Vec<u64> = deaths_ms.iter().copied().filter(|d| d.saturating_add(DEATH_WINDOW_MS) > now_ms).collect();
    if recent.len() < DEATHS_TO_LOCK {
        return None;
    }
    let latest = recent.iter().copied().max()?;
    let doublings = u32::try_from(recent.len() - DEATHS_TO_LOCK).unwrap_or(u32::MAX).min(16);
    let span = LOCKOUT_BASE_MS.saturating_mul(1 << doublings).min(LOCKOUT_CAP_MS);
    let until = latest.saturating_add(span);
    (until > now_ms).then_some(until)
}

// ---------------------------------------------------------------------------
// State in the data dir.

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn read_deaths(data_dir: &Path) -> Vec<(u32, u64)> {
    fs::read_to_string(data_dir.join(DEATHS_FILE))
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (pid, ms) = line.split_once(' ')?;
            Some((pid.parse().ok()?, ms.parse().ok()?))
        })
        .collect()
}

fn read_ready(data_dir: &Path) -> Option<u32> {
    fs::read_to_string(data_dir.join(READY_FILE)).ok()?.trim().parse().ok()
}

/// A client saw this hub answer. Written only when it changes.
fn note_ready(data_dir: &Path, pid: u32) {
    if read_ready(data_dir) != Some(pid) {
        let _ = fs::write(data_dir.join(READY_FILE), pid.to_string());
    }
}

/// Charge a hub that died after it was ready, once.
fn note_death(data_dir: &Path, pid: u32, at_ms: u64) {
    let mut deaths = read_deaths(data_dir);
    if !deaths.iter().any(|(seen, _)| *seen == pid) {
        deaths.push((pid, at_ms));
        let from = deaths.len().saturating_sub(MAX_DEATH_RECORDS);
        let text: String = deaths[from..].iter().map(|(p, ms)| format!("{p} {ms}\n")).collect();
        let _ = fs::write(data_dir.join(DEATHS_FILE), text);
    }
    let _ = fs::remove_file(data_dir.join(READY_FILE));
}

/// A hub that said hello and then did not answer a request, once.
fn note_wedged(data_dir: &Path, pid: u32, at_ms: u64) {
    let _ = fs::write(data_dir.join(WEDGED_FILE), format!("{pid} {at_ms}\n"));
}

fn clear_wedged(data_dir: &Path) {
    let _ = fs::remove_file(data_dir.join(WEDGED_FILE));
}

/// Is `pid` the hub last found wedged, less than a minute ago?
fn is_wedged(data_dir: &Path, pid: u32, now_ms: u64) -> bool {
    let Ok(text) = fs::read_to_string(data_dir.join(WEDGED_FILE)) else { return false };
    let Some((seen, at)) = text.trim().split_once(' ') else { return false };
    seen.parse::<u32>() == Ok(pid) && at.parse::<u64>().is_ok_and(|at| at.saturating_add(WEDGED_MS) > now_ms)
}

/// How much longer spawning is locked out, if it is.
#[must_use]
pub fn lockout_remaining(paths: &Paths) -> Option<Duration> {
    let now = now_ms();
    let deaths: Vec<u64> = read_deaths(&paths.data_dir).into_iter().map(|(_, ms)| ms).collect();
    lockout_until(&deaths, now).map(|until| Duration::from_millis(until - now))
}

/// Hubs that died abnormally within the last `window_ms`.
#[must_use]
pub fn deaths_within(paths: &Paths, window_ms: u64) -> usize {
    let now = now_ms();
    read_deaths(&paths.data_dir).iter().filter(|(_, at)| at.saturating_add(window_ms) > now).count()
}

/// Take the right to spawn a hub for this window. One winner among any
/// number of clients: `create_new` on a name made of the ten-second window
/// and the generation of the socket (a hub that died leaves a socket with a
/// new mtime, so its successor is not shut out by its own predecessor's
/// marker). Old markers are swept.
fn claim(data_dir: &Path, generation: u64, now_secs: u64) -> bool {
    let window = now_secs / SPAWN_WINDOW_SECS;
    if let Ok(entries) = fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let old = name
                .strip_prefix(SPAWN_PREFIX)
                .and_then(|rest| rest.split('-').next())
                .and_then(|w| w.parse::<u64>().ok())
                .is_some_and(|w| w + 6 < window);
            if old {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    let marker = data_dir.join(format!("{SPAWN_PREFIX}{window}-{generation}"));
    fs::OpenOptions::new().write(true).create_new(true).open(marker).is_ok()
}

fn generation(ep: &Endpoint) -> u64 {
    fs::symlink_metadata(&ep.socket).map_or(0, |meta| u64::try_from(meta.mtime()).unwrap_or(0))
}

/// The executable a hub is started from, when it can be started at all.
/// A missing or non-executable file is not a crash and is never counted.
fn spawn_exe() -> Option<PathBuf> {
    #[cfg(debug_assertions)]
    let exe = std::env::var_os("ROLEPOD_BRAIN_HUB_SPAWN_EXE").map(PathBuf::from).or_else(|| std::env::current_exe().ok());
    #[cfg(not(debug_assertions))]
    let exe = std::env::current_exe().ok();
    let exe = exe?;
    let meta = fs::metadata(&exe).ok()?;
    (meta.is_file() && meta.mode() & 0o111 != 0).then_some(exe)
}

/// The one place this process starts another: `brain hub serve`, detached
/// (its own process group, no stdio), inheriting the environment so it finds
/// the same data dir.
fn start_hub_process(exe: &Path) -> bool {
    let child = Command::new(exe)
        .args(["hub", "serve"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn();
    match child {
        Ok(mut child) => {
            // Reaped when it exits, so a long session leaves no zombie behind.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            true
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Talking to a hub.

/// A connection that has said hello.
pub(super) struct Conn {
    pub reader: BufReader<UnixStream>,
    pub writer: UnixStream,
    pub pid: u32,
    /// The hub is an older build than this one.
    pub older: bool,
    /// The hub welcomed us (false: it refused as older and waits for `Retire`).
    pub usable: bool,
}

pub(super) enum Probe {
    /// No hub. `stale`: a socket was left behind with nobody holding the lock.
    Absent { stale: bool },
    Unsafe(&'static str),
    Down,
    Hub(Conn),
}

enum ReadFail {
    /// The read timeout ran out.
    TimedOut,
    /// Closed or reset.
    Down,
    /// Too long, not text, not a reply.
    Hostile,
}

fn read_reply(reader: &mut BufReader<UnixStream>) -> Result<Reply, ReadFail> {
    match proto::read_line(reader) {
        Ok(Some(line)) => serde_json::from_str(&line).map_err(|_| ReadFail::Hostile),
        Ok(None) | Err(LineError::Io) => Err(ReadFail::Down),
        Err(LineError::TimedOut) => Err(ReadFail::TimedOut),
        Err(LineError::TooLong | LineError::NotUtf8) => Err(ReadFail::Hostile),
    }
}

/// What is left of `until` at `now`, as a socket timeout: never zero, which
/// the OS reads as "no timeout".
fn left(until: Instant, now: Instant) -> Duration {
    until.saturating_duration_since(now).max(Duration::from_millis(1))
}

/// Judge the endpoint, then talk to what is behind it. Nothing is written to
/// the socket until the directory, the socket, the lock and the pid agree.
///
/// Every wait inside is cut to what is left of `until`; `hello_wait` caps the
/// wait for `Welcome` alone. `who` is the pid the hub counts as a session; a
/// probe that only looks (status, doctor) passes `None` and is not one.
pub(super) fn probe(ep: &Endpoint, hello_wait: Duration, until: Instant, who: Option<u32>) -> Probe {
    match fs::symlink_metadata(&ep.dir) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Probe::Absent { stale: false },
        Err(_) => return Probe::Unsafe("dir-unreadable"),
        Ok(_) => {}
    }
    if let Err(unsafe_dir) = endpoint::check_dir(&ep.dir, ep.euid) {
        return Probe::Unsafe(unsafe_dir.0);
    }
    match fs::symlink_metadata(&ep.socket) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Probe::Absent { stale: false },
        Err(_) => return Probe::Unsafe("socket-unreadable"),
        Ok(_) => {}
    }
    if let Err(unsafe_path) = ep.verify() {
        return Probe::Unsafe(unsafe_path.0);
    }
    let Some(holder) = ep.holder_pid() else { return Probe::Absent { stale: true } };
    let Some(stream) = connect_within(&ep.socket, CONNECT_WAIT.min(left(until, Instant::now()))) else { return Probe::Down };
    let hello_wait = hello_wait.min(left(until, Instant::now()));
    let write_wait = Duration::from_secs(2).min(left(until, Instant::now()));
    if stream.set_read_timeout(Some(hello_wait)).is_err() || stream.set_write_timeout(Some(write_wait)).is_err() {
        return Probe::Down;
    }
    let (Ok(clone), writer) = (stream.try_clone(), stream) else { return Probe::Down };
    let mut conn = Conn { reader: BufReader::new(clone), writer, pid: holder, older: false, usable: true };
    let ours = env!("CARGO_PKG_VERSION");
    let hello = Request::Hello { proto: PROTO, build: ours.to_owned(), client: who };
    if proto::write_line(&mut conn.writer, &hello).is_err() {
        return Probe::Down;
    }
    match read_reply(&mut conn.reader) {
        Ok(Reply::Welcome { pid, build, .. }) => {
            if !welcome_matches_lock(pid, holder) {
                return Probe::Unsafe("pid-mismatch");
            }
            conn.older = should_retire(ours, &build);
            Probe::Hub(conn)
        }
        Ok(Reply::Refused { reason: said }) if said == reason::OLDER => {
            conn.older = true;
            conn.usable = false;
            Probe::Hub(conn)
        }
        Ok(Reply::Refused { .. }) | Err(ReadFail::Down | ReadFail::TimedOut) => Probe::Down,
        Ok(_) | Err(ReadFail::Hostile) => Probe::Unsafe("bad-reply"),
    }
}

/// How long a connect may take before the hub counts as down.
const CONNECT_WAIT: Duration = Duration::from_millis(200);

/// Connect threads still blocked, process-wide.
static CONNECT_THREADS: AtomicUsize = AtomicUsize::new(0);

/// One of the few connect threads that may exist; frees its place on drop.
struct ThreadSlot<'a>(&'a AtomicUsize);

impl<'a> ThreadSlot<'a> {
    fn take(count: &'a AtomicUsize, max: usize) -> Option<Self> {
        if count.fetch_add(1, Ordering::AcqRel) >= max {
            count.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(Self(count))
    }
}

impl Drop for ThreadSlot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// `connect` has no timeout in std, and on Linux it blocks while a wedged
/// hub's backlog is full. A helper thread carries the call; if it is still
/// blocked at the bound the search goes on without it (the thread ends when
/// the connect does). No more than [`MAX_CONNECT_THREADS`] are ever blocked at
/// once: past that, or when a thread cannot be made, there is no connect.
fn connect_within(socket: &Path, wait: Duration) -> Option<UnixStream> {
    let slot = ThreadSlot::take(&CONNECT_THREADS, MAX_CONNECT_THREADS)?;
    let (tx, rx) = mpsc::channel();
    let socket = socket.to_path_buf();
    std::thread::Builder::new()
        .name("brain-hub-connect".into())
        .spawn(move || {
            let _slot = slot;
            let _ = tx.send(UnixStream::connect(socket));
        })
        .ok()?;
    rx.recv_timeout(wait).ok()?.ok()
}

/// Ask an older hub to finish and go. Best effort: what matters is that the
/// next probe finds it gone.
fn retire(mut conn: Conn, until: Instant) {
    if proto::write_line(&mut conn.writer, &Request::Retire).is_ok() {
        let _ = conn.writer.set_read_timeout(Some(Duration::from_millis(500).min(left(until, Instant::now()))));
        let _ = read_reply(&mut conn.reader);
    }
}

fn next_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let salt = u64::from(std::process::id()) << 32;
    salt ^ COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Ask the hub one rerank. `until` ends the whole search (connect and hello
/// included); the write and the read each get what is left of it.
fn ask(mut conn: Conn, data_dir: &Path, query: &str, entries: &[String], remaining: Duration, until: Instant) -> HubOrder {
    let id = next_id();
    let deadline_ms = u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX).max(1);
    let request = Request::Rerank { id, query: query.to_owned(), entries: entries.to_vec(), deadline_ms };
    if conn.writer.set_write_timeout(Some(left(until, Instant::now()))).is_err() || proto::write_line(&mut conn.writer, &request).is_err() {
        return HubOrder::Down(reason::DOWN);
    }
    if conn.writer.set_read_timeout(Some(left(until, Instant::now()))).is_err() {
        return HubOrder::Down(reason::DOWN);
    }
    match read_reply(&mut conn.reader) {
        Ok(reply) => {
            let order = judge(&reply, id, entries.len());
            if matches!(order, HubOrder::Ranked(_) | HubOrder::Busy) {
                clear_wedged(data_dir);
            }
            order
        }
        Err(ReadFail::TimedOut) => {
            note_wedged(data_dir, conn.pid, now_ms());
            HubOrder::Down(reason::DOWN)
        }
        Err(ReadFail::Down) => HubOrder::Down(reason::DOWN),
        Err(ReadFail::Hostile) => HubOrder::Down(reason::UNSAFE),
    }
}

/// Rerank `entries` for `query` on the shared hub, within `deadline`.
///
/// The caller has decided the hub is on (see [`mode`]); nothing is read from
/// the config here.
///
/// Starts a hub when there is none (see the module doc), waits for it for a
/// few seconds at most and always inside the deadline, and otherwise answers
/// at once. Nothing here loads a model or asks a vendor CLI.
#[must_use]
pub fn rerank(paths: &Paths, query: &str, entries: &[String], deadline: Duration) -> HubOrder {
    if !proto::within_caps(query, entries) {
        return HubOrder::Down(reason::DOWN);
    }
    let Ok(ep) = endpoint::resolve(&paths.data_dir) else { return HubOrder::Down(reason::UNSAFE) };
    let started = Instant::now();
    let remaining = || deadline.saturating_sub(started.elapsed());
    // One bound for the whole search: connect, hello, request and reply.
    let until = started + deadline + REPLY_SLACK;
    let mut wait_until: Option<Instant> = None;
    let mut retired = false;
    loop {
        if remaining().is_zero() {
            return HubOrder::Down(reason::STARTING);
        }
        match probe(&ep, HELLO_WAIT.min(remaining()), until, Some(std::process::id())) {
            Probe::Unsafe(_) => return HubOrder::Down(reason::UNSAFE),
            Probe::Down => return HubOrder::Down(reason::DOWN),
            Probe::Hub(conn) if conn.older => {
                if retired {
                    return HubOrder::Down(reason::STARTING);
                }
                retired = true;
                retire(conn, until);
            }
            Probe::Hub(conn) => {
                if is_wedged(&paths.data_dir, conn.pid, now_ms()) {
                    return HubOrder::Down(reason::DOWN);
                }
                note_ready(&paths.data_dir, conn.pid);
                return ask(conn, &paths.data_dir, query, entries, remaining(), until);
            }
            Probe::Absent { stale } => {
                if stale {
                    let ready = read_ready(&paths.data_dir);
                    if let Some(pid) = abnormal_death(ready, ep.holder_pid(), true) {
                        note_death(&paths.data_dir, pid, now_ms());
                    }
                }
                if wait_until.is_none() {
                    if lockout_remaining(paths).is_some() {
                        return HubOrder::Down(reason::DOWN);
                    }
                    let Some(exe) = spawn_exe() else { return HubOrder::Down(reason::DOWN) };
                    let now_secs = now_ms() / 1000;
                    if claim(&paths.data_dir, generation(&ep), now_secs) {
                        start_hub_process(&exe);
                    }
                    wait_until = Some(Instant::now() + WAIT_FOR_HUB.min(remaining()));
                }
                if wait_until.is_some_and(|until| Instant::now() >= until) {
                    return HubOrder::Down(reason::STARTING);
                }
                std::thread::sleep(POLL);
            }
        }
    }
}

/// What the running hub says about itself, or `None` when
/// not running, not safe or not answering. Starts nothing.
#[must_use]
pub fn status(paths: &Paths) -> Option<HubStatus> {
    let ep = endpoint::resolve(&paths.data_dir).ok()?;
    let Probe::Hub(mut conn) = probe(&ep, HELLO_WAIT, Instant::now() + CONNECT_WAIT + HELLO_WAIT, None) else { return None };
    if !conn.usable {
        return None;
    }
    proto::write_line(&mut conn.writer, &Request::Status).ok()?;
    match read_reply(&mut conn.reader).ok()? {
        Reply::Status { pid, build, proto, uptime_ms, footprint_kb, clients, sessions, queued, model_loaded, retiring } => {
            Some(HubStatus { pid, build, proto, uptime_ms, footprint_kb, clients, sessions, queued, model_loaded, retiring })
        }
        _ => None,
    }
}

/// Write one request on an open connection and read the reply; for the
/// operator's `status` and `stop`.
pub(super) fn exchange(conn: &mut Conn, request: &Request) -> Result<Reply, &'static str> {
    proto::write_line(&mut conn.writer, request).map_err(|_| "the hub did not take the request")?;
    let _ = conn.writer.flush();
    read_reply(&mut conn.reader).map_err(|_| "the hub sent nothing usable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn tmp(name: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = PathBuf::from("/tmp").join(format!("hubcl-{name}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn welcome_pid_mismatch_rejected() {
        assert!(welcome_matches_lock(42, 42));
        assert!(!welcome_matches_lock(42, 43));
    }

    #[test]
    fn order_valid_permutation_ok() {
        assert!(valid_order(&[2, 0, 1], 3));
        assert!(valid_order(&[1], 3), "a subset is fine, the caller keeps the rest");
        assert_eq!(judge(&Reply::Order { id: 9, indices: vec![1, 0] }, 9, 2), HubOrder::Ranked(vec![1, 0]));
    }

    #[test]
    fn order_out_of_range_rejected() {
        assert!(!valid_order(&[0, 3], 3));
        assert!(!valid_order(&[usize::MAX], 3));
        assert_eq!(judge(&Reply::Order { id: 9, indices: vec![0, 5] }, 9, 2), HubOrder::Down(reason::UNSAFE));
    }

    #[test]
    fn order_duplicate_index_rejected() {
        assert!(!valid_order(&[1, 1], 3));
        assert!(!valid_order(&[0, 1, 2, 0], 3));
        assert!(!valid_order(&[], 3));
        assert_eq!(judge(&Reply::Order { id: 9, indices: vec![0, 0] }, 9, 2), HubOrder::Down(reason::UNSAFE));
    }

    #[test]
    fn order_wrong_id_rejected() {
        assert_eq!(judge(&Reply::Order { id: 8, indices: vec![0, 1] }, 9, 2), HubOrder::Down(reason::UNSAFE));
        assert_eq!(judge(&Reply::Busy { id: 8 }, 9, 2), HubOrder::Down(reason::UNSAFE));
    }

    #[test]
    fn reply_variant_mismatch_rejected() {
        assert_eq!(judge(&Reply::Ack, 9, 2), HubOrder::Down(reason::UNSAFE));
        assert_eq!(judge(&Reply::Welcome { proto: 1, build: "x".into(), pid: 1 }, 9, 2), HubOrder::Down(reason::UNSAFE));
        assert_eq!(judge(&Reply::Busy { id: 9 }, 9, 2), HubOrder::Busy);
        assert_eq!(judge(&Reply::Unavailable { id: 9 }, 9, 2), HubOrder::Down(reason::DOWN));
    }

    #[test]
    fn retire_only_toward_an_older_hub() {
        assert!(should_retire("0.68.1", "0.68.0"));
        assert!(!should_retire("0.68.0", "0.68.0"));
        assert!(!should_retire("0.68.0", "0.68.1"));
        assert!(!should_retire("0.68.0", "garbage"));
    }

    #[test]
    fn marker_create_new_single_winner() {
        let dir = tmp("marker");
        let wins: usize = (0..16)
            .map(|_| {
                let dir = dir.clone();
                std::thread::spawn(move || claim(&dir, 7, 1_000))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| usize::from(t.join().unwrap()))
            .sum();
        assert_eq!(wins, 1);
        assert!(!claim(&dir, 7, 1_005), "same window, same generation: taken");
        assert!(claim(&dir, 8, 1_005), "a new socket generation is a new claim");
        assert!(claim(&dir, 7, 1_010), "ten seconds later the marker has expired");
        assert!(claim(&dir, 7, 1_100));
        let left = fs::read_dir(&dir).unwrap().count();
        assert!(left <= 3, "old markers are swept, {left} left");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn accounting_table() {
        // lost lock, ENOENT, exec failure: never ready, never charged.
        assert_eq!(abnormal_death(None, None, false), None);
        assert_eq!(abnormal_death(None, None, true), None);
        // ready, then gone with the socket left behind: charged.
        assert_eq!(abnormal_death(Some(5), None, true), Some(5));
        // clean exit unlinks the socket.
        assert_eq!(abnormal_death(Some(5), None, false), None);
        // still alive.
        assert_eq!(abnormal_death(Some(5), Some(5), true), None);
        let dir = tmp("deaths");
        note_death(&dir, 5, 100);
        note_death(&dir, 5, 200);
        assert_eq!(read_deaths(&dir), vec![(5, 100)], "one death per pid");
        for pid in 10..40 {
            note_death(&dir, pid, 300);
        }
        assert_eq!(read_deaths(&dir).len(), MAX_DEATH_RECORDS);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lockout_backoff_table() {
        let m = 60_000;
        let now = 1_000 * m;
        assert_eq!(lockout_until(&[], now), None);
        assert_eq!(lockout_until(&[now - m, now - m / 2], now), None, "two are not enough");
        assert_eq!(lockout_until(&[now - 3 * m, now - 2 * m, now - 30_000], now), Some(now - 30_000 + 2 * m));
        let four = [now - 4 * m, now - 3 * m, now - 2 * m, now - 30_000];
        assert_eq!(lockout_until(&four, now), Some(now - 30_000 + 4 * m));
        let six: Vec<u64> = (1..=6).map(|i| now - i * 1000).collect();
        assert_eq!(lockout_until(&six, now), Some(now - 1000 + 10 * m), "capped at ten minutes");
        let old = [now - 20 * m, now - 19 * m, now - 18 * m];
        assert_eq!(lockout_until(&old, now), None, "deaths age out");
        let expired = [now - 9 * m, now - 8 * m, now - 7 * m];
        assert_eq!(lockout_until(&expired, now), None, "the lockout ended two minutes after the latest");
    }

    #[test]
    fn mode_off_is_read_from_the_config_and_touches_nothing() {
        let dir = tmp("off");
        fs::write(dir.join("config.toml"), "[hub]\nmode = \"off\"\n").unwrap();
        let paths = Paths { data_dir: dir.clone() };
        assert_eq!(mode(&paths), HubMode::Off);
        assert_eq!(mode_of(&Config::load(&paths.config_file()).unwrap()), HubMode::Off);
        let names: Vec<_> = fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(names.len(), 1, "nothing but the config: {names:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn connect_threads_are_capped_and_freed() {
        let count = AtomicUsize::new(0);
        let slots: Vec<_> = (0..MAX_CONNECT_THREADS).map(|_| ThreadSlot::take(&count, MAX_CONNECT_THREADS).unwrap()).collect();
        assert!(ThreadSlot::take(&count, MAX_CONNECT_THREADS).is_none(), "the fifth is refused");
        assert_eq!(count.load(Ordering::Acquire), MAX_CONNECT_THREADS, "a refusal takes no place");
        drop(slots);
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert!(ThreadSlot::take(&count, MAX_CONNECT_THREADS).is_some());
    }

    #[test]
    fn every_timeout_is_what_is_left_of_one_deadline() {
        let start = Instant::now();
        let until = start + Duration::from_secs(14);
        assert_eq!(left(until, start), Duration::from_secs(14));
        assert_eq!(left(until, start + Duration::from_secs(13)), Duration::from_secs(1));
        assert_eq!(left(until, start + Duration::from_secs(14)), Duration::from_millis(1), "never zero");
        assert_eq!(left(until, start + Duration::from_secs(20)), Duration::from_millis(1), "never past");
    }

    #[test]
    fn a_wedged_mark_is_for_that_pid_for_a_minute() {
        let dir = tmp("wedged");
        assert!(!is_wedged(&dir, 7, 1_000));
        note_wedged(&dir, 7, 1_000);
        assert!(is_wedged(&dir, 7, 1_000 + WEDGED_MS - 1));
        assert!(!is_wedged(&dir, 7, 1_000 + WEDGED_MS), "a minute later it is tried again");
        assert!(!is_wedged(&dir, 8, 1_001), "another pid is another hub");
        clear_wedged(&dir);
        assert!(!is_wedged(&dir, 7, 1_001));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn over_cap_request_is_not_sent() {
        let dir = tmp("cap");
        fs::write(dir.join("config.toml"), "[hub]\nmode = \"on\"\n").unwrap();
        let paths = Paths { data_dir: dir.clone() };
        let entries = vec!["a".repeat(proto::MAX_ENTRY_BYTES + 1), "b".into()];
        assert_eq!(mode(&paths), HubMode::On, "unset ROLEPOD_BRAIN_HUB to run this test");
        assert_eq!(rerank(&paths, "q", &entries, Duration::from_secs(1)), HubOrder::Down(reason::DOWN));
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1, "no marker, no process");
        let _ = fs::remove_dir_all(&dir);
    }
}
