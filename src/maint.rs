//! Keeping the index file small without anyone asking.
//!
//! Dropped bodies leave holes a rewrite alone gives back. [`after_run`] is
//! called when a consolidation run has ended and let go of its lock; once the
//! index is due for a rewrite and the machine has been quiet, it opens a short
//! window: a marker file the hooks read, the rewrite, then a catch-up for what
//! a hook gave up on meanwhile. `brain compact` is the same code, run now.
//!
//! Three promises hold throughout. Nothing here writes the event log, or
//! `brain.log` (doctor counts every line there as a failure): a refusal is
//! written to `schema_state`, with its reason. The marker always expires on its
//! own, so a killed compactor cannot leave the hooks standing aside. And a
//! store that is not due pays one small read.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::config::Paths;
use crate::consolidate::{self, RunLock};
use crate::event::EventLog;
use crate::store::{CompactState, Ledger, Store, SurfacedLine};

/// How long a hook waits on the database while the window is open, before it
/// leaves its event to the catch-up.
pub const FAIL_FAST: Duration = Duration::from_millis(100);

/// How long the event logs must have been untouched before a rewrite.
pub(crate) const QUIET: Duration = Duration::from_secs(60);

const MARKER: &str = ".brain-maintenance";
const GATE: &str = ".brain-maintain.lock";
const YIELDED: &str = ".brain-maint-yielded";

/// The window needs this much of the file to be worth the rewrite...
const MIN_RECLAIM: u64 = 32 * 1024 * 1024;
/// ...and a try is not repeated before this many seconds.
const RETRY_AFTER_SECS: i64 = 24 * 3600;
/// Free disk required beyond two and a half copies of the index.
const DISK_SLACK: u64 = 1 << 30;
/// Doctor warns when the index has wanted a rewrite this long.
const OVERDUE_SECS: i64 = 14 * 24 * 3600;
/// A gate file older than this belongs to a dead compactor.
const GATE_STALE: Duration = Duration::from_secs(15 * 60);

/// The clocks of one window. [`Timing::live`] is the real one; tests shorten it.
#[derive(Debug, Clone, Copy)]
struct Timing {
    quiet: Duration,
    poll: Duration,
    /// How long from the run's start a wait for quiet may last.
    budget: Duration,
    heartbeat: Duration,
    /// How far ahead each heartbeat puts the marker's expiry.
    lifetime: Duration,
    /// How long between two tries to cut the log while a hook holds it.
    cut_wait: Duration,
}

impl Timing {
    fn live() -> Self {
        // A test binary never waits; an end-to-end test that wants the wait
        // sets the variable itself.
        let default = if cfg!(test) { 0 } else { 6 * 60 };
        let budget = std::env::var("ROLEPOD_BRAIN_MAINT_WAIT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default);
        // How long a killed compactor's marker may go on claiming a window: a
        // test that kills one sets it to seconds instead of waiting out 30.
        let lifetime = std::env::var("ROLEPOD_BRAIN_MAINT_LIFETIME_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(Duration::from_secs(30), Duration::from_millis);
        Self {
            quiet: QUIET,
            poll: Duration::from_secs(10),
            budget: Duration::from_secs(budget),
            heartbeat: (lifetime / 6).min(Duration::from_secs(5)),
            lifetime,
            cut_wait: Duration::from_secs(1),
        }
    }
}

impl CompactState {
    /// What a rewrite would give back, as a guess: the free pages, plus
    /// roughly 45% of the body bytes retention dropped since the last rewrite
    /// (their text-index entries are still there to fold away).
    pub fn reclaimable(&self) -> u64 {
        let unsettled = self.retention_dropped_bytes.saturating_sub(self.compact_dropped_bytes);
        let share = (unsettled / 100).saturating_mul(45).saturating_add(unsettled % 100 * 45 / 100);
        self.free_bytes.saturating_add(share)
    }

    /// Would a rewrite pay, whatever the last try did?
    pub fn wanted(&self) -> bool {
        if self.retention_pending {
            return false;
        }
        // A store upgraded from before the window has never compacted by
        // itself; retention having dropped anything is reason enough once.
        let first = self.done_at.is_none() && self.retention_dropped > 0;
        let guess = self.reclaimable();
        first || (guess >= MIN_RECLAIM && guess.saturating_mul(10) >= self.file_bytes)
    }

    /// `wanted`, and the last try is more than a day old.
    pub fn due(&self, now_unix: i64) -> bool {
        self.wanted() && self.tried_at.is_none_or(|at| now_unix.saturating_sub(at) >= RETRY_AFTER_SECS)
    }
}

/// What doctor shows about the index's space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactStatus {
    pub last: Option<String>,
    pub reclaimable: u64,
    pub wanted: bool,
    /// Wanted for longer than doctor lets pass quietly.
    pub overdue: bool,
    pub skip: Option<String>,
}

/// # Errors
/// Returns an error when the index cannot be read.
pub fn status(_paths: &Paths, store: &Store) -> Result<CompactStatus> {
    Ok(status_of(&store.compact_state()?, unix_now()))
}

fn status_of(state: &CompactState, now_unix: i64) -> CompactStatus {
    let wanted = state.wanted();
    let since = state.done_at.as_ref().or(state.retention_done_at.as_ref());
    let overdue = wanted
        && since
            .and_then(|at| at.parse::<jiff::Timestamp>().ok())
            .is_some_and(|at| now_unix.saturating_sub(at.as_second()) > OVERDUE_SECS);
    CompactStatus {
        last: state.done_at.clone(),
        reclaimable: state.reclaimable(),
        wanted,
        overdue,
        skip: state.skip.clone(),
    }
}

fn unix_now() -> i64 {
    jiff::Timestamp::now().as_second()
}

fn now_ms() -> u64 {
    u64::try_from(jiff::Timestamp::now().as_millisecond()).unwrap_or(0)
}

/// Is a compactor holding the window open? One open and at most 64 bytes read:
/// a hook asks this on its way past, so it never touches SQLite and never fails.
pub fn active(paths: &Paths) -> bool {
    active_at(&paths.data_dir, now_ms())
}

fn active_at(dir: &Path, now_ms: u64) -> bool {
    let Ok(file) = std::fs::File::open(dir.join(MARKER)) else { return false };
    let mut text = String::new();
    if file.take(64).read_to_string(&mut text).is_err() {
        return false;
    }
    text.split_whitespace().nth(1).and_then(|until| until.parse::<u64>().ok()).is_some_and(|until| until > now_ms)
}

const SURFACED: &str = "surfaced.jsonl";
const FOLDING: &str = "surfaced.jsonl.folding";
const WIPES: &str = "pending-wipe";

/// Remember ids a session was shown that the ledger could not take. One line
/// per call, opened, appended and closed (like `brain.log`), and no SQLite:
/// the database is exactly what was out of reach. Best effort; there is
/// nowhere left to report a failure to.
pub fn spill_surfaced(paths: &Paths, session: &str, kind: Ledger, ids: &[String]) {
    spill_line(paths, session, kind, ids, 0);
}

/// [`spill_surfaced`] for an injection, with the bytes it spent, so the fold
/// can give them back to the session's budget.
pub fn spill_injected(paths: &Paths, session: &str, ids: &[String], bytes: usize) {
    spill_line(paths, session, Ledger::Injected, ids, bytes);
}

fn spill_line(paths: &Paths, session: &str, kind: Ledger, ids: &[String], bytes: usize) {
    if ids.is_empty() {
        return;
    }
    let line = if bytes == 0 {
        serde_json::json!({ "session": session, "kind": kind_tag(kind), "ids": ids })
    } else {
        serde_json::json!({ "session": session, "kind": kind_tag(kind), "ids": ids, "bytes": bytes })
    };
    let _ = std::fs::create_dir_all(&paths.data_dir);
    if let Ok(mut file) =
        std::fs::OpenOptions::new().create(true).append(true).open(paths.data_dir.join(SURFACED))
    {
        let _ = std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes());
    }
}

fn kind_tag(kind: Ledger) -> &'static str {
    match kind {
        Ledger::Recalled => "recalled",
        Ledger::Opened => "opened",
        Ledger::Injected => "injected",
        Ledger::File => "file",
    }
}

fn parse_surfaced(text: &str) -> Vec<SurfacedLine> {
    text.lines()
        .filter_map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).ok()?;
            let kind = match value.get("kind")?.as_str()? {
                "recalled" => Ledger::Recalled,
                "opened" => Ledger::Opened,
                "injected" => Ledger::Injected,
                "file" => Ledger::File,
                _ => return None,
            };
            Some(SurfacedLine {
                session: value.get("session")?.as_str()?.to_string(),
                kind,
                ids: value
                    .get("ids")?
                    .as_array()?
                    .iter()
                    .filter_map(|id| id.as_str().map(str::to_string))
                    .collect(),
                bytes: value.get("bytes").and_then(serde_json::Value::as_u64).map_or(0, |n| usize::try_from(n).unwrap_or(0)),
            })
        })
        .collect()
}

/// Was this id spilled for this session and not yet folded back? What the
/// forget/correct/feedback guard asks after the ledger itself says no.
pub fn spilled(paths: &Paths, session: &str, id: &str) -> bool {
    [SURFACED, FOLDING].iter().any(|name| {
        std::fs::read_to_string(paths.data_dir.join(name)).is_ok_and(|text| {
            parse_surfaced(&text)
                .iter()
                .any(|line| {
                    line.session == session
                        && line.kind != Ledger::File
                        && line.ids.iter().any(|spilled| spilled == id)
                })
        })
    })
}

/// Fold one spill file and empty it. The file stays (emptied, not removed): a
/// hook that had it open before the rename can still append to it, and that
/// line is folded by the next fold. Emptying rather than replaying keeps a
/// fold from re-arming what a context wipe cleared in between. `false` means
/// the store refused a line and the file is left whole.
fn fold_and_empty(path: &Path, store: &Store) -> bool {
    let mut done = 0;
    // A line landing between the read and the emptying makes the length
    // differ; it is folded before the file is emptied.
    for _ in 0..4 {
        let Ok(bytes) = std::fs::read(path) else { return true };
        let text = String::from_utf8_lossy(bytes.get(done..).unwrap_or_default());
        if store.fold_surfaced(parse_surfaced(&text).into_iter()).is_err() {
            return false;
        }
        done = bytes.len();
        if done == 0 {
            return true;
        }
        if std::fs::metadata(path).is_ok_and(|meta| meta.len() == done as u64) {
            let _ = std::fs::OpenOptions::new().write(true).truncate(true).open(path);
            return true;
        }
    }
    true
}

/// Fold the spill back into the ledger. The file is renamed first so a hook
/// appending meanwhile starts a fresh one; the renamed `.folding` is folded
/// and emptied here, and again by the next fold for any late line. Folding is
/// idempotent for a line seen twice. Best effort: a fold that fails leaves the
/// files for the next run, and says so: `false` means something is still
/// unfolded.
pub fn fold_surfaced(paths: &Paths, store: &Store) -> bool {
    let spill = paths.data_dir.join(SURFACED);
    let folding = paths.data_dir.join(FOLDING);
    if !fold_and_empty(&folding, store) {
        return false;
    }
    if spill.exists() && std::fs::rename(&spill, &folding).is_err() {
        return false;
    }
    if !fold_and_empty(&folding, store) {
        return false;
    }
    // Whether anything is left unfolded: retention must not run while a body
    // that only the spill remembers having been shown could be dropped.
    !spill.exists()
}

/// The file name of a session's pending wipe: the session id with anything
/// that is not a safe file-name character replaced.
fn wipe_path(paths: &Paths, session: &str) -> PathBuf {
    let name: String = session
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    paths.data_dir.join(WIPES).join(name)
}

/// A context wipe the store could not take: leave a file for the next hook of
/// this session that can write.
pub fn pending_wipe(paths: &Paths, session: &str) {
    let path = wipe_path(paths, session);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, b"");
}

/// Is a wipe waiting for this session? One stat.
pub fn wipe_pending(paths: &Paths, session: &str) -> bool {
    wipe_path(paths, session).exists()
}

/// The wipe was applied; take the file away.
pub fn clear_pending_wipe(paths: &Paths, session: &str) {
    let _ = std::fs::remove_file(wipe_path(paths, session));
}

/// `pid until_unix_ms`, written beside the marker and renamed over it, so a
/// reader sees the old line or the new one and never half of either.
fn write_marker(dir: &Path, until_ms: u64) -> std::io::Result<()> {
    let tmp = dir.join(format!("{MARKER}.tmp"));
    std::fs::write(&tmp, format!("{} {until_ms}", std::process::id()))?;
    std::fs::rename(tmp, dir.join(MARKER))
}

/// The open window: a marker a thread keeps renewing for as long as this value
/// lives. The expiry is always near, so a process that dies takes the marker's
/// meaning with it within `lifetime`.
struct Window {
    dir: PathBuf,
    stop: Option<mpsc::Sender<()>>,
    beat: Option<std::thread::JoinHandle<()>>,
}

impl Window {
    fn open(dir: &Path, timing: &Timing) -> Result<Self> {
        let life = u64::try_from(timing.lifetime.as_millis()).unwrap_or(u64::MAX);
        write_marker(dir, now_ms().saturating_add(life)).context("open the maintenance window")?;
        let (stop, rx) = mpsc::channel::<()>();
        let (path, every) = (dir.to_path_buf(), timing.heartbeat);
        let beat = std::thread::spawn(move || {
            while let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(every) {
                let _ = write_marker(&path, now_ms().saturating_add(life));
            }
        });
        Ok(Self { dir: dir.to_path_buf(), stop: Some(stop), beat: Some(beat) })
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(beat) = self.beat.take() {
            let _ = beat.join();
        }
        let _ = std::fs::remove_file(self.dir.join(MARKER));
    }
}

/// One compactor at a time, across processes. A file made with `create_new`.
struct Gate(PathBuf);

impl Gate {
    fn take(dir: &Path, stale: Duration) -> Option<Self> {
        let path = dir.join(GATE);
        for _ in 0..2 {
            match std::fs::OpenOptions::new().create_new(true).write(true).open(&path) {
                Ok(_) => return Some(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let old = std::fs::metadata(&path)
                        .and_then(|meta| meta.modified())
                        .is_ok_and(|at| at.elapsed().is_ok_and(|age| age > stale));
                    if !old {
                        return None;
                    }
                    let _ = std::fs::remove_file(&path);
                }
                Err(_) => return None,
            }
        }
        None
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Wait {
    Quiet,
    NotQuiet,
    AskOpen,
}

/// Poll until the logs have been untouched for `timing.quiet`, giving up when
/// the budget would be spent or someone's ask is waiting. The first look is
/// immediate. `newest` is the age of the newest log write (`None`: no log).
fn wait_quiet(
    timing: &Timing,
    mut elapsed: Duration,
    newest: &mut dyn FnMut() -> Option<Duration>,
    ask_open: &mut dyn FnMut() -> bool,
    sleep: &mut dyn FnMut(Duration),
) -> Wait {
    loop {
        if ask_open() {
            return Wait::AskOpen;
        }
        if newest().is_none_or(|age| age >= timing.quiet) {
            return Wait::Quiet;
        }
        if elapsed.saturating_add(timing.poll) > timing.budget {
            return Wait::NotQuiet;
        }
        sleep(timing.poll);
        elapsed = elapsed.saturating_add(timing.poll);
    }
}

/// Room for a second copy of the index plus the log, and a gigabyte besides.
fn check_room(free: Option<u64>, index: u64) -> Result<(), String> {
    let Some(free) = free else { return Err("free-disk-unknown".to_string()) };
    let need = (index / 2).saturating_mul(5).saturating_add(DISK_SLACK);
    if free >= need {
        Ok(())
    } else {
        Err(format!("low-disk: {} MB free, {} MB needed", free >> 20, need >> 20))
    }
}

/// `Err(reason)` when the disk cannot take a rewrite.
pub(crate) fn guard(paths: &Paths) -> Result<(), String> {
    check_room(free_disk_bytes(&paths.data_dir), index_bytes(paths))
}

/// What a finished rewrite did.
#[derive(Debug)]
pub struct CompactReport {
    pub before: u64,
    pub after: u64,
    pub pages: i64,
    pub free_pages: i64,
    /// False when a reader kept the log from being cut; the next checkpoint will.
    pub truncated: bool,
    pub caught: usize,
}

/// Rewrite the index now, holding the run lock the caller took. The caller has
/// checked the quiet and the disk; this checks the disk once more before the
/// rewrite itself. On failure the log is cut back, the reason is recorded in
/// `schema_state`, and the error is returned.
///
/// # Errors
/// Returns an error when the index cannot be folded, rewritten or checkpointed.
pub fn compact_now(paths: &Paths, store: &Store, lock: &RunLock) -> Result<CompactReport> {
    compact_with(paths, store, lock, &Timing::live())
}

fn compact_with(paths: &Paths, store: &Store, lock: &RunLock, timing: &Timing) -> Result<CompactReport> {
    // SQLite sorts and rewrites through temp files; point them at the disk
    // that was checked, not at wherever `/tmp` is.
    std::env::set_var("SQLITE_TMPDIR", &paths.data_dir);
    let (pages, free_pages) = store.page_use()?;
    let state = store.compact_state()?;
    // Both sizes are the index's own pages, so a log that has grown does not
    // read as a saving.
    let before = state.file_bytes;
    let truncated = match rewrite(paths, store, lock, timing) {
        Ok(truncated) => truncated,
        Err(error) => {
            let _ = store.truncate_wal();
            let reason: String = format!("{error:#}").chars().take(160).collect();
            record_skip(store, &reason, true);
            return Err(error);
        }
    };
    // The rewritten index's own size: a reader can keep the log from being cut,
    // and then the file on disk and the log still hold the old pages.
    // The rewrite is done: a failed read here falls back, it never loses the record.
    let after = store.compact_state().map_or_else(|_| index_bytes(paths), |state| state.file_bytes);
    // A hook that met the rewrite and gave up left its event in the log only.
    let caught = consolidate::catch_up_all(paths, store, lock).unwrap_or(0);
    let done = CompactState {
        done_at: Some(jiff::Timestamp::now().to_string()),
        tried_at: Some(unix_now()),
        compact_dropped_bytes: state.retention_dropped_bytes,
        before_bytes: Some(before),
        after_bytes: Some(after),
        ..CompactState::default()
    };
    let _ = store.record_compact(&done);
    Ok(CompactReport { before, after, pages, free_pages, truncated, caught })
}

/// The window itself: fold, check the disk again, rewrite, cut the log. The
/// marker is gone when this returns, however it returns.
fn rewrite(paths: &Paths, store: &Store, lock: &RunLock, timing: &Timing) -> Result<bool> {
    let window = Window::open(&paths.data_dir, timing)?;
    store.optimize_text_indexes()?;
    lock.touch();
    guard(paths).map_err(|reason| anyhow::anyhow!(reason))?;
    store.vacuum()?;
    lock.touch();
    // A hook holding the database sends the checkpoint away; wait for it
    // one second at a time rather than blocking inside SQLite.
    store.set_busy_timeout(Duration::ZERO)?;
    let cut = (|| {
        for attempt in 0..10 {
            if store.truncate_wal()? {
                return Ok(true);
            }
            if attempt < 9 {
                std::thread::sleep(timing.cut_wait);
            }
        }
        Ok(false)
    })();
    // A hook the window held off may have let go by now: one more try, which
    // is not allowed to fail the run.
    drop(window);
    let cut = match cut {
        Ok(false) => Ok(cut_once(store)),
        other => other,
    };
    store.set_busy_timeout(Duration::from_secs(5))?;
    cut
}

/// One try to cut the log, at once; a failure is a `false`.
fn cut_once(store: &Store) -> bool {
    store.truncate_wal().unwrap_or(false)
}

/// Write down why a window did not open, and end the attempt quietly.
fn stand_aside(store: &Store, reason: &str, tried: bool) -> Result<()> {
    record_skip(store, reason, tried);
    Ok(())
}

fn record_skip(store: &Store, reason: &str, tried: bool) {
    let state = CompactState {
        skip: Some(format!("{reason}@{}", jiff::Timestamp::now())),
        tried_at: tried.then(unix_now),
        ..CompactState::default()
    };
    let _ = store.record_compact(&state);
}

/// Called by a consolidation run once it has ended and let go of its lock.
/// Best effort: nothing here fails the run, and nothing is written to
/// `brain.log`. A store that is not due costs one read.
pub fn after_run(paths: &Paths, started: Instant) {
    let _ = try_window(paths, started, &Timing::live(), &mut |_| {});
}

/// `between` runs once the run lock is held and before the second look at
/// whether the rewrite is still due: the gap a test needs to change the store.
fn try_window(
    paths: &Paths,
    started: Instant,
    timing: &Timing,
    between: &mut dyn FnMut(&Store),
) -> Result<()> {
    if !paths.db().is_file() {
        return Ok(());
    }
    let store = Store::open(&paths.db())?;
    if !store.compact_state()?.due(unix_now()) {
        return Ok(());
    }
    let Some(_gate) = Gate::take(&paths.data_dir, GATE_STALE) else { return Ok(()) };
    let waited = wait_quiet(
        timing,
        started.elapsed(),
        &mut || newest_log_write_age(paths).unwrap_or(Some(Duration::ZERO)),
        &mut || store.next_consolidation_request().is_ok_and(|ask| ask.is_some()),
        &mut std::thread::sleep,
    );
    match waited {
        Wait::Quiet => {}
        Wait::NotQuiet => return stand_aside(&store, "not-quiet", false),
        Wait::AskOpen => return stand_aside(&store, "ask-open", false),
    }
    let Some(lock) = RunLock::take(&consolidate::run_lock_path(paths))? else {
        return stand_aside(&store, "run-lock-busy", false);
    };
    // Held now: what was true while waiting may not be.
    between(&store);
    if !store.compact_state()?.due(unix_now()) {
        return stand_aside(&store, "no-longer-due", false);
    }
    if let Err(reason) = guard(paths) {
        return stand_aside(&store, &reason, true);
    }
    let result = compact_now(paths, &store, &lock);
    drop(lock);
    finish_yield(paths);
    result.map(drop)
}

/// A run that met the open window stood aside and left a file saying so. Now
/// the window is closed and the lock free: do what it would have done.
pub(crate) fn finish_yield(paths: &Paths) {
    let file = paths.data_dir.join(YIELDED);
    if file.exists() {
        let _ = std::fs::remove_file(&file);
        spawn_detached(&["consolidate", "--all"]);
    }
}

/// Run `brain <args>` as a child with no stdio, and do not wait for it.
/// Best-effort: a child that cannot start is simply not started.
fn spawn_detached(args: &[&str]) {
    let Ok(exe) = std::env::current_exe() else { return };
    let _ = std::process::Command::new(exe)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Say that a run stood aside for the window.
pub(crate) fn note_yield(paths: &Paths) {
    let _ = std::fs::write(paths.data_dir.join(YIELDED), "");
}

/// Bytes the index takes on disk: the database and its write-ahead log.
pub(crate) fn index_bytes(paths: &Paths) -> u64 {
    let db = paths.db();
    let mut wal = db.clone().into_os_string();
    wal.push("-wal");
    [db, wal.into()]
        .iter()
        .map(|file| std::fs::metadata(file).map_or(0, |meta| meta.len()))
        .sum()
}

/// How long ago any project's event log was last appended to. A capture hook
/// appends before it does anything else, so this is the age of the last hook;
/// `None` when no log exists yet.
pub(crate) fn newest_log_write_age(paths: &Paths) -> Result<Option<Duration>> {
    let mut newest = None;
    for (_, dir) in consolidate::known_projects(paths)? {
        for file in EventLog::open(&dir)?.files()? {
            if let Ok(modified) = std::fs::metadata(&file).and_then(|meta| meta.modified()) {
                newest = newest.max(Some(modified));
            }
        }
    }
    // A clock set back reads as "just now": refuse rather than guess.
    Ok(newest.map(|at| at.elapsed().unwrap_or_default()))
}

/// Bytes free on the disk holding `dir`, from `df`; `None` when it cannot say
/// (Windows has no `df`).
pub(crate) fn free_disk_bytes(dir: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-Pk").arg(dir).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let kilobytes: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
    kilobytes.checked_mul(1024)
}

const DAY_SECS: i64 = 24 * 3600;
/// `brain.log` is set aside when its first line is older than this.
const LOG_KEEP_SECS: i64 = 30 * DAY_SECS;
/// A parked session is given its attempts back this many times, then stays parked.
const UNPARK_LIMIT: u32 = 3;
/// A model fetch that left its marker this long ago is tried again.
const FETCH_RETRY: Duration = Duration::from_secs(24 * 3600);

/// The upkeep that needs doing once a day, not once a run: ageing
/// `brain.log`, giving parked sessions another go, repairing the files brain
/// owns, and fetching a missing embedding model. Called under the run lock,
/// never from a hook or MCP. Best effort, and silent: doctor counts every
/// `brain.log` line as a failure, so a refusal waits for tomorrow instead.
pub fn daily(paths: &Paths, store: &Store, lock: &RunLock) {
    // A unit test must not describe, let alone repair, the machine it runs on.
    let home = if cfg!(test) { None } else { dirs::home_dir() };
    daily_in(paths, store, lock, home.as_deref(), unix_now());
}

fn daily_in(paths: &Paths, store: &Store, lock: &RunLock, home: Option<&Path>, now: i64) {
    if day_passed(store, "brain_log_aged_at", now) {
        age_log(&paths.log_file(), now);
        stamp(store, "brain_log_aged_at", now);
    }
    lock.touch();
    if day_passed(store, "unpark_at", now) {
        let _ = unpark(store, now);
        stamp(store, "unpark_at", now);
    }
    if let (Some(home), Ok(exe)) = (home, std::env::current_exe()) {
        if day_passed(store, "files_healed_at", now) {
            crate::setup::heal_owned_files(home, &exe, true);
            stamp(store, "files_healed_at", now);
        }
    }
    fetch_model(paths);
    spawn_update(paths, store, now);
}

/// Has a day gone by since `key` was last stamped? A store that cannot be
/// read says no: better one day late than a repair on a guess.
fn day_passed(store: &Store, key: &str, now: i64) -> bool {
    match store.state(key) {
        Ok(value) => value
            .and_then(|at| at.parse::<i64>().ok())
            .is_none_or(|at| now.saturating_sub(at) >= DAY_SECS),
        Err(_) => false,
    }
}

fn stamp(store: &Store, key: &str, now: i64) {
    let _ = store.set_state(key, &now.to_string());
}

/// Is a project's relink due? Once a day each, read by the per-project pass.
pub(crate) fn relink_due(store: &Store, project: &str) -> bool {
    day_passed(store, &format!("relinked_at:{project}"), unix_now())
}

pub(crate) fn relink_done(store: &Store, project: &str) {
    stamp(store, &format!("relinked_at:{project}"), unix_now());
}

/// Set `brain.log` aside as `brain.log.1` (replacing the last one) when its
/// first line is older than a month. Every writer opens, appends and closes
/// per line, so a line written after the rename starts a new file.
fn age_log(log: &Path, now: i64) {
    use std::io::BufRead as _;
    let Ok(file) = std::fs::File::open(log) else { return };
    // The first line that carries a timestamp: a stray untimed line at the
    // top must not keep the file from ever ageing.
    let first = std::io::BufReader::new(file)
        .lines()
        .map_while(std::result::Result::ok)
        .take(20)
        .find_map(|line| line.split_whitespace().next()?.parse::<jiff::Timestamp>().ok());
    let old = first.is_some_and(|at| now.saturating_sub(at.as_second()) > LOG_KEEP_SECS);
    if old {
        let name = log.file_name().unwrap_or_default().to_string_lossy().into_owned();
        // A rename that fails (Windows, a reader holding it) is tomorrow's.
        let _ = std::fs::rename(log, log.with_file_name(format!("{name}.1")));
    }
}

/// Give the sessions parked for a day their attempts back, at most
/// [`UNPARK_LIMIT`] times each; past that they stay parked. Returns how many
/// it revived.
fn unpark(store: &Store, now: i64) -> Result<usize> {
    let cutoff = jiff::Timestamp::from_second(now.saturating_sub(DAY_SECS))?.to_string();
    let mut revived = 0;
    for session in store.parked_before(&cutoff)? {
        let key = format!("unparked:{session}");
        let tries = store.state(&key)?.and_then(|n| n.parse::<u32>().ok()).unwrap_or(0);
        if tries >= UNPARK_LIMIT {
            continue;
        }
        store.set_state(&key, &(tries + 1).to_string())?;
        consolidate::revive_session(store, &session)?;
        revived += 1;
    }
    Ok(revived)
}

/// Ask the installer, detached, for the embedding model this machine lacks.
///
/// The same shape as the reranker's fetch: the marker is written first, so a
/// second run does not start a second download, and nothing waits. Unlike it,
/// the marker stays when the fetch fails and goes only when it worked, so an
/// offline machine tries once a day rather than once a run. None on Windows,
/// which has no installer script yet; doctor names the manual step there.
fn fetch_model(paths: &Paths) {
    // Not under a unit test either: it would reach the network.
    if cfg!(windows) || cfg!(test) || std::env::var_os("ROLEPOD_BRAIN_NO_FETCH").is_some() {
        return;
    }
    let model = paths.model_dir();
    if model.join(crate::embed::WEIGHTS_FILE).is_file() {
        return;
    }
    let Some(parent) = model.parent() else { return };
    let marker = parent.join(format!("{}.fetching", model.file_name().unwrap_or_default().to_string_lossy()));
    let young = std::fs::metadata(&marker)
        .and_then(|meta| meta.modified())
        .is_ok_and(|at| at.elapsed().is_ok_and(|age| age < FETCH_RETRY));
    if young {
        return;
    }
    let _ = std::fs::create_dir_all(parent);
    if std::fs::write(&marker, "").is_err() {
        return;
    }
    // The paths are arguments of the shell, not part of its text: a home
    // with a quote in its name cannot break out of the command.
    let script = "(curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh \
                  | sh -s -- --model-only --into \"$1\") && rm -f \"$2\"";
    let _ = std::process::Command::new("sh")
        .args(["-c", script, "sh"])
        .arg(parent)
        .arg(&marker)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Once a day, start `brain update` detached and go on: nothing here waits
/// for the network. `update_checked_at` is stamped at the spawn, so a machine
/// where the updater has nothing to do still asks once a day, not once a run;
/// `update.running` keeps a second one from starting while one is out (a
/// marker a day old is a dead one). Silent, like the rest of `daily`.
fn spawn_update(paths: &Paths, store: &Store, now: i64) {
    if cfg!(windows) || cfg!(test) {
        return;
    }
    #[allow(unused_mut)]
    let mut no_fetch = std::env::var_os("ROLEPOD_BRAIN_NO_FETCH").is_some();
    // A debug build can be told to spawn anyway, for the end-to-end test.
    #[cfg(debug_assertions)]
    if std::env::var_os("ROLEPOD_BRAIN_UPDATE_SPAWN").is_some() {
        no_fetch = false;
    }
    if no_fetch {
        return;
    }
    let Ok(config) = crate::config::Config::load(&paths.config_file()) else { return };
    if !crate::update::spawnable(&config) || !day_passed(store, crate::update::STATE_CHECKED_AT, now) {
        return;
    }
    let marker = paths.data_dir.join(crate::update::FILE_RUNNING);
    let young = std::fs::metadata(&marker)
        .and_then(|meta| meta.modified())
        .is_ok_and(|at| at.elapsed().is_ok_and(|age| age < FETCH_RETRY));
    let Ok(exe) = std::env::current_exe() else { return };
    if young || std::fs::write(&marker, "").is_err() {
        return;
    }
    let spawned = std::process::Command::new(exe)
        .arg("update")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    if spawned.is_ok() {
        stamp(store, crate::update::STATE_CHECKED_AT, now);
    } else {
        let _ = std::fs::remove_file(&marker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1024 * 1024;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-maint-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A 1 GB file that retention has emptied and nobody has compacted.
    fn dropped() -> CompactState {
        CompactState {
            retention_dropped: 10,
            retention_dropped_bytes: 600 * MB,
            file_bytes: 1000 * MB,
            ..CompactState::default()
        }
    }

    #[test]
    fn never_compacted_after_a_drop_is_due_once() {
        let mut state = CompactState { retention_dropped: 3, file_bytes: 1000 * MB, ..CompactState::default() };
        assert!(state.due(1_000_000), "an upgraded store that dropped bodies must compact once");
        state.done_at = Some("2026-10-01T00:00:00Z".into());
        assert!(!state.due(1_000_000), "rule (a) fires only while no compact has happened");
        state.retention_dropped = 0;
        state.done_at = None;
        assert!(!state.due(1_000_000), "nothing dropped, nothing to give back");
    }

    #[test]
    fn a_pass_in_progress_a_small_gain_or_a_recent_try_is_not_due() {
        let mut state = dropped();
        assert!(state.due(1_000_000));
        state.retention_pending = true;
        assert!(!state.due(1_000_000), "the cursor is set");

        let mut state = dropped();
        state.done_at = Some("x".into());
        state.retention_dropped_bytes = 40 * MB; // ~18 MB
        assert!(!state.due(1_000_000), "under 32 MB");
        state.retention_dropped_bytes = 100 * MB; // 45 MB of 1000 MB
        state.file_bytes = 1000 * MB;
        assert!(!state.due(1_000_000), "under 10% of the file");
        state.file_bytes = 400 * MB;
        assert!(state.due(1_000_000), "45 MB of 400 MB passes both");

        state.tried_at = Some(1_000_000 - 3600);
        assert!(!state.due(1_000_000), "tried an hour ago");
        assert!(state.wanted(), "a recent try does not make the space unwanted");
        assert!(state.due(1_000_000 + 24 * 3600), "a day later it tries again");
    }

    #[test]
    fn free_pages_count_and_the_estimate_never_goes_negative() {
        let state = CompactState {
            done_at: Some("x".into()),
            free_bytes: 50 * MB,
            retention_dropped_bytes: 10,
            compact_dropped_bytes: 900 * MB,
            file_bytes: 300 * MB,
            ..CompactState::default()
        };
        assert_eq!(state.reclaimable(), 50 * MB, "a counter behind the compact one adds nothing");
        assert!(state.due(0));
    }

    #[test]
    fn a_clear_resets_the_compact_marks_with_the_retention_ones() {
        let store = Store::open_memory().unwrap();
        for (key, value) in [
            ("retention_dropped_bytes", "500"),
            ("compact_dropped_bytes", "400"),
            ("compact_done_at", "2026-10-01T00:00:00Z"),
            ("compact_skip", "not-quiet@x"),
        ] {
            store.set_state(key, value).unwrap();
        }
        store.clear().unwrap();
        let state = store.compact_state().unwrap();
        assert_eq!((state.retention_dropped_bytes, state.compact_dropped_bytes), (0, 0));
        assert!(state.done_at.is_none() && state.skip.is_none());
    }

    #[test]
    fn record_compact_settles_the_counter_and_clears_the_last_skip() {
        let store = Store::open_memory().unwrap();
        store.set_state("retention_dropped_bytes", "700").unwrap();
        store.set_state("compact_skip", "low-disk@x").unwrap();
        let done = CompactState {
            done_at: Some("2026-10-08T00:00:00Z".into()),
            compact_dropped_bytes: 700,
            before_bytes: Some(9),
            after_bytes: Some(5),
            ..CompactState::default()
        };
        store.record_compact(&done).unwrap();
        let state = store.compact_state().unwrap();
        assert_eq!(state.done_at.as_deref(), Some("2026-10-08T00:00:00Z"));
        assert_eq!((state.compact_dropped_bytes, state.skip), (700, None));
    }

    #[test]
    fn a_marker_that_is_missing_junk_or_expired_is_not_active() {
        let dir = scratch();
        assert!(!active_at(&dir, 1000), "no marker");
        std::fs::write(dir.join(MARKER), "garbage").unwrap();
        assert!(!active_at(&dir, 1000), "junk");
        std::fs::write(dir.join(MARKER), [0xff, 0xfe, 0xfd]).unwrap();
        assert!(!active_at(&dir, 1000), "not text");
        std::fs::write(dir.join(MARKER), "42 1000").unwrap();
        assert!(!active_at(&dir, 1000), "expiring this instant");
        assert!(!active_at(&dir, 2000), "expired");
        assert!(active_at(&dir, 999), "an expiry still ahead");
    }

    #[test]
    fn the_heartbeat_keeps_the_marker_alive_past_one_lifetime() {
        let dir = scratch();
        let timing = Timing {
            heartbeat: Duration::from_millis(40),
            lifetime: Duration::from_millis(400),
            ..Timing::live()
        };
        let window = Window::open(&dir, &timing).unwrap();
        // Work three lifetimes long, watched the whole way.
        let began = Instant::now();
        while began.elapsed() < Duration::from_millis(1200) {
            assert!(active_at(&dir, now_ms()), "the marker lapsed at {:?}", began.elapsed());
            std::thread::sleep(Duration::from_millis(25));
        }
        drop(window);
        assert!(!dir.join(MARKER).exists(), "the window left its marker behind");
        assert!(!active_at(&dir, now_ms()));
    }

    #[test]
    fn a_marker_nobody_renews_expires_by_itself() {
        // What a SIGKILLed compactor leaves: a marker and no heartbeat.
        let dir = scratch();
        write_marker(&dir, now_ms() + 300).unwrap();
        assert!(active_at(&dir, now_ms()));
        std::thread::sleep(Duration::from_millis(400));
        assert!(!active_at(&dir, now_ms()), "an orphaned marker outlived its expiry");
    }

    fn timing() -> Timing {
        Timing { quiet: Duration::from_secs(60), poll: Duration::from_secs(10), budget: Duration::from_secs(360), ..Timing::live() }
    }

    #[test]
    fn the_wait_goes_on_while_the_log_is_written_and_ends_when_it_goes_quiet() {
        let mut ages = vec![Some(Duration::from_secs(5)), Some(Duration::from_secs(2)), Some(Duration::from_secs(70))].into_iter();
        let mut slept = Vec::new();
        let waited = wait_quiet(
            &timing(),
            Duration::ZERO,
            &mut || ages.next().unwrap(),
            &mut || false,
            &mut |d| slept.push(d),
        );
        assert_eq!(waited, Wait::Quiet);
        assert_eq!(slept.len(), 2, "two polls before the quiet one");
    }

    #[test]
    fn a_new_ask_ends_the_wait() {
        let mut asks = vec![false, true].into_iter();
        let waited = wait_quiet(
            &timing(),
            Duration::ZERO,
            &mut || Some(Duration::from_secs(1)),
            &mut || asks.next().unwrap(),
            &mut |_| {},
        );
        assert_eq!(waited, Wait::AskOpen);
    }

    #[test]
    fn a_spent_budget_ends_the_wait_as_not_quiet() {
        let mut slept = Duration::ZERO;
        let waited = wait_quiet(
            &timing(),
            Duration::from_secs(300),
            &mut || Some(Duration::from_secs(1)),
            &mut || false,
            &mut |d| slept += d,
        );
        assert_eq!(waited, Wait::NotQuiet);
        assert!(slept <= Duration::from_secs(60), "waited {slept:?} past the budget");
        let none = wait_quiet(&timing(), Duration::ZERO, &mut || None, &mut || false, &mut |_| {});
        assert_eq!(none, Wait::Quiet, "no log at all is quiet");
    }

    #[test]
    fn the_disk_guard_wants_two_and_a_half_copies_and_a_gigabyte() {
        let index = 2000 * MB;
        let need = 5000 * MB + 1024 * MB;
        assert!(check_room(Some(need), index).is_ok());
        assert!(check_room(Some(need - 1), index).unwrap_err().starts_with("low-disk"));
        assert_eq!(check_room(None, index).unwrap_err(), "free-disk-unknown");
    }

    #[test]
    fn the_gate_admits_one_and_a_stale_one_is_taken_over() {
        let dir = scratch();
        let first = Gate::take(&dir, GATE_STALE).expect("first");
        assert!(Gate::take(&dir, GATE_STALE).is_none(), "second while the first lives");
        assert!(Gate::take(&dir, Duration::ZERO).is_some(), "a gate past its staleness is reclaimed");
        drop(first);
    }

    fn paths_in(dir: &Path) -> Paths {
        Paths { data_dir: dir.to_path_buf() }
    }

    fn indexed_event(store: &Store) -> String {
        let event = crate::event::Event::new(
            uuid::Uuid::nil(),
            uuid::Uuid::from_u128(1),
            uuid::Uuid::nil(),
            crate::event::Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
            crate::event::EventKind::Observation,
            "title".into(),
            "body".into(),
        );
        store.index(&event).unwrap();
        event.id
    }

    #[test]
    fn a_spill_is_remembered_for_its_session_until_it_is_folded() {
        let dir = scratch();
        let paths = paths_in(&dir);
        spill_surfaced(&paths, "s1", Ledger::Recalled, &["a".into(), "b".into()]);
        spill_surfaced(&paths, "s1", Ledger::File, &["src/x.rs".into()]);
        spill_surfaced(&paths, "s1", Ledger::Opened, &[]);
        assert!(spilled(&paths, "s1", "b"));
        assert!(!spilled(&paths, "s2", "b"), "another session's spill counted");
        assert!(!spilled(&paths, "s1", "src/x.rs"), "a file path is not an id");
        assert_eq!(std::fs::read_to_string(dir.join(SURFACED)).unwrap().lines().count(), 2);
    }

    #[test]
    fn folding_twice_counts_once_and_a_leftover_folding_file_is_taken_too() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open_memory().unwrap();
        let (one, two) = (indexed_event(&store), indexed_event(&store));
        spill_surfaced(&paths, "s1", Ledger::Recalled, std::slice::from_ref(&one));
        spill_surfaced(&paths, "s1", Ledger::Opened, std::slice::from_ref(&one));
        spill_surfaced(&paths, "s1", Ledger::Injected, std::slice::from_ref(&one));
        spill_surfaced(&paths, "s1", Ledger::File, &["src/x.rs".into()]);
        // A fold that died after committing: its file is still there, and the
        // next spill lands beside it.
        std::fs::rename(dir.join(SURFACED), dir.join(FOLDING)).unwrap();
        spill_surfaced(&paths, "s1", Ledger::Recalled, std::slice::from_ref(&two));

        fold_surfaced(&paths, &store);
        let count = |sql: &str| -> i64 { store.raw_count(sql) };
        assert_eq!(count("SELECT COUNT(*) FROM recalled"), 2);
        assert_eq!(count("SELECT COUNT(*) FROM recalled WHERE opened = 1"), 1);
        assert_eq!(count("SELECT COUNT(*) FROM injected"), 1);
        assert_eq!(count("SELECT COUNT(*) FROM injected_files"), 1);
        assert_eq!(count("SELECT read_count FROM events WHERE read_count > 0 LIMIT 1"), 1);
        assert!(!dir.join(SURFACED).exists(), "the spill was left");

        // The same lines again: nothing changes.
        spill_surfaced(&paths, "s1", Ledger::Recalled, &[one.clone(), two]);
        spill_surfaced(&paths, "s1", Ledger::Injected, &[one]);
        fold_surfaced(&paths, &store);
        assert_eq!(count("SELECT COUNT(*) FROM recalled"), 2);
        assert_eq!(count("SELECT COUNT(*) FROM injected"), 1);
        assert_eq!(count("SELECT MAX(read_count) FROM events"), 1, "a second fold counted a second read");
    }

    #[test]
    fn an_injection_spill_gives_its_bytes_back_once_and_an_old_spill_still_folds() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open_memory().unwrap();
        let one = indexed_event(&store);
        spill_injected(&paths, "s1", std::slice::from_ref(&one), 700);
        // A line from before the field existed.
        let old = indexed_event(&store);
        let line = serde_json::json!({ "session": "s1", "kind": "injected", "ids": [old] });
        let mut file = std::fs::OpenOptions::new().append(true).open(dir.join(SURFACED)).unwrap();
        std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes()).unwrap();
        assert!(fold_surfaced(&paths, &store));
        assert_eq!(store.session_injected_bytes("s1").unwrap(), 700);
        assert_eq!(store.raw_count("SELECT COUNT(*) FROM injected"), 2);

        spill_injected(&paths, "s1", std::slice::from_ref(&one), 700);
        assert!(fold_surfaced(&paths, &store));
        assert_eq!(store.session_injected_bytes("s1").unwrap(), 700, "a second fold spent again");
    }

    #[test]
    fn a_line_appended_after_the_folds_last_read_is_taken_by_the_next_fold() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open_memory().unwrap();
        let (one, two) = (indexed_event(&store), indexed_event(&store));
        spill_surfaced(&paths, "s1", Ledger::Recalled, std::slice::from_ref(&one));
        assert!(fold_surfaced(&paths, &store));
        // A hook that opened the file before the rename appends to it now.
        let line = serde_json::json!({ "session": "s1", "kind": "recalled", "ids": [two] });
        let mut file = std::fs::OpenOptions::new().append(true).open(dir.join(FOLDING)).unwrap();
        std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes()).unwrap();
        assert_eq!(store.raw_count("SELECT COUNT(*) FROM recalled"), 1);
        assert!(fold_surfaced(&paths, &store));
        assert_eq!(store.raw_count("SELECT COUNT(*) FROM recalled"), 2, "the late line was dropped");
        assert_eq!(std::fs::metadata(dir.join(FOLDING)).unwrap().len(), 0, "a folded file is emptied");
    }

    #[test]
    fn a_fold_after_a_wipe_does_not_re_arm_what_the_wipe_cleared() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open_memory().unwrap();
        let one = indexed_event(&store);
        spill_injected(&paths, "s1", std::slice::from_ref(&one), 300);
        spill_surfaced(&paths, "s1", Ledger::File, &["src/x.rs".into()]);
        assert!(fold_surfaced(&paths, &store));
        store.reset_injection_state("s1").unwrap();
        assert!(fold_surfaced(&paths, &store));
        assert_eq!(store.raw_count("SELECT COUNT(*) FROM injected WHERE active = 1"), 0);
        assert_eq!(store.raw_count("SELECT COUNT(*) FROM injected_files"), 0);
    }

    #[test]
    fn folding_the_same_file_again_changes_nothing_bytes_included() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open_memory().unwrap();
        let one = indexed_event(&store);
        spill_injected(&paths, "s1", std::slice::from_ref(&one), 300);
        let kept = std::fs::read(dir.join(SURFACED)).unwrap();
        assert!(fold_surfaced(&paths, &store));
        std::fs::write(dir.join(FOLDING), kept).unwrap();
        assert!(fold_surfaced(&paths, &store));
        assert_eq!(store.session_injected_bytes("s1").unwrap(), 300);
        assert_eq!(store.raw_count("SELECT MAX(injected_count) FROM events"), 1);
    }

    #[test]
    fn the_log_is_cut_once_more_when_the_reader_lets_go() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open(&paths.db()).unwrap();
        let raw = rusqlite::Connection::open(paths.db()).unwrap();
        raw.execute_batch("CREATE TABLE junk(b BLOB); INSERT INTO junk VALUES (zeroblob(65536));").unwrap();
        let reader = rusqlite::Connection::open(paths.db()).unwrap();
        reader.execute_batch("BEGIN; SELECT count(*) FROM junk;").unwrap();
        raw.execute_batch("INSERT INTO junk VALUES (zeroblob(65536));").unwrap();
        store.set_busy_timeout(Duration::ZERO).unwrap();
        assert!(!cut_once(&store), "a reader still holds the log");
        reader.execute_batch("COMMIT;").unwrap();
        assert!(cut_once(&store), "the reader let go");
    }

    #[test]
    fn a_pending_wipe_is_one_file_per_session_and_leaves_with_its_clear() {
        let dir = scratch();
        let paths = paths_in(&dir);
        assert!(!wipe_pending(&paths, "s1"));
        pending_wipe(&paths, "s1");
        pending_wipe(&paths, "../escape");
        assert!(wipe_pending(&paths, "s1") && !wipe_pending(&paths, "s2"));
        assert!(!dir.join("escape").exists() && dir.join(WIPES).join("___escape").exists());
        clear_pending_wipe(&paths, "s1");
        assert!(!wipe_pending(&paths, "s1"));
    }

    #[test]
    fn status_reports_never_due_overdue_and_done() {
        let now = jiff::Timestamp::now().as_second();
        let never = status_of(&CompactState::default(), now);
        assert_eq!((never.last, never.wanted, never.overdue), (None, false, false));

        let due = status_of(&dropped(), now);
        assert!(due.wanted && !due.overdue, "due, but only just");

        let long_ago = jiff::Timestamp::from_second(now - 20 * 24 * 3600).unwrap().to_string();
        let mut stale = dropped();
        stale.retention_done_at = Some(long_ago);
        stale.skip = Some("low-disk@x".into());
        let overdue = status_of(&stale, now);
        assert!(overdue.overdue);
        assert_eq!(overdue.skip.as_deref(), Some("low-disk@x"));

        let mut done = dropped();
        done.done_at = Some(jiff::Timestamp::now().to_string());
        done.compact_dropped_bytes = done.retention_dropped_bytes;
        let done = status_of(&done, now);
        assert!(!done.wanted && done.last.is_some());
    }

    #[test]
    fn a_daily_item_waits_a_day_on_an_injected_clock() {
        let store = Store::open_memory().unwrap();
        let t0 = 1_800_000_000;
        assert!(day_passed(&store, "k", t0), "never run is due");
        stamp(&store, "k", t0);
        assert!(!day_passed(&store, "k", t0 + DAY_SECS - 1));
        assert!(day_passed(&store, "k", t0 + DAY_SECS));
    }

    #[test]
    fn the_log_is_set_aside_only_when_its_first_line_is_a_month_old() {
        let dir = scratch();
        let log = dir.join("brain.log");
        let now = jiff::Timestamp::now().as_second();
        let at = |secs: i64| jiff::Timestamp::from_second(secs).unwrap();
        std::fs::write(&log, format!("{} hook x: boom\n", at(now - 29 * DAY_SECS))).unwrap();
        age_log(&log, now);
        assert!(log.is_file() && !dir.join("brain.log.1").exists(), "29 days stays");

        std::fs::write(&log, format!("{} hook x: boom\n", at(now - 31 * DAY_SECS))).unwrap();
        std::fs::write(dir.join("brain.log.1"), "older").unwrap();
        age_log(&log, now);
        assert!(!log.exists(), "31 days is set aside");
        assert!(std::fs::read_to_string(dir.join("brain.log.1")).unwrap().contains("boom"), "replaced the last one");

        // A line appended afterwards starts a fresh file, and loses nothing.
        std::fs::write(&log, "fresh\n").unwrap();
        assert!(std::fs::read_to_string(dir.join("brain.log.1")).unwrap().contains("boom"));
    }

    #[test]
    fn a_parked_session_is_given_three_more_tries_and_then_stays_parked() {
        let store = Store::open_memory().unwrap();
        let mut now = jiff::Timestamp::now().as_second();
        let park = |store: &Store| {
            for _ in 0..Store::PARK_AFTER {
                store.record_session_failure("s1", "p", "boom").unwrap();
            }
        };
        park(&store);
        assert_eq!(unpark(&store, now).unwrap(), 0, "parked a moment ago stays");
        for round in 1..=UNPARK_LIMIT {
            now += 2 * DAY_SECS;
            assert_eq!(unpark(&store, now).unwrap(), 1, "round {round}");
            assert_eq!(store.failing_sessions(Store::PARK_AFTER).unwrap().len(), 0, "attempts given back");
            park(&store);
        }
        now += 2 * DAY_SECS;
        assert_eq!(unpark(&store, now).unwrap(), 0, "the fourth time it stays parked");
        assert_eq!(store.failing_sessions(Store::PARK_AFTER).unwrap().len(), 1);
    }

    #[test]
    fn a_rewrite_that_fails_closes_the_window_and_writes_down_why() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open(&paths.db()).unwrap();
        let lock = RunLock::take(&consolidate::run_lock_path(&paths)).unwrap().expect("lock");
        // A directory where the marker belongs: the window cannot open.
        std::fs::create_dir(dir.join(MARKER)).unwrap();

        let result = compact_with(&paths, &store, &lock, &Timing::live());
        assert!(result.is_err(), "a window that cannot open fails the rewrite");
        assert!(!active(&paths), "no marker claims a window that never opened");
        let state = store.compact_state().unwrap();
        assert!(state.skip.is_some(), "the reason is written to schema_state");
        assert!(state.tried_at.is_some(), "a failed try counts, so it is not repeated at once");
        assert!(state.done_at.is_none());
    }

    #[test]
    fn a_reader_that_keeps_the_log_does_not_inflate_the_size_after() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open(&paths.db()).unwrap();
        let lock = RunLock::take(&consolidate::run_lock_path(&paths)).unwrap().expect("lock");
        // Bloat the file, then free it all and settle the log.
        let raw = rusqlite::Connection::open(paths.db()).unwrap();
        raw.execute_batch("CREATE TABLE junk(b BLOB);").unwrap();
        for _ in 0..200 {
            raw.execute("INSERT INTO junk VALUES (zeroblob(65536))", []).unwrap();
        }
        raw.execute_batch("DROP TABLE junk;").unwrap();
        assert!(store.truncate_wal().unwrap());
        // A second connection reads throughout, so the log cannot be cut.
        let reader = rusqlite::Connection::open(paths.db()).unwrap();
        reader.execute_batch("BEGIN; SELECT count(*) FROM sqlite_master;").unwrap();

        let began = Instant::now();
        let timing = Timing { cut_wait: Duration::from_millis(10), ..Timing::live() };
        let report = compact_with(&paths, &store, &lock, &timing).unwrap();
        assert!(began.elapsed() < Duration::from_secs(2), "the cut waited {:?}", began.elapsed());
        assert!(!report.truncated, "the reader kept the log");
        let (pages, _) = store.page_use().unwrap();
        let page_size: i64 = raw.pragma_query_value(None, "page_size", |row| row.get(0)).unwrap();
        assert_eq!(report.after, (pages * page_size) as u64);
        assert!(report.after < report.before, "{} < {}", report.after, report.before);
        assert_eq!(store.state("compact_bytes_after").unwrap(), Some(report.after.to_string()));
        assert_eq!(store.state("compact_bytes_before").unwrap(), Some(report.before.to_string()));
    }

    #[test]
    fn a_state_that_changes_after_the_lock_stops_the_rewrite_and_says_why() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open(&paths.db()).unwrap();
        store.set_state("retention_dropped", "3").unwrap();
        assert!(store.compact_state().unwrap().due(unix_now()), "precondition: due");

        // Retention restarts between the first look and the lock being held.
        let timing = Timing::live();
        try_window(&paths, Instant::now(), &timing, &mut |store| {
            store.set_state("retention_cursor", "x").unwrap();
        })
        .unwrap();
        let state = store.compact_state().unwrap();
        assert!(state.done_at.is_none(), "no rewrite ran");
        assert!(!dir.join(MARKER).exists(), "no window was opened");
        assert!(state.skip.is_some_and(|skip| skip.starts_with("no-longer-due")));

        // And the untouched case does rewrite, so the test above proves the recheck.
        store.clear_state("retention_cursor").unwrap();
        try_window(&paths, Instant::now(), &timing, &mut |_| {}).unwrap();
        assert!(store.compact_state().unwrap().done_at.is_some(), "a still-due store is rewritten");
    }

    #[test]
    fn a_fold_that_cannot_finish_says_so() {
        let dir = scratch();
        let paths = paths_in(&dir);
        let store = Store::open(&paths.db()).unwrap();
        assert!(fold_surfaced(&paths, &store), "nothing spilled is nothing left");

        std::fs::write(dir.join(SURFACED), "").unwrap();
        // A directory where the half-folded file belongs: it can be neither read nor joined.
        std::fs::create_dir(dir.join(FOLDING)).unwrap();
        assert!(!fold_surfaced(&paths, &store), "an unfolded spill must stop retention");
    }
}
