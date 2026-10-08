//! Self-update of the installed `brain` binary.
//!
//! The one rule everything here leans on: no byte of a file whose signature
//! did not verify is ever executed, and the binary's path is runnable at every
//! instant. A step that does not pass changes nothing on disk except
//! `schema_state` (and the data dir's own `update.bad`), and never writes
//! `brain.log`. Each threat the design answers has a test named after it here
//! and in `tests/update_e2e.rs`.
//!
//! Order of a run: opt-out (before any `curl`), lock, shape of the install,
//! key, release metadata, decision on the tag, `.minisig`, binary into a
//! `O_EXCL` staging file, signature + trusted comment, `chmod`, `codesign`,
//! self-test, then `link(bin, brain.prev)` and one `rename` over the path.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::config::{Config, Paths};
use crate::store::Store;

/// The release public key: the second line (the base64 one) of `minisign.pub`
/// at the repository root, which is also the file the release workflow
/// verifies every signature against. `None` would mean nothing is ever
/// installed. The test `pubkey_const_matches_minisign_pub` holds the two
/// together.
pub const UPDATE_PUBKEY: Option<&str> = Some("RWQzSHiV/5G9L0o/+oGzL//Htn9M5PNPsfEbUZ7oC4g3vqeqf96/zbjr");

const RELEASES_API: &str = "https://api.github.com/repos/nuttaruj/rolepod-brain/releases/latest";
const DOWNLOAD_PREFIX: &str = "https://github.com/nuttaruj/rolepod-brain/releases/download/";
const MIN_AGE_SECS: i64 = 24 * 3600;
const MAX_JSON: u64 = 1 << 20;
const MAX_SIG: u64 = 64 << 10;
const MAX_BIN: u64 = 200 << 20;
const SELFTEST_SECS: u64 = 10;
const BAD_CAP: usize = 20;
const STALE_SECS: u64 = 3600;

// One spelling for every file the updater keeps in the data dir and every key
// it keeps in `schema_state`, shared with maint.rs and doctor.rs.
pub(crate) const FILE_INSTALLED: &str = "update.installed";
pub(crate) const FILE_PREV: &str = "update.prev";
pub(crate) const FILE_BAD: &str = "update.bad";
pub(crate) const FILE_RUNNING: &str = "update.running";
pub(crate) const FILE_FAILURES: &str = "update.failures";
pub(crate) const FILE_LOCK: &str = "update.lock";
pub(crate) const STATE_INSTALLED: &str = "update_installed";
pub(crate) const STATE_PREV: &str = "update_prev";
pub(crate) const STATE_BAD: &str = "update_bad";
pub(crate) const STATE_CHECKED_AT: &str = "update_checked_at";
pub(crate) const STATE_SKIP: &str = "update_skip";

/// Where releases are read from: the metadata URL and the base asset URLs
/// are joined to (`<base><tag>/<name>` on GitHub, `<base><name>` in a test).
struct Source {
    meta: String,
    base: String,
    per_tag: bool,
    local: bool,
}

impl Source {
    fn release() -> Self {
        Self { meta: RELEASES_API.into(), base: DOWNLOAD_PREFIX.into(), per_tag: true, local: false }
    }

    /// A URL for one asset of `tag`, or None when it is not one this source
    /// may hand to curl.
    fn asset_url(&self, tag: &str, name: &str) -> Option<String> {
        let url = if self.per_tag { format!("{}{tag}/{name}", self.base) } else { format!("{}{name}", self.base) };
        self.url_allowed(&url).then_some(url)
    }

    fn url_allowed(&self, url: &str) -> bool {
        if url == self.meta {
            return true;
        }
        let Some(rest) = url.strip_prefix(&self.base) else { return false };
        !rest.is_empty()
            && !rest.starts_with('/')
            && !rest.contains("..")
            && rest.chars().all(|c| c.is_ascii_alphanumeric() || ".-_/".contains(c))
    }
}

/// The source in use. A debug build can be pointed at a directory
/// (`latest.json` + assets); a release build reads constants only.
fn source() -> Source {
    #[cfg(debug_assertions)]
    if let Some(dir) = std::env::var_os("ROLEPOD_BRAIN_UPDATE_URL") {
        let dir = dir.to_string_lossy();
        let dir = dir.strip_prefix("file://").unwrap_or(&dir).trim_end_matches('/').to_string();
        return Source {
            meta: format!("file://{dir}/latest.json"),
            base: format!("file://{dir}/"),
            per_tag: false,
            local: true,
        };
    }
    Source::release()
}

/// The key in use: the constant, or in a debug build the env override.
fn pubkey() -> Option<String> {
    #[cfg(debug_assertions)]
    let over = std::env::var("ROLEPOD_BRAIN_UPDATE_PUBKEY").ok();
    #[cfg(not(debug_assertions))]
    let over = None;
    resolve_key(over)
}

/// An override wins; an empty one means "no key" so a test can stand in for a
/// build that has none whatever the constant holds.
fn resolve_key(over: Option<String>) -> Option<String> {
    match over {
        Some(key) => Some(key).filter(|k| !k.is_empty()),
        None => UPDATE_PUBKEY.map(str::to_string),
    }
}

fn max_bin() -> u64 {
    #[cfg(debug_assertions)]
    if let Some(n) = std::env::var("ROLEPOD_BRAIN_UPDATE_MAX_BYTES").ok().and_then(|n| n.parse().ok()) {
        return n;
    }
    MAX_BIN
}

/// The asset target of a platform: the four unix names `release.yml` builds.
fn platform_target(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        ("macos", "x86_64") => Some("x86_64-apple-darwin"),
        ("linux", "x86_64") => Some("x86_64-unknown-linux-gnu"),
        ("linux", "aarch64") => Some("aarch64-unknown-linux-gnu"),
        _ => None,
    }
}

fn target() -> Option<&'static str> {
    platform_target(std::env::consts::OS, std::env::consts::ARCH)
}

// ---- versions and the decision --------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Ver {
    nums: [u64; 3],
    pre: Option<String>,
}

impl Ord for Ver {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.nums.cmp(&other.nums).then_with(|| match (&self.pre, &other.pre) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            (Some(a), Some(b)) => a.cmp(b),
        })
    }
}

impl PartialOrd for Ver {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// `0.68.1`, `v0.68.1`, `1.0.0-rc.1`. Anything else is None.
fn parse_ver(text: &str) -> Option<Ver> {
    let text = text.strip_prefix('v').unwrap_or(text);
    let (core, pre) = match text.split_once('-') {
        Some((core, pre)) => (core, Some(pre)),
        None => (text, None),
    };
    let mut parts = core.split('.');
    let mut nums = [0u64; 3];
    for slot in &mut nums {
        let part = parts.next()?;
        if part.is_empty() || part.len() > 9 || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        *slot = part.parse().ok()?;
    }
    if parts.next().is_some() {
        return None;
    }
    if let Some(pre) = pre {
        if pre.is_empty() || !pre.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-') {
            return None;
        }
    }
    Some(Ver { nums, pre: pre.map(str::to_string) })
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Install,
    Skip(&'static str),
}

/// Install `candidate` over `running`? Strictly newer, a release (no
/// prerelease), not on the bad list, and published more than 24 h ago. A
/// missing or unparsable `published_at` only ever delays.
fn decide(running: &str, candidate: &str, published_at: Option<&str>, bad: &[String], now: i64) -> Verdict {
    let Some(cand) = parse_ver(candidate) else { return Verdict::Skip("bad-tag") };
    if cand.pre.is_some() {
        return Verdict::Skip("prerelease");
    }
    let Some(run) = parse_ver(running) else { return Verdict::Skip("running-unparsable") };
    if cand <= run {
        return Verdict::Skip("not-newer");
    }
    let plain = candidate.strip_prefix('v').unwrap_or(candidate);
    if bad.iter().any(|b| b.strip_prefix('v').unwrap_or(b) == plain) {
        return Verdict::Skip("marked-bad");
    }
    let Some(at) = published_at.and_then(|p| p.parse::<jiff::Timestamp>().ok()) else {
        return Verdict::Skip("published-unknown");
    };
    if now.saturating_sub(at.as_second()) < MIN_AGE_SECS {
        return Verdict::Skip("too-young");
    }
    Verdict::Install
}

/// `brain 1.2.3` and nothing else.
fn version_output_matches(output: &str, version: &str) -> bool {
    output.trim() == format!("brain {version}")
}

// ---- signature -------------------------------------------------------------

/// Check `bytes` against `sig` with `key`, then the signed trusted comment:
/// it must be exactly `brain-<triple> <version>` and the version must be the
/// tag's. Returns the signed version (the one the decision is made on).
fn accept(bytes: &[u8], sig: &str, key: &str, triple: &str, tag: &str) -> Result<String, &'static str> {
    if sig.trim().is_empty() {
        return Err("unsigned");
    }
    let key = minisign_verify::PublicKey::from_base64(key).map_err(|_| "bad-key")?;
    let sig = minisign_verify::Signature::decode(sig).map_err(|_| "bad-signature")?;
    key.verify(bytes, &sig, false).map_err(|_| "bad-signature")?;
    let comment = sig.trusted_comment();
    let rest = comment.strip_prefix("brain-").ok_or("signed-target-mismatch")?;
    let (comment_target, version) = rest.split_once(' ').ok_or("signed-target-mismatch")?;
    if comment_target != triple {
        return Err("signed-target-mismatch");
    }
    let parsed = parse_ver(version).filter(|v| v.pre.is_none() && !version.starts_with('v')).ok_or("signed-version-mismatch")?;
    let tag_ver = parse_ver(tag).ok_or("bad-tag")?;
    if parsed != tag_ver {
        return Err("signed-version-mismatch");
    }
    Ok(version.to_string())
}

// ---- curl ------------------------------------------------------------------

fn curl_args(url: &str, out: &Path, max: u64, secs: u64, local: bool) -> Vec<OsString> {
    let proto = if local { "=file" } else { "=https" };
    // `-q` first, or curl reads the user's ~/.curlrc.
    let mut args: Vec<OsString> = ["-q", "-fsSL", "--proto", proto].iter().map(OsString::from).collect();
    if !local {
        args.extend(["--proto-redir", proto].map(OsString::from));
    }
    args.extend(["--max-filesize".into(), max.to_string().into(), "--max-time".into(), secs.to_string().into()]);
    args.extend(["-A".into(), format!("rolepod-brain-update/{}", env!("CARGO_PKG_VERSION")).into()]);
    args.extend(["-o".into(), out.as_os_str().to_owned(), "--url".into(), url.into()]);
    args
}

// ---- tools -----------------------------------------------------------------

/// Where `name` may be run from. Never `$PATH`: a directory earlier on it is
/// the first place a planted `curl` or `codesign` would sit. A debug build can
/// name one directory of stubs for the tests.
fn tool_path(name: &str) -> Option<PathBuf> {
    #[cfg(debug_assertions)]
    if let Some(dir) = std::env::var_os("ROLEPOD_BRAIN_UPDATE_TOOLS") {
        let path = Path::new(&dir).join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    fixed_tool(name)
}

fn fixed_tool(name: &str) -> Option<PathBuf> {
    let candidates: &[&str] = match name {
        "curl" if cfg!(target_os = "macos") => &["/usr/bin/curl"],
        "curl" => &["/usr/bin/curl", "/bin/curl", "/usr/local/bin/curl"],
        "codesign" => &["/usr/bin/codesign"],
        "kill" => &["/bin/kill", "/usr/bin/kill"],
        _ => &[],
    };
    candidates.iter().map(PathBuf::from).find(|path| path.is_file())
}

/// A child gets nothing of ours: an empty environment and a fixed `PATH`.
fn clean(cmd: &mut Command) -> &mut Command {
    cmd.env_clear().env("PATH", "/usr/bin:/bin")
}

fn tool_command(name: &str) -> Option<Command> {
    let mut cmd = Command::new(tool_path(name)?);
    clean(&mut cmd);
    if name == "curl" {
        pass_proxy(&mut cmd, std::env::vars_os());
    }
    Some(cmd)
}

/// The proxy settings a machine behind one needs, for `curl` only. Nothing
/// that changes what curl trusts (`CURL_*`, `SSL_CERT_*`) or reads (`HOME`):
/// TLS validation and the signature keep a proxy harmless.
const PROXY_VARS: [&str; 6] = ["https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"];

fn pass_proxy(cmd: &mut Command, vars: impl Iterator<Item = (OsString, OsString)>) {
    for (key, value) in vars {
        if PROXY_VARS.iter().any(|name| key == *name) {
            cmd.env(key, value);
        }
    }
}

/// Fetch `url` into `out` (a file this run owns). False on any failure, and
/// `out` is removed when what arrived is over `max`.
fn fetch(src: &Source, url: &str, out: &Path, max: u64, secs: u64) -> bool {
    if !src.url_allowed(url) {
        return false;
    }
    let Some(mut curl) = tool_command("curl") else { return false };
    let ok = curl
        .args(curl_args(url, out, max, secs, src.local))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let size = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    if !ok || size > max {
        let _ = std::fs::remove_file(out);
        return false;
    }
    true
}

// ---- files this run owns ---------------------------------------------------

/// A file removed when the run ends, however it ends.
struct Owned(PathBuf);

impl Drop for Owned {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The staging file, created `O_EXCL`: a planted symlink or file at the name
/// makes this fail rather than be followed.
#[cfg(unix)]
fn create_staging(dir: &Path) -> std::io::Result<Owned> {
    use std::os::unix::fs::OpenOptionsExt as _;
    let path = dir.join(format!(".brain.new-{}", std::process::id()));
    std::fs::OpenOptions::new().write(true).create_new(true).mode(0o700).open(&path)?;
    Ok(Owned(path))
}

/// What identifies a file's bytes cheaply: compared right before the rename.
#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
struct Snap(u64, u64, u64, i64, i64);

#[cfg(unix)]
fn snap(path: &Path) -> Option<Snap> {
    use std::os::unix::fs::MetadataExt as _;
    let m = std::fs::symlink_metadata(path).ok()?;
    m.is_file().then(|| Snap(m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec()))
}

/// Is `pid` a running process? Unknown counts as running: a sweep must not
/// delete a live updater's file on a guess.
#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    let Some(mut kill) = tool_command("kill") else { return true };
    kill.args(["-0", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_or(true, |s| s.success())
}

/// Remove `.brain.new-`, `.brain.prev-` and `.brain.rollback-` files older
/// than an hour whose `<pid>` is gone.
#[cfg(unix)]
fn sweep_stale(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(pid) = [".brain.new-", ".brain.prev-", ".brain.rollback-"]
            .iter()
            .find_map(|prefix| name.strip_prefix(prefix).and_then(|p| p.parse::<u32>().ok()))
        else {
            continue;
        };
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|at| at.elapsed().is_ok_and(|age| age > Duration::from_secs(STALE_SECS)));
        if old && pid != std::process::id() && !pid_alive(pid) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The exclusive update lock, or None when another updater holds it.
fn take_lock(data_dir: &Path) -> Option<std::fs::File> {
    let _ = std::fs::create_dir_all(data_dir);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(data_dir.join(FILE_LOCK))
        .ok()?;
    file.try_lock().ok()?;
    Some(file)
}

/// The uid this process creates files as: a fresh `O_EXCL` file in the data
/// dir, read and removed. (A new file's owner is the euid on Linux and macOS.)
#[cfg(unix)]
fn probe_euid(data_dir: &Path) -> Option<u32> {
    use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
    let path = data_dir.join(format!(".euid-probe-{}", std::process::id()));
    let file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).ok()?;
    let uid = file.metadata().ok().map(|m| m.uid());
    drop(file);
    let _ = std::fs::remove_file(&path);
    uid
}

// ---- where the binary lives ------------------------------------------------

fn bin_dir() -> Option<PathBuf> {
    match std::env::var_os("BRAIN_BIN_DIR") {
        Some(dir) if !dir.is_empty() => Some(PathBuf::from(dir)),
        _ => dirs::home_dir().map(|home| home.join(".local").join("bin")),
    }
}

/// The canonical bootstrap directory, when `exe` is `brain` in it.
fn install_dir(exe: &Path, bin_dir: &Path) -> Result<PathBuf, &'static str> {
    let dir = std::fs::canonicalize(bin_dir).map_err(|_| "not-bootstrap-install")?;
    let exe = std::fs::canonicalize(exe).map_err(|_| "not-bootstrap-install")?;
    if exe.parent() != Some(dir.as_path()) || exe.file_name().is_none_or(|n| n != "brain") {
        return Err("not-bootstrap-install");
    }
    Ok(dir)
}

/// The directory and `brain` must be ours, `brain` a plain file (never a
/// symlink), and neither writable by others. Group write is allowed: whoever
/// can write the directory can replace `brain` without the updater, and a
/// umask of 002 makes `~/.local/bin` 775 on many distributions.
#[cfg(unix)]
fn bin_safety(dir: &Path, euid: u32) -> Result<(), &'static str> {
    use std::os::unix::fs::MetadataExt as _;
    let d = std::fs::symlink_metadata(dir).map_err(|_| "bin-unsafe")?;
    let b = std::fs::symlink_metadata(dir.join("brain")).map_err(|_| "bin-unsafe")?;
    let sound = d.is_dir() && d.uid() == euid && d.mode() & 0o002 == 0;
    let plain = b.is_file() && !b.file_type().is_symlink() && b.uid() == euid && b.mode() & 0o002 == 0;
    if sound && plain { Ok(()) } else { Err("bin-unsafe") }
}

// ---- self-test and placement -----------------------------------------------

/// Run `cmd` for at most `secs`; kill it past that. Some(stdout) on a clean
/// exit, None on a failure or a timeout.
fn run_bounded(mut cmd: Command, secs: u64) -> Option<String> {
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut out = String::new();
                if let Some(mut pipe) = child.stdout.take() {
                    use std::io::Read as _;
                    let _ = pipe.by_ref().take(4096).read_to_string(&mut out);
                }
                return Some(out);
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

fn child_command(exe: &Path, data_dir: &Path, arg: &str) -> Command {
    let mut cmd = Command::new(exe);
    clean(&mut cmd);
    if let Some(home) = std::env::var_os("HOME") {
        cmd.env("HOME", home);
    }
    cmd.arg(arg)
        .env("ROLEPOD_BRAIN_HOME", data_dir)
        .env("ROLEPOD_BRAIN_NO_FETCH", "1")
        .env("ROLEPOD_BRAIN_HUB", "off")
        .env("ROLEPOD_BRAIN_NO_UPDATE", "1");
    cmd
}

#[derive(Debug, PartialEq, Eq)]
enum SelfTest {
    Pass,
    VersionMismatch,
    Failed,
}

fn self_test_file(exe: &Path, data_dir: &Path, version: &str) -> SelfTest {
    match run_bounded(child_command(exe, data_dir, "--version"), SELFTEST_SECS) {
        Some(out) if version_output_matches(&out, version) => {}
        Some(_) => return SelfTest::VersionMismatch,
        None => return SelfTest::Failed,
    }
    if run_bounded(child_command(exe, data_dir, "self-test"), SELFTEST_SECS).is_some() {
        SelfTest::Pass
    } else {
        SelfTest::Failed
    }
}

/// `codesign -s - -f` on macOS (an unsigned Mach-O is never placed); `needed`
/// is false elsewhere.
fn sign_step(path: &Path, needed: bool, run: &mut dyn FnMut(&Path) -> bool) -> Result<(), &'static str> {
    if needed && !run(path) { Err("codesign-failed") } else { Ok(()) }
}

fn run_codesign(path: &Path) -> bool {
    let Some(mut codesign) = tool_command("codesign") else { return false };
    codesign
        .args(["-s", "-", "-f"])
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The old binary becomes `brain.prev` (a hard link, so the same inode a
/// running process holds), then one `rename` puts the new one over `bin`.
/// `bin` exists at every step; `seen` is told each step after it ran.
#[cfg(unix)]
fn place(dir: &Path, staging: &Path, seen: &mut dyn FnMut(&str)) -> std::io::Result<()> {
    let bin = dir.join("brain");
    let tmp = dir.join(format!(".brain.prev-{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    if std::fs::hard_link(&bin, &tmp).is_err() {
        std::fs::copy(&bin, &tmp)?;
    }
    seen("linked");
    std::fs::rename(&tmp, dir.join("brain.prev"))?;
    seen("prev");
    std::fs::rename(staging, &bin)?;
    seen("placed");
    Ok(())
}

/// The last look before the rename: `staging` must still be the file that was
/// verified and self-tested (`before`), else nothing is placed.
#[cfg(unix)]
fn place_if_unchanged(dir: &Path, staging: &Path, before: &Snap, seen: &mut dyn FnMut(&str)) -> Result<(), &'static str> {
    if snap(staging).as_ref() != Some(before) {
        return Err("staging-changed");
    }
    place(dir, staging, seen).map_err(|_| "place-failed")
}

// ---- state -----------------------------------------------------------------

pub(crate) fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0))
}

/// The bad versions: `stored` (the `update_bad` state, possibly empty) and
/// `update.bad` in the data dir, merged without repeats.
pub(crate) fn bad_versions(stored: &str, data_dir: &Path) -> Vec<String> {
    let filed = std::fs::read_to_string(data_dir.join(FILE_BAD)).unwrap_or_default();
    let mut list: Vec<String> = Vec::new();
    for v in stored.split_whitespace().chain(filed.split_whitespace()) {
        if !list.iter().any(|x| x == v) {
            list.push(v.to_string());
        }
    }
    list
}

/// What the updater tells `schema_state`. A store that cannot be opened is
/// not an error: `update.bad` in the data dir still carries the bad list.
struct Record {
    store: Option<Store>,
    data_dir: PathBuf,
}

impl Record {
    fn open(paths: &Paths) -> Self {
        Self { store: Store::open(&paths.db()).ok(), data_dir: paths.data_dir.clone() }
    }

    fn set(&self, key: &str, value: &str) {
        if let Some(store) = &self.store {
            let _ = store.set_state(key, value);
        }
    }

    fn skip(&self, reason: &str) {
        self.set(STATE_SKIP, &format!("{reason}@{}", now_secs()));
    }

    fn bad_list(&self) -> Vec<String> {
        let stored = self.store.as_ref().and_then(|s| s.state(STATE_BAD).ok().flatten()).unwrap_or_default();
        bad_versions(&stored, &self.data_dir)
    }

    /// A new install, recorded BEFORE the rename: in the store and in plain
    /// files a rollback can read when the store is what broke. A record for a
    /// version that never got placed is harmless, because the hook gate
    /// compares it with the running version.
    fn installing(&self, version: &str) {
        let installed = format!("{version} {}", now_secs() * 1000);
        self.set(STATE_INSTALLED, &installed);
        let _ = std::fs::write(self.data_dir.join(FILE_INSTALLED), &installed);
        let prev = env!("CARGO_PKG_VERSION");
        self.set(STATE_PREV, prev);
        let _ = std::fs::write(self.data_dir.join(FILE_PREV), prev);
    }

    fn mark_bad(&self, version: &str) {
        let mut list = self.bad_list();
        if !list.iter().any(|v| v == version) {
            list.push(version.to_string());
        }
        if list.len() > BAD_CAP {
            list.drain(..list.len() - BAD_CAP);
        }
        let joined = list.join(" ");
        self.set(STATE_BAD, &joined);
        let _ = std::fs::write(self.data_dir.join(FILE_BAD), joined);
    }
}

// ---- the run ---------------------------------------------------------------

/// Why a run stopped without installing.
enum Stop {
    Skip(&'static str),
    /// A release not yet a day old, with its age in seconds.
    Young(i64),
    Bad(&'static str, String),
}

/// The stop for a `decide` skip: a too-young release carries its age.
fn stop_for(reason: &'static str, published: Option<&str>, now: i64) -> Stop {
    let age = published.and_then(|p| p.parse::<jiff::Timestamp>().ok()).map(|at| now.saturating_sub(at.as_second()));
    match (reason, age) {
        ("too-young", Some(age)) => Stop::Young(age.max(0)),
        _ => Stop::Skip(reason),
    }
}

/// What a reader acts on, for a stop. Plain words: no URL, no path.
fn stop_text(stop: &Stop) -> String {
    let why = match stop {
        Stop::Young(age) => {
            let ago = if *age < 3600 { "under an hour ago".to_string() } else { format!("{} h ago", age / 3600) };
            format!("the latest release was published {ago}; it installs once it is a day old")
        }
        Stop::Skip(reason) | Stop::Bad(reason, _) => match *reason {
            "bad-tag" => "the latest release has a tag brain cannot read",
            "prerelease" => "the latest release is a pre-release",
            "running-unparsable" => "this brain's own version cannot be read",
            "marked-bad" => "the latest release failed its self-test here before, so it is skipped",
            "published-unknown" => "the latest release has no readable publish time; it waits until it has one",
            "not-bootstrap-install" => "this brain was not installed by the bootstrap script, so it is not replaced",
            "bin-unsafe" => "the install directory or the brain file is not yours or is writable by others",
            "staging-failed" => "could not create a staging file next to brain",
            "no-key" => "this build has no signing key to check releases with",
            "bad-key" => "this build's signing key cannot be read",
            "fetch-failed" => "could not fetch the release information",
            "unsigned" => "the latest release has no signature",
            "download-failed" => "the download failed or was too large",
            "bad-signature" => "the signature does not match the download",
            "signed-target-mismatch" => "the signature is for another platform",
            "signed-version-mismatch" => "the signed version is not the release's version",
            "codesign-failed" => "the new binary could not be code-signed",
            "staging-changed" => "the downloaded file changed before it was placed",
            "place-failed" => "the new binary could not be put in place",
            "selftest-version" => "the new binary reports a different version than it was signed as",
            "selftest-failed" => "the new binary failed its self-test",
            _ => "the update stopped for an unexpected reason",
        }
        .to_string(),
    };
    format!("not updated: {why}")
}

impl From<&'static str> for Stop {
    fn from(reason: &'static str) -> Self {
        Self::Skip(reason)
    }
}

/// `brain update`: install the newest signed release, or change nothing.
/// Never fails loudly and never touches `brain.log`. Returns the one line
/// that says what happened, for the caller to print.
pub fn run(paths: &Paths, config: &Config) -> String {
    // Opt-out returns before anything else, curl above all.
    if let Readiness::Off(why) = off_reason(config) {
        return format!("updates are off: {why}");
    }
    #[cfg(unix)]
    return unix_run(paths);
    #[cfg(not(unix))]
    {
        let _ = paths;
        String::new()
    }
}

#[cfg(unix)]
fn unix_run(paths: &Paths) -> String {
    let Some(triple) = target() else { return "updates are off: no release build for this platform".into() };
    let Some(_lock) = take_lock(&paths.data_dir) else { return "another update is running".into() };
    let record = Record::open(paths);
    let outcome = attempt(paths, triple, &record);
    let line = match outcome {
        Ok(version) => {
            let _ = record.store.as_ref().map(|s| s.clear_state(STATE_SKIP));
            format!("installed {version} (was {})", env!("CARGO_PKG_VERSION"))
        }
        Err(stop) => {
            let line = match &stop {
                Stop::Skip("not-newer") => format!("already on {}, the latest release", env!("CARGO_PKG_VERSION")),
                other => stop_text(other),
            };
            match stop {
                Stop::Skip(reason) => record.skip(reason),
                Stop::Young(_) => record.skip("too-young"),
                Stop::Bad(reason, version) => {
                    record.mark_bad(&version);
                    record.skip(reason);
                }
            }
            line
        }
    };
    let _ = std::fs::remove_file(paths.data_dir.join(FILE_RUNNING));
    line
}

#[cfg(unix)]
fn attempt(paths: &Paths, triple: &str, record: &Record) -> Result<String, Stop> {
    use std::os::unix::fs::PermissionsExt as _;
    let data = &paths.data_dir;
    let exe = std::env::current_exe().map_err(|_| Stop::Skip("not-bootstrap-install"))?;
    let dir = install_dir(&exe, &bin_dir().ok_or("not-bootstrap-install")?)?;
    let euid = probe_euid(data).ok_or("bin-unsafe")?;
    bin_safety(&dir, euid)?;
    // Only a directory that passed the check is swept.
    sweep_stale(&dir);
    record.set(STATE_CHECKED_AT, &now_secs().to_string());
    let key = pubkey().ok_or("no-key")?;
    let src = source();
    let pid = std::process::id();

    // Release metadata, small and bounded.
    let meta_file = Owned(data.join(format!("update.meta-{pid}")));
    if !fetch(&src, &src.meta, &meta_file.0, MAX_JSON, 30) {
        return Err("fetch-failed".into());
    }
    let meta: serde_json::Value = std::fs::read(&meta_file.0)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or("fetch-failed")?;
    let tag = meta["tag_name"].as_str().ok_or("bad-tag")?.to_string();
    let published = meta["published_at"].as_str().map(str::to_string);
    let running = env!("CARGO_PKG_VERSION");
    // tag_name only decides whether to download; the version used after
    // that is the signed one.
    let bad = record.bad_list();
    if let Verdict::Skip(reason) = decide(running, &tag, published.as_deref(), &bad, now_secs()) {
        return Err(stop_for(reason, published.as_deref(), now_secs()));
    }

    let name = format!("brain-{triple}");
    let sig_url = src.asset_url(&tag, &format!("{name}.minisig")).ok_or("bad-tag")?;
    let bin_url = src.asset_url(&tag, &name).ok_or("bad-tag")?;
    let sig_file = Owned(data.join(format!("update.sig-{pid}")));
    if !fetch(&src, &sig_url, &sig_file.0, MAX_SIG, 30) {
        return Err("unsigned".into());
    }
    let sig = std::fs::read_to_string(&sig_file.0).unwrap_or_default();

    let staging = create_staging(&dir).map_err(|_| Stop::Skip("staging-failed"))?;
    if !fetch(&src, &bin_url, &staging.0, max_bin(), 300) {
        return Err("download-failed".into());
    }
    let bytes = std::fs::read(&staging.0).map_err(|_| Stop::Skip("download-failed"))?;
    let version = accept(&bytes, &sig, &key, triple, &tag)?;
    drop(bytes);
    // The version that counts is the signed one: decide again on it.
    if let Verdict::Skip(reason) = decide(running, &version, published.as_deref(), &bad, now_secs()) {
        return Err(stop_for(reason, published.as_deref(), now_secs()));
    }

    std::fs::set_permissions(&staging.0, std::fs::Permissions::from_mode(0o755)).map_err(|_| Stop::Skip("staging-failed"))?;
    sign_step(&staging.0, cfg!(target_os = "macos"), &mut |p| run_codesign(p))?;
    // The snapshot is taken AFTER codesign on purpose: codesign rewrites the
    // Mach-O, so a snapshot from before it could never match. What makes that
    // step trusted is the absolute SIP path of `codesign`, not the snapshot.
    let before = snap(&staging.0).ok_or("staging-changed")?;
    match self_test_file(&staging.0, data, &version) {
        SelfTest::Pass => {}
        SelfTest::VersionMismatch => return Err(Stop::Bad("selftest-version", version)),
        SelfTest::Failed => return Err(Stop::Bad("selftest-failed", version)),
    }
    record.installing(&version);
    place_if_unchanged(&dir, &staging.0, &before, &mut |_| {})?;
    Ok(version)
}

/// `brain self-test`: open the store read-only. Run by the updater on a
/// binary it is about to place; it writes nothing, WAL included.
///
/// # Errors
/// Returns an error when an existing store cannot be read.
pub fn self_test(paths: &Paths) -> Result<()> {
    let db = paths.db();
    if !db.exists() {
        return Ok(());
    }
    // `immutable` is what keeps a read-only open from creating `-wal` and
    // `-shm` beside a store nobody else has open.
    let mut uri = String::from("file:");
    for byte in db.to_string_lossy().bytes() {
        if byte.is_ascii_alphanumeric() || b"/-_.".contains(&byte) {
            uri.push(char::from(byte));
        } else {
            uri.push_str(&format!("%{byte:02X}"));
        }
    }
    uri.push_str("?immutable=1");
    let conn = rusqlite::Connection::open_with_flags(
        &uri,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context("open the store read-only")?;
    let _: i64 = conn
        .query_row("SELECT count(*) FROM schema_state", [], |row| row.get(0))
        .context("read schema_state")?;
    println!("ok");
    Ok(())
}

// ---- rollback --------------------------------------------------------------

/// Hooks in a row that must fail on a fresh install before it is rolled back.
#[cfg(unix)]
const FAIL_LIMIT: u32 = 3;
/// A rollback only ever happens in the first hour after an install.
#[cfg(unix)]
const WINDOW_MS: i64 = 3600 * 1000;

/// `update.failures`: line one `<version> <count>`, then one `pid <n>` line
/// for every hook of that version that has begun and not yet ended.
#[cfg(unix)]
#[derive(Debug, Default, PartialEq, Eq)]
struct Failures {
    version: String,
    count: u32,
    pids: Vec<u32>,
}

#[cfg(unix)]
impl Failures {
    fn parse(text: &str) -> Self {
        let mut lines = text.lines();
        let mut head = lines.next().unwrap_or_default().split_whitespace();
        let version = head.next().unwrap_or_default().to_string();
        let count = head.next().and_then(|n| n.parse().ok()).unwrap_or(0);
        let pids = lines.filter_map(|l| l.strip_prefix("pid ")?.trim().parse().ok()).collect();
        Self { version, count, pids }
    }

    fn render(&self) -> String {
        let mut out = format!("{} {}\n", self.version, self.count);
        for pid in &self.pids {
            out.push_str(&format!("pid {pid}\n"));
        }
        out
    }

    /// A hook of `running` starts as `me`. An entry whose process is gone
    /// ended without a word (abort, kill): it counts as one failure. A live
    /// one is still running and counts for nothing. Returns the count.
    fn begin(&mut self, running: &str, me: u32, alive: &dyn Fn(u32) -> bool) -> u32 {
        if self.version != running {
            *self = Self { version: running.to_string(), ..Self::default() };
        }
        let pids = std::mem::take(&mut self.pids);
        for pid in pids {
            if pid == me || alive(pid) {
                self.pids.push(pid);
            } else {
                self.count += 1;
            }
        }
        if !self.pids.contains(&me) {
            self.pids.push(me);
        }
        self.count
    }

    /// The hook `me` ended: a success clears the run of failures, an error
    /// adds one. Returns the count.
    fn end(&mut self, running: &str, me: u32, ok: bool) -> u32 {
        if self.version != running {
            *self = Self { version: running.to_string(), ..Self::default() };
        }
        self.pids.retain(|p| *p != me);
        self.count = if ok { 0 } else { self.count + 1 };
        self.count
    }
}

/// Run `f` on the failures file under a short exclusive lock. None when the
/// file or the lock cannot be had: a hook never waits on it.
#[cfg(unix)]
fn locked_failures<T>(path: &Path, f: impl FnOnce(&mut Failures) -> T) -> Option<T> {
    use std::io::{Read as _, Seek as _, Write as _};
    let mut file = std::fs::OpenOptions::new().create(true).read(true).write(true).truncate(false).open(path).ok()?;
    let mut tries = 0;
    while file.try_lock().is_err() {
        tries += 1;
        if tries > 100 {
            return None;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let mut text = String::new();
    file.read_to_string(&mut text).ok()?;
    let mut fails = Failures::parse(&text);
    let out = f(&mut fails);
    file.set_len(0).ok()?;
    file.rewind().ok()?;
    file.write_all(fails.render().as_bytes()).ok()?;
    Some(out)
}

/// Is `update_installed` (`<version> <unix_ms>`) this version, put there less
/// than an hour ago?
#[cfg(unix)]
fn installed_recently(value: &str, running: &str, now_ms: i64) -> bool {
    let mut parts = value.split_whitespace();
    let (Some(version), Some(at)) = (parts.next(), parts.next().and_then(|n| n.parse::<i64>().ok())) else { return false };
    version == running && at > 0 && now_ms.saturating_sub(at) < WINDOW_MS
}

/// Watches one hook of a fresh install: see [`hook_begin`].
pub struct HookGuard {
    #[cfg(unix)]
    paths: Paths,
    /// The count was already at the limit when this hook began.
    #[cfg(unix)]
    due: bool,
}

/// Called first thing by a hook. Only a binary put in place within the last
/// hour, and recorded in `update.installed`, is watched. Any other hook does
/// one `stat` of its own executable and nothing else: no `update.*` file is
/// touched. The store is never opened here (capture must reach its own log
/// before it waits on the store); a rollback waits for [`HookGuard::end`].
/// Never an error: None means no watch.
#[cfg(unix)]
#[must_use]
pub fn hook_begin() -> Option<HookGuard> {
    let exe = std::env::current_exe().ok()?;
    let fresh = std::fs::metadata(exe).and_then(|m| m.modified()).is_ok_and(|at| match at.elapsed() {
        Ok(age) => age.as_millis() < WINDOW_MS.unsigned_abs().into(),
        Err(_) => true,
    });
    if !fresh {
        return None;
    }
    let paths = Paths::resolve().ok()?;
    let running = env!("CARGO_PKG_VERSION");
    let installed = std::fs::read_to_string(paths.data_dir.join(FILE_INSTALLED)).ok()?;
    if !installed_recently(&installed, running, now_secs() * 1000) {
        return None;
    }
    let me = std::process::id();
    let count = locked_failures(&paths.data_dir.join(FILE_FAILURES), |f| f.begin(running, me, &pid_alive))?;
    Some(HookGuard { paths, due: count >= FAIL_LIMIT })
}

#[cfg(not(unix))]
#[must_use]
pub fn hook_begin() -> Option<HookGuard> {
    None
}

impl HookGuard {
    /// The hook is over: `ok` is whether capture returned without an error.
    pub fn end(self, ok: bool) {
        #[cfg(unix)]
        {
            let me = std::process::id();
            let path = self.paths.data_dir.join(FILE_FAILURES);
            let count = locked_failures(&path, |f| f.end(env!("CARGO_PKG_VERSION"), me, ok));
            if self.due || count.is_some_and(|c| c >= FAIL_LIMIT) {
                rollback(&self.paths);
            }
        }
        #[cfg(not(unix))]
        let _ = (self, ok);
    }
}

/// Restore `brain.prev` over the fresh install. One at a time (`update.lock`);
/// busy means the next hook tries again with the count kept. It reads what it
/// needs from plain files in the data dir and works without the store: a
/// store the new build cannot open is exactly when a rollback is wanted.
#[cfg(unix)]
fn rollback(paths: &Paths) {
    let data = &paths.data_dir;
    let Some(_lock) = take_lock(data) else { return };
    let record = Record::open(paths);
    let running = env!("CARGO_PKG_VERSION");
    // Under the lock: another hook may have rolled back while this one waited.
    let installed = std::fs::read_to_string(data.join(FILE_INSTALLED)).unwrap_or_default();
    if installed_recently(&installed, running, now_secs() * 1000) {
        let prev = std::fs::read_to_string(data.join(FILE_PREV)).unwrap_or_default().trim().to_string();
        let done = (|| {
            let exe = std::env::current_exe().map_err(|_| "rollback-unavailable")?;
            let dir = install_dir(&exe, &bin_dir().ok_or("rollback-unavailable")?).map_err(|_| "rollback-unavailable")?;
            let euid = probe_euid(data).ok_or("rollback-unavailable")?;
            bin_safety(&dir, euid).map_err(|_| "rollback-unavailable")?;
            check_prev(&dir, data, &prev, euid)?;
            // Bad before the swap: a crash in between still never reinstalls it.
            record.mark_bad(running);
            restore_prev(&dir)
        })();
        match done {
            Ok(()) => {
                // Window closed: the restored version is the installed one.
                record.set(STATE_INSTALLED, &format!("{prev} 0"));
                let _ = std::fs::write(data.join(FILE_INSTALLED), format!("{prev} 0"));
                record.skip("rolled-back");
            }
            Err(reason) => record.skip(reason),
        }
    }
    let _ = std::fs::remove_file(data.join(FILE_FAILURES));
}

/// `brain.prev` is ours, a plain file, and says it is the version the updater
/// recorded as replaced. A missing or other `prev` is never invented.
#[cfg(unix)]
fn check_prev(dir: &Path, data: &Path, prev_version: &str, euid: u32) -> Result<(), &'static str> {
    use std::os::unix::fs::MetadataExt as _;
    let path = dir.join("brain.prev");
    let meta = std::fs::symlink_metadata(&path).map_err(|_| "rollback-unavailable")?;
    if !meta.is_file() || meta.uid() != euid || prev_version.is_empty() {
        return Err("rollback-unavailable");
    }
    match run_bounded(child_command(&path, data, "--version"), SELFTEST_SECS) {
        Some(out) if version_output_matches(&out, prev_version) => Ok(()),
        _ => Err("rollback-unavailable"),
    }
}

/// Put `brain.prev` over `brain` without ever opening `brain` for writing
/// (a running executable cannot be, on Linux): hard link to a temp name,
/// then one rename.
#[cfg(unix)]
fn restore_prev(dir: &Path) -> Result<(), &'static str> {
    use std::os::unix::fs::PermissionsExt as _;
    let prev = dir.join("brain.prev");
    let tmp = dir.join(format!(".brain.rollback-{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    if std::fs::hard_link(&prev, &tmp).is_err() {
        std::fs::copy(&prev, &tmp).map_err(|_| "rollback-unavailable")?;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755));
    }
    std::fs::rename(&tmp, dir.join("brain")).map_err(|_| {
        let _ = std::fs::remove_file(&tmp);
        "rollback-unavailable"
    })
}

/// `Off` with the reason when config, env or the platform turn the updater
/// off; `Ready` here only means none of those did.
fn off_reason(config: &Config) -> Readiness {
    if !cfg!(unix) {
        return Readiness::Off("not available on this platform");
    }
    if !config.update.enabled() {
        return Readiness::Off(if config.update.auto { "ROLEPOD_BRAIN_NO_UPDATE is set" } else { "auto = false in config" });
    }
    Readiness::Ready
}

/// Why the updater is or is not able to act on this machine, for `doctor`.
/// Reads config, env, the key and the path of this executable; no network.
#[derive(Debug, PartialEq, Eq)]
pub enum Readiness {
    Off(&'static str),
    NoKey,
    Elsewhere,
    Ready,
}

#[must_use]
pub fn readiness(config: &Config) -> Readiness {
    let off = off_reason(config);
    if off != Readiness::Ready {
        return off;
    }
    if target().is_none() {
        return Readiness::Off("no release build for this platform");
    }
    if pubkey().is_none() {
        return Readiness::NoKey;
    }
    let installed_here = match (std::env::current_exe(), bin_dir()) {
        (Ok(exe), Some(bin)) => install_dir(&exe, &bin).is_ok(),
        _ => false,
    };
    if installed_here { Readiness::Ready } else { Readiness::Elsewhere }
}

/// Is a daily `brain update` worth starting here? Only when the updater is
/// [`Readiness::Ready`]: opted in, a supported platform, a key to verify with,
/// and this executable being `brain` in the bootstrap dir. Otherwise the run
/// would only record a skip.
#[must_use]
pub fn spawnable(config: &Config) -> bool {
    readiness(config) == Readiness::Ready
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/update")
    }

    fn tmp(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!("upd-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_key() -> String {
        std::fs::read_to_string(fixtures().join("test.pub")).unwrap().lines().nth(1).unwrap().to_string()
    }

    fn triple() -> &'static str {
        target().expect("a unix test host")
    }

    fn sig(kind: &str) -> String {
        std::fs::read_to_string(fixtures().join(format!("{kind}.{}.minisig", triple()))).unwrap()
    }

    fn good() -> Vec<u8> {
        std::fs::read(fixtures().join("bin-good")).unwrap()
    }

    #[test]
    fn tampered_binary_byte_rejected() {
        let mut bytes = good();
        assert_eq!(accept(&bytes, &sig("good"), &test_key(), triple(), "v9.0.0"), Ok("9.0.0".to_string()));
        bytes[3] ^= 1;
        assert_eq!(accept(&bytes, &sig("good"), &test_key(), triple(), "v9.0.0"), Err("bad-signature"));
    }

    #[test]
    fn trusted_comment_other_target_rejected() {
        assert_eq!(accept(&good(), &sig("wrongtarget"), &test_key(), triple(), "v9.0.0"), Err("signed-target-mismatch"));
    }

    #[test]
    fn trusted_comment_other_version_rejected() {
        assert_eq!(accept(&good(), &sig("wrongversion"), &test_key(), triple(), "v9.0.0"), Err("signed-version-mismatch"));
    }

    #[test]
    fn trusted_comment_exact_ok() {
        assert_eq!(accept(&good(), &sig("good"), &test_key(), triple(), "9.0.0"), Ok("9.0.0".to_string()));
    }

    #[test]
    fn comment_version_beats_tag_name() {
        // A forged far-future tag with an older signed binary is refused.
        assert_eq!(accept(&good(), &sig("good"), &test_key(), triple(), "v99.0.0"), Err("signed-version-mismatch"));
        // And a tag the decision would accept does not make an unsigned thing pass.
        assert_eq!(accept(&good(), "", &test_key(), triple(), "v9.0.0"), Err("unsigned"));
    }

    #[test]
    fn empty_minisig_rejected() {
        assert_eq!(accept(&good(), "  \n", &test_key(), triple(), "v9.0.0"), Err("unsigned"));
        assert_eq!(accept(&good(), "garbage", &test_key(), triple(), "v9.0.0"), Err("bad-signature"));
    }

    #[test]
    fn decide_table() {
        let now = 1_000_000_000_i64;
        let ts = |ago: i64| jiff::Timestamp::from_second(now - ago).unwrap().to_string();
        let old = ts(25 * 3600);
        let d = |run: &str, cand: &str, at: Option<&str>, bad: &[&str]| {
            let bad: Vec<String> = bad.iter().map(ToString::to_string).collect();
            decide(run, cand, at, &bad, now)
        };
        assert_eq!(d("0.68.1", "v0.69.0", Some(&old), &[]), Verdict::Install);
        assert_eq!(d("0.68.1", "v0.68.1", Some(&old), &[]), Verdict::Skip("not-newer"));
        assert_eq!(d("0.68.1", "v0.68.0", Some(&old), &[]), Verdict::Skip("not-newer"));
        assert_eq!(d("0.68.1", "v0.70.0-rc.1", Some(&old), &[]), Verdict::Skip("prerelease"));
        assert_eq!(d("0.70.0", "v0.69.0", Some(&old), &[]), Verdict::Skip("not-newer"), "a source build ahead of the release");
        assert_eq!(d("0.68.1", "v0.69.0", Some(&old), &["0.69.0"]), Verdict::Skip("marked-bad"));
        assert_eq!(d("0.68.1", "v0.69.0", Some(&ts(23 * 3600)), &[]), Verdict::Skip("too-young"));
        assert_eq!(d("0.68.1", "v0.69.0", Some(&ts(25 * 3600)), &[]), Verdict::Install);
        assert_eq!(d("0.68.1", "v0.69.0", None, &[]), Verdict::Skip("published-unknown"));
        assert_eq!(d("0.68.1", "v0.69.0", Some("garbage"), &[]), Verdict::Skip("published-unknown"));
        assert_eq!(d("0.68.1", "latest", Some(&old), &[]), Verdict::Skip("bad-tag"));
        assert_eq!(d("0.9.0", "0.10.0", Some(&old), &[]), Verdict::Install, "numeric, not text, order");
    }

    #[test]
    fn decision_off_on_windows() {
        assert_eq!(platform_target("windows", "x86_64"), None);
        assert_eq!(platform_target("linux", "riscv64"), None);
        assert_eq!(platform_target("macos", "aarch64"), Some("aarch64-apple-darwin"));
        assert_eq!(platform_target("linux", "x86_64"), Some("x86_64-unknown-linux-gnu"));
        assert_eq!(platform_target("macos", "x86_64"), Some("x86_64-apple-darwin"));
        assert_eq!(platform_target("linux", "aarch64"), Some("aarch64-unknown-linux-gnu"));
    }

    #[test]
    fn version_output_parse_exact() {
        assert!(version_output_matches("brain 9.0.0\n", "9.0.0"));
        assert!(!version_output_matches("brain 9.0.0-evil", "9.0.0"));
        assert!(!version_output_matches("brain 9.0.1", "9.0.0"));
        assert!(!version_output_matches("brain 9.0.0\nextra", "9.0.0"));
        assert!(!version_output_matches("9.0.0", "9.0.0"));
    }

    #[test]
    fn no_embedded_key_installs_nothing() {
        // An explicit empty override is "no key" even when the constant holds one.
        assert_eq!(resolve_key(Some(String::new())), None);
        assert_eq!(resolve_key(None).as_deref(), UPDATE_PUBKEY);
        assert_eq!(resolve_key(Some("k".into())).as_deref(), Some("k"));
    }

    #[test]
    fn pubkey_const_matches_minisign_pub() {
        // One source of truth: `minisign.pub` is what CI verifies against and
        // what the binary embeds; one without the other is a broken release.
        let file = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/minisign.pub")).ok();
        match (UPDATE_PUBKEY, file) {
            (None, None) => {}
            (Some(key), Some(file)) => {
                assert_eq!(file.lines().nth(1), Some(key), "UPDATE_PUBKEY differs from minisign.pub");
                assert!(minisign_verify::PublicKey::from_base64(key).is_ok());
            }
            (Some(_), None) => panic!("UPDATE_PUBKEY is set but minisign.pub is missing"),
            (None, Some(_)) => panic!("minisign.pub exists but UPDATE_PUBKEY is None"),
        }
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn env_seam_absent_in_release() {
        std::env::set_var("ROLEPOD_BRAIN_UPDATE_URL", "/tmp/x");
        std::env::set_var("ROLEPOD_BRAIN_UPDATE_PUBKEY", "x");
        let s = source();
        assert_eq!(s.meta, RELEASES_API);
        assert_eq!(pubkey().as_deref(), UPDATE_PUBKEY);
    }

    #[test]
    fn asset_url_prefix_enforced() {
        let s = Source::release();
        let ok = s.asset_url("v1.2.3", "brain-x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(ok, "https://github.com/nuttaruj/rolepod-brain/releases/download/v1.2.3/brain-x86_64-unknown-linux-gnu");
        assert!(s.url_allowed(RELEASES_API));
        for bad in [
            "https://evil.example/nuttaruj/rolepod-brain/releases/download/v1/brain",
            "http://github.com/nuttaruj/rolepod-brain/releases/download/v1/brain",
            "file:///etc/passwd",
            "-o/tmp/x",
            "--output",
            "https://github.com/nuttaruj/rolepod-brain/releases/download/../../x",
            "https://github.com/nuttaruj/rolepod-brain/releases/download/v1/a b",
            "https://github.com/nuttaruj/rolepod-brain/releases/download/",
        ] {
            assert!(!s.url_allowed(bad), "{bad}");
        }
        assert!(s.asset_url("v1/../..", "x").is_none());
    }

    fn arg_strings(args: &[OsString]) -> Vec<String> {
        args.iter().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn curl_args_have_proto_and_size_caps() {
        let a = arg_strings(&curl_args("https://x/y", Path::new("/o"), 1234, 60, false));
        let joined = a.join(" ");
        for need in ["--fail", "--proto =https", "--proto-redir =https", "--max-filesize 1234", "--max-time 60"] {
            let ok = joined.contains(need) || (need == "--fail" && a[1].contains('f'));
            assert!(ok, "{need} in {joined}");
        }
    }

    #[test]
    fn curl_args_exact_no_user_data() {
        let a = arg_strings(&curl_args("https://x/y", Path::new("/o"), 10, 5, false));
        let ua = format!("rolepod-brain-update/{}", env!("CARGO_PKG_VERSION"));
        let want = [
            "-q", "-fsSL", "--proto", "=https", "--proto-redir", "=https", "--max-filesize", "10", "--max-time", "5", "-A", &ua, "-o", "/o",
            "--url", "https://x/y",
        ];
        assert_eq!(a, want);
        assert_eq!(a[0], "-q", "-q must come first or curl reads ~/.curlrc");
    }

    #[test]
    fn tool_resolver_never_returns_a_path_entry() {
        // The resolver takes no PATH: a fake tool first on it cannot be chosen.
        let fake = tmp("tools");
        for name in ["curl", "codesign", "kill"] {
            std::fs::write(fake.join(name), "#!/bin/sh\nexit 0\n").unwrap();
            std::fs::set_permissions(fake.join(name), std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        for name in ["curl", "codesign", "kill"] {
            let got = fixed_tool(name);
            let fixed = ["/usr/bin/", "/bin/", "/usr/local/bin/"];
            assert!(got.as_ref().is_none_or(|p| fixed.iter().any(|d| p.starts_with(d)) && !p.starts_with(&fake)), "{name}: {got:?}");
        }
        assert_eq!(fixed_tool("rm"), None, "an unnamed tool is not resolved");
        // And a child is run with nothing of ours.
        let mut cmd = Command::new("/usr/bin/env");
        cmd.env("EVIL", "1");
        clean(&mut cmd);
        let out = cmd.output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "PATH=/usr/bin:/bin");
    }

    #[test]
    fn curl_carries_proxy_vars_and_not_ca_or_curlrc_vars() {
        let vars = ["https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY", "CURL_CA_BUNDLE", "CURL_HOME", "SSL_CERT_FILE", "SSL_CERT_DIR", "HOME", "http_proxy"];
        let mut cmd = Command::new("/usr/bin/curl");
        clean(&mut cmd);
        pass_proxy(&mut cmd, vars.iter().map(|v| (OsString::from(v), OsString::from("x"))));
        let carried: Vec<String> = cmd.get_envs().map(|(k, _)| k.to_string_lossy().into_owned()).collect();
        for want in ["https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY", "PATH"] {
            assert!(carried.iter().any(|k| k == want), "{want} in {carried:?}");
        }
        for out in ["CURL_CA_BUNDLE", "CURL_HOME", "SSL_CERT_FILE", "SSL_CERT_DIR", "HOME", "http_proxy"] {
            assert!(!carried.iter().any(|k| k == out), "{out} in {carried:?}");
        }
    }

    #[test]
    fn euid_probe_matches_and_leaves_nothing() {
        let dir = tmp("euid");
        assert_eq!(probe_euid(&dir), Some(me(&dir)));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    }

    #[test]
    fn place_rechecks_the_snapshot() {
        let dir = fake_install("recheck");
        let staging = create_staging(&dir).unwrap();
        std::fs::write(&staging.0, "verified").unwrap();
        let before = snap(&staging.0).unwrap();
        // Swapped for another file after the self-test: nothing is placed.
        let other = dir.join("other");
        std::fs::write(&other, "verified").unwrap();
        std::fs::rename(&other, &staging.0).unwrap();
        assert_eq!(place_if_unchanged(&dir, &staging.0, &before, &mut |_| {}), Err("staging-changed"));
        assert_eq!(std::fs::read_to_string(dir.join("brain")).unwrap(), "old");
        assert!(!dir.join("brain.prev").exists());
        // Unchanged: placed.
        let before = snap(&staging.0).unwrap();
        assert_eq!(place_if_unchanged(&dir, &staging.0, &before, &mut |_| {}), Ok(()));
        assert_eq!(std::fs::read_to_string(dir.join("brain")).unwrap(), "verified");
    }

    #[test]
    fn swapped_staging_file_detected_before_rename() {
        let dir = tmp("swap");
        let staging = create_staging(&dir).unwrap();
        std::fs::write(&staging.0, "verified").unwrap();
        let before = snap(&staging.0).unwrap();
        assert_eq!(snap(&staging.0).unwrap(), before);
        // Same bytes, another file under the name: the inode differs.
        let other = dir.join("other");
        std::fs::write(&other, "verified").unwrap();
        std::fs::rename(&other, &staging.0).unwrap();
        assert_ne!(snap(&staging.0).unwrap(), before);
        // Same inode, bytes changed: size or mtime differs.
        let before = snap(&staging.0).unwrap();
        std::fs::write(&staging.0, "verified, then edited").unwrap();
        assert_ne!(snap(&staging.0).unwrap(), before);
    }

    #[test]
    fn staging_created_excl_and_not_followed() {
        let dir = tmp("excl");
        let target = dir.join("victim");
        std::fs::write(&target, "keep").unwrap();
        let link = dir.join(format!(".brain.new-{}", std::process::id()));
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(create_staging(&dir).is_err());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep");
        std::fs::remove_file(&link).unwrap();
        let made = create_staging(&dir).unwrap();
        assert_eq!(std::fs::metadata(&made.0).unwrap().permissions().mode() & 0o777, 0o700);
        assert!(create_staging(&dir).is_err(), "a second create does not reuse the file");
    }

    fn fake_install(tag: &str) -> PathBuf {
        let dir = std::fs::canonicalize(tmp(tag)).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(dir.join("brain"), "old").unwrap();
        std::fs::set_permissions(dir.join("brain"), std::fs::Permissions::from_mode(0o755)).unwrap();
        dir
    }

    fn me(dir: &Path) -> u32 {
        std::fs::metadata(dir).unwrap().uid()
    }

    #[test]
    fn bin_dir_world_writable_refused_group_writable_allowed() {
        let dir = fake_install("gw");
        assert_eq!(bin_safety(&dir, me(&dir)), Ok(()));
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o775)).unwrap();
        assert_eq!(bin_safety(&dir, me(&dir)), Ok(()), "group write is allowed (umask 002)");
        std::fs::set_permissions(dir.join("brain"), std::fs::Permissions::from_mode(0o775)).unwrap();
        assert_eq!(bin_safety(&dir, me(&dir)), Ok(()));
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o707)).unwrap();
        assert_eq!(bin_safety(&dir, me(&dir)), Err("bin-unsafe"), "world-writable dir");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::set_permissions(dir.join("brain"), std::fs::Permissions::from_mode(0o757)).unwrap();
        assert_eq!(bin_safety(&dir, me(&dir)), Err("bin-unsafe"), "world-writable brain");
    }

    #[test]
    fn bin_dir_other_owner_refused() {
        let dir = fake_install("own");
        assert_eq!(bin_safety(&dir, me(&dir) + 1), Err("bin-unsafe"));
    }

    #[test]
    fn brain_is_symlink_refused() {
        let dir = fake_install("sym");
        std::fs::rename(dir.join("brain"), dir.join("real")).unwrap();
        std::os::unix::fs::symlink(dir.join("real"), dir.join("brain")).unwrap();
        assert_eq!(bin_safety(&dir, me(&dir)), Err("bin-unsafe"));
    }

    #[test]
    fn current_exe_not_bootstrap_dir_skipped() {
        let a = fake_install("exe-a");
        let b = fake_install("exe-b");
        assert_eq!(install_dir(&a.join("brain"), &a), Ok(a.clone()));
        assert_eq!(install_dir(&a.join("brain"), &b), Err("not-bootstrap-install"));
        std::fs::write(a.join("other"), "x").unwrap();
        assert_eq!(install_dir(&a.join("other"), &a), Err("not-bootstrap-install"));
        // A symlink named brain resolves elsewhere: not the bootstrap file.
        std::fs::rename(a.join("brain"), a.join("real")).unwrap();
        std::os::unix::fs::symlink(a.join("real"), a.join("brain")).unwrap();
        assert_eq!(install_dir(&a.join("brain"), &a), Err("not-bootstrap-install"));
    }

    #[test]
    fn stale_staging_swept_live_pid_kept() {
        let dir = tmp("sweep");
        let old = std::time::SystemTime::now() - Duration::from_secs(2 * STALE_SECS);
        let mk = |name: &str, aged: bool| {
            let path = dir.join(name);
            let file = std::fs::File::create(&path).unwrap();
            if aged {
                file.set_modified(old).unwrap();
            }
            path
        };
        let dead = mk(".brain.new-4000000", true);
        let young = mk(".brain.new-4000001", false);
        let live = mk(&format!(".brain.new-{}", std::process::id()), true);
        // A live pid other than ours: the parent test runner.
        let parent = mk(&format!(".brain.new-{}", std::os::unix::process::parent_id()), true);
        let unrelated = mk("brain", true);
        let prev_dead = mk(".brain.prev-4000002", true);
        let rollback_dead = mk(".brain.rollback-4000003", true);
        let rollback_young = mk(".brain.rollback-4000004", false);
        let prev_file = mk("brain.prev", true);
        sweep_stale(&dir);
        assert!(!prev_dead.exists() && !rollback_dead.exists(), "dead .brain.prev-/.brain.rollback-: swept");
        assert!(rollback_young.exists() && prev_file.exists(), "young temp and brain.prev itself: kept");
        assert!(!dead.exists(), "dead pid, old: swept");
        assert!(young.exists(), "young: kept");
        assert!(live.exists() && parent.exists(), "live pid: kept");
        assert!(unrelated.exists());
    }

    #[test]
    fn codesign_failure_aborts_before_rename() {
        let dir = fake_install("cs");
        let staging = create_staging(&dir).unwrap();
        std::fs::write(&staging.0, "new").unwrap();
        let mut ran = false;
        let result = sign_step(&staging.0, true, &mut |_| {
            ran = true;
            false
        });
        assert!(ran);
        assert_eq!(result, Err("codesign-failed"));
        assert_eq!(std::fs::read_to_string(dir.join("brain")).unwrap(), "old", "the installed binary is untouched");
        assert_eq!(sign_step(&staging.0, false, &mut |_| false), Ok(()), "not needed off macOS");
    }

    #[test]
    fn place_never_removes_target() {
        let dir = fake_install("place");
        let old_ino = std::fs::metadata(dir.join("brain")).unwrap().ino();
        let staging = create_staging(&dir).unwrap();
        std::fs::write(&staging.0, "new").unwrap();
        let mut steps = Vec::new();
        place(&dir, &staging.0, &mut |s| {
            assert!(dir.join("brain").exists(), "path empty after {s}");
            assert!(std::fs::read(dir.join("brain")).is_ok_and(|b| !b.is_empty()));
            steps.push(s.to_string());
        })
        .unwrap();
        assert_eq!(steps, ["linked", "prev", "placed"]);
        assert_eq!(std::fs::read_to_string(dir.join("brain")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(dir.join("brain.prev")).unwrap(), "old");
        assert_eq!(std::fs::metadata(dir.join("brain.prev")).unwrap().ino(), old_ino, "prev is the original inode");
        assert!(!staging.0.exists());
    }

    #[test]
    fn second_updater_exits_without_work() {
        let dir = tmp("lock");
        let first = take_lock(&dir);
        assert!(first.is_some());
        assert!(take_lock(&dir).is_none());
        drop(first);
        // A process forked by a parallel test holds the descriptor until its
        // exec, so the release can take a moment.
        let freed = (0..100).any(|_| {
            take_lock(&dir).is_some() || {
                std::thread::sleep(Duration::from_millis(20));
                false
            }
        });
        assert!(freed);
    }

    #[test]
    fn selftest_timeout_kills_child() {
        let started = Instant::now();
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        assert!(run_bounded(cmd, 1).is_none());
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!(run_bounded(Command::new("true"), 5), Some(String::new()));
        assert!(run_bounded(Command::new("false"), 5).is_none());
    }

    fn dead(_: u32) -> bool {
        false
    }

    fn alive(_: u32) -> bool {
        true
    }

    #[test]
    fn counter_success_between_failures_resets() {
        let mut f = Failures::default();
        f.begin("1.0.0", 10, &alive);
        assert_eq!(f.end("1.0.0", 10, false), 1);
        f.begin("1.0.0", 11, &alive);
        assert_eq!(f.end("1.0.0", 11, false), 2);
        f.begin("1.0.0", 12, &alive);
        assert_eq!(f.end("1.0.0", 12, true), 0);
        f.begin("1.0.0", 13, &alive);
        assert_eq!(f.end("1.0.0", 13, false), 1);
        assert!(f.pids.is_empty());
        assert_eq!(Failures::parse(&f.render()), f);
    }

    #[test]
    fn counter_ignores_other_version_and_expired_window() {
        let mut f = Failures { version: "0.9.0".into(), count: 2, pids: vec![] };
        assert_eq!(f.begin("1.0.0", 10, &alive), 0, "another version's count is dropped");
        let now = 10_000_000_i64;
        assert!(installed_recently("1.0.0 9999000", "1.0.0", now));
        assert!(!installed_recently("1.0.0 9999000", "1.0.1", now), "not the running version");
        assert!(!installed_recently(&format!("1.0.0 {}", now - WINDOW_MS), "1.0.0", now), "an hour old");
        assert!(!installed_recently("1.0.0", "1.0.0", now));
        assert!(!installed_recently("", "1.0.0", now));
    }

    #[test]
    fn pending_entry_dead_pid_counts_failure() {
        let mut f = Failures { version: "1.0.0".into(), count: 0, pids: vec![4_000_001, 4_000_002] };
        assert_eq!(f.begin("1.0.0", 10, &dead), 2);
        assert_eq!(f.pids, [10]);
    }

    #[test]
    fn pending_entry_live_pid_not_counted() {
        let mut f = Failures { version: "1.0.0".into(), count: 0, pids: vec![4_000_001, 4_000_002] };
        assert_eq!(f.begin("1.0.0", 10, &alive), 0);
        assert_eq!(f.pids, [4_000_001, 4_000_002, 10]);
    }

    #[test]
    fn bad_list_survives_when_store_unwritable() {
        let dir = tmp("badlist");
        let record = Record { store: None, data_dir: dir.clone() };
        record.mark_bad("9.9.9");
        assert_eq!(std::fs::read_to_string(dir.join("update.bad")).unwrap(), "9.9.9");
        let again = Record { store: None, data_dir: dir };
        assert_eq!(again.bad_list(), ["9.9.9"]);
    }

    fn with_prev(tag: &str, script: Option<&str>) -> PathBuf {
        let dir = fake_install(tag);
        if let Some(script) = script {
            std::fs::write(dir.join("brain.prev"), script).unwrap();
            std::fs::set_permissions(dir.join("brain.prev"), std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    }

    #[test]
    fn rollback_refuses_missing_prev() {
        let dir = with_prev("rb-missing", None);
        assert_eq!(check_prev(&dir, &dir, "8.0.0", me(&dir)), Err("rollback-unavailable"));
    }

    #[test]
    fn rollback_refuses_wrong_version_prev() {
        let dir = with_prev("rb-wrong", Some("#!/bin/sh\necho 'brain 7.7.7'\n"));
        assert_eq!(check_prev(&dir, &dir, "8.0.0", me(&dir)), Err("rollback-unavailable"));
        let ok = with_prev("rb-ok", Some("#!/bin/sh\necho 'brain 8.0.0'\n"));
        assert_eq!(check_prev(&ok, &ok, "8.0.0", me(&ok)), Ok(()));
        assert_eq!(check_prev(&ok, &ok, "8.0.0", me(&ok) + 1), Err("rollback-unavailable"), "not ours");
        assert_eq!(check_prev(&ok, &ok, "", me(&ok)), Err("rollback-unavailable"));
    }

    #[test]
    fn two_rollbacks_race_single_rename() {
        let dir = with_prev("rb-twice", Some("restored"));
        let before = std::fs::metadata(dir.join("brain.prev")).unwrap().ino();
        restore_prev(&dir).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("brain")).unwrap(), "restored");
        assert_eq!(std::fs::metadata(dir.join("brain")).unwrap().ino(), before, "a link, not a write in place");
        assert!(std::fs::read_dir(&dir).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().starts_with(".brain.rollback")));
        // The reset the winner writes closes the window for whoever comes second.
        let now = 10_000_000_i64;
        assert!(installed_recently("1.0.0 9999000", "1.0.0", now));
        assert!(!installed_recently("8.0.0 0", "1.0.0", now));
        assert!(!installed_recently("8.0.0 0", "8.0.0", now));
    }
}
