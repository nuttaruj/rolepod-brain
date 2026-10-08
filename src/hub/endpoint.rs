//! Where the hub's socket lives, and whether it may be trusted.
//!
//! The socket sits in a private directory, `rolepod-brain-<uid>/`, under
//! `$XDG_RUNTIME_DIR` (Linux) or `$TMPDIR` (macOS). There is no fallback to a
//! shared `/tmp`: no private directory means no hub. Control is the directory
//! (owner = us, mode 0700, not a symlink) and the socket (owner = us, 0600);
//! there is no peer-credential check, which std does not offer.
//!
//! The server and every client call the same [`check_dir`], and a client
//! calls it before it sends a byte.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A socket path must stay under this, so `sun_path` (104 on macOS, 108 on
/// Linux) is never reached. Over it, the hub is off.
pub const MAX_SOCKET_PATH: usize = 100;

/// The lock file's name in the data dir. Held for the hub's life.
pub const LOCK_FILE: &str = "hub.lock";

/// What a path is, by `lstat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Dir,
    Socket,
    Symlink,
    File,
    Other,
}

/// The three facts every check needs, so a test can hand in a verdict it
/// could not make for real (a directory owned by someone else).
#[derive(Debug, Clone, Copy)]
pub struct Meta {
    pub uid: u32,
    pub mode: u32,
    pub kind: Kind,
}

impl Meta {
    /// `lstat` - never follows a symlink.
    ///
    /// # Errors
    /// Whatever `symlink_metadata` says, including `NotFound`.
    pub fn of(path: &Path) -> io::Result<Self> {
        let meta = fs::symlink_metadata(path)?;
        let kind = match meta.file_type() {
            t if t.is_symlink() => Kind::Symlink,
            t if t.is_dir() => Kind::Dir,
            t if t.is_socket() => Kind::Socket,
            t if t.is_file() => Kind::File,
            _ => Kind::Other,
        };
        Ok(Self { uid: meta.uid(), mode: meta.mode() & 0o7777, kind })
    }
}

/// Why the endpoint cannot be trusted. The client reports every one of these
/// as `hub-unsafe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsafe(pub &'static str);

impl std::fmt::Display for Unsafe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", super::reason::UNSAFE, self.0)
    }
}

/// Is this a directory we may put a socket in?
///
/// # Errors
/// A symlink, a non-directory, another owner, or any group/other bit.
pub fn judge_dir(meta: &Meta, euid: u32) -> Result<(), Unsafe> {
    match meta.kind {
        Kind::Symlink => Err(Unsafe("symlink")),
        Kind::Dir if meta.uid != euid => Err(Unsafe("wrong-owner")),
        Kind::Dir if meta.mode & 0o077 != 0 => Err(Unsafe("loose-mode")),
        Kind::Dir => Ok(()),
        _ => Err(Unsafe("not-a-directory")),
    }
}

/// Is this a socket we made?
///
/// # Errors
/// A symlink, a non-socket, another owner, or any group/other bit.
pub fn judge_socket(meta: &Meta, euid: u32) -> Result<(), Unsafe> {
    match meta.kind {
        Kind::Symlink => Err(Unsafe("symlink")),
        Kind::Socket if meta.uid != euid => Err(Unsafe("wrong-owner")),
        Kind::Socket if meta.mode & 0o077 != 0 => Err(Unsafe("loose-mode")),
        Kind::Socket => Ok(()),
        _ => Err(Unsafe("not-a-socket")),
    }
}

/// [`judge_dir`] on the real path.
///
/// # Errors
/// [`Unsafe`], also when the path cannot be read at all.
pub fn check_dir(dir: &Path, euid: u32) -> Result<(), Unsafe> {
    judge_dir(&Meta::of(dir).map_err(|_| Unsafe("missing"))?, euid)
}

/// [`judge_socket`] on the real path.
///
/// # Errors
/// [`Unsafe`], also when the path cannot be read at all.
pub fn check_socket(path: &Path, euid: u32) -> Result<(), Unsafe> {
    judge_socket(&Meta::of(path).map_err(|_| Unsafe("missing"))?, euid)
}

/// The directory private sockets go under, or `None`. Never `/tmp` by
/// default: Linux takes only `XDG_RUNTIME_DIR`, macOS only `TMPDIR`, and both
/// must be absolute.
#[must_use]
pub fn base_dir(xdg: Option<OsString>, tmpdir: Option<OsString>, macos: bool) -> Option<PathBuf> {
    let chosen = if macos { tmpdir } else { xdg }?;
    let path = PathBuf::from(chosen);
    (path.is_absolute() && path.as_os_str().len() > 1).then_some(path)
}

/// FNV-1a: a hash that stays the same across builds, which `DefaultHasher`
/// does not promise.
pub(super) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3))
}

/// `hub-<16 hex>.sock`, from the canonical data dir: one hub per data dir.
#[must_use]
pub fn socket_name(data_dir: &Path) -> String {
    let canonical = fs::canonicalize(data_dir).unwrap_or_else(|_| data_dir.to_path_buf());
    format!("hub-{:016x}.sock", fnv1a(canonical.as_os_str().as_encoded_bytes()))
}

/// The effective uid, from the owner of a file this process creates and
/// removes in `data_dir` (std has no `geteuid`, and no libc crate is allowed).
///
/// # Errors
/// [`Unsafe`] when the probe file cannot be made.
pub fn euid(data_dir: &Path) -> Result<u32, Unsafe> {
    // The uid of a process does not change; only a success is kept, so a data
    // dir that could not be written to is tried again.
    static CACHED: OnceLock<u32> = OnceLock::new();
    if let Some(uid) = CACHED.get() {
        return Ok(*uid);
    }
    let uid = probe_euid(data_dir)?;
    Ok(*CACHED.get_or_init(|| uid))
}

fn probe_euid(data_dir: &Path) -> Result<u32, Unsafe> {
    let probe = data_dir.join(format!(".hub-uid-{}", std::process::id()));
    let _ = fs::remove_file(&probe);
    let made = fs::OpenOptions::new().write(true).create_new(true).open(&probe);
    let uid = made.and_then(|_| fs::symlink_metadata(&probe)).map(|meta| meta.uid());
    let _ = fs::remove_file(&probe);
    uid.map_err(|_| Unsafe("no-data-dir"))
}

/// Where one data dir's hub lives.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub dir: PathBuf,
    pub socket: PathBuf,
    pub lock: PathBuf,
    pub euid: u32,
}

/// Work out the paths for `data_dir` under `base`. Touches the file system
/// only to canonicalize the data dir.
///
/// # Errors
/// No private base, or a socket path that would not fit `sun_path`.
pub fn locate(data_dir: &Path, base: Option<PathBuf>, euid: u32) -> Result<Endpoint, Unsafe> {
    let base = base.ok_or(Unsafe("no-private-dir"))?;
    let dir = base.join(format!("rolepod-brain-{euid}"));
    let socket = dir.join(socket_name(data_dir));
    if socket.as_os_str().len() >= MAX_SOCKET_PATH {
        return Err(Unsafe("path-too-long"));
    }
    Ok(Endpoint { dir, socket, lock: data_dir.join(LOCK_FILE), euid })
}

/// [`locate`] for the running process: reads the environment and the euid.
///
/// # Errors
/// [`Unsafe`] as above.
pub fn resolve(data_dir: &Path) -> Result<Endpoint, Unsafe> {
    let euid = euid(data_dir)?;
    let base = base_dir(
        std::env::var_os("XDG_RUNTIME_DIR"),
        std::env::var_os("TMPDIR"),
        cfg!(target_os = "macos"),
    );
    locate(data_dir, base, euid)
}

impl Endpoint {
    /// Make the private directory (0700) if it is missing, then check it.
    /// A directory that already exists is never chmod'ed, only judged.
    ///
    /// # Errors
    /// [`Unsafe`] when it cannot be made or does not pass.
    pub fn prepare(&self) -> Result<(), Unsafe> {
        match fs::DirBuilder::new().mode(0o700).create(&self.dir) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(_) => return Err(Unsafe("cannot-create-dir")),
        }
        check_dir(&self.dir, self.euid)
    }

    /// What a client checks before it connects: the directory, then the socket.
    ///
    /// # Errors
    /// [`Unsafe`].
    pub fn verify(&self) -> Result<(), Unsafe> {
        check_dir(&self.dir, self.euid)?;
        check_socket(&self.socket, self.euid)
    }

    /// The pid in `hub.lock`, if the lock is held right now. A free lock, or
    /// a pid nobody holds the lock for, is `None`: the pid is advisory and the
    /// lock is the only proof of life.
    #[must_use]
    pub fn holder_pid(&self) -> Option<u32> {
        let file = fs::OpenOptions::new().read(true).open(&self.lock).ok()?;
        match file.try_lock_shared() {
            Err(fs::TryLockError::WouldBlock) => {}
            _ => return None,
        }
        fs::read_to_string(&self.lock).ok()?.trim().parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn tmp(name: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        // Short on purpose: a socket path must fit `sun_path`, and macOS temp dirs are long.
        let dir = PathBuf::from("/tmp").join(format!("hubep-{name}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn mine(dir: &Path) -> u32 {
        euid(dir).unwrap()
    }

    #[test]
    fn dir_wrong_owner_refused() {
        let meta = Meta { uid: 1234, mode: 0o700, kind: Kind::Dir };
        assert_eq!(judge_dir(&meta, 501), Err(Unsafe("wrong-owner")));
        assert_eq!(judge_dir(&Meta { uid: 501, ..meta }, 501), Ok(()));
        let sock = Meta { uid: 1234, mode: 0o600, kind: Kind::Socket };
        assert_eq!(judge_socket(&sock, 501), Err(Unsafe("wrong-owner")));
    }

    #[test]
    fn dir_mode_0755_0770_0707_refused() {
        for mode in [0o755, 0o770, 0o707, 0o701] {
            let meta = Meta { uid: 5, mode, kind: Kind::Dir };
            assert_eq!(judge_dir(&meta, 5), Err(Unsafe("loose-mode")), "{mode:o}");
        }
        let dir = tmp("loose");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(check_dir(&dir, mine(&dir)), Err(Unsafe("loose-mode")));
    }

    #[test]
    fn dir_mode_0700_ok() {
        let base = tmp("ok");
        let ep = locate(&base, Some(base.clone()), mine(&base)).unwrap();
        ep.prepare().unwrap();
        assert_eq!(Meta::of(&ep.dir).unwrap().mode, 0o700);
        // A second prepare finds it and judges it, without changing anything.
        ep.prepare().unwrap();
    }

    #[test]
    fn symlink_dir_refused() {
        let base = tmp("symdir");
        let real = base.join("real");
        fs::create_dir(&real).unwrap();
        fs::set_permissions(&real, fs::Permissions::from_mode(0o700)).unwrap();
        let link = base.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert_eq!(check_dir(&link, mine(&base)), Err(Unsafe("symlink")));
    }

    #[test]
    fn symlink_socket_refused() {
        let base = tmp("symsock");
        let target = base.join("t");
        fs::write(&target, b"x").unwrap();
        let link = base.join("s.sock");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert_eq!(check_socket(&link, mine(&base)), Err(Unsafe("symlink")));
    }

    #[test]
    fn regular_file_as_socket_refused() {
        let base = tmp("filesock");
        let file = base.join("s.sock");
        fs::write(&file, b"x").unwrap();
        assert_eq!(check_socket(&file, mine(&base)), Err(Unsafe("not-a-socket")));
    }

    #[test]
    fn no_xdg_no_tmp_fallback() {
        let tmp = Some(OsString::from("/tmp"));
        assert_eq!(base_dir(None, tmp.clone(), false), None);
        assert_eq!(base_dir(None, None, true), None);
        assert_eq!(base_dir(Some("relative".into()), None, false), None);
        assert_eq!(base_dir(Some("/run/user/1".into()), tmp, false), Some(PathBuf::from("/run/user/1")));
        let missing = locate(Path::new("/x"), None, 1);
        assert_eq!(missing.unwrap_err(), Unsafe("no-private-dir"));
    }

    #[test]
    fn path_too_long_is_none_not_panic() {
        let long = PathBuf::from(format!("/{}", "a".repeat(120)));
        assert_eq!(locate(Path::new("/d"), Some(long), 501).unwrap_err(), Unsafe("path-too-long"));
        assert!(locate(Path::new("/d"), Some("/tmp/x".into()), 501).is_ok());
    }

    #[test]
    fn socket_name_fixed_length_from_hash() {
        let a = socket_name(Path::new("/some/where"));
        let b = socket_name(Path::new("/some/where/else"));
        assert_ne!(a, b);
        assert_eq!(a, socket_name(Path::new("/some/where")));
        assert_eq!(a.len(), "hub-".len() + 16 + ".sock".len());
        assert!(a.starts_with("hub-") && a.ends_with(".sock"));
        // Pinned: a build that changed the hash would orphan every running hub.
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn bound_socket_mode_0600() {
        use std::os::unix::net::UnixListener;
        let base = tmp("bind");
        let ep = locate(&base, Some(base.clone()), mine(&base)).unwrap();
        ep.prepare().unwrap();
        let _listener = UnixListener::bind(&ep.socket).unwrap();
        fs::set_permissions(&ep.socket, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(Meta::of(&ep.socket).unwrap().mode, 0o600);
        ep.verify().unwrap();
    }

    #[test]
    fn holder_pid_needs_a_held_lock() {
        let base = tmp("holder");
        let ep = locate(&base, Some(base.clone()), mine(&base)).unwrap();
        assert_eq!(ep.holder_pid(), None);
        let mut file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&ep.lock).unwrap();
        io::Write::write_all(&mut file, b"4242").unwrap();
        assert_eq!(ep.holder_pid(), None, "a pid with no lock behind it is advisory");
        file.try_lock().unwrap();
        assert_eq!(ep.holder_pid(), Some(4242));
    }
}
