//! Multi-device sync over a folder the owner already trusts.
//!
//! There is no server here, and that is the design rather than a gap. The
//! user points brain at any directory that already moves between their
//! machines — iCloud Drive, Dropbox, a NAS mount, a USB stick — and brain
//! writes one encrypted bundle per store into it. The folder only ever
//! holds ciphertext: "the brain never leaves the machine" stays true in
//! the only sense that matters, because what leaves is unreadable without
//! a key that never does.
//!
//! One bundle per store, named by the store's origin id, written
//! atomically. Each store only ever writes its own bundle, so a dumb
//! shared folder needs no locking and conflicts cannot happen. Pull runs
//! before push: merge what the other machines said, then publish the
//! merged state.
//!
//! An earlier version of this file was a signpost that refused to sync at
//! all: the wiki holds the architecture, decisions and dead ends of every
//! project it watched - enough to reconstruct one without its source - so
//! nothing could be allowed to leave. That reasoning has not softened; it
//! is WHY a bundle is sealed before it exists anywhere outside the data
//! dir, why the key never enters the folder that moves, and why all of
//! this stays off until the owner runs `brain sync init` themselves.
//!
//! Everything underneath is machinery that already existed and is already
//! tested — `portable::export` / `import --merge` (id-keyed union),
//! two-pass replay, origin stamps, root-commit project identity. Sync is
//! those pieces plus one cipher and one loop.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chacha20poly1305::aead::Aead;
use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce};

use crate::config::{Config, Paths};

/// Bundle filename suffix. The stem is the writing store's origin id —
/// opaque, random, and meaningless to whoever hosts the folder.
const BUNDLE_SUFFIX: &str = ".brain.enc";

/// What one sync did, for the caller to report.
pub struct Outcome {
    pub pulled: usize,
    pub skipped: Vec<String>,
    pub gained: usize,
    pub pushed_bytes: u64,
}

/// Point this store at a shared directory and mint the key if needed.
///
/// # Errors
/// Returns an error when the directory cannot be used or config cannot be
/// written.
pub fn init(dir: &str) -> Result<String> {
    let paths = Paths::resolve()?;
    paths.ensure()?;
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let dir = dir.canonicalize().with_context(|| format!("resolve {}", dir.display()))?;

    let mut config = Config::load(&paths.config_file())?;
    config.sync.dir = Some(dir.clone());
    let rendered = toml::to_string(&config).context("render config")?;
    std::fs::write(paths.config_file(), rendered).context("write config")?;

    let key_path = paths.data_dir.join("sync.key");
    let minted = if key_path.is_file() {
        false
    } else {
        let mut key = [0u8; 32];
        getrandom::fill(&mut key).map_err(|error| anyhow::anyhow!("no randomness: {error}"))?;
        let hex: String = key.iter().map(|byte| format!("{byte:02x}")).collect();
        std::fs::write(&key_path, hex).context("write sync.key")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600));
        }
        true
    };

    Ok(format!(
        "sync dir: {}\nkey: {} ({})\n\n\
         To pair another machine:\n  \
         1. install brain there and run `brain sync init <same dir>`\n  \
         2. REPLACE its {} with this machine's — the key is what makes\n     \
            the bundles one brain; a folder full of bundles under\n     \
            different keys is several brains ignoring each other\n  \
         3. run `brain sync` on both\n\n\
         The folder only ever holds ciphertext. Nothing leaves this machine\n\
         readable, and nothing syncs until `brain sync` is run.",
        dir.display(),
        key_path.display(),
        if minted { "minted now — copy it to your other machines" } else { "already present" },
        key_path.display(),
    ))
}

/// Pull every peer bundle, then push our own.
///
/// # Errors
/// Returns an error when sync is unconfigured, the key is missing, or the
/// push fails. A peer bundle that cannot be decrypted is reported in the
/// outcome and skipped — one corrupt or foreign file must not stop the
/// owner's own machines from converging.
pub fn run() -> Result<Outcome> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file())?;
    let dir = config.sync.dir.as_ref().context(
        "sync is not configured on this machine - run `brain sync init <shared dir>` first",
    )?;
    anyhow::ensure!(dir.is_dir(), "sync dir {} does not exist", dir.display());
    let key = load_key(&paths)?;
    let origin = crate::ids::origin().context("this store has no origin id yet")?;

    // Pull before push: merge what the other machines said, then publish
    // the merged state - one round trip fewer to convergence.
    let mut outcome =
        Outcome { pulled: 0, skipped: Vec::new(), gained: 0, pushed_bytes: 0 };
    let staging = std::env::temp_dir().join(format!("brain-sync-{}", std::process::id()));
    std::fs::create_dir_all(&staging).context("create sync staging")?;
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let path = entry.context("read sync dir entry")?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else { continue };
        let Some(stem) = name.strip_suffix(BUNDLE_SUFFIX) else { continue };
        if stem == origin {
            continue;
        }
        let sealed = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let Ok(archive_bytes) = open_bundle(&key, &sealed) else {
            // Wrong key or corruption. Either way this file is not ours to
            // merge, and saying so beats both crashing and silence.
            outcome.skipped.push(name.to_string());
            continue;
        };
        let archive = staging.join(format!("{stem}.tar.gz"));
        std::fs::write(&archive, archive_bytes)
            .with_context(|| format!("write {}", archive.display()))?;
        let (_, gained) =
            crate::portable::import_counted(&archive, crate::portable::Existing::Merge)?;
        outcome.pulled += 1;
        outcome.gained += gained;
    }
    if outcome.pulled > 0 {
        crate::reindex()?;
    }

    // A machine that has captured nothing yet - the freshly paired laptop,
    // mid-first-sync - has nothing to push and every reason to pull. Push
    // when there is something to say.
    if !paths.wiki().is_dir() {
        let _ = std::fs::remove_dir_all(&staging);
        return Ok(outcome);
    }
    let archive = staging.join("push.tar.gz");
    crate::portable::export_wiki_only(&archive)?;
    let plain = std::fs::read(&archive).context("read export for push")?;
    let sealed = seal_bundle(&key, &plain)?;
    outcome.pushed_bytes = sealed.len() as u64;
    let target = dir.join(format!("{origin}{BUNDLE_SUFFIX}"));
    let tmp = dir.join(format!("{origin}{BUNDLE_SUFFIX}.tmp"));
    std::fs::write(&tmp, sealed).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &target).with_context(|| format!("publish {}", target.display()))?;

    let _ = std::fs::remove_dir_all(&staging);
    Ok(outcome)
}

/// The shared key, 32 bytes as hex on disk.
fn load_key(paths: &Paths) -> Result<[u8; 32]> {
    let path = paths.data_dir.join("sync.key");
    let hex = std::fs::read_to_string(&path)
        .with_context(|| format!("no sync key at {} - run `brain sync init`", path.display()))?;
    let hex = hex.trim();
    anyhow::ensure!(hex.len() == 64, "sync.key is not a 32-byte hex key");
    let mut key = [0u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)
            .context("sync.key is not valid hex")?;
    }
    Ok(key)
}

/// `[24-byte nonce][ciphertext]`. A fresh random nonce per seal: XChaCha's
/// nonce is wide enough that random never collides in practice, which is
/// the property that makes "no counter state to sync" safe.
fn seal_bundle(key: &[u8; 32], plain: &[u8]) -> Result<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new(key.into());
    let mut nonce = [0u8; 24];
    getrandom::fill(&mut nonce).map_err(|error| anyhow::anyhow!("no randomness: {error}"))?;
    let sealed = cipher
        .encrypt(XNonce::from_slice(&nonce), plain)
        .map_err(|_| anyhow::anyhow!("encryption failed"))?;
    let mut out = nonce.to_vec();
    out.extend(sealed);
    Ok(out)
}

/// The inverse of [`seal_bundle`]. Fails on a wrong key, a truncated file,
/// or any tampering - the AEAD tag covers all of it.
fn open_bundle(key: &[u8; 32], sealed: &[u8]) -> Result<Vec<u8>> {
    anyhow::ensure!(sealed.len() > 24, "bundle too short to hold a nonce");
    let cipher = XChaCha20Poly1305::new(key.into());
    cipher
        .decrypt(XNonce::from_slice(&sealed[..24]), &sealed[24..])
        .map_err(|_| anyhow::anyhow!("wrong key or corrupt bundle"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bundle_round_trips_and_a_wrong_key_opens_nothing() {
        let key = [7u8; 32];
        let sealed = seal_bundle(&key, b"the memory").unwrap();
        assert_eq!(open_bundle(&key, &sealed).unwrap(), b"the memory");
        assert!(open_bundle(&[8u8; 32], &sealed).is_err(), "a wrong key must open nothing");
        // Tampering is detected, not absorbed.
        let mut bent = sealed.clone();
        let last = bent.len() - 1;
        bent[last] ^= 1;
        assert!(open_bundle(&key, &bent).is_err(), "a flipped bit must fail the tag");
    }

    #[test]
    fn two_seals_of_one_plaintext_never_share_a_nonce() {
        let key = [7u8; 32];
        let a = seal_bundle(&key, b"same").unwrap();
        let b = seal_bundle(&key, b"same").unwrap();
        assert_ne!(a[..24], b[..24], "nonces must be fresh per seal");
        assert_ne!(a[24..], b[24..], "same nonce would mean same ciphertext");
    }
}
