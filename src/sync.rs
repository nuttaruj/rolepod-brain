//! `brain sync` — kept as a signpost, because the answer is "no", not "how".
//!
//! This command once pushed the wiki to a git remote. It does not any more,
//! and the code to do it is gone rather than disabled.
//!
//! The reason is what the wiki *is*. It accumulates the architecture, the
//! decision sequence, and the failed approaches — the internally-known weak
//! points — of every project it watched. Whoever holds it can reconstruct a
//! project, and the thinking behind it, without ever seeing the source. For
//! client or confidential work that is worse than a source leak.
//!
//! So the brain does not leave the machine. The wiki stays a local git
//! repository, which is where its real value is anyway — history, diffs,
//! rollback, conflict-free merges — with no remote to exfiltrate to. Backups
//! are the same problem as backing up anything else on the disk, and the user
//! already has a way to do that.

use anyhow::Result;

/// Explain the local-only stance.
///
/// # Errors
/// Always returns an error: there is nothing to sync, and reporting success
/// would imply a backup exists somewhere that does not.
pub fn run() -> Result<()> {
    anyhow::bail!(
        "brain has no remote sync, by design.\n\n\
         This memory holds the architecture, decisions and dead ends of every \n\
         project it has watched — enough to reconstruct them without the source. \n\
         It stays on this machine.\n\n\
         The wiki is already a local git repository, so history, diffs and \n\
         rollback all work; `git -C ~/.rolepod-brain/wiki log` shows it. To keep \n\
         a copy, back it up the way you back up the rest of your disk."
    )
}
