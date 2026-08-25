#!/bin/sh
# Build rolepod-brain FROM SOURCE and wire the CLIs on this machine.
#
# This is the contributor path: it expects you to be standing in a clone.
# Installing does not require one — `bootstrap.sh` fetches a prebuilt binary,
# checks it against the release's published checksum, and leaves no working
# tree behind:
#
#   curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh
#
# Safe to re-run: setup backs up each config before writing and only replaces
# entries it created itself.
set -eu

BIN_DIR="${BRAIN_BIN_DIR:-$HOME/.local/bin}"
BIN="$BIN_DIR/brain"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v cargo >/dev/null 2>&1 || die "cargo not found. Install Rust: https://rustup.rs"
command -v git >/dev/null 2>&1 || die "git not found."

say "Building release binary..."
cargo build --release --quiet

mkdir -p "$BIN_DIR"

# Replace, never overwrite in place. On macOS, writing over a running binary
# invalidates its code signature and the kernel then kills it on exec - which
# looks exactly like a broken install. Removing first gives a fresh inode, and
# re-signing keeps it launchable.
rm -f "$BIN"
cp target/release/brain "$BIN"
if [ "$(uname -s)" = "Darwin" ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$BIN" >/dev/null 2>&1 || true
fi

say "Installed $("$BIN" --version) to $BIN"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) say ""; say "NOTE: $BIN_DIR is not on your PATH. Add it to your shell profile:";
       say "  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

say ""
say "Planned changes:"
say ""
"$BIN" setup

say ""
printf 'Apply these changes? [y/N] '
read -r reply
case "$reply" in
    [yY]*)
        "$BIN" setup --apply
        say ""
        "$BIN" doctor || true
        say ""
        say "Done. Start a session in any wired CLI; nothing else to run."
        ;;
    *)
        say "Nothing changed. Run '$BIN setup --apply' when you are ready."
        ;;
esac
