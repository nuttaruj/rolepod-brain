#!/bin/sh
# rolepod-brain bootstrap — fetch the binary, wire the CLIs, leave nothing else.
#
#   curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh
#
# What lands on your machine is one binary. No repository, no toolchain, no
# resident process. If no prebuilt binary matches your platform this falls
# back to building from source, which needs Rust.
#
# Options:
#   --target=all        wire every supported CLI found here (the default)
#   --target=<cli>      wire one: claude-code, codex, cursor, gemini-cli,
#                       antigravity, opencode
#   --yes, -y           apply without asking — for CI, or for a pipe with no
#                       terminal to ask at
#   --uninstall         remove brain from every CLI it wired
#
# Working on brain itself? There is no install script to run: `cargo build
# --release` and then `target/release/brain setup`. This file is the only
# supported way to INSTALL, so that installing never means keeping a clone.
#
# `--binary-only` (also BRAIN_NO_SETUP) is not in the list above because it is
# not a choice worth making: it installs the binary and wires nothing, which on
# its own captures nothing and recalls nothing. It exists for the plugin, whose
# SessionStart hook needs the binary and supplies the wiring itself.
#
# Env:
#   BRAIN_BIN_DIR   where to install (default $HOME/.local/bin)
#   BRAIN_VERSION   a tag to install (default: the latest release)
set -eu

target=""
assume_yes=""
uninstall=""
binary_only="${BRAIN_NO_SETUP:-}"

for arg in "$@"; do
    case "$arg" in
        --target=all) target="" ;;
        --target=*)   target="${arg#--target=}" ;;
        --yes|-y)     assume_yes=1 ;;
        --uninstall)  uninstall=1 ;;
        --binary-only) binary_only=1 ;;
--model-only) model_only=1 ;;
--reranker-only) reranker_only=1 ;;
--into)       shift; model_into="$1" ;;
        -h|--help)
            # The header block: from the second line to the first that is
            # not a comment. A fixed line range went stale the moment the
            # header grew and printed `set -eu` at the reader as if it were
            # documentation; printing every comment in the file instead would
            # hand them the script's internal notes.
            awk 'NR > 1 { if (/^#/) print; else exit }' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            printf 'error: unknown option %s (try --help)\n' "$arg" >&2
            exit 2
            ;;
    esac
done

REPO="nuttaruj/rolepod-brain"
BIN_DIR="${BRAIN_BIN_DIR:-$HOME/.local/bin}"
BIN="$BIN_DIR/brain"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required"; }
need curl

case "$(uname -s)" in
    Darwin) os="apple-darwin" ;;
    Linux)  os="unknown-linux-gnu" ;;
    *)      os="" ;;
esac
case "$(uname -m)" in
    arm64|aarch64) arch="aarch64" ;;
    x86_64|amd64)  arch="x86_64" ;;
    *)             arch="" ;;
esac
# Named apart from `$target` on purpose. They were one variable once: the
# option parser stored the user's CLI choice in it and this line overwrote it
# with the platform triple, so `--target=codex` — and the bare one-liner the
# README leads with — asked `brain setup` to wire a CLI named
# `aarch64-apple-darwin`.
platform=""
[ -n "$os" ] && [ -n "$arch" ] && platform="$arch-$os"

version="${BRAIN_VERSION:-}"
if [ -z "$version" ] && [ -n "$platform" ]; then
    version=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
        | sed -n 's/.*"tag_name" *: *"\([^"]*\)".*/\1/p' | head -n 1) || true
fi

from_source() {
    say "Building from source (this needs Rust and git)."
    command -v cargo >/dev/null 2>&1 || die "cargo not found. Install Rust: https://rustup.rs"
    command -v git >/dev/null 2>&1 || die "git not found."
    # A working tree the user did not ask for is not left behind: this is
    # cloned into a temporary directory and removed on the way out.
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT INT TERM
    git clone --depth 1 "https://github.com/$REPO.git" "$work/src" >/dev/null 2>&1 \
        || die "could not clone https://github.com/$REPO"
    # The reranker goes in. Nothing links ONNX Runtime at build time any more,
    # so this compiles wherever `brain` itself compiles - the runtime is a file
    # downloaded later, and a machine that cannot load it falls through to the
    # host CLI at that point rather than failing here. The bare build is kept
    # as a second chance only because a build failure at install time should
    # cost a feature, not the install.
    ( cd "$work/src" && cargo build --release --quiet --features local-rerank ) \
        || ( cd "$work/src" && cargo build --release --quiet ) \
        || die "build failed"
    install_binary "$work/src/target/release/brain"
}

install_binary() {
    src="$1"
    mkdir -p "$BIN_DIR"
    # Replaced, never written over in place. On macOS, writing over a running
    # binary invalidates its code signature and the kernel then kills it on
    # exec — which looks exactly like a broken install, with no output to say
    # so. A fresh inode plus a re-sign keeps it launchable.
    rm -f "$BIN"
    cp "$src" "$BIN"
    chmod +x "$BIN"
    if [ "$(uname -s)" = "Darwin" ] && command -v codesign >/dev/null 2>&1; then
        codesign -s - -f "$BIN" >/dev/null 2>&1 || true
    fi
}

verify() {
    file="$1"
    name="$2"
    sums="$3"
    # Either separator: two spaces, or the ` *` that `sha256sum` writes in
    # binary mode. A published file has carried both.
    want=$(grep -E "[ *]$name\$" "$sums" 2>/dev/null | awk '{print $1}' | head -n 1)
    [ -n "$want" ] || die "no checksum published for $name — refusing to install"
    if command -v shasum >/dev/null 2>&1; then
        got=$(shasum -a 256 "$file" | awk '{print $1}')
    elif command -v sha256sum >/dev/null 2>&1; then
        got=$(sha256sum "$file" | awk '{print $1}')
    else
        die "no sha256 tool available — refusing to install unverified"
    fi
    # This binary reads what you type into your editor. It is not installed
    # without matching what the release says it should be.
    [ "$want" = "$got" ] || die "checksum mismatch for $name (expected $want, got $got)"
}

if [ -n "$platform" ] && [ -n "$version" ]; then
    name="brain-$platform"
    base="https://github.com/$REPO/releases/download/$version"
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT INT TERM
    say "Fetching $name $version..."
    # Failure here is expected and handled - a platform with no published
    # binary, or no network - so curl's own diagnostics are noise in front of
    # the message that actually tells the reader what happens next.
    if curl -fsSL -o "$work/$name" "$base/$name" 2>/dev/null \
        && curl -fsSL -o "$work/SHA256SUMS" "$base/SHA256SUMS" 2>/dev/null; then
        verify "$work/$name" "$name" "$work/SHA256SUMS"
        install_binary "$work/$name"
    else
        say "No prebuilt binary for $platform at $version."
        from_source
    fi
else
    say "Unrecognised platform ($(uname -s) $(uname -m)) or no published release."
    from_source
fi

fetch_model() {
    # The embedding table is 122 MB - past what a repository file may be, and
    # not something to make every install of every version download again. It
    # is a release asset, fetched once into a directory named for the model, so
    # a later version that needs different weights finds nothing here rather
    # than reading the wrong ones.
    model_dir="$1"
    [ -n "$model_dir" ] || model_dir="$($BIN where --models 2>/dev/null)"
    [ -n "$model_dir" ] || return 1
    if [ -s "$model_dir/model-int8.safetensors" ] && [ -s "$model_dir/tokenizer.json" ]; then
        return 0
    fi
    mkdir -p "$model_dir" || return 1
    work=$(mktemp -d) || return 1
    say "Fetching the embedding model (122 MB, once) ..."
    if [ -z "$base" ]; then
        base="https://github.com/$REPO/releases/download/$version"
    fi
    curl -fsSL -o "$work/SHA256SUMS" "$base/SHA256SUMS" 2>/dev/null || {
        rm -rf "$work"
        return 1
    }
    for f in model-int8.safetensors tokenizer.json; do
        curl -fsSL -o "$work/$f" "$base/$f" 2>/dev/null || {
            rm -rf "$work"
            return 1
        }
        verify "$work/$f" "$f" "$work/SHA256SUMS"
    done
    # Renamed into place only once both are whole, so a broken download is
    # never a half-installed model that loads and answers differently.
    for f in model-int8.safetensors tokenizer.json; do
        mv "$work/$f" "$model_dir/$f" || { rm -rf "$work"; return 1; }
    done
    rm -rf "$work"
    say "Semantic search is ready."
}

fetch_reranker() {
    # About 600 MB all told and reranking is off by default, so this is never
    # part of an install. `brain` calls it the first time someone actually asks
    # for a rerank, in the background, and that search falls through to the
    # host CLI while it runs. The one after it has the model.
    #
    # Three files, not two. ONNX Runtime is no longer linked into the binary -
    # that is what let this feature reach Intel macOS and older Linux at all -
    # so the runtime is downloaded here like the weights, per platform, and
    # verified the same way.
    model_dir="$1"
    [ -n "$model_dir" ] || model_dir="$($BIN where --reranker 2>/dev/null)"
    [ -n "$model_dir" ] || return 1
    case "$(uname -s)" in
        Darwin) runtime_name="libonnxruntime.dylib" ;;
        *)      runtime_name="libonnxruntime.so" ;;
    esac
    if [ -s "$model_dir/model.onnx" ] && [ -s "$model_dir/tokenizer.json" ] \
        && [ -s "$model_dir/$runtime_name" ]; then
        return 0
    fi
    [ -n "$platform" ] || return 1
    mkdir -p "$model_dir" || return 1
    work=$(mktemp -d) || return 1
    say "Fetching the reranker (up to 600 MB, once) ..."
    if [ -z "$base" ]; then
        base="https://github.com/$REPO/releases/download/$version"
    fi
    curl -fsSL -o "$work/SHA256SUMS" "$base/SHA256SUMS" 2>/dev/null || {
        rm -rf "$work"
        return 1
    }
    # Each file is skipped when it is already here, because they do not always
    # go missing together. A machine that reranked under a build with its own
    # linked runtime has the 568 MB model and no library at all, and pulling
    # the model again to collect a 37 MB dylib is 568 MB of nothing.
    #
    # The remote name is not the local one. The asset carries the platform
    # because which ONNX Runtime a machine can load is a per-platform answer -
    # 1.28 nearly everywhere, 1.23 on Intel macOS, where 1.23 is the last one
    # Microsoft built - and the loader only ever wants a library.
    #
    # Verified and renamed one at a time. Each file is whole or absent, never
    # half-written, and a set that is missing one of the three simply does not
    # report ready: `brain` checks all three before it believes it has a
    # reranker, so an interrupted fetch costs a retry rather than a model that
    # loads and answers differently.
    fetch_one() {
        remote="$1"
        local_name="$2"
        [ -s "$model_dir/$local_name" ] && return 0
        curl -fsSL -o "$work/$remote" "$base/$remote" 2>/dev/null || return 1
        verify "$work/$remote" "$remote" "$work/SHA256SUMS"
        mv "$work/$remote" "$model_dir/$local_name" || return 1
    }
    fetch_one reranker-int8.onnx       model.onnx      || { rm -rf "$work"; return 1; }
    fetch_one reranker-tokenizer.json  tokenizer.json  || { rm -rf "$work"; return 1; }
    fetch_one "onnxruntime-$platform"  "$runtime_name" || { rm -rf "$work"; return 1; }
    chmod +x "$model_dir/$runtime_name" 2>/dev/null || true
    rm -rf "$work"
    say "Local reranking is ready."
}

if [ -n "$reranker_only" ]; then
    fetch_reranker "" || die "could not fetch the reranker"
    exit 0
fi

if [ -n "$model_only" ]; then
    if [ -n "$model_into" ]; then
        into_dir="$model_into/potion-multilingual-128M"
    else
        into_dir=""
    fi
    fetch_model "$into_dir" || die "could not fetch the embedding model"
    exit 0
fi

say "Installed $("$BIN" --version) to $BIN"

# Semantic search needs the model; everything else does not. A failure here
# is reported and stepped over - capture, keyword, entity and neighbour
# recall all work without it, and `brain doctor` says what is missing.
fetch_model "" || say "Could not fetch the embedding model; run 'brain doctor' for what that costs."

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        say ""
        say "NOTE: $BIN_DIR is not on your PATH. Add it to your shell profile:"
        say "  export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

if [ -n "$uninstall" ]; then
    say ""
    "$BIN" uninstall --apply
    say ""
    say "Removed from every CLI. The binary and your memory are untouched:"
    say "  rm $BIN        # the binary"
    say "  $BIN where     # where the memory lives, if you want it gone too"
    exit 0
fi

if [ -n "$binary_only" ]; then
    say ""
    say "Skipped hook registration. Run '$BIN setup' to see what it would change."
    exit 0
fi

# No `--cli` means every supported CLI found here, which is what `--target=all`
# asks for and what someone piping this into a shell almost always wants.
set -- setup
[ -n "$target" ] && set -- "$@" --cli "$target"

say ""
say "Planned changes:"
say ""
"$BIN" "$@"

# Piped into a shell, stdin is the script — so a prompt has to read the
# terminal directly, and where there is no terminal there is nobody to ask.
if [ -n "$assume_yes" ]; then
    reply="y"
elif [ -r /dev/tty ]; then
    say ""
    printf 'Apply these changes? [y/N] '
    read -r reply < /dev/tty || reply=""
else
    reply=""
fi

case "$reply" in
    [yY]*)
        "$BIN" "$@" --apply
        say ""
        "$BIN" doctor || true
        say ""
        say "Done. Start a session in any wired CLI; nothing else to run."
        ;;
    *)
        say ""
        say "Nothing changed. Run '$BIN setup --apply' when you are ready."
        ;;
esac
