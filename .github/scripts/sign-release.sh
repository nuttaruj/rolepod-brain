#!/bin/sh
# Signs the five release binaries with minisign and checks every signature
# against the public key committed in the repository before anything leaves
# the job. The release workflow runs this file; it is not a copy of a step.
#
#   sign-release.sh <artifacts-dir> <out-dir> <pubkey-file>
#
# Environment:
#   TAG                  the release tag, e.g. v0.69.0 (plain x.y.z only)
#   MINISIGN_SECRET_KEY  the secret key file's text (required)
#   MINISIGN_PASSWORD    the key's password, only when it has one
#   KEY_FILE             where the key is written (default: a temp file)
#
# <artifacts-dir>/brain-<target>/ holds each build's artifact. Output is
# <out-dir>/brain-<target>.minisig, trusted comment "brain-<target> <version>".
# Any problem stops the script: a release is never published unsigned.
set -eu
set +x

TARGETS="aarch64-apple-darwin x86_64-apple-darwin x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-pc-windows-msvc"

die() { echo "sign-release: $*" >&2; exit 1; }

[ $# -eq 3 ] || die "usage: sign-release.sh <artifacts-dir> <out-dir> <pubkey-file>"
art=$1; out=$2; pub=$3

[ -s "$pub" ] || die "public key file $pub is missing or empty"
[ -n "${MINISIGN_SECRET_KEY:-}" ] || die "MINISIGN_SECRET_KEY is not set"
[ -n "${TAG:-}" ] || die "TAG is not set"
version=${TAG#v}
printf '%s' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
    || die "tag '$TAG' is not vX.Y.Z; the updater only accepts a plain version"
command -v minisign >/dev/null 2>&1 || die "minisign is not installed"

# Only the five known artifacts: anything else in the folder stops here.
for d in "$art"/*; do
    [ -d "$d" ] || die "unexpected file $d in $art"
    name=$(basename "$d")
    ok=0
    for t in $TARGETS; do [ "$name" = "brain-$t" ] && ok=1; done
    [ "$ok" = 1 ] || die "unexpected artifact $name"
done

[ ! -e "$out" ] || [ -z "$(ls -A "$out")" ] || die "$out is not empty"
mkdir -p "$out"

umask 077
key=${KEY_FILE:-$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/minisign-key.XXXXXX")}
trap 'rm -f "$key"' EXIT
printf '%s\n' "$MINISIGN_SECRET_KEY" > "$key"

for t in $TARGETS; do
    dir="$art/brain-$t"
    if [ -f "$dir/brain-$t" ]; then bin="$dir/brain-$t"
    elif [ -f "$dir/brain-$t.exe" ]; then bin="$dir/brain-$t.exe"
    else die "no binary for $t in $dir"; fi
    # A signature that arrived with the build is never kept.
    [ -z "$(ls "$dir" | grep '\.minisig$' || true)" ] || die "$dir carries a .minisig"
    comment="brain-$t $version"
    sig="$out/brain-$t.minisig"
    if [ -n "${MINISIGN_PASSWORD:-}" ]; then
        printf '%s\n' "$MINISIGN_PASSWORD" | minisign -S -s "$key" -x "$sig" -t "$comment" -m "$bin" >/dev/null
    else
        minisign -S -s "$key" -x "$sig" -t "$comment" -m "$bin" </dev/null >/dev/null
    fi
    [ -s "$sig" ] || die "no signature written for $t"
    # The public key from the repository, not the one the secret implies.
    said=$(minisign -V -p "$pub" -x "$sig" -m "$bin") || die "verification failed for $t"
    printf '%s\n' "$said" | grep -Fxq "Trusted comment: $comment" \
        || die "trusted comment for $t is not '$comment'"
    echo "signed and verified: brain-$t $version"
done

[ "$(ls "$out" | wc -l | tr -d ' ')" = 5 ] || die "expected exactly 5 signatures in $out"
