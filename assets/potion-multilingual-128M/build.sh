#!/bin/sh
# Rebuild the embedding model from its pinned upstream revision, and hold the
# result to the checksums in PROVENANCE.md.
#
#   sh assets/potion-multilingual-128M/build.sh            build if missing, then check
#   sh assets/potion-multilingual-128M/build.sh --check    check only; fetch nothing
#
# The release workflow and the CI lane both run this, so there is one recipe
# rather than two that can drift. Anything already present is checked rather
# than rebuilt, which is what makes a restored cache as trustworthy as a fresh
# build. Needs curl and a python with pip; runs where the workflows do - bash
# on Linux and macOS, Git Bash on Windows.
set -eu
MODEL=$(dirname "$0")
REVISION=73908c3438cf03b6a01bcb9611d62b23d0726f08
UP=https://huggingface.co/minishlab/potion-multilingual-128M/resolve/$REVISION

# Neither tool is on every runner: Git Bash on Windows ships `sha256sum` and no
# `shasum`, and macOS ships the reverse.
sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}
# `<hash>  <name>` lines follow a heading in PROVENANCE.md; the upstream file's
# hash is the line after its own heading.
recorded() {
    grep -A4 "$1" "$MODEL/PROVENANCE.md" | grep -E "^ +[0-9a-f]{64}( |$)" \
        | awk -v f="$2" 'f == "" || $2 == f { print $1; exit }'
}
check() {
    got=$(sha256 "$1")
    [ "$got" = "$2" ] || { echo "$1 is $got, PROVENANCE.md says $2" >&2; exit 1; }
}

if [ "${1:-}" != "--check" ]; then
[ -s "$MODEL/tokenizer.json" ] || curl -fsSL -o "$MODEL/tokenizer.json" "$UP/tokenizer.json"
if [ ! -s "$MODEL/model-int8.safetensors" ]; then
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT INT TERM
    curl -fsSL -o "$work/upstream.safetensors" "$UP/model.safetensors"
    # Checked before anything runs on it, like every other input.
    check "$work/upstream.safetensors" "$(recorded 'upstream model.safetensors' '')"
    python -m pip install --quiet --require-hashes --only-binary :all: -r "$MODEL/requirements.txt"
    python "$MODEL/quantize.py" "$work/upstream.safetensors" "$MODEL/model-int8.safetensors"
fi
fi
for f in model-int8.safetensors tokenizer.json; do
    check "$MODEL/$f" "$(recorded 'vendored here' "$f")"
done
echo "embedding model matches PROVENANCE.md"
