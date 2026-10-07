#!/bin/sh
# Codex runs each of brain's hooks through this script: codex-hook.sh <Event>.
#
# Codex runs a plugin hook only while a hash of its command text matches what
# the person approved, and it takes that hash before it substitutes
# ${PLUGIN_ROOT}. Every command in codex-hooks.json is therefore the fixed line
# `sh "${PLUGIN_ROOT}/hooks/codex-hook.sh" <Event>`, and a fix made here
# reaches every Codex user without asking them to approve the hooks again.
# Change this script, never that line.

# Git's auto-maintenance stays off for brain and everything it starts. The
# plugin updates apart from the binary, and a brain older than 0.64.0 commits
# once per wiki page; each commit starts `git maintenance run --auto`, which
# recent git detaches, and that is how a machine ended up with hundreds of git
# processes on 2026-10-07.
export GIT_CONFIG_PARAMETERS="'maintenance.auto=false' 'gc.auto=0'"
# Codex clears the environment before it runs a hook.
export PATH="$HOME/.local/bin:$PATH"

# Only SessionStart checks that the binary exists. It fires once a session,
# where PostToolUse fires thousands of times and is the path a person waits on.
if [ "$1" != SessionStart ]; then
    exec brain hook --cli codex --event "$1"
fi

command -v brain >/dev/null 2>&1 || {
    echo 'rolepod-brain: the brain binary is not installed yet. Fetching it (checksum-verified) from github.com/nuttaruj/rolepod-brain — this happens once.' >&2
    curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh -s -- --binary-only >&2 ||
        echo 'rolepod-brain: could not fetch the binary. Install it yourself: curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh' >&2
}
# The embedding model is large, so it comes down in the background and the
# session does not wait for it.
command -v brain >/dev/null 2>&1 && [ ! -s "$(brain where --models 2>/dev/null)/model-int8.safetensors" ] && {
    nohup sh -c 'curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh -s -- --model-only' >/dev/null 2>&1 &
}
command -v brain >/dev/null 2>&1 && brain hook --cli codex --event SessionStart || echo '{}'
