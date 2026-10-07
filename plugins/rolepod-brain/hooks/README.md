# Hook manifests

These two files carried an explanatory `_comment` key until 0.52.2. JSON has
no comments, and Claude Code now warns about the unknown key at every session
start — a line that reads like an error in a tool whose whole promise is that
you install it and forget it. The notes live here instead.

## hooks.json — Claude Code

Claude Code hooks, auto-discovered from hooks/hooks.json (plugins reference:
'Location: hooks/hooks.json in plugin root, or inline in plugin.json').
Event names are PascalCase here; Cursor's are camelCase (postToolUse,
beforeSubmitPrompt), so this file cannot be picked up by the wrong host.
Codex declares its own file explicitly in .codex-plugin/plugin.json.

PreToolUse is scoped to Read and injects only - it is never captured, or it
would duplicate PostToolUse. PostCompact is absent on purpose: Claude Code
rejects injected context under that event. Both reasons are in src/setup.rs.

Only SessionStart checks that the binary exists. It fires once a session,
where PostToolUse fires thousands of times and is the path a person waits on.
The `Setup` event is NOT an install hook - it fires on --init/--maintenance -
so it cannot be the thing that puts the binary in place.

Every command sets `GIT_CONFIG_PARAMETERS` to turn git's auto-maintenance
off. The plugin updates apart from the binary, and a brain older than 0.64.0
commits once per wiki page; each commit starts `git maintenance run --auto`,
which recent git detaches, and that is how a machine ended up with hundreds of
git processes on 2026-10-07. `brain hook` and the consolidate it starts inherit
the variable, so even an old binary's commits start nothing.

## codex-hooks.json — Codex

Declared explicitly in `.codex-plugin/plugin.json` rather than discovered, so
the name is free.

Every command is the same fixed line, `sh "${PLUGIN_ROOT}/hooks/codex-hook.sh"
<Event>`, and the work happens in `codex-hook.sh`. Codex trusts a hook by a
sha256 of its event, command and timeout (`hooks.state.<key>.trusted_hash` in
`config.toml`), taken before it substitutes `${PLUGIN_ROOT}`, so a changed
command reads as modified and stops running until the person approves it
again. The line holds no version and no path, so the hash stays the same while
the script changes.

Moving to the script in 0.64.0 changed every command once. Each Codex user is
asked once to review the hooks ("Hooks need review"), and brain captures
nothing from Codex until they choose Trust. That one approval was accepted on
2026-10-07 in exchange for never needing another.

The rule from here on: a fix goes into `codex-hook.sh`, never into the command
text, the event names or the timeouts in `codex-hooks.json`.

The script sets what the commands used to set: `PATH` (Codex clears the
environment before running a hook) and the same `GIT_CONFIG_PARAMETERS` guard
as `hooks.json`. SessionStart still fetches a missing binary and the model,
and answers `{}` when there is no binary to run.
