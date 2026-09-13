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

## codex-hooks.json — Codex

Declared explicitly in `.codex-plugin/plugin.json` rather than discovered, so
the name is free. Codex clears the environment before running a hook, which is
why each command sets `PATH` itself.
