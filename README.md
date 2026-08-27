# rolepod-brain

Persistent memory for AI coding agents. One binary, one SQLite index, a
git-versioned markdown wiki — and **nothing running between events**.

Your coding CLI forgets everything when the session ends. rolepod-brain
captures what happened, consolidates it into readable pages, and hands the
next session a short list of pointers it can pull from. Install it once and
stop thinking about it.

## What makes it different

**Your brain never leaves your machine.** No cloud, no remote, no telemetry,
no account. There is no sync command that uploads anything, because there is
nowhere for it to upload to — see below for why that is a design decision
rather than a missing feature.

**No resident process, ever.** No daemon, no server, no supervised worker, no
port. Hooks spawn a short-lived process that exits; the MCP server lives
exactly as long as one session; consolidation runs when a session boundary
fires. After a reboot there is nothing to start.

```
$ brain doctor
ok   no resident process  nothing of ours is running
```

**No API keys. No tokens. No provider configuration.** Summaries are written
by whichever CLI you are already signed into, through that vendor's own
supported headless entry point (`claude -p --model haiku`, `codex exec`). This
project never holds a credential, because it never has one to hold. If no
model is reachable it writes rule-based summaries instead — a permanent,
first-class mode, not a broken one.

**Pointers in, content pulled.** Automatic injection carries titles and ids
against a hard byte budget. Full content is never pushed into your context;
the agent calls `brain_search` / `brain_get` when the task actually needs a
body. `brain doctor` reports the real bytes spent, not the configured limit.

**Found by meaning, not only by words.** A session that recorded `login token
expiry` is one someone later searching `auth` needs, and no keyword index will
ever connect those two. So a search asks five questions at once — which
memory used these words, which one means this, which session declared it was
about this, what else touched the same things, and (for scripts written
without spaces between words) which title contains this run of characters —
and fuses whatever each one ranked. A memory several of them agree on
outranks one a single ranking felt strongly about. Each is equal weight: a
per-stream tuning knob is a number nobody can justify a value for.

None of the five needs a model to be reachable, which is what keeps recall
wide when none is.

**In whatever language you work in.** The question and the memory do not have
to be in the same one: ask in Thai about work recorded in English and the
right memory still comes back. The model covers 101 languages, and this is
measured rather than assumed — 40 pairs of a Thai question and the English
memory answering it, the right one ranked first 80% of the time and inside the
top five 97%, measured on the quantized file that actually ships and not on
the original it was made from. The English-only model this replaced managed
2.5%, which is one in forty: chance.

The model is 122 MB of static embeddings — no Python, no second process, no
API key, and no service to be down. It is fetched once when brain is
installed, checksum-verified, and never touched over the network again; the
binary itself is under 4 MB on every platform. Until the model arrives, and
if it never does, recall
runs on its other four rankings and `brain doctor` says so.

It is never loaded into memory either. A search needs about ten rows out of
half a million, so those rows are read where they sit and the operating
system's page cache shares them between every brain process. Segmenting the
query is written out here rather than taken from the usual library for the
same reason: that library holds this vocabulary as a trie of one hash map per
byte, which is 710 MB that never goes away.

It is never loaded during capture, which is why hooks still answer in ~13ms;
vectors are written by consolidation, and `brain doctor` reports how much of
the corpus has one yet.

Anything that DELETES on what it finds — `forget --entity` — searches by words
only. A semantic ranking always has a closest answer, and a bulk withdrawal
must never be handed a ranking that always returns something.

**A file's memory arrives before the file does.** Open a file and what we
already know about it lands ahead of its contents, while that can still change
what the turn does — not after the agent has read it and already has an answer.
The pre-read hook only injects; it never captures, so it adds nothing to store
and nothing to summarize.

**Secrets are scrubbed before anything is written.** Redaction happens in the
capture process, ahead of the log. There is no later stage that could catch a
leak, so there is no window where one exists.

**The log is the truth.** Every observation is an append-only, fsynced JSONL
line keyed by a ULID. SQLite and the wiki pages are derived: delete the index
and `brain reindex` rebuilds it. It is also what makes the wiki safe to copy,
merge or roll back with ordinary git: entries are keyed by ULID, so two logs
combine without conflicts.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh
```

That fetches the binary for your platform, checks it against the checksum
published with the release, installs it to `~/.local/bin/brain`, fetches the
embedding model once, and wires every supported CLI it finds. It prints a plan first and asks before touching
anything. No repository is left on your machine, and no Rust toolchain is
needed — what lands is one binary.

This binary reads what you type into your editor, so it refuses to install a
download whose checksum does not match, and `bootstrap.sh` is short enough to
read before you run it. If you would rather not pipe a script to a shell, take
the binary from [Releases](https://github.com/nuttaruj/rolepod-brain/releases)
yourself and run `brain setup`.

### Every CLI on the machine

The default, and what the command above already does. `--yes` skips the
confirmation, for a scripted or headless install.

```sh
curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh -s -- --target=all
curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh -s -- --target=all --yes
```

### One CLI

`claude-code`, `codex`, `cursor`, `gemini-cli`, `antigravity`, or `opencode`.
A name it does not recognise is refused rather than quietly wiring nothing.

```sh
curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh -s -- --target=codex
```

### Update

The same command that installs. It fetches whatever the latest release is,
verifies the checksum, and replaces the binary in place — your memory and the
wiring are untouched, so there is nothing else to re-run. `brain --version`
says what you have; the [Releases](https://github.com/nuttaruj/rolepod-brain/releases)
page says what is current.

```sh
curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh -s -- --yes
```

### Uninstall

Unwires every CLI it wired. The binary and your memory are left alone — `brain
where` prints where the memory lives if you want that gone too.

```sh
curl -fsSL https://raw.githubusercontent.com/nuttaruj/rolepod-brain/main/bootstrap.sh | sh -s -- --uninstall
```

### Or install it as a plugin

On Claude Code and Codex the plugin is a complete install on its own — it
carries the hooks, the MCP tools and the skills, and on your first session it
fetches the binary if it is not already on your PATH and starts the embedding
model download in the background. Nothing waits on that: the session opens
immediately and recall gains its fifth ranking when the model lands. Every
other CLI is wired by the one-liner above.

Codex is installed this way and no other. Its plugin flow is what grants the
hooks permission to run, and brain does not write its own trust entry.

```sh
# Claude Code
claude plugin marketplace add nuttaruj/rolepod-brain
claude plugin install rolepod-brain@rolepod-brain

# Codex
codex plugin marketplace add nuttaruj/rolepod-brain
codex plugin add rolepod-brain@rolepod-brain
```

Updating:

```sh
# Claude Code
claude plugin marketplace update rolepod-brain
claude plugin update rolepod-brain@rolepod-brain

# Codex
codex plugin marketplace upgrade rolepod-brain
codex plugin add rolepod-brain@rolepod-brain

brain setup --apply     # only needed the first time the plugin takes over the hooks
```

The plugin carries the hooks and tools; the binary updates itself separately —
the one-liner under [Update](#update) at any time, and the plugin's own first
session fetches it when it is missing entirely.

That last line matters exactly once. Before the plugin carried hooks, `setup`
had written its own into `settings.json`; with both in place every event would
be captured twice. `setup` notices the plugin, stands down, and takes its own
entries back out — and `brain doctor` then reports capture as coming "via the
plugin" rather than claiming the machine is unwired.

The two paths are not exclusive. Whichever you use, `brain setup` is the thing
that reconciles them, and it is safe to re-run at any time.

### Seeing what it would do

```sh
brain setup            # dry run; --apply performs it
brain doctor           # what is actually wired, and what is not working
brain where            # which project you are in, and where its memory lives
```

`setup` backs up each config before writing and only ever replaces entries it
put there itself. Hooks belonging to other tools are left exactly where they
are.

## Supported CLIs

| CLI | Capture | MCP recall | Status |
|---|---|---|---|
| Claude Code | 8 lifecycle events | registered automatically | verified by our tests and daily use |
| Codex | 7 lifecycle events | via the plugin | installs as a plugin; capture needs one approval — see below |
| Gemini CLI | 5 lifecycle events | register manually | capture works; its own summarizer tier is unavailable here |
| Antigravity (`agy`) | 2 lifecycle events | register manually | capture verified; needs an explicit workspace, see below |
| OpenCode | 4 lifecycle events | register manually | session capture verified; tool capture wired, not yet exercised |
| Cursor | 3 lifecycle events | registered automatically | capture verified |

Cursor reports its project under `workspace_roots`. It also sends a `cwd`, but
that field arrives **empty** — which is why the lookup requires a non-empty
value rather than merely a present key. Without `workspace_roots` its hooks
would have been unplaceable, since Cursor runs them from its own config
directory.

Only `postToolUse` is wired for tool activity: `afterShellExecution` fires for
the *same* execution, so wiring both would record every shell command twice.
In headless `cursor-agent -p` runs, tool events are the only ones observed;
`beforeSubmitPrompt` and `stop` are wired for interactive sessions.

OpenCode has no hook configuration file — `setup` installs a small plugin into
`~/.config/opencode/plugins/` instead. The plugin takes the project path from
OpenCode's own plugin factory, so it has none of the placement problem below.
Its session events are verified live; its tool events are wired against a
handler signature read from a working plugin, but OpenCode's model provider is
failing on the author's machine, so that path has not been exercised end to end.

Antigravity gives a hook no working directory and runs it from its own config
directory, so it can only be placed in a project when the workspace is explicit
— `agy --add-dir <project>`, or launching it from an added workspace. Without
that, its events are **skipped rather than filed under a guess**: a memory in
the wrong project is worse than a missing one.

All CLIs write to **one store per project**. Work in one CLI in the morning
and another in the afternoon; it is a single memory, tagged by which CLI
observed what. Codex exposes no session-end event, so consolidation there
triggers on `Stop` with a debounce.

The table says "verified" only where our own tests cover it. If a CLI is
missing here, it is not supported yet.

## Use

Nothing, normally. That is the point. When you want to look:

```sh
brain search "auth" --topic decision   # only what was DECIDED about auth
brain history decisions --diff        # what a page used to say, and when it changed
brain doctor            # is capture actually working?
brain stats             # what it has captured, consolidated, and injected
brain search "auth"     # full-text search this project's memory
brain where             # which project am I in, and where does it live
```

Your agent gets ten MCP tools. Four to read — `brain_search` by words and
meaning, `brain_get` for a full body, `brain_recent` to re-orient, and
`brain_timeline` for a stretch of history. Two to move around memory rather
than query it: `brain_outline`, for what a project IS before you know what to
ask about it, and `brain_related`, for what sits beside a memory you are
already holding. Four to write back — `brain_note`, `brain_correct`,
`brain_feedback`, and `brain_forget`.

On Codex the plugin also ships two skills:
`using-brain`, describing when to reach for those tools — MCP tools that
nothing tells the model about tend not to get called — and `brain-report`, for
turning the memory into something a person reads.

Ask for "a report on this project's history" or "a digest for last week" and
`brain-report` writes a narrative from the decisions and findings memory holds,
ranked by what actually carries a story. It is pull-only: produced when asked,
never injected.

## Where things live

```
~/.rolepod-brain/
  brain.db                       # derived index (FTS5) - disposable
  Rolepod Brain/                 # git repository - the durable memory
    <project>/
      events/YYYY-MM.jsonl       # append-only log - the source of truth
      pages/sessions/*.md        # one page per consolidated session
      knowledge/gotchas/*.md     # what stayed true across many sessions
      entities/*.md              # the things sessions kept being about
      <project>.md, <topic>.md   # hub notes linking the rest together
  config.toml                    # optional; defaults are deliberately light
```

Projects live directly under `wiki/` by their plain names. Two exceptions,
both earned rather than default: a project whose basename collides with
another's gets a `--<id>` suffix so two projects never share a directory, and
a project assigned to a named workspace (via the marker file below) nests
under `wiki/<workspace>/`. Trees written by older versions lived at `wiki/` with every project under
`wiki/default/` and a permanent suffix; they keep working untouched, and
`brain reindex` moves the whole tree to its human-first shape without losing
a line.

Projects are keyed by the main git repository root, so every worktree of one
repo shares one memory. Drop a `.rolepod-brain.toml` in any ancestor directory
to override the project or workspace explicitly — useful for monorepos and for
keeping work and personal memory apart.

## Configuration

Everything has a default. `brain setup` leaves a fully-commented
`~/.rolepod-brain/config.toml` with every knob visible at its default —
uncomment a line to change it; it never overwrites a file you have edited.

```toml
[summarizer]
mode = "auto"          # auto | claude-code | codex | gemini | off
                       # "off" = permanent rule-based summaries, fully functional

[summarizer.models]    # optional per-CLI model overrides. Memory quality is
                       # a spend decision: the default is each CLI's cheap
                       # tier, and naming a better model here buys better
                       # summaries at that CLI's price. Per-CLI because model
                       # names do not travel between vendors.
# "claude-code" = "sonnet"

[injection]
primer_budget = 4096   # bytes pushed at session start (~1k tokens)
session_budget = 8192  # ceiling for ALL automatic injection in one session

# Raising these buys the agent more memory up front - and costs input tokens
# in EVERY future session, which is the definition of a hidden recurring
# spend. Lowering them keeps only the top-ranked lines (durable knowledge
# first, then summaries); the agent can always pull more through
# brain_search, which has no budget because the agent asked. `brain doctor`
# reports what sessions actually spend against the cap - tune from that
# number, not from a guess.

[search]
rerank = false         # true reranks every search: local model, else CLI
                       # results by what the query was asking

[sanitize]
extra_patterns = []    # additional regexes to redact (see Redaction below)
allowlist = []         # substrings that survive redaction
```

### `search.rerank`

FTS5 ranks by term statistics, which is a good proxy for relevance and a poor
one for intent. Ask *why did we stop using the queue* and every entry that
mentions a queue scores; the one that explains the decision can sit seventh.

Reranking reads the candidates and puts the ones that answer the question
first. Two engines do it, and which one you get depends on the machine:

| | how | cost |
|---|---|---|
| local | a cross-encoder in this process | **~1.7s** |
| host CLI | one call to the CLI whose work is being searched | ~12s |
| neither | the index's own order | 0.2s |

The local one is about 600 MB — the model, its tokenizer, and ONNX Runtime
itself — fetched the first time something actually asks for a rerank, never at
install, because reranking is off by default and most installs will never turn
it on. That first search falls through to the CLI while the download runs; the
next one has the model. `brain doctor` says which state a machine is in.

Every platform can run it. Nothing links ONNX Runtime into the binary: the
runtime is downloaded beside the weights, per platform, checksum-verified the
same way. Which one a machine receives depends on what exists for it — 1.28
nearly everywhere, 1.23 on Intel macOS, which is the last one Microsoft built
for that architecture and is measurably close behind.

A machine that cannot load its runtime is not a broken install. Reranking
there falls through to the host CLI, which is where every machine starts
anyway, and the search still returns.

Ask for it per search — `brain_search(query, rerank: true)` — when a question
is worth the wait, or set `rerank = true` in config to make it the default for
every search. Either way it is bounded: one call, no second vendor if the
first cannot answer, and any failure at all leaves the order exactly as the
index ranked it. The worst case is the search you already had.

## How the plugin fits

This repository is a plugin marketplace, and on Claude Code and Codex a plugin
is enough on its own. The commands are under
[Install](#or-install-it-as-a-plugin); this is what makes them sufficient.

The plugin carries the hooks as well as the MCP tools and the skills, and its
`SessionStart` hook fetches the binary — announced, checksum-verified, once —
if `brain` is not already on your PATH. `brain setup` then stands down for that
CLI rather than writing a second set of hooks beside the plugin's, which would
capture every event twice; `brain doctor` reports capture as coming "via the
plugin".

That only works where the host loads a hooks file from a plugin. Every other
CLI is wired by the one-liner, which is why the plugin route is documented for
these two and nothing else.

### Codex is the exception

Codex will not run a hook it has not been told to trust, and it does this
silently. Entries written straight into `~/.codex/hooks.json` are therefore
useless — nothing runs them and there is no reliable way to approve them. What
Codex does have a trust path for is a plugin's own bundled hooks, so on Codex
the plugin carries capture as well.

Then **open Codex interactively once and approve the plugin's hooks**. Until
that approval exists, the plugin is installed and enabled and still captures
nothing; a non-interactive `codex exec` cannot grant it. `brain doctor` reports
whether the plugin is installed, and the event counts in its capture line are
what tell you whether approval actually took.

`brain setup` does not write Codex hooks itself; it removes any raw entries an
older version left behind and points here.

## Surviving a context wipe

`/compact` and `/clear` destroy what the agent knows while the session itself
continues. Memory has to come straight back, so both paths re-inject the primer
— Claude Code reports each of them as a session start that names its source.

Claude Code also has a `PostCompact` hook, and we deliberately do not register
it: that event will not accept injected context, so a hook that answers it fails
validation, drops the primer, and prints a schema dump at the user instead. The
session start already covers compaction, and it works.

The subtle part is that per-session de-duplication has to reset at the same
moment. A session id survives a compaction; the context does not. Without the
reset, the guard that stops us repeating ourselves would suppress exactly the
memory the fresh context needs — turning our own safeguard into the amnesia it
exists to prevent. Compaction also kicks consolidation first, so the primer
that lands a moment later carries a narrative rather than a list of commands.

Codex was previously documented here as having no compaction or session-end
hooks. That was wrong, and worth saying plainly: the claim came from reading
this machine's `hooks.json`, which lists what somebody had configured — not
what Codex supports. A probe settled it. `SessionEnd` fires and is now wired;
`PreCompact` is wired too, on the weaker evidence that Codex's own trust store
holds a `pre_compact` entry belonging to another tool. Forcing a real
compaction to watch it fire was out of scope, so treat that one as wired rather
than witnessed.

Either way capture is continuous — events land as they happen rather than being
gathered at session end — so a compaction costs context, never memory.

## Headless runs

A one-shot invocation — `claude -p`, `codex exec` — usually is not a person
working. It is an orchestrated step: a reviewer, a judge, a summarizer. So
those runs receive **no automatic injection**. Handing a reviewer the author's
own narrative quietly destroys its independence, and nothing downstream can see
that it happened.

They still capture, tagged, and a headless run's observations rank below a
person's in the primer. For a completely clean room — no injection *and* no
capture — set `ROLEPOD_BRAIN_SILENT=1` in the environment of the process you
want left alone. That variable is a stable public contract; orchestrators are
meant to set it directly.

## When it calls a model

Consolidation is the only thing that spends a model call, and it happens at
session boundaries — never mid-session, never per turn. What counts as a
boundary differs per CLI, because their lifecycle surfaces differ:

| CLI | Consolidates on |
|---|---|
| Claude Code, Codex | session end, compaction |
| OpenCode | session idle, compaction |
| Antigravity, Cursor | end of turn — they expose no session-end event |
| Gemini CLI | the backstop only; no boundary event reaches us |

`brain doctor` prints this for the CLIs you actually have installed. Whatever a
boundary misses, the backstop below finishes when the next session opens.

## Nothing runs in the background

There is no launchd agent, no login item, no timer — not off by default,
**gone**: no code path in this product can register anything to run in the
background. Consolidation happens when a session ends, and a session *opening*
finishes anything stale left over — which covers every case that matters,
because consolidated memory only has value when a next session reads it, and
that session fires hooks. (Early versions had an opt-in wall-clock timer; it
was removed rather than left disabled, because enabling it put a launchd job
in Login Items — "brain can run in the background" on your own screen, from
the product whose promise is that nothing does. `brain setup --apply` removes
the job from machines that once enabled it.)

## What the summaries are written from

Lifecycle hooks see tool calls and prompts. They never see the model's own
prose — the reasoning it wrote, the decision it explained, the dead end it
described — which is usually where the *why* lives.

Your CLI already writes that prose to a transcript on disk for its own
purposes. At consolidation time we read the recent tail of it, hand it to the
summarizer beside the captured events, and **persist only the summary**. No
transcript content is ever copied into this memory; the store keeps a path, and
that is all. If the transcript has been cleaned up by the CLI, consolidation
quietly proceeds without it.

Claude Code and Codex provide transcripts. OpenCode, Antigravity and Cursor do
not, and their summaries are written from events alone.

## What outlives a session

A session page answers *what happened on Tuesday*. Most of what you actually
want back is narrower and longer-lived: that vitest has to run file-by-file in
this repo, that the retry wrapper was a deliberate choice and not an accident,
that a migration here needs two steps in a particular order.

Every fifth consolidated session, brain spends one extra cheap-tier call over
the recent session summaries and asks what has become durably true. What comes
back is written as pages under `knowledge/` — `gotchas/`, `decisions/`,
`procedures/` — and recorded in memory proper, so it turns up in `brain_search`
and in the primer alongside everything else, tagged `KNW`.

Three rules keep these honest:

- **Provenance.** Every page names the session summaries it was drawn from. A
  durable claim you cannot trace is indistinguishable from an invented one.
- **It has to recur.** An entry has to cite at least two session summaries or
  it is discarded — a claim one session supports is a session summary wearing
  a promotion, and knowledge outranks summaries in the primer. The prompt asks
  for this too, but the filter is what enforces it. At most five entries are
  kept per round, so one talkative synthesis cannot crowd out the primer.
- **No duplicates.** Later rounds rediscover what they found before. Something
  already known is skipped rather than appended again.

Redaction and the anti-invention rules apply here exactly as they do to session
summaries. Without a reachable CLI, synthesis simply does not run — deciding
what recurs is a judgement, and a rule-based stand-in would produce confident
nonsense.

## `ROLEPOD_BRAIN_WORKER` — a contract for other tools

When brain consolidates, it runs your CLI headlessly. Every process it spawns
carries `ROLEPOD_BRAIN_WORKER=1`, and so does everything below it — including
the lifecycle hooks that CLI fires on itself.

brain uses this on itself first: its own capture hook exits immediately when it
sees the variable, which is what stops consolidation from recording its own
model calls as if they were your work.

**The same signal is available to anyone else.** If you run orchestrators,
notifiers, or gates on the same lifecycle hooks, they will otherwise fire once
per consolidation — a desktop notification for a summary you never asked to
see. One line at the top of such a hook is enough:

```sh
[ -n "${ROLEPOD_BRAIN_WORKER:-}" ] && exit 0
```

This is a stable public contract: the variable name will not change for
internal convenience.

Note what brain deliberately does *not* do: it does not strip the environment
it inherits. A summarizer child therefore carries whatever session identity
your orchestrator set. Guessing which variables belong to which tool would
couple brain to tools it does not own, and pruning the environment down to a
guessed minimum would break CLIs in ways that fail silently — consolidation
would simply fall back to rule-based summaries with nobody the wiser. The flag
above is the honest interface instead.

## When a CLI runs out

If the CLI that produced the work cannot summarize it — rate limited, logged
out, a model id that no longer exists — consolidation moves to the next CLI you
are signed into, then to rule-based summaries. Events that only got the
rule-based treatment stay marked unconsolidated, so the next working run
rewrites them properly. The ladder loses quality, never data.

Both failure shapes advance: a crash or non-zero exit, and an answer that comes
back with exit 0 but is unusable — a login prompt, a quota banner, empty
output. That second case is the one that matters in practice, because an
exhausted subscription often looks like success. Each attempt is charged to
that CLI's circuit breaker, and one call tries at most two CLIs, so a prompt
none of them can handle costs two calls rather than one per CLI you own.

## Redaction

Three layers, in order of how much they can be trusted:

1. **Patterns, at capture.** Secrets are scrubbed before anything is written.
2. **Instruction, at consolidation.** The summarizer is told never to reproduce
   a credential — to say "configured the API key", never its value.
3. **Patterns again, over the model's output.** Everything a model writes goes
   back through the same scrub before it is persisted.

Layer 3 exists because layer 2 is a request, not a guarantee. Small models
misread instructions — one returned bare strings where the schema asked for
objects — and security cannot rest on a model complying. There is a test that
plants credentials in a summarizer's output and proves they never reach disk.

### Extending the patterns

The built-in patterns cover the generic shapes — cloud keys, `*_TOKEN=`,
`*_PASSWORD=`, credential paths like `.ssh/`. What they cannot know is what
counts as a secret *in your organisation*. `extra_patterns` adds your own
regexes on top; anything they match is replaced with `[REDACTED]` before it
is written anywhere:

```toml
[sanitize]
extra_patterns = [
  "EMP-[0-9]{6}",                        # employee ids
  "[a-z0-9-]+\\.internal\\.acme\\.com",  # internal hostnames
  "ACME-(PROD|STAGE)-[A-Za-z0-9]{16}",   # your license-key format
]
```

`allowlist` is the opposite valve — for **false positives**, strings a
built-in pattern catches that are not secrets at all. The classic case is
code that assigns something *named* like a credential:

```toml
[sanitize]
# `DESIGN_TOKEN = "spacing-4"` is a CSS value, but the generic `*_TOKEN=`
# pattern cannot know that. Allowlisted substrings survive redaction.
allowlist = ["DESIGN_TOKEN", "CSRF_TOKEN_HEADER"]
```

If your session pages keep showing `[REDACTED]` where real content should be,
an allowlist entry is usually the fix; if you spot something in a page that
should never have been written, an extra pattern is. An invalid regex fails
loudly at startup rather than silently disabling itself.

### `<private>` — the optional escape hatch

Some things no pattern can recognize: a client's name, a figure from a
contract. Wrap them and they are removed before anything is stored or shown to
a summarizer:

```
deploy for <private>Acme Holdings, 4.2M contract</private> next week
```

An unclosed `<private>` drops everything after it. That is deliberate: you
typed the tag because what follows must not be kept, and a missing closer is far
more likely a typo than an invitation.

You never have to learn this. The three layers above run either way.

## Reading it in Obsidian

There is no sync step, because none is needed:

1. In Obsidian, choose **Open folder as vault**.
2. In the folder picker, press **Cmd+Shift+G** (macOS) and paste
   `~/.rolepod-brain/Rolepod Brain` — the `.rolepod-brain` directory is
   hidden, so it will not appear in the list on its own. (Cmd+Shift+.
   toggles hidden folders if you prefer to browse; on Linux, type the path
   into the location bar.)
3. Open it. That is the whole setup.

Obsidian reads the markdown in place, new pages appear as consolidation
writes them, and the vault shows up in the switcher under the product's name
— the directory is named for exactly that reason, because Obsidian names a
vault after its folder. One caution that follows from the same fact: renaming
the vault inside Obsidian renames the real directory, and brain looks for its
memory by name. If you have renamed it, rename it back — `brain doctor` will
tell you if capture has started a second tree in the meantime.

Three things worth knowing before you do:

**You can correct a summary where you read it.** If a session page says
something wrong, fix the text under `## Summary` in Obsidian. The next
consolidation reads your wording back into the log as a correction, so it
survives every later rewrite — and every `brain reindex`, because the log is
what pages are rendered from. `brain consolidate` says how many edits it
adopted.

Only the summary section works this way. The rest of a page — timeline,
files, frontmatter — is rendered from the log verbatim, so changes there are
simply rewritten. Hub notes, entity pages and `index.md` are regenerated
whole.

**Write alongside instead.** Consolidation writes `pages/`, `knowledge/`,
`entities/`, the hub notes at the top of a project directory, and `index.md`.
Anything outside those is left alone — a `notes/` folder inside a project's
directory is never touched. For notes you want the *agent* to see later, use
`brain_note`, which puts them in memory proper rather than beside it.

Obsidian writes its own configuration into any folder it opens; the wiki's
`.gitignore` already keeps that out of the history.

## If you want it on more than one machine

**We never sync your brain. The format makes it trivially yours to sync if you
choose.** It is a folder of markdown in a git repository — Syncthing, `rsync`
to a NAS you own, a private git remote you push by hand, or Obsidian Sync all
work on it without this project shipping a line of network code.

That is a different offer from a memory product with a cloud tier: there, your
memory sits on someone's server in a form they can read, and you pay for the
privilege. Here the transport is your choice, and so is who can read it.

Choose the channel accordingly, because of what the wiki is: a
reverse-engineering blueprint of your projects, complete with the reasoning and
the dead ends. Pick something you trust end to end.

For Obsidian Sync specifically, end-to-end encryption is the default when you
create a remote vault, and it depends on an encryption password you set and
keep — the documentation is explicit that losing it means the data stays
encrypted and unusable, with no recovery by anyone including Obsidian. Verify
the current terms yourself before trusting any summary of them, including this
one.

## Fixing what it remembers

A summary written by a cheap model can be wrong, and a wrong memory is injected
into every later session with exactly the confidence of a right one — which
nobody notices, because nobody re-reads a wiki looking for sentences that were
never true.

```sh
brain forget  <id>                    # withdraw one entry
brain correct <id> "text"             # replace what it says
brain forget --entity acme-corp       # list everything that mentions it
brain forget --entity acme-corp --apply   # …then withdraw all of it
```

`--entity` is for "forget everything about this customer / this key / this
repo", where you do not know the ids and should not have to list them. It
matches at the entry level so **unrelated memories from the same sessions
survive** — the whole point of forgetting one thing rather than one session.
It prints what would go and does nothing without `--apply`, and it is honest
about its reach: matching is by text, so a mention under a different name or
in another script is not found. It withdraws what it listed; it does not
claim the list is complete.

Neither deletes anything. Both append an entry that is *about* the earlier one,
and recall stops showing the old version. The log keeps the original wording,
the withdrawal, and the correction, so the history of what memory believed
stays honest. Your agent can do the same through `brain_forget` and
`brain_correct` when you say something is wrong mid-session — though it may
only withdraw entries it has actually been shown, not ids it guessed at.

## Moving to another machine

There is no sync, so migration is a command rather than a hope:

```sh
brain export brain.tar.gz          # on the old machine
brain import brain.tar.gz          # on the new one
```

The archive carries the log and the pages; the index is left behind and rebuilt
on arrival, which also proves the log really is the source of truth. Importing
onto a machine that already has memory refuses until you pick `--merge` or
`--replace`, and `--replace` moves the old wiki aside rather than deleting it.

One thing to know: a project's identity normally follows its path, so the same
repository checked out somewhere else is a different project. Put a
`.rolepod-brain.toml` with a `name` in it and identity follows the name instead
— which is what makes an imported brain attach to the work rather than sit
beside it.

## Removing it

```sh
brain uninstall            # prints a plan
brain uninstall --apply    # unwires every CLI
brain uninstall --apply --wipe   # ...and deletes the memory, after you type DELETE
```

Hooks belonging to other tools are left exactly where they are. Without
`--wipe`, your memory stays on disk and the command tells you where.

## Why it stays local

The wiki is the most sensitive file set this machine holds, and that is not an
exaggeration about privacy in general — it is what the contents actually are.

It accumulates the architecture of every project it watched, the sequence of
decisions that produced it, and the approaches that were tried and abandoned:
the weak points already known internally. Someone holding this wiki can
reconstruct a project *and the reasoning behind it* without ever seeing the
source. For client or confidential work, that is worse than a source leak.

So:

- **No remote sync.** The code to push anywhere was removed, not disabled.
  `brain sync` exists only to say so.
- **No cloud, no account, no telemetry.** Nothing is sent anywhere, ever.
- **No API keys held.** Summarization borrows a CLI you are already signed
  into, through its own supported entry point.

What remains is a plain-markdown git repository at `~/.rolepod-brain/Rolepod Brain` —
`grep` it, open it in Obsidian, read its history with `git log`, roll it back
with `git revert`. Back it up the way you back up the rest of your disk; Time
Machine and `rsync` to a disk you control both work, and the choice is yours
rather than ours.

Everything that reduces what the wiki holds in the first place — the capture
sanitizer, the redaction pass over model output, `<private>`, and declining to
file an event whose project cannot be determined — is defence in depth for the
day the machine itself is the breach.

## Development

```sh
cargo test              # unit and end-to-end
cargo test --release    # also enforces the 50ms hook latency budget
cargo clippy --all-targets
```

The end-to-end suite runs against the real binary in an isolated data
directory and covers the claims above: two CLIs merging into one store,
cross-CLI recall, secrets never reaching the log, index rebuild from the log,
and graceful degradation when no model is reachable.

## License

MIT. See `LICENSE`, and `NOTICE` for third-party attribution.
