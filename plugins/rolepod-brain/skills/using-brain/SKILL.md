---
name: using-brain
description: How this project's memory works and when to search it. Use when you need context from earlier sessions, when the user refers to a past decision or an earlier fix, when you are about to re-investigate something that may already be known, or when the user asks what brain remembers. Also covers keeping something out of memory.
---

# Using brain

Memory of this project persists across sessions and across CLIs. You did not
see those sessions, so treat it as a colleague's notebook: worth checking
before you re-derive something.

Across CLIs is literal: one brain per machine, written by every agent that
runs here. A benchmark codex ran an hour ago, a file cursor edited yesterday —
those are in the same memory you are reading, and nothing else will tell you
they happened.

## When to search before working

Search first when any of these is true:

- The user refers to something as already decided, already fixed, or already
  tried — the reasoning is probably recorded, and re-deriving it wastes their
  time and risks contradicting it.
- You are about to investigate a subsystem you have not seen this session.
- Something looks wrong in a way that suggests it was deliberate. A prior
  session may have made it that way on purpose.
- The user asks what happened before, or asks you to continue earlier work.

Do **not** search for things the current session already answers, or for
general knowledge. Memory is about *this project's history*, not about how
software works.

## Tools

- `brain_search(query, k)` — full-text over everything remembered. Bare words
  are ANDed; `"quoted phrase"` matches exactly; `OR` and `NOT` work. Start
  here.
- `brain_get(ids)` — full text of specific entries. Search returns one-line
  titles with ids; call this on the ones worth reading.
- `brain_recent(k, cli, kind, session)` — what happened most recently. Good for
  re-orienting at the start of a session or after a context compaction. Pass
  `cli` to read one agent's work on its own: `brain_recent(cli="codex")` is
  what codex did here, whether or not you were running.

  Pass `kind` with it, because agents run several sessions at once and the
  unfiltered list interleaves them:

  - `kind="session_summary"` — one line per finished session. This is the one
    to reach for first: "what has codex been doing" is a question about
    sessions, not about events.
  - `kind="raw"` — what a session is doing *right now*. A session that is still
    running has not been summarized yet, so this is the only way to see it.

  Every entry carries `session`, and `session` narrows to that one piece of
  work. When you do read an unfiltered list, group by it before drawing any
  conclusion — two adjacent lines are often two different pieces of work.

## Answering "what did the other agent do about X"

The user asks about work another CLI did — "find where codex analyzed this
project", "what did cursor change yesterday". Three calls:

1. `brain_search("<what they asked about>")` — find it by meaning. Hits from
   every CLI are in the same index.
2. Take `session` off the hit that matches.
3. `brain_recent(session=..., kind="session_summary")` for the conclusion, and
   `brain_recent(session=..., kind="raw")` for what it actually did.

A session that is still running has no summary yet; `raw` is then the whole
answer, and worth saying so rather than reporting the work as absent. Report
what the session concluded, not a list of its tool calls.
- `brain_timeline(since, k)` — chronological slice. Use when the question is
  about *ordering* or about when something changed, rather than about a topic.
- `brain_note(text, files)` — record something worth remembering that no tool
  call would show: a decision and its reason, a constraint, a dead end worth
  not repeating. Capture is automatic, so this is only for the *why*.

## What arrives without you asking

At session start, and again after a context wipe, you receive a short list of
pointers: an id, a time, a three-letter type tag, and a one-line title.

`DEC` decision · `FND` finding · `FIX` bugfix · `NEW` feature · `CFG` config ·
`TST` test · `SUM` session summary · `NTE` note · lowercase `raw` means it has
not been through consolidation yet.

Those lines are **titles only, never content**. If one looks relevant, call
`brain_get` with its id. When you touch a file that has history, one to three
of its pointers arrive the same way.

## When memory is wrong

The entries you are given were true when someone wrote them. Some of them
stopped being true afterwards, and nothing in this system notices on its own:
there is a path that writes durable claims and no path that retires them.

Measured on a real store: a page saying the release targeted four platforms
outlived the fifth by three days, still arriving in every session as current
fact — while a second page about adding that fifth target sat beside it,
unconnected. Three more had been fixed the same morning they were written.

So when a claim you were given disagrees with what is in front of you:

- **The file wins.** A memory is a record of what someone concluded once; the
  code is what is true now. Do not edit around a stale claim to keep it true.
- **Correct it while you are here.** `brain_correct(id, text)` replaces what
  recall returns; the original stays in the log. This is the whole retirement
  mechanism — if you skip it, the next session is told the same wrong thing,
  and so is the one after that.
- **Write it as a claim, not a note.** The first line becomes the title, so
  put the corrected fact there and the detail underneath.
- **Say what changed, not just what is right.** "This said four until the
  Windows target landed" is worth more than "five", because it tells the next
  reader the claim moves.
- **Unsure whether it is stale or you are wrong?** `brain_feedback(id)` sinks
  it for review without destroying anything, and the flagged list is written
  to the vault. Reach for that rather than leaving a claim you distrust
  ranking first.

What this is NOT: an invitation to rewrite memory to match an opinion. Correct
what you have checked — the file, the command output, the test — and leave the
rest alone.

## Keeping something out of memory

Text wrapped in `<private>` … `</private>` is removed before anything is
stored. It exists for what no pattern can recognize — a client's name, a figure
from a contract. Credentials are already scrubbed automatically; the tag is for
things only a human knows are sensitive.

If the user asks you to remember something *without* recording specifics, put
the general shape in `brain_note` and leave the specifics out entirely, rather
than wrapping them.
