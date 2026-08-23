---
name: using-brain
description: How this project's memory works and when to search it. Use when you need context from earlier sessions, when the user refers to a past decision or an earlier fix, when you are about to re-investigate something that may already be known, or when the user asks what brain remembers. Also covers keeping something out of memory.
---

# Using brain

Memory of this project persists across sessions and across CLIs. You did not
see those sessions, so treat it as a colleague's notebook: worth checking
before you re-derive something.

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
- `brain_recent(k)` — what happened most recently. Good for re-orienting at
  the start of a session or after a context compaction.
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

## Keeping something out of memory

Text wrapped in `<private>` … `</private>` is removed before anything is
stored. It exists for what no pattern can recognize — a client's name, a figure
from a contract. Credentials are already scrubbed automatically; the tag is for
things only a human knows are sensitive.

If the user asks you to remember something *without* recording specifics, put
the general shape in `brain_note` and leave the specifics out entirely, rather
than wrapping them.
