---
name: brain-report
description: Write a narrative report of this project's development history from its memory. Use when asked for a project history, a development journey, a retrospective, a weekly digest, a summary of what has happened, or "what have we been doing". Takes a mode - full for the whole history, weekly for a single ISO week.
---

# brain-report

Turn this project's memory into something a person reads. Two modes:

- **full** — the journey of the project so far. Default.
- **weekly** — one ISO week. The user may name a week; otherwise use the one
  that just ended.

## Gathering

1. `brain_timeline(since, k)` for the range. For **full**, start at the
   beginning and page forward with a generous `k`; for **weekly**, bound it to
   that week.
2. Read what carries the story. The timeline returns one-line titles with type
   tags — pull the full text with `brain_get` for the entries that matter, not
   for all of them.
3. `brain_search` when a thread needs following: a decision that references a
   bug, a fix whose cause was found earlier.

## What carries a narrative

The type tags rank the material, and the ranking is the report's spine:

- `DEC` decisions and `FND` findings — **the story**. A project's history is
  what it decided and what it learned. Lead with these.
- `SUM` session summaries — the connective tissue between them.
- `FIX` bugfixes and `NEW` features — what actually changed. The events.
- `CFG` config and `TST` test — seasoning. Mention them where they explain
  something; never make them a section.

A week with three decisions and forty config entries is a week about three
decisions.

## Writing it

Chronological, but not a list. A list of commits is something the user already
has; what memory adds is *why*, and why only reads well as prose.

- Open with what the period was actually about. One paragraph.
- Then the arc: what was tried, what was learned, what changed as a result.
  Group by thread rather than by day — a bug found Monday and fixed Thursday
  is one story, not two entries.
- Name files, decisions and errors concretely. Vague history is not worth
  reading.
- Close with where things stand, and what is unresolved if the memory says so.

Use headings and short paragraphs. This is a document, not a chat reply, so
length is allowed — but every paragraph should earn its place.

## The one hard rule

**State only what the observations state.** Do not infer a motive nobody
recorded, do not resolve a decision the memory leaves open, do not supply a
number, a name, or a date that is not there. If the record is thin for a
period, say it is thin. A history that quietly invents its connective tissue is
worse than a short one, because nobody can tell which parts were real.

Where memory is genuinely ambiguous, write the ambiguity: "the reason is not
recorded" is a true sentence and a useful one.

## What this is not

Not a status report on the working tree, and not a code review. Read the
repository only to check a detail the memory already raised. If the user wants
to know what the code does now, that is a different question and the code is
the source for it.
