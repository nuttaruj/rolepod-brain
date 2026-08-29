---
name: brain-doctor
description: Check that this project's memory is actually working, and read the result back plainly. Use when the user asks whether brain is working, whether it is capturing, why recall came back empty or stale, where their memory is kept, or after they install it. Also use when memory behaves oddly and nobody has looked at the health report yet.
---

# brain-doctor

Almost everything that goes wrong here goes wrong quietly. Capture stops, a
summarizer rung loses its login, a rebuild leaves a backlog behind — and the
memory keeps answering, just with less in it. `brain_doctor` is the only thing
that says so out loud, and it exists as a tool precisely so nobody has to open
a terminal to hear it.

## Do this

Call `brain_doctor`. It takes no arguments and returns the checks as data:

    ok            every check passed
    failing       how many did not
    checks[]      name, ok, detail — one per check
    data_directory, wiki, project

Then show it. A table of name / status / detail reads best; put the failing
rows first, because those are the answer. Do not paste the raw JSON at
someone.

## Reading it back

**All passing** — say so in a line, give the capture count and where the wiki
lives, and stop. A healthy report does not need paragraphs.

**Something failing** — lead with what it means for them, not with the check
name. The details are written to be quoted, so quote the one that matters and
say what it costs:

- `capture` failing — nothing is being recorded. Everything else is moot until
  this is fixed; it usually means hooks are not wired, so `brain setup --apply`
  is the next step.
- `summarizer` failing, or a rung in cooldown — sessions still get pages, but
  rule-based ones: a list of what was touched rather than a narrative, and no
  durable knowledge at all. Nothing is lost. Those events stay pending and are
  redone the moment a model can be reached again, so the fix is usually just
  logging that CLI back in.
- `semantic` behind — search still works on words; meaning-based search is
  thinner until consolidation catches up.
- `processes` — this one is informational. It lists the MCP servers belonging
  to open sessions. There is no daemon; a number here is not a leak.

**Where is my memory** — `wiki` is the folder to open in Obsidian, and
`data_directory` holds the log and the index. Both come back with every call,
so this needs no second question.

## What not to do

Do not fix anything without being asked. This skill reports; `brain setup
--apply` and logging a CLI back in are the user's calls, and the difference
between "your summarizer is logged out" and running commands on their machine
matters.

Do not treat a rule-based summarizer as breakage. It is a supported mode — some
people set `mode = "off"` deliberately to spend no tokens — and saying "your
memory is broken" when it is doing exactly what it was configured to do is
worse than saying nothing.
