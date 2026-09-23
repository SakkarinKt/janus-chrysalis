# Learning journal — Claude (daily loop)

The loop's own journal: one short entry on the first run of each ISO week (`loop/GOAL.md`,
"Learning journals and retro"). Append-only. Covers the loop's process failures, what the
human's reviews taught it, and assumptions that broke. The human's journal is root
[`LEARNING.md`](../LEARNING.md). The two are read side by side at the twice-monthly retro
(`reports/retro/`).

*(Created 2026-09-23 by the journal split. The first entry below moved here, unedited, from
root `LEARNING.md`. No weekly entries were written between 2026-07-04 and 2026-09-23, because
root `LEARNING.md` was outside the loop's write paths. That gap is recorded as session audit P5,
`reports/quality/2026-09-23-session-audit.md`.)*

---

## 2026-07-04 — Claude (setup day)

The project brief said "multi-agent world model" and I initially saw three plausible projects in that phrase (MARL world models, a multi-agent research pipeline, or just co-authorship). Asking four targeted questions before planning collapsed the space in minutes — the human wanted *all three dimensions at once*: MARL as subject, agents assisting the process, and co-authorship as method. Assumption that broke: I treated the daily autonomous loop as optional infrastructure to defer; for the human it is itself one of the experiments ("Loop Engineering") and a hard requirement — the work must advance while their Mac is offline. Lesson: when a collaborator names a constraint that looks like tooling, check whether it's actually a research goal.

## 2026-09-23 — week 39 (interactive session, written at the split)

**What went wrong in the process.** From 2026-09-06 on, the loop ran out of authorized work. The
priority list was exhausted and everything left was human-reserved. I kept filing 800–1,300-word
stand-ups, each re-proving the same exhaustion. Every one was *correct*, but a correct stand-up that
asks nothing of the human costs them reading time and moves nothing forward.

**What reviews taught me.** The human's reviews are precise: line numbers, exact numbers, and
which items to fold into one PR. The fastest loop turn was always "do exactly what the review
enumerated, plus nothing". The PR #80 review also widened the reserved list (`Actor`/`Critic`) in a
comment rather than in GOAL.md. Rules can change in review threads, and the contract file only
catches up at an amendment.

**Assumption that broke.** I assumed "explain-before-implement" was working because the
explainers kept landing. Its actual merge condition, a human summary in `LEARNING.md`, was met
0 times in 14. A process can look healthy from the side that produces output.

**Do differently.** Put the one human action at the top of every stand-up, and on quiet days
say less.
