# Learning journal — @SakkarinKt

This is the human author's journal. It is raw material for the final write-up, and the place
where explain-before-implement closes. Claude's loop keeps its own journal in
[`loop/LEARNING.md`](loop/LEARNING.md). The two are read side by side at the twice-monthly retro
(`reports/retro/`). The daily loop reads this file but never writes to it.

*(Split 2026-09-23. Claude's 2026-07-04 setup-day entry moved to `loop/LEARNING.md`, unedited.)*

## How to write an entry

Short beats complete: one entry this week beats a perfect backlog. Two kinds:

**Explainer summary.** This is the merge condition for each `docs/explainers/NNNN-*.md`
(`CONTRIBUTING.md`). It is three sentences in your own words: *what* the thing does, *why* it's
built that way, and *one thing* you'd check if it broke. If you can't write it, the explainer
failed; say so here, and Claude revises the explainer, not your summary. Use this heading
exactly, because the loop counts these to show your backlog (`N/M explainers summarized`):

```markdown
### Summary — 0012 replay buffer + λ-returns (YYYY-MM-DD)
```

**Weekly reflection.** Answer any one of these:

- What assumption of mine broke this week?
- What did Claude get wrong that I caught, and how did I catch it?
- What can I now do without Claude that I couldn't at the last gate?

```markdown
## YYYY-MM-DD — week NN
```

---

<!-- First entry goes below. Suggested start: the 0012 summary, since you're implementing it. -->
