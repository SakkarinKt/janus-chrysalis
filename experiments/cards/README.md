# Experiment cards

A card is one experiment, written down before it is run: what will run, the one number it is
about, and the rule that turns that number into pass or fail. The rule is fixed when the card is
merged, so it cannot be chosen after seeing the result.

Each card is a directory, `NNNN-short-slug/`, with two files:

- `card.json`: the hypothesis, configuration, seeds and decision rule;
- `run.ts`: the code, exporting one function, `run`.

`npm run exp` runs cards and records them. `../README.md` covers the commands.

## Lifecycle

1. **Propose.** Draft the card under `proposed/`. Nothing there can be run. The daily loop may
   add at most one proposal per run.
2. **Approve.** The human moves it to `cards/NNNN-slug/` with the next free number. Merging that
   PR *is* the approval. Use the checklist at the end of this file.
3. **Run.** From a checkout of `main`, after the card has merged: `npm run exp -- NNNN`, or
   `npm run exp -- --next` for the lowest-numbered card that is ready.
4. **Record.** The run writes its manifest and appends one ledger line. Commit those and open a
   PR. That PR contains results only: no card or code changes.
5. **Reproduce**, whenever wanted: `npm run repro -- <ledger id>`.

A card is recorded once. There is no re-run with different settings and no editing a card to
fit what happened. To try something else, write a new card.

## `card.json`

| Field | Meaning |
| --- | --- |
| `id` | Four digits, equal to the directory's prefix. |
| `title` | One line. |
| `hypothesis` | What you expect, stated so the decision rule can contradict it. |
| `kind` | `confirmatory` (tests a hypothesis), `calibration` (measures something later cards read, with its own pass rule), or `repro` (re-runs something and compares it with a stored reference; runs on demand, any number of times). |
| `config` | Everything the run is parameterised by. `run.ts` reads it from here. |
| `seeds` | The seeds. Fresh ones for a confirmatory card: not seeds an earlier card already looked at. |
| `primaryMetric` | Name of the one number the rule is about. |
| `secondaryMetrics` | Reported alongside. Never part of the rule. |
| `decisionRule.text` | The rule in words, complete enough to apply without reading `run.ts`. |
| `decisionRule.thresholds` | The numbers the rule uses. `run.ts` reads them from here, never from its own constants. |
| `onFail` | What a `fail` changes in the research plan. If the answer is "nothing", the card is not worth running. |
| `dependsOn` | Cards that must have reached a given verdict first, and what this card reads from them. |
| `budget.maxWallMinutes` | The run is stopped, and nothing recorded, past this. |
| `supersedes` | The id of a card this one replaces before that card ran, or `null`. |

`TEMPLATE/` holds a skeleton of both files.

## `run.ts`

```ts
import type { RunContext, RunResult } from "../../../src/experiment/card.ts";

export function run(context: RunContext): RunResult {
  // context.card, context.seeds, context.outDir, context.dependencies, context.log
  return { primary: { name: "…", value: 0 }, verdict: "pass", resultSummary: {} };
}
```

- **Write outputs to `context.outDir`.** Every file there is hashed into the manifest. Name
  summaries `*.summary.csv`: those are committed and are what a reproduction compares.
- **Take every parameter from the card.** A constant in `run.ts` that duplicates a card field
  will drift from it.
- **Return a verdict or throw.** A throw means "no result": nothing is recorded and the card can
  run again. So if the rule says a NaN halt counts as failure, catch it and return `"fail"`.
- **Seed everything through `src/env/rng.ts`.** No `Math.random`, no clock, no network.
- **A dependency's result arrives in `context.dependencies[id].resultSummary`.** Do not read
  another card's files directly.

Cards are typechecked with zero errors allowed (`npm run typecheck:harness`).

## What CI enforces

`scripts/check-preregistration.ts` runs on every PR and fails when, compared with `main`:

1. a ledger line was removed, altered or reordered;
2. a recorded manifest no longer has the hash its ledger line recorded;
3. a card directory that is on `main` changed in any way;
4. a new ledger line's run did not execute, from a clean tree, at a commit that is on `main`
   with the card as it was at that commit;
5. a confirmatory or calibration card has a second result;
6. a new result's dependency has no earlier line with the required verdict;
7. a file under `artifacts/` that is on `main` was modified or deleted.

Rule 4 is the one that makes "pre-registered" checkable. This repository squash-merges, which
erases the order of commits inside a PR, so "the card was committed before the result" cannot
be verified from history. "The card was on `main`, and the run happened there" can.

In practice:

- **Merge the card first, then run it from `main`.** A card added and run in the same PR fails.
- **Do not commit between runs.** A run's own new files under `artifacts/` and its ledger line
  do not count as a dirty tree, so several cards can be run back to back and committed together.
  Committing in between moves `HEAD` off `main`, and the next run then fails rule 4.
- **Nothing else may be uncommitted.** A modified source file, card or committed artifact makes
  the run refuse. Use `--scratch` to explore on a dirty tree; scratch runs are never recorded.

## Changing your mind

- **Before a card has run:** add a new card with `"supersedes": "<old id>"`. The old card stays
  in the tree and will never run.
- **After it has run:** the result stands. Write a new card for the new question.
- **A card found to be wrong after it ran** (a bug in `run.ts`, say): the ledger line stays.
  Say what was wrong in the new card's `hypothesis`, and supersede nothing: the old card has a
  result, and that result is part of the record.

## Checklist before approving a card

- [ ] The hypothesis could come out false.
- [ ] The decision rule can be applied from `decisionRule.text` and `thresholds` alone.
- [ ] Every threshold has a stated basis (an earlier calibration card, or a measurement recorded in `config`).
- [ ] There is a control: the run also measures something the effect should *not* move.
- [ ] If the card claims to detect an effect, an earlier card showed the instrument detects a known one.
- [ ] A trivial baseline (persistence, a mean predictor) is among the secondary metrics.
- [ ] The seeds are new.
- [ ] `onFail` names a consequence.
- [ ] The budget is realistic: `npm run exp -- NNNN --scratch --seeds <one seed>` finished well inside it.
- [ ] A fresh session with no authoring context reviewed the card as a skeptic (`BLOCKING` / `NOTE` / `NIT`), and every `BLOCKING` is resolved.
