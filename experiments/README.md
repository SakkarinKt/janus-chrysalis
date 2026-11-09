# experiments/

Two generations of experiment live here.

- **Cards** (`cards/NNNN-slug/`), from 2026-11-09 on. One command runs a card and records it.
  New experiments are cards.
- **Dated scripts** (`YYYY-MM-DD-<slug>/run.ts`), before that. Frozen records of what ran.

## Cards: run, record, reproduce

A card is a directory holding `card.json` (what will run and the rule it will be judged by) and
`run.ts` (the code). It is written and merged *before* it is run.

```bash
npm run exp -- --list
```

```bash
npm run exp -- 0000
```

```bash
npm run exp -- --next
```

`--list` shows every card and where it stands. A card id runs that card. `--next` runs the
lowest-numbered card that is ready, which is how the daily loop picks its work.

A recorded run:

- needs a clean working tree, so the commit in the manifest describes exactly what ran;
- writes `artifacts/<card>/<runId>/manifest.json` (schema v2) beside the run's outputs;
- appends one line to `ledger.jsonl`.

A `confirmatory` or `calibration` card is recorded **once**. Asking again is refused: the rule
was fixed before the run, and a second run would be choosing between results. A `repro` card
(such as the canary, `0000`) can run any number of times.

To explore without recording, on any tree and with any seeds:

```bash
npm run exp -- 0000 --scratch --seeds 1001
```

Scratch output goes to `artifacts/_scratch/`, which is gitignored, and the ledger is not touched.

If a run crashes, exceeds the card's `budget.maxWallMinutes`, or returns no verdict, nothing is
recorded and the card can be run again. A crash is not a result.

### The ledger

`ledger.jsonl` is append-only, one JSON object per line. It is the only place a card's status
is stored; a card directory never changes after it is merged. Each line names the card, the
card's hash, the commit, the manifest and its hash, the primary value and the verdict.

### Reproducing a result

```bash
npm run repro -- <ledger id>
```

re-runs that card at HEAD and compares its summary CSVs, primary value and verdict with the
original. The outcome is its own manifest under `artifacts/repro/` and a `repro` line in the
ledger. By default the comparison is exact; `--tol <n>` allows a difference per number, measured
as `|a − b| / max(1, |a|, |b|)`.

### What a manifest records

`schemaVersion`, `runId`, the card id, directory and hash, `kind`, the command line, the
environment (`gitCommit`, `gitDirty`, Node version, tfjs version and backend, platform, arch,
lockfile hash, timestamp), `config` and its hash, `seeds`, the decision rule, the primary value,
the verdict, `resultSummary`, a hash of every output file (including the gitignored telemetry),
and wall time. The type is `RunManifest` in `src/experiment/harness.ts`.

### Typecheck

`scripts/` and `cards/` are typechecked with zero errors allowed (`npm run typecheck:harness`).

## Dated scripts are frozen at their run's commit

A `run.ts` in a dated directory is the record of what ran, not maintained code. It is **not**
kept compiling against `HEAD`: `src/` keeps evolving, and editing an old script would make it
disagree with the manifest it produced. Outputs are under `artifacts/<same-name>/`; only
`manifest.json` and `*.summary.csv` are committed (`.gitignore`), and per-step `telemetry.jsonl`
stays local.

To reproduce one, run it at the commit its manifest recorded:

```bash
npm run repro -- artifacts/2026-09-23-g2-loss-curve/seed-1001/manifest.json --at-commit
```

That checks the recorded commit out into a temporary worktree, runs the script there, and
compares the regenerated `*.summary.csv` files with the committed ones. Every recorded commit
is kept reachable: CI fails if a manifest cites a commit that is neither on `main` nor held by
an `evidence/*` tag (`scripts/check-evidence-reachable.mjs`).

Known break at `HEAD` (session audit B1, `reports/quality/2026-09-23-session-audit.md`): the 12
`run.ts` scripts from `2026-08-12-*` through `2026-09-08-*` construct `WorldModel` without
`rewardScale`, which PR #77 made required (`src/model/worldModel.ts`, `WorldModelConfig`). They
fail on their first `runEpisode` step at `HEAD`. Even with `rewardScale` added they would not
reproduce their committed numbers, because the reward and continue loss terms changed training
since those runs. The human's call on PR #77's open item (2026-09-23 session): leave them frozen.

Runs before 2026-09-08 did not seed their weight initialisation, so they are not expected to
reproduce even at their own commit. A `repro` line saying so is a correct record, not a bug.

`tsconfig.json` includes only `src/` and `test/`, so CI's typecheck does not see the dated
directories.
