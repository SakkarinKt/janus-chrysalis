# experiments/

One directory per run, named `YYYY-MM-DD-<slug>/`, holding the script that produced it. Outputs
land under `artifacts/<same-name>/` — only `manifest.json` and `*.summary.csv` are committed
(`.gitignore`); per-step `telemetry.jsonl` stays local.

## Scripts are frozen at their run's commit

A `run.ts` here is the record of what ran, not maintained code. It is **not** kept compiling
against `HEAD`: `src/` keeps evolving, and editing an old script would make it disagree with the
manifest it produced. To reproduce a run, check out the `gitCommit` recorded in its
`artifacts/<run>/**/manifest.json` and run the script there.

Known break at `HEAD` (session audit B1, `reports/quality/2026-09-23-session-audit.md`): the 12
`run.ts` scripts from `2026-08-12-*` through `2026-09-08-*` construct `WorldModel` without
`rewardScale`, which PR #77 made required (`src/model/worldModel.ts`, `WorldModelConfig`). They
fail on their first `runEpisode` step at `HEAD`. Even with `rewardScale` added they would not
reproduce their committed numbers, because the reward and continue loss terms changed training
since those runs. The human's call on PR #77's open item (2026-09-23 session): leave them frozen.

A new script copies the nearest predecessor into a new dated directory and passes
`rewardScale: env.rewardScale` — e.g. `2026-09-23-g2-loss-curve/run.ts`.

`tsconfig.json` includes only `src/` and `test/`, so CI's typecheck does not see this directory.
