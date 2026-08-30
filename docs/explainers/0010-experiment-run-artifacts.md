# Explainer: shared experiment run-artifact helpers

`src/experiment/telemetry.ts` — `loop/GOAL.md` priority 5's first sub-increment ("JSONL
telemetry + `manifest.json` per run"), which `PLAN.html` Phase 2 has listed since the phase
started ("a run without a manifest didn't happen. No dashboard yet.") but that had, until this
run, no `src/` implementation at all — every `experiments/*/run.ts` script that writes a
`manifest.json` and a `telemetry.jsonl` reimplements the mechanics inline instead.

## Why this is drift worth naming, not just a missing feature

`git log`/`grep -l "manifest.json" experiments/**/*.ts` (this run) turns up **eleven** scripts
that already do this, going back to `2026-08-12-arm-a-instrument-validation/run.ts`:

- `2026-08-12-arm-a-instrument-validation/run.ts`
- `2026-08-13-paired-init-instrument-validation/run.ts`
- `2026-08-20-post-freeze-action-divergence/run.ts`
- `2026-08-21-agent0-observation-divergence/run.ts`
- `2026-08-22-viewradius-sweep/run.ts`
- `2026-08-23-post-freeze-viewradius-switch/run.ts`
- `2026-08-24-partner-only-viewradius-switch/run.ts`
- `2026-08-25-radius6-seed-spread/run.ts`
- `2026-08-26-radius4-matched-seeds/run.ts`
- `2026-08-26-radius6-more-seeds/run.ts`
- `2026-08-27-high-visibility-preregistered/run.ts`

Each independently: shells out to `git rev-parse HEAD`, reads `process.version` and
`tf.getBackend()`, stamps `new Date().toISOString()`, joins telemetry records with `"\n"` and
writes them, and calls `JSON.stringify(manifest, null, 2) + "\n"`. None of that is
experiment-specific — it's the same four-line block, copy-pasted and re-verified fresh (correctly)
every time a new `run.ts` is written. This is a **recorded-but-undone** drift item under
`loop/GOAL.md` priority 5: the plan named this as shared infrastructure to build; what actually
happened is it got reinvented ad hoc, once per run, for six weeks, and never factored out.

## What this run does and doesn't do about it

**Does**: extracts the identity-and-mechanics shell — `captureRunEnvironment()` (git commit, Node
version, tfjs backend, timestamp), `ensureRunDir()`, `writeTelemetry()`, `writeManifest()` — into
one tested module, so the *next* experiment script has something to import instead of something
to copy.

**Does not**: touch any of the eleven existing scripts. Two reasons, both boundary-driven, not
just caution:

1. **`loop/GOAL.md`'s "Never" list**: "delete or rewrite existing artifacts, reports, or notes
   written by others." These scripts are dated, narrative records of specific past runs (each
   with its own pre-registration reasoning, seed choice, and findings in its header comment) —
   PR #24's review already established this precedent for `experiments/*/summary.json` itself
   ("overwriting it twice... destroyed the evidence behind numbers already cited"), and the same
   logic extends to the scripts that produced that evidence. Refactoring their manifest-writing
   internals to call `writeManifest`/`writeTelemetry` would touch working, already-run,
   already-cited code for a cosmetic dedup with no new behavior — risk with no research payoff.
2. **Scope**: eleven call-site migrations, each needing its own re-verification that the
   refactored script still produces byte-for-byte the same `manifest.json` shape it originally
   did (some of these manifests are cited by number in `docs/proposals/0001`), is not "one bounded
   increment."

Future experiment scripts are the intended callers. This module doesn't impose a manifest
*schema* — every run's manifest fields differ (seed, condition, model config, per-run result
summary, findings) — only the writing convention every existing script already converged on by
hand.

## What each function does

- `captureRunEnvironment()` — the four run-identity fields (`gitCommit`, `nodeVersion`,
  `tfjsBackend`, `createdAt`), one call instead of four inline statements.
- `ensureRunDir(dir)` — `mkdirSync(dir, { recursive: true })`, named for what callers use it for.
- `writeTelemetry(dir, records, fileName?)` — JSONL write, returns the file name written so a
  manifest's `telemetryFile` field can reference it by the same value that was actually written,
  not a separately-typed string literal that could drift from it.
- `writeManifest(dir, manifest, fileName?)` — pretty-printed JSON write (2-space indent, trailing
  newline), matching every existing script's `JSON.stringify(manifest, null, 2) + "\n"`.

## Test coverage

`test/experiment/telemetry.test.ts`, against `node:fs.mkdtempSync` scratch directories (never
under `artifacts/`, cleaned up after each test): `captureRunEnvironment`'s `gitCommit` matches a
direct `git rev-parse HEAD`, `nodeVersion` matches `process.version`, `tfjsBackend` is a non-empty
string, `createdAt` round-trips through `Date`; `ensureRunDir` creates missing nested directories
and doesn't throw when the directory already exists; `writeTelemetry` writes one JSON-encoded
record per line in order, returns the file name it wrote (default and custom); `writeManifest`
writes pretty-printed, trailing-newline JSON that parses back to an equal object (default and
custom file name).

## What's deliberately not here yet

- **No manifest schema/interface.** Every existing script's manifest shape differs by what that
  run measured; imposing one now, from eleven examples that were never designed to a common
  shape, risks forcing a schema that fits none of them well. If a common shape emerges
  organically once 2–3 *new* scripts use this module, that's a future extraction.
  - **A `.gitignore` note this run confirmed still holds**: `artifacts/**` with
    `!artifacts/**/manifest.json` and `!artifacts/**/*.summary.csv` already commits manifests
    while leaving `telemetry.jsonl` uncommitted-but-on-disk, per `loop/GOAL.md` ("commit only
    manifests + summary artifacts") — this module's defaults don't change that; it only writes
    what a caller tells it to, where it tells it to.
- **Migrating existing `run.ts` scripts** — see "Does not," above.
- **A numbering note**: this file is `0010`, skipping `0009` deliberately — PR #58 (open as of
  this run, not yet merged) already claims `docs/explainers/0009-worldmodel-nan-halt.md` on its
  own branch. Filenames don't collide either way (different basenames), so this is a courtesy
  against later renumbering churn, not a correctness requirement — recorded under "Assumptions
  made" in this run's stand-up.
