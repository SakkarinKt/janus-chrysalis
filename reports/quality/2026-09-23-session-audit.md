# Session audit — 2026-09-23

Human-directed interactive session (@SakkarinKt + Claude), run ahead of Gate G2. **Not** a loop
quality pass: the name avoids `loop/GOAL.md` priority 2's self-limiting `*-quality-pass.md` glob on
purpose, so this report does not re-arm or consume the loop's quality slot. Same evidence rules
as `2026-08-03-quality-pass.md`: every finding carries `file:line`, severity
(`BLOCKING`/`NOTE`/`NIT`), and a confidence tag. Findings marked **repro'd** were reproduced in
this session with a command or a failing test. The rest come from a static read and are
`self_checked`.

**Baseline** (`b104ccd`, fresh `npm ci`): `npm test` 177 = 165 pass / 12 todo / 0 fail;
`npm run typecheck:ratchet` 44/44. **After this report's fixes**: see "Fixes applied".

**Method**: three read-only sub-agents (code audit across `src/` `test/` `experiments/` `scripts/`;
Gate G2 evidence inventory; research-direction review), then main-session reproduction of every
finding fixed here. Scope: 2,322 LOC `src/`, 2,534 `test/`, 4,778 `experiments/`, 201 `scripts/`.

## Findings summary

| # | Sev | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| B1 | BLOCKING | 12× `experiments/*/run.ts` (`WorldModel` config literals) | Omit `rewardScale`, required since PR #77 → fail at the first `runEpisode` step at HEAD | **Resolved by decision**: left frozen, documented (`experiments/README.md`) |
| B2 | BLOCKING | `src/agent/lambdaReturns.ts` (`continues` doc), `docs/explainers/0012`, `src/model/worldModel.ts` (`continueTargetTensor`) | Spec tells callers to pass `done ? 0 : 1`, but `done` is always a time-limit truncation → severs bootstrapping (PR #43's bias) | **Spec half fixed**; model half carried to Phase 3 |
| N1 | NOTE | `src/model/worldModel.ts` `dispose()` | Disposed only recurrent state; every weight + all Adam state leaked | **Fixed** (repro'd: 41 tensors leaked → 0) |
| N2 | NOTE → promote? | `src/env/rng.ts` `deriveSeed` | XOR mix is commutative/self-inverse under nesting → agents 0/1 share GRU-kernel init seeds | **Fixed** (repro'd) — see promotion note |
| N3 | NOTE | `src/env/gridworld.ts:33,57`, `src/experiment/freeze.ts:130` | Env spawn RNG and policy action RNG both `new Rng(seed)` → first ~8 draws identical | Open (low impact) |
| N4 | NOTE | `src/experiment/freeze.ts:134-175`, `src/agent/policy.ts:15-24` | Structural blockers for an actor-critic (update-before-WM-step ordering, reset obs never reaches WM, `Policy` has no latent-state input, shared action RNG, freeze gate bypassable by imagination) | Open → Phase 3 design input |
| N5 | NOTE | `docs/explainers/0013` | Stale "no reward head"; separation test omits `rewardHead` | Open — **owned by open PR #81**, not touched here |
| N6 | NOTE | `test/model/losses.test.ts` free-bits test | Batch-size-1 only; `max(fb, mean(KL))` regression would pass | **Fixed** (mutation-checked) |
| N7 | NOTE | `test/experiment/metrics.test.ts` fixture | `klLoss: 0`, `rewardLoss: 0` → dropping KL / adding reward passes | **Fixed** (mutation-checked) |
| N8 | NOTE | `test/model/worldModel.test.ts:160,187,254`; `test/experiment/freeze.test.ts:198-209` | "Loss decreases" tests use unseeded weight init | Open |
| N9 | NOTE | `src/model/worldModel.ts` NaN check; `docs/explainers/0009` | Checks forward values only, after `applyGradients` (non-finite grads corrupt silently); `rewardScale` never validated (0 → NaN); 0009 still describes a three-field check | Open |
| N10 | NOTE (low conf.) | `src/model/losses.ts:17` | `DEFAULT_BETA_DYN = 1.0`: DreamerV3's own config (as recalled, unverified) uses 0.5; 0004 should name the revision it follows | Open |
| N11 | NOTE | `docs/explainers/0007:107-118` | Stated reason for excluding `continueLoss` contradicts freeze semantics (agent 0 frozen in both arms); "freeze-37" vs `FREEZE_STEP = 38` | Open |
| N12 | NOTE | drift metric | `klLoss` is floored at ≈1.1 in every experiment → post-freeze KL change below 1 nat is invisible | Open → instrument v2 input |
| N13 | NOTE | `src/model/losses.ts` `rewardLoss`; `src/env/gridworld.ts` `rewardScale` getter | Reward term effectively weighted 1/16; getter's "reproduces exactly" bound (−4.0) is unreachable (worst achievable −3.875) | Open |
| N14 | NOTE | `src/experiment/telemetry.ts:32-77`; `src/model/rssm.ts` `sampleStraightThrough` | Dead code: telemetry helpers only called by their own test (2026-09-08 re-implements them inline) | Open |
| N15 | NOTE | `test/env/gridworld.test.ts:94-111` | Collision test never checks `reward`; nothing pins `rewardScale` or `|reward| ≤ rewardScale` | Open (subsumes 08-03 #7) |
| N16 | NOTE | `src/agent/lambdaReturns.ts`, `test/agent/lambdaReturns.test.ts:31` | Indexing consistent (no off-by-one), but `values[0]` unused and unpinned; λ=1 test name says "Monte-Carlo" while the recursion keeps a bootstrap tail; empty input / ranges unspecified | **Addressed** in `0012` errata; tests land with Gate G2 |

**BLOCKING slots**: 2 of 3 used (B1, B2). **Promotion suggestion for the human**: N2. It
silently correlates the two "independent" per-agent models' initial weights, which is exactly the
contrast Phase 3's Arm A vs Arm D comparison needs clean. It's fixed here, so promoting it only
changes how the record reads.

## Code findings — detail

### B1 — frozen experiments vs. required `rewardScale` [BLOCKING · self_checked · high]

`src/model/worldModel.ts` `WorldModelConfig.rewardScale: number` (required since PR #77,
`ebeade5`); the constructor stores it unvalidated. All 12 scripts `2026-08-12-*` … `2026-09-08-*`
build configs like `{ rssm: RSSM_CONFIG, observationSize, seed: deriveSeed(seed, agentIndex) }`.
Repro: `for f in experiments/*/run.ts; do grep -c rewardScale $f; done` → `0` for all 12.
Construction succeeds; the first `step()` reaches `rewardLoss` → `tf.scalar(undefined)`. CI
cannot see it (`tsconfig.json` `include: ["src","test"]`). The human's decision (this session):
leave the scripts frozen at their run commit, documented in `experiments/README.md`. A re-run
at HEAD would not reproduce the committed manifests anyway, since the reward and continue terms
changed training. Closes the "Decisions needed" item carried on PR #77.

### B2 — truncation treated as termination [BLOCKING · self_checked · high]

`src/env/gridworld.ts:81` `const done = this.currentStep >= this.config.horizon;` means `done` is
always a time-limit truncation. `src/agent/policy.ts:125-129` already encodes the consequence
("always bootstrap", PR #43 review). But:

- `src/agent/lambdaReturns.ts` (`continues` doc) and `0012`'s λ-returns section told callers to
  pass `done ? 0 : 1`, and `test/agent/lambdaReturns.test.ts:35`'s todo tied the severing
  property to that convention. An actor-critic following the spec would stop bootstrapping at
  every episode's final step.
- `src/model/worldModel.ts` `continueTargetTensor = tf.tensor2d([[done ? 0 : 1]])` trains the
  continue head to predict the horizon. Without time in the observation that target isn't a
  Markov function of state, and in imagination it would inject the same truncation bias.

**Fixed (spec half)**: the `continues` doc comment and a dated errata section in `0012` now say
`1` for truncation and `0` only for a true terminal. **Carried (model half)**: separating
terminal from truncated in `StepResult` and the continue target needs its own explainer. It's a
Phase-3 prerequisite, required before any actor-critic bootstraps through imagination.

### N1 — `WorldModel.dispose()` leak [NOTE · repro'd]

The pre-fix `dispose()` freed only `state.deterministic`/`state.stochastic`. Repro: the new test
`WorldModel: dispose() releases every tensor…` run against the pre-fix source reported
`got 41 leaked`; post-fix it reports 0. Fix: dispose every `trainableVars` entry plus
`this.optimizer`.

### N2 — nested `deriveSeed` cancels [NOTE · repro'd]

`(seed ^ Math.imul(0x9e3779b9, salt + 1)) >>> 0` is commutative and self-inverse under
composition. Repro (scratch script over the real call chain, seed 1001): agent 0's RSSM seed is
the raw seed (`true`). Agent 0 kernels `2654435920,1013903515,…` and agent 1 kernels
`1013903515,2654435920,…` swap, so agent 0's GRU input kernel (glorot) shares a seed with agent
1's recurrent kernel (orthogonal), and the reverse. Agent 1's decoder seed is the raw seed. Fix:
MurmurHash3 `fmix32` over `seed + golden·(salt+1)`. Two new tests (order-sensitivity/no-collapse;
no duplicates in the full 3-seed × 2-agent seed tree) both fail on the old XOR and pass now.
Full suite still green, so no existing test depended on a specific seeded outcome. Every seeded
stream changes from this commit on. Old manifests stay reproducible at their recorded `gitCommit`.

### N6 / N7 — tests that could not fail [NOTE · mutation-checked]

- N6: new batch-2 test with one row below and one above the floor. Mutating `losses.ts` to
  `tf.maximum(tf.mean(dynKL), freeBits)` → fails with "expected per-row floor 1.728…, got
  1.228…". Restored.
- N7: fixture `rewardLoss` is now a non-zero offset, plus a dedicated `0.5 + 0.25 = 0.75` test.
  Mutating `metrics.ts` to drop `klLoss` makes it fail. Restored.

### N3–N5, N8–N15 — logged, not fixed

Detail as in the summary table. N4's five structural points are the design input for whoever
wires an actor-critic into `runEpisode` (Phase 3). N12 is the design input for instrument v2's
metric.

### NITs (logged)

- **Stale line refs**:
  - `src/agent/lambdaReturns.ts` (fixed here), `0012:141`, and the test name at
    `test/model/worldModel.test.ts:210` cite `worldModel.ts:184` for the continue target. It moved
    to `:228`, then to `:238` after this session's `dispose()` docstring. Prefer symbol references
    (`continueTargetTensor`) over line numbers in long-lived docs.
  - `0011:12,96,123-125`, `0014:52,198,224` (PR #81's file), `0007:111`.
- **`rssm.ts` tensor-contract text**: `:86` "every non-training call site wraps its call in
  `tf.tidy`" is false for `test/model/rssm.test.ts:33-170`.
- **Tautological or weak tests**:
  - `test/env/rng.test.ts:47-48` built a fresh `Rng` per draw (fixed here).
  - `rng.test.ts:37-40`, `policy.test.ts:18-25,44-51`, `rssm.test.ts:172-185`,
    `rssm.test.ts:145-149` (incidental `TypeError`, no message check), `freeze.test.ts:84,156-161`,
    `losses.test.ts:127-133,176-186`.
- **Unseeded randomness**: `experiments/2026-07-21-week3-stack-spike/benchmark.ts:129`
  `tf.randomNormal` (historical). No `Math.random` anywhere in the repo.
- **`package.json` `test` glob**: `node --test test/**/*.test.ts` under `sh` has no globstar, so it
  matches exactly one directory level. Fine today, but a deeper test file would be silently
  skipped.
- **Silent no-ops**: `isFrozen` with an out-of-range agent index freezes nobody; a `freezeStep`
  outside `[1, horizon]` never fires.

## Process and drift findings

| # | Finding | Evidence | Proposed reconciliation |
| --- | --- | --- | --- |
| P1 | Loop exhausted its authorized work | `grep -l exhaust reports/standup/*.md` → 09-06, 09-09…12, 09-16…18, 09-20…22 | Gate G2 → Phase 3 gives it a new priority list (`loop/GOAL.md` v3) |
| P2 | Stand-up bloat | Sept stand-ups 775–1,336 words; quiet days re-narrate the prior day's re-verification | Stand-up v3 (mid-phase amendment 2026-09-23): ≤400 words, quiet-day form ≤150, evidence folded |
| P3 | Run-number drift | 2026-09-22 "run 1" vs PR #81's 2026-09-23 "run 32" | v3 defines `run N` = count of `reports/standup/*.md` on `main` + 1 |
| P4 | Explain-before-implement merge condition never met | `LEARNING.md`: 0 human entries; 14 explainers (`CONTRIBUTING.md:12`) | LEARNING split + one-bullet Learning nudge per stand-up + twice-monthly retro |
| P5 | Claude's weekly journal stopped after 07-04 | `LEARNING.md` outside loop write paths | `loop/LEARNING.md` (inside `loop/`, already writable) |
| P6 | Role flips have no issue trail | `user-implements` label exists; 0 issues in the repo; no issue template | Two issues opened for the G2 modules this session |
| P7 | Reserved list incomplete | `loop/GOAL.md` reserves replay buffer + λ-returns only; the human reserved `Actor`/`Critic` in the PR #80 review | Added in the 2026-09-23 amendment |
| P8 | CI `test` job not a required check | `.github/workflows/ci.yml:5-8` header comment | Repo-settings change, human action (unchanged since 2026-07-16) |
| P9 | Schedule | Phase 2 planned weeks 3–7; today ≈ week 12 | Recorded as a deviation in the Gate G2 record |
| P10 | Arm-A instrument gate never passed | 09-08 diffMean −0.0938/+0.0046/+0.0000; control slopes all positive; 08-27 pre-registered replication failed | Research-direction decision at Phase 3 promotion (not a code finding) |

## Carry-over from `2026-08-03-quality-pass.md`

- **#3** (no `tf.tidy` contract documented on `RSSMCell`): actually fixed by PR #39 (`fb1cbf6`,
  `src/model/rssm.ts:82-95`). Stand-ups through 2026-09-21 still list it as open. Close it, and
  note the inaccurate sentence at `:86` (NIT above).
- **#4** (implicit `any`, `benchmark.ts`): code unchanged, line moved 261 → 269. Unverified whether
  `tsc` still flags it (outside `tsconfig`).
- **#7** (always-true assert, `gridworld.test.ts`): still present (`:110`). Subsumed by N15.

## Fixes applied (this session)

| Finding | Files | Verification |
| --- | --- | --- |
| N1 | `src/model/worldModel.ts`, `test/model/worldModel.test.ts` | new test fails pre-fix (41 leaked), passes post-fix |
| N2 (+ `rng.test.ts:47-48` NIT) | `src/env/rng.ts`, `test/env/rng.test.ts` | 2 new tests fail pre-fix, pass post-fix; full suite green |
| N6 | `test/model/losses.test.ts` | mutation → fails; restored → passes |
| N7 | `test/experiment/metrics.test.ts` | mutation → fails; restored → passes |
| B2 (spec) | `src/agent/lambdaReturns.ts`, `docs/explainers/0012` | doc-only; `markdownlint-cli2` 0 issues |
| B1 (decision) | `experiments/README.md` | doc-only |

After fixes: `npm test` 182 = 170 pass / 12 todo / 0 fail (+5 tests: N1 ×1, N2 ×2, N6 ×1, N7 ×1);
`npm run typecheck:ratchet` 44/44 (baseline unchanged).
