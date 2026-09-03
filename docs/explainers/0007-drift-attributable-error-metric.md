# Explainer: the drift-attributable-error metric

`src/experiment/metrics.ts` — `loop/GOAL.md` priority 4's first sub-increment ("Arm-A metric
plumbing"), per PR #41's review (@SakkarinKt, 2026-08-08: "Next: priority 4 — Arm-A metric
plumbing, with PR #39 follow-up (2)'s per-agent Rng folded in at the start, plus the post-keep
test above"). This is the computation proposal `0001`'s "Minimal experiment" section defines as
the primary metric — not yet the 3-seed instrument-validation runs that exercise it end-to-end
(see "What's deliberately not here yet" below).

## What it is

Two functions:

- `postFreezeLossSeries(records, freezeStep, agentIndex)` — pulls one agent's per-step
  `reconstructionLoss + klLoss` (`EpisodeStepRecord.worldModelLossBreakdown`,
  `src/experiment/freeze.ts`) out of a completed episode's records, restricted to steps at or
  after `freezeStep`, in order. This is "the frozen agent's world-model prediction error... on
  newly collected transitions" from proposal `0001`'s definition — for exactly one run (one
  condition, one seed). See the addendum below for why this reads the recon+KL breakdown rather
  than the summed `worldModelLoss` total.
- `driftAttributableError(interventionLosses, controlLosses)` — the elementwise difference between
  two such series, one from an `"intervention"`-condition episode and one from a `"control"`-
  condition episode (both `src/experiment/freeze.ts`'s `FreezeConfig`). This is proposal `0001`'s
  primary metric itself: "(frozen-agent one-step prediction error under the freeze intervention)
  minus (frozen-agent prediction error under the both-frozen control)."

## Why aligned by steps-since-freeze, not absolute env step

`intervention` and `control` are two separate `runEpisode` calls — ordinarily different seeds
(the milestone's 3-seed design runs each condition multiple times independently), so their
trajectories diverge from the freeze point onward regardless: the partner agent's actions differ
between conditions (still training vs. also frozen), which changes the environment's evolution,
which changes the frozen agent's own observations even though its *policy* is identical and
frozen in both. There is no meaningful way to pair "env step 47 of the intervention run" with "env
step 47 of the control run" as the *same* transition — they aren't. What proposal `0001`'s
"tracked over post-freeze steps" phrasing does support is comparing the two runs' error curves by
how far post-freeze they are: "the first post-freeze prediction," "the second," and so on. That's
what `driftAttributableError` aligns on — array index, i.e. `postFreezeLossSeries`'s output
position, not the `EpisodeStepRecord.step` value.

A consequence: `driftAttributableError` requires its two inputs to be the same length and throws
otherwise, rather than truncating to the shorter one. A length mismatch means the two runs used a
different horizon or `freezeStep` — silently truncating would compare, say, the intervention run's
steps-since-freeze 0-9 against the control run's 0-9 even if the control run's horizon actually
only produced steps-since-freeze 0-6, which is not a bug in the metric, but truncating instead of
throwing would hide that the caller passed mismatched runs.

## Relationship to the milestone's two-sided gate

Proposal `0001`'s L2 promotion request gate (revised per PR #7 review, two-sided):

- (a) **Both-frozen control stays flat** — the control run's own `postFreezeLossSeries` should not
  show a rising trend beyond a seed-variance tolerance band.
- (b) **Freeze intervention shows a detectable rising signal** — the intervention run's
  `postFreezeLossSeries` should rise measurably above that same tolerance band.

`driftAttributableError` is not itself the gate check — it's the diff that would make a rising
signal visually obvious on a plotted curve (a flat-vs-flat pair gives an all-zero or near-zero
diff; a rising intervention against a flat control gives a visibly rising diff), but the gate as
proposal `0001` states it is actually phrased over each condition's *own* series and a
variance-derived tolerance band, not over the diff series directly. Computing that tolerance band
needs multiple control-condition seeds' variance — a 3-seed-run concern, not something a single
pair of series can produce. This module deliberately stops at "compute the two series, compute
their diff" and leaves the gate's statistical check (tolerance-band estimation, trend test) to
whatever consumes the 3-seed runs' output, since that check can't be written or tested
meaningfully against fabricated single-run data — it needs the actual seed-to-seed variance this
milestone hasn't produced yet.

## What's deliberately not here yet

The 3-seed freeze-vs-both-frozen validation runs themselves (`loop/GOAL.md` priority 4's second
half) — no `experiments/0001/...` scaffold, no manifest-producing training run, no seed-variance
tolerance-band computation or gate-pass/fail check. Blocking prerequisite, not addressed by this
sub-increment: **there is no trainable policy in this repo yet.** `src/agent/policy.ts`'s only
implementation, `RandomPolicy`, has no `update()` — it cannot learn, so a partner agent running it
never actually drifts, and the milestone's whole premise ("prediction error should show this error
*rising* as the still-training partner's policy drifts away from what the frozen agent's world
model was fit to," proposal `0001`) has nothing to produce a signal from. This is a real
prerequisite gap, not a design choice within this sub-increment's scope to resolve unilaterally —
flagged as a "Decisions needed" item in this run's stand-up report rather than addressed here.

## Test coverage

`test/experiment/metrics.test.ts`: `postFreezeLossSeries` extracts the right slice in order for
several `freezeStep`/`agentIndex` combinations (including `freezeStep` equal to the records' first
step, i.e. the whole series); throws when `freezeStep` never occurs in the given records; throws
when the requested agent's `worldModelLossBreakdown` is `undefined` at a post-freeze step.
`driftAttributableError`: elementwise diff on a rising-vs-flat pair; an all-zero result when both
series are identical (the flat-control-vs-flat-control sanity case gate (a) cares about); throws
on a length mismatch rather than truncating.

## Addendum (2026-09-03): excluding `continueLoss` from the instrument's series

`docs/explainers/0011-continue-termination-head.md` landed `WorldModel.step()`'s continue head
(PR #63, merged 2026-09-02) after this metric already existed — its "what's deliberately not here
yet" section named "feeding `drift-attributable-error`" as explicitly out of scope, but didn't flag
that the continue head's loss term had already folded itself in by construction: `WorldModelStepResult.loss`
(and therefore the old `EpisodeStepRecord.worldModelLoss` this module originally read) is
`reconstructionLoss + klLoss + continueLoss`, unweighted, so once the continue head existed,
`postFreezeLossSeries` started including it without any code in this file changing. PR #63's review
(@SakkarinKt, 2026-09-02) caught this before any new experiment run collected data on the mixed
number (the 114 manifests already committed all predate the continue head and are unaffected):

> `freeze.ts:148` records only `.loss`, so `worldModelLoss` — and therefore `postFreezeLossSeries`
> and `driftAttributableError` — now measures recon + KL + continue, where all 114 committed
> manifests measured recon + KL. Pairing doesn't cancel it: post-freeze the frozen arm's continue
> term is static while control's keeps training, and the target flips to 0 once inside the 38-step
> window.

Concretely: proposal `0001`'s Arm-A gridworld's `done` target is `currentStep >= horizon`
(`src/env/gridworld.ts:62,65` — see `docs/explainers/0011`), so once a rollout is inside its final
38 post-freeze steps (the milestone's horizon-75/freeze-37 design), *every* remaining step has
`done: false` until the last one, i.e. the continue head's target is a near-constant `1` for the
whole post-freeze window regardless of condition. `continueLoss` isn't therefore identical across
control and intervention at a given steps-since-freeze offset: the frozen arm's continue head stops
training at freeze (per `isFrozen`/`policy.update()` gating — the world model's `train` flag follows
the same `frozen[i]`, `src/experiment/freeze.ts`), so its loss on that near-constant target drifts
away from the still-training arm's, purely as an artifact of which arm keeps optimizing — not a
signal proposal `0001` is trying to measure at all. Adding that artifact into the series a rising
`driftAttributableError` is supposed to attribute to non-stationarity would contaminate the
instrument with a second, unrelated source of drift.

**Fix**: `EpisodeStepRecord` gains `worldModelLossBreakdown` (the per-agent
`reconstructionLoss`/`klLoss`/`continueLoss` triple, `src/experiment/freeze.ts`), and
`postFreezeLossSeries` now sums `reconstructionLoss + klLoss` from that breakdown instead of
reading the summed `worldModelLoss` total — restoring the recon+KL-only series the 114 existing
manifests were measured on. `worldModelLoss` (the total) is left in place on `EpisodeStepRecord`
for other consumers (e.g. the bit-identical determinism checks in several `experiments/*/run.ts`
scripts) since removing it would be an unrelated, unrequested change; it is simply no longer what
this metric reads. `driftAttributableError` itself needed no change — it only ever consumes
`postFreezeLossSeries`'s output, not `EpisodeStepRecord` directly.

**Still open, not addressed by this fix**: whether the continue head's loss should be surfaced as
*its own* tracked series (a genuine "does the frozen agent's termination-boundary prediction drift
too" question) is a new-metric question, not this fix's — `docs/explainers/0011`'s "not part of
this vertical slice" stance on using the continue head's output stands.
