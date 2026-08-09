# Explainer: the drift-attributable-error metric

`src/experiment/metrics.ts` — `loop/GOAL.md` priority 4's first sub-increment ("Arm-A metric
plumbing"), per PR #41's review (@SakkarinKt, 2026-08-08: "Next: priority 4 — Arm-A metric
plumbing, with PR #39 follow-up (2)'s per-agent Rng folded in at the start, plus the post-keep
test above"). This is the computation proposal `0001`'s "Minimal experiment" section defines as
the primary metric — not yet the 3-seed instrument-validation runs that exercise it end-to-end
(see "What's deliberately not here yet" below).

## What it is

Two functions:

- `postFreezeLossSeries(records, freezeStep, agentIndex)` — pulls one agent's raw per-step
  `worldModelLoss` (`EpisodeStepRecord`, `src/experiment/freeze.ts`) out of a completed episode's
  records, restricted to steps at or after `freezeStep`, in order. This is "the frozen agent's
  world-model prediction error... on newly collected transitions" from proposal `0001`'s
  definition — for exactly one run (one condition, one seed).
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
when the requested agent's `worldModelLoss` is `undefined` at a post-freeze step.
`driftAttributableError`: elementwise diff on a rising-vs-flat pair; an all-zero result when both
series are identical (the flat-control-vs-flat-control sanity case gate (a) cares about); throws
on a length mismatch rather than truncating.
