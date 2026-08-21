# Proposal 0001: Directly measuring co-learning non-stationarity across world-model sharing topologies

- **Status**: **selected-primary** — Gate G1
  (`docs/adr/0003-gate-g1-research-question-selection.md`, 2026-07-20); backup = this proposal's
  comms-content pivot (see kill criteria). History: draft, full 4-arm sweep design, with **L2
  promotion approved for a scoped Arm-A instrument-validation milestone** (PR #7 review,
  @SakkarinKt, 2026-07-08 — gate revised two-sided, see "L2 promotion request" below). Everything
  beyond the Phase-2 vertical-slice scope now defined in `loop/GOAL.md` stays gated (design-only).
- **Author**: Claude (loop run 4, 2026-07-07; revised run 6, 2026-07-08 per PR #5 review; revised
  run 7, 2026-07-09 per PR #7 review) · **Reviewed by**: @SakkarinKt (PR #5, 2026-07-07 — approved
  the core idea, requested the two text fixes plus this scoped re-proposal; PR #7, 2026-07-08 —
  approved the scoped re-proposal with a two-sided gate revision)

## Hypothesis

The component of world-model prediction error attributable specifically to a co-learning partner's
policy drift (as opposed to ordinary early-training model inaccuracy) can be isolated with a
controlled freeze intervention, and its magnitude differs systematically across world-model
sharing topologies — independent per-agent world models accrue significantly more
drift-attributable error than topologies with explicit sharing or aggregation, holding
environment, world-model backbone, and compute budget fixed.

## Why novel

Per `docs/proposals/gap-analysis.md` §2: every empirical paper in `notes/lit-map.md` (n=8,
multi-agent) evaluates non-stationarity's downstream effects (return, sample efficiency) but none
directly measures it. `marie-2024` and `matwm-2025` name co-learning non-stationarity as a design
target (see their notes' claims), but both report only aggregate performance, not an isolated
non-stationarity signal. `gawm-2025`'s "world-model training instability" is a related but distinct
framing (input-distribution mismatch, not agent-policy drift — see its note). No paper runs the
same world-model backbone across a topology sweep on one fixed codebase/environment
(gap-analysis §3) — each compares against prior papers' numbers under different codebases and
budgets.

## Minimal experiment (laptop-scale)

**Environment**: a small custom 2-agent cooperative grid-world implemented in TS (simplified
predator-prey / cooperative-navigation, in the spirit of MPE's `simple_spread`/`simple_tag` but
reimplemented natively — no PettingZoo/JAX dependency, keeping the stack JS/TS-only per the
mission). Discrete actions, partial observability (local view window), episode length ~50-100
steps. Small enough for CPU-only or single-consumer-GPU laptop training.

**World-model backbone**: fixed across all arms (RSSM-style, pending ADR-0002 — using whatever
backbone that ADR settles on keeps this proposal decoupled from the backbone question). Only the
**sharing topology** varies between arms:

- **Arm A — independent** (per-agent world model, no sharing, no comms). The unfilled "fully
  independent" axis point (`notes/lit-map.md` axis point 6) — the non-stationarity stress case.
- **Arm B — peer comms** (`codreamer-2024`-style: per-agent world model, raw latent shared via a
  lightweight message-passing step).
- **Arm C — centralized aggregation** (`marie-2024`/`gawm-2025`-style, simplified: per-agent world
  models plus a shared aggregation bottleneck).
- **Arm D — shared/joint** (`matwm-2025`/`mmsa-2026`-style: single parameter-shared world model,
  decentralized execution).

  **Reviewer note (PR #7, @SakkarinKt, 2026-07-08 — deferred, does not gate the Arm-A milestone):**
  the cross-arm metric definition below is under-controlled as written for Arm D. Arms A–C measure a
  static model's error under input drift only; Arm D's shared weights *also* change from the
  partner's continued updates, so its number conflates two effects — "sharing reduced
  non-stationarity" and "the partner's updates happened to help/hurt this stream." Fix for the future
  Arms B–D proposal: add an Arm-D sub-condition with the shared weights frozen at the freeze point
  (only the partner's data-collection policy drifts, not the shared parameters) — the live-vs-frozen
  difference then cleanly attributes the parameter-coupling effect. Recorded here for whoever drafts
  the B–D follow-up; Arm A has no shared model to exercise this, so it does not block the current
  milestone.

**Training budget per run**: target ≤200K environment steps per arm per seed — small enough for a
laptop-CPU overnight run at this environment's scale (rough estimate; to be calibrated once the
environment and backbone are implemented, since no compute-scale figures could be confirmed from
any paper in the map — see each note's "Compute scale" section).

**The measurement intervention** (the actual novel contribution, distinct from the 4 architecture
arms): partway through training, freeze one agent's **policy and world model together** — both
stop updating — while the partner agent continues training its policy (and, in the independent
arms, its own separate world model). Freezing the policy alone is not sufficient: if the measured
agent's world model kept training, it would adapt to the partner's drifting behavior and the signal
we're trying to isolate would be absorbed into that adaptation rather than showing up as prediction
error. Track the frozen agent's world-model one-step (and n-step) prediction error on newly
collected transitions over the following steps. A genuine co-learning non-stationarity signal
should show this error *rising* as the still-training partner's policy drifts away from what the
frozen agent's world model was fit to — versus a control condition where **both** agents (policy
and world model) are frozen, so no partner drift can occur (prediction error should stay flat,
isolating ordinary evaluation noise from drift-attributable error).

## Ablations

1. **Freeze-intervention on vs. off** — confirms the measurement isolates co-learning drift rather
   than just general model staleness (the control condition above).
2. **Sharing topology (Arms A–D)** — the primary comparison; tests whether topology changes the
   magnitude/rate of drift-attributable error.
3. **MATWM-style prioritized recency-weighted replay, layered on top of each arm** — tests whether
   this training-procedure-level mechanism (flagged in `notes/papers/matwm-2025.md` as
   architecture-agnostic and factorizable) reduces drift-attributable error independent of
   topology, as the lit map's gap observations (2026-07-06) speculate it should.

## Metrics

- **Primary**: drift-attributable world-model prediction error — (frozen-agent one-step prediction
  error under the freeze intervention) minus (frozen-agent prediction error under the both-frozen
  control), tracked over post-freeze steps. Logged per arm × seed; artifacts under
  `experiments/0001/<arm>/<seed>/metrics.json` (path is a plan, not yet created — no experiment
  code exists at L1).

  **Cross-arm definition, made explicit** (this metric must mean the same thing in every arm, or
  the topology comparison is invalid): "frozen-agent one-step prediction error" is always the
  prediction error of **the world model responsible for predicting the frozen agent's local
  observation stream**, evaluated only on transitions collected *after* the freeze point, using the
  frozen agent's own trajectory. In Arms A–C each agent has its own world-model instance, so this is
  unambiguous — it is that instance's error. **Arm D has a single parameter-shared world model
  serving both agents**, so "the frozen agent's world model" means: the shared model's prediction
  error evaluated specifically on the frozen agent's observation stream (its inputs/targets),
  holding the model's parameters fixed at the freeze point exactly as in the other arms. The shared
  model's parameters still drift in Arm D only insofar as the *other* (still-training) agent's
  experience continues to update the shared weights — which is precisely the topology difference
  under test (a frozen per-agent model in Arms A–C cannot drift at all post-freeze from either
  agent's updates, whereas Arm D's nominally-frozen evaluation stream can still be perturbed by the
  partner's continued updates to the shared weights). This asymmetry is the intended comparison, not
  a confound to be normalized away — Arm D's whole hypothesis is that shared parameters change this
  dynamic, so the metric must be left free to reflect that rather than artificially frozen in a way
  that would erase the effect being measured.
- **Secondary**: standard return/sample-efficiency curves per arm, for comparability with the
  existing literature's evaluation convention.

## Success criteria

A statistically significant separation (across ≥5 seeds per arm) in the primary metric's
post-freeze growth rate between Arm A (independent) and at least one of Arms B–D, with an effect
size large enough to be visually distinguishable on a plotted curve, not just a p-value — matching
the mission's "findings first" bar. For ablation 3: replay-reweighting reduces Arm A's
drift-attributable error by a meaningful fraction (target: ≥25%, to be refined once baseline
variance is known) without requiring a topology change.

## Kill criteria

- If the freeze-intervention's both-frozen control shows prediction error drifting anyway (e.g.
  from stochastic environment dynamics or model capacity noise), the measurement methodology itself
  doesn't isolate the intended signal — stop and redesign the intervention before running the full
  topology sweep.
- If post-freeze error growth is statistically indistinguishable across all 4 topology arms (no
  detectable effect of sharing), switch to the backup question: comms-*content* as the independent
  variable (raw latent state vs. compressed self-intention, per `plans-not-percepts-2025` and
  `codreamer-2024` — see gap-analysis §1.3), holding topology fixed at Arm B instead.
  **This pivot is the designated G1 backup question** (ADR-0003, 2026-07-20).

## Estimated cost

4 topology arms × 2 replay conditions (ablation 3) × ≥5 seeds = 40 runs, ≤200K steps each. At a
rough laptop-CPU budget this targets ≤40 overnight-runs total (one run per arm×replay×seed
combination, most completing well under a full night at this environment's scale) — to be
recalibrated once the environment and backbone exist and an actual wall-clock-per-step figure is
measured, since no paper in the lit map yielded a confirmed compute-scale number to calibrate
against in advance (see `notes/lit-map.md`, every paper's "Compute scale" section).

## L2 promotion request — Arm-A instrument-validation milestone (scoped)

Per PR #5 review (@SakkarinKt, 2026-07-07): the full 4-arm sweep is **not** being proposed for L2
yet. This section proposes a narrower, first milestone — **approval required, not yet acted on**.

**Scope of the request** — build and run only:

1. The grid-world environment (as specified above).
2. **One** world-model backbone, fixed for the milestone. Blocking dependency: ADR-0002 needs to be
   settled at least to the point of naming a single backbone choice (see the new `notes/js-ml-stack.md`
   research below, which advances but does not close that question — the ADR itself still needs
   human signoff, per `loop/GOAL.md` boundaries).
3. **Arm A only** (fully independent per-agent world models, no sharing) — the topology point with
   no cross-agent machinery to build, so it is the cheapest arm that can validate the measurement
   instrument.
4. The freeze intervention and both-frozen control, as redefined above (policy **and** world model
   both frozen).

**Explicitly out of scope for this milestone**: Arms B–D, ablation 3 (replay-reweighting), and the
full ≥5-seed/40-run sweep. Those stay design-only (L1) until this milestone validates the
instrument.

**Gate** (revised per PR #7 review, @SakkarinKt, 2026-07-08 — two-sided, both required):

(a) **Both-frozen control stays flat** — prediction error under the both-frozen control condition
does not rise beyond a tolerance band set from seed variance (kill criterion #1, unchanged).

(b) **Freeze intervention shows a detectable rising signal on Arm A** — prediction error under the
freeze intervention (partner still training) rises measurably above the both-frozen control's
baseline/tolerance band.

Both must pass. (a) alone proves the instrument has no false positives; it does not prove the
instrument can *detect* drift at all — under the old single-sided gate, a flat control plus a flat
intervention would also pass, while providing zero evidence the metric is sensitive to the effect it
exists to measure. That would silently greenlight Arms B–D on a powerless metric.

**Milestone seed count**: 3 seeds (both-frozen control + freeze intervention, Arm A only) — enough
to estimate seed variance for (a)'s tolerance band and confirm (b)'s effect isn't a single-seed
fluke, without paying the full sweep's cost. Distinct from the full 4-arm sweep's ≥5-seed target
("Success criteria" above, unchanged).

A pass on both (a) and (b) validates the instrument and justifies extending to Arms B–D as a
follow-up L2 proposal. A fail on either means the measurement methodology needs to be redesigned
before any topology comparison is meaningful — that redesign happens *before* Arms B–D are
attempted, not in parallel with them.

**What "L2" means here in practice**: writing and running experiment code for the first time on
this project (environment + one backbone + Arm A + the freeze mechanism), under whatever compute
and safety constraints the human attaches to the promotion. No code exists yet — this section is a
request to start writing it, not a claim that it has been written.

**Status of this request**: **approved**, gate revised two-sided per above (PR #7 review,
@SakkarinKt, 2026-07-08). The backbone dependency (ADR-0002 §6) is separately approved to build
against as a *fixture*, gated on a short RSSM-vs-SSM/Mamba implementation-robustness note landing
before the world-model cell itself is written — see `notes/adr-0002-js-ml-stack.md` §7. As of this
revision the loop has still taken no L2 action (no experiment code, no dependency additions, no
training runs) — see the run's stand-up report for two open questions (dependency-addition approval
scope; whether code work starts before or after `loop/GOAL.md`'s status text is updated) before that
starts.

**2026-07-14 update**: environment (PR #10), freeze mechanism (PR #12/#13) landed since this
revision. The world-model-cell gate itself is now `self_checked`-cleared —
`notes/rssm-vs-ssm-implementation-robustness.md` recommends proceeding with RSSM from TF.js
primitives; see `notes/adr-0002-js-ml-stack.md` §7's 2026-07-14 addendum. Metric plumbing and the
cell itself remain unwritten.

**2026-07-15 update**: PR #14 review (@SakkarinKt, 2026-07-14) approved starting the world-model
cell, split struct/forward-pass first, then STE + gradient-check. Prerequisite landed first (its own
PR per `loop/GOAL.md`'s dependency carve-out, not bundled with the cell): `@tensorflow/tfjs-node`
pinned to `4.22.0` now in `package.json`, smoke-tested — see `notes/adr-0002-js-ml-stack.md` §7's
2026-07-15 addendum. The cell's struct/forward-pass sub-increment itself is next, still unwritten.

**2026-07-17 update**: the cell's struct/forward-pass sub-increment landed —
`src/model/rssm.ts`'s `RSSMCell`, the GRU-based deterministic recurrent state update only (see
`docs/explainers/0003-rssm-world-model-cell.md`). No stochastic latent, no straight-through
estimator, no gradient-check test, no wiring into `src/experiment/freeze.ts`'s rollout, and no
metric computation yet — all still the next sub-increment(s).

**2026-07-18 update**: no new cell code this run — processed PR #17's review reply instead (per
`loop/GOAL.md` priority 1). The reviewer reported `@tensorflow/tfjs-node@4.22.0` (the pinned
dependency version) fails to install on Apple Silicon (404 on the prebuilt `darwin-arm64` binary,
no working source fallback); documented in `notes/adr-0002-js-ml-stack.md` §3/§7. Doesn't block
progress in this project's Linux x64 sandbox, but does block running this milestone's code on the
human's own machine if it's Apple Silicon — see the 2026-07-18 stand-up's "Decisions needed."

**2026-07-19 update**: no new cell code this run either — processed PR #18's review reply (per
`loop/GOAL.md` priority 1), which named a concrete next step for the install failure: probe
`tfjs-node` versions for one with a working darwin prebuilt. Ran that probe; the answer is that no
such version exists at this repo's Node/napi tier (darwin support was dropped from the whole
project after a 2021-era release and never restored) — see `notes/adr-0002-js-ml-stack.md` §3/§7's
2026-07-19 entries for the full evidence. Doesn't change this sandbox's progress, but narrows the
Apple Silicon path to two options needing a human decision — see the 2026-07-19 stand-up's
"Decisions needed."

**2026-07-21 update**: week-3 stack-validation spike (`loop/GOAL.md` priority 2, ADR-0002 decision
5). Extended the RSSM gradient-check to a full training step; found that differentiating
`RSSMCell.step()` chained with `.prior()` across ≥2 timesteps crashes `tf.variableGrads` — a
long-standing upstream tfjs-layers bug (`tensorflow/tfjs#1529`, `#3550`), not fixable from this
proposal's side, and one that affects exactly the recurrence this proposal's Arm-A world model
needs (see `notes/adr-0002-js-ml-stack.md` §8 for the full root-cause). Steps/sec at Arm-A dims
(h=256, z=32, batch 16) is otherwise healthy — ≈372/s forward rollout, ≈101/s single-step gradient
— so the ≤200K-steps/arm/seed budget is fine on raw throughput; the open question is whether
multi-step BPTT training can happen on this stack at all. Raised as a "Decisions needed" item in
the 2026-07-21 stand-up rather than resolved here, per `loop/GOAL.md`'s hard-kill-criterion
handling (no unilateral custom-autograd fallback).

**2026-07-22 update**: resolved, favorably. Processing the reviewer's reply on that "Decisions
needed" item — try calling the GRU cell's step function directly, bypassing the `tf.layers.rnn`
wrapper, before considering the custom-autograd fallback — the bypass works cleanly: no crash,
finite-difference gradients match across every chain length tried (2-4 steps). `RSSMCell.step()`
now uses this approach in production. The week-3 kill criterion did not fire; multi-step BPTT
training is unblocked on this stack with no new gradient math. Full root-cause and fix details in
`notes/adr-0002-js-ml-stack.md` §9. Multi-step BPTT steps/sec (as opposed to the single-step number
above) is still unmeasured — a natural next increment now that it's actually computable.

**2026-07-23 update**: measured, then corrected same day (PR #24 review). `benchmark.ts` now sweeps
truncated-BPTT chain lengths {2, 4, 8, 16, 32} at Arm-A dims. The first version of this update
converted the result straight to "environment-steps/sec" by multiplying chain length × gradient-
steps/sec — wrong, since it drops the batch dimension (`BATCH = 16`) entirely. What's actually
measured is `modelTimestepsPerSec` (chain length × batch × gradient-steps/sec): ~1,700–1,900/sec
this run, roughly flat across chain lengths as expected (compute is ~linear in chain length). **On
raw stack throughput, the kill criterion's steps/sec half doesn't fire** — that's not a slow stack.
But converting to this proposal's own ≤200K-env-steps-per-arm-per-seed budget needs a **replay
ratio**, not fixed anywhere yet: at replay ratio 16 it's ≈30 min/arm/seed (comfortably overnight);
at replay ratio 512 it's ≈16 h (right at the edge of "overnight"). Both are plausible training
configurations. So: stack throughput is usable: confirmed. Whether it clears *this proposal's*
overnight-run budget specifically: contingent on a replay ratio this proposal doesn't fix yet —
worth pinning down before the real Arm-A training runs, not before now. Full numbers, the
correction, and caveats (container-to-container variance, synthetic-throughput-only scope — no real
env stepping, replay sampling, or real losses yet) in `notes/adr-0002-js-ml-stack.md` §10. Medium
confidence, not high: 30 timed iterations per chain length is an order-of-magnitude read, and the
replay buffer (human's G2 module) and real losses (priority 3) aren't wired in yet, so this bounds
RSSM forward/backward cost specifically, not the eventual full training loop's throughput.

**2026-07-24 correction (same PR, later same-day comment):** the paragraph above's "silently
assumed every batch row is a distinct, never-replayed environment step" framing was itself backwards
— dropping `BATCH` from the formula is arithmetically `modelTimestepsPerSec ÷ BATCH`, which is what
environment-steps/sec equals at replay ratio = `BATCH` (16), not ratio 1. So the pre-correction
figure was a conservative (~16×-too-low) read of raw stack throughput, coincidentally valid at
replay ratio 16 specifically — not an optimistic ratio-1 assumption. The replay-ratio bracket table
itself (≈30 min at ratio 16, ≈16 h at ratio 512) was independently verified against the raw data and
stands unchanged; only the diagnosis of *why* the first version's formula was wrong needed fixing.
Full correction in `notes/adr-0002-js-ml-stack.md` §10.

**2026-07-25 update**: per PR #25's review ("root-cause [the flaky multi-step gradient test] before
priority 3 touches RSSM code"), root-caused it. **Not a bug, and not blocking** — an initial same-day
diagnosis on PR #26 concluded otherwise (a real `tf.customGrad` gradient-correctness bug re-tripping
ADR-0002 decision 5); that was wrong, caught in review, and corrected in the same PR before merge.
The actual mechanism: `straightThroughEstimator`'s forward pass is defined to return exactly the
fixed `hard` sample regardless of any weight, so once its output (`z_{t-1}`) feeds back into the next
timestep, that path is numerically invariant to the weights being differentiated — while the STE's
backward pass still injects its intentional proxy gradient (the softmax Jacobian) through it.
Finite-differencing the chained forward pass can't see that injected gradient by construction — the
"mismatch" (growing with chain length, non-vanishing across an epsilon sweep) is exactly the STE's
by-design behavior, not a computation error. Confirmed directly (bumping a weight by `0.5` leaves the
fed-back sample bit-for-bit unchanged; a construction with the same loss *value* but a detached,
non-differentiable constant in place of the STE sample matches finite differences cleanly at every
chain length). Full evidence: `notes/adr-0002-js-ml-stack.md` §11, reproducible via
`experiments/2026-07-25-ste-chained-finite-difference-trap/repro.ts`.

§9's 2026-07-22 "kill criterion did not fire" read stands — **ADR-0002 decision 5 stays closed,
priority 3 is not blocked**. `test/model/rssm.test.ts`'s multi-step test now checks the GRU cell's
gradient using the detached-constant construction (a genuine correctness check that passes because
the computation is correct), rather than either the original flaky check or the briefly-shipped
bug-pin. No "Decisions needed" item from this.

**2026-08-12 update**: `loop/GOAL.md` priority 4's second half — the 3-seed freeze-vs-both-frozen
instrument-validation runs (metric plumbing itself landed PR #42; `QLearningPolicy` landed
PR #43/#44, per PR #44's review, @SakkarinKt, 2026-08-12: "Next: priority 4 — wire it into the
freeze-vs-both-frozen episode pairing and land the 3-seed instrument-validation runs"). Runner:
`experiments/2026-08-12-arm-a-instrument-validation/run.ts`. Per seed, one `"control"`-condition
episode and one `"intervention"`-condition episode (`FreezeConfig`, `src/experiment/freeze.ts`),
each a single 75-step episode (`DEFAULT_CONFIG.horizon`), `freezeStep: 38`, `frozenAgentIndex: 0`,
Arm-A dims (`{deterministicSize: 256, latentCategoricals: 8, latentClasses: 4}`, same placeholder
as `benchmark.ts`'s `ARM_A_CONFIG`), both agents driven by `QLearningPolicy` (default config) with
one independent `WorldModel` each. Manifests + `results.summary.csv` committed under
`artifacts/2026-08-12-arm-a-instrument-validation/` (per-seed `telemetry.jsonl` written but not
committed, per `loop/GOAL.md` boundaries and `.gitignore`'s existing `artifacts/**` rule).

**Result: inconclusive, not a clean pass on either half of the gate** (`self_checked, medium
confidence`) — reported honestly rather than called a pass:

| seed | control slope | intervention slope | diffMean (intervention − control) |
| --- | --- | --- | --- |
| 1001 | +0.0086 | +0.0096 | −0.3022 |
| 1002 | +0.0354 | +0.0301 | +0.1066 |
| 1003 | −0.0071 | −0.0026 | +0.1839 |

(`slope` = OLS slope of `postFreezeLossSeries` against steps-since-freeze index; `diffMean` = mean
of `driftAttributableError`, both from `src/experiment/metrics.ts`, computed over each run's 38
post-freeze steps.) Gate (a) (control stays flat): control's slope sign flips across seeds and its
magnitude is comparable to the intervention slope's — not a clean "flat," though n=3 makes "flat
vs. noisy" hard to distinguish either way. Gate (b) (intervention rises measurably above control):
`diffMean` is mixed-sign across seeds (negative for seed 1001, positive for 1002/1003) — no
consistent rising signal.

Leading hypothesis, not confirmed (`self_checked, medium confidence`): **37 pre-freeze steps is
very little training** for a from-zero tabular `QLearningPolicy` whose state key is the (near-)
continuous raw observation vector (`src/agent/policy.ts`'s `qRow`) — most of those 37 steps likely
hit previously-unseen keys rather than updating a revisited one, so the still-training partner
agent has barely started to drift from near-random behavior by the freeze point, leaving little
co-learning drift for the post-freeze window to detect. This would make today's result an
underpowered pilot, not evidence the measurement mechanism itself is broken — but that's a
hypothesis about *why*, not a confirmed diagnosis; distinguishing it from "the instrument doesn't
work" needs a follow-up run at a longer pre-freeze training horizon (many episodes of training
before the freeze point, not steps 1-37 of one episode) before this milestone's gate can be
evaluated for real. Not proposed as this run's fix — sizing that follow-up (episode count, whether
`freezeStep` needs to move to a multi-episode training-then-freeze structure at all) is next
increment's design call, not decided unilaterally here.

**Second finding, orthogonal to the above** (`self_checked, high confidence`): `RSSMCell`'s
`tf.layers.gruCell`/`tf.layers.dense` and `ObservationDecoder`'s `tf.layers.dense` are constructed
with no `kernelInitializer` seed (`src/model/rssm.ts:95-98`, `src/model/decoder.ts:26`) — two
fresh `WorldModel`s with identical config draw different initial weights from tfjs's own unseeded
global initializer, confirmed directly by this run (`checkWeightInitDeterminism` in `run.ts`,
recorded in every manifest's `findings` field) and independently by re-running the script twice at
the same 3 seeds and getting different numbers both times (the table above is one of those two
runs). This means `runEpisode`'s `seed` does not make a `WorldModel`-involving run fully
bit-reproducible today — only the policy-action and world-model-*sampling* streams are seeded
(finding #1 from the 2026-08-03 quality pass, fixed 2026-08-04), not layer weight *initialization*.
Not fixed in this run: the fix touches `RSSMCell`/`ObservationDecoder`'s public construction
surface (threading a seed into every `tf.layers.*` call), which is `src/model/` API-surface work
for whoever next touches that surface, not an experiment-script fix. Flagged here and in this run's
stand-up report rather than raised as "Decisions needed" — it doesn't block interpreting today's
result (the inconclusive read above doesn't depend on run-to-run reproducibility), but it does mean
any future run of this exact script should not be expected to reproduce today's exact numbers.

Full detail, both findings, and the raw per-seed manifests: `artifacts/2026-08-12-arm-a-instrument-validation/`.

**2026-08-13 update**: processing PR #45's review (@SakkarinKt, 2026-08-12: "gate (b) isn't
readable until init is seeded or paired across conditions... seed or pair world-model init before
spending compute on a longer pre-freeze horizon"). Fixed the weight-init-determinism gap the
2026-08-12 update flagged: `RSSMConfig.seed`/`DecoderConfig.seed`/`WorldModelConfig.seed`
(`src/model/rssm.ts`, `src/model/decoder.ts`, `src/model/worldModel.ts`) derive per-layer seeds
via `deriveSeed` and pass them to every `tf.layers.*` call's `kernelInitializer`/
`recurrentInitializer` (bias stays the tfjs default `zeros`, already deterministic). New tests:
`test/model/rssm.test.ts`, `test/model/decoder.test.ts`, `test/model/worldModel.test.ts` (same
seed → identical initial weights; different seeds → different; omitted seed → unchanged unseeded
fallback). `npm test`: 100/100 (95 prior + 5 new). `npm run typecheck:ratchet`: 44/44, unchanged.

Re-ran the identical 3-seed Arm-A validation —
`experiments/2026-08-13-paired-init-instrument-validation/run.ts`, same `SEEDS`, `freezeStep: 38`,
`frozenAgentIndex: 0`, Arm-A dims — but now passing the trial's `seed` into both conditions'
`WorldModel`s (`WorldModelConfig.seed`), pairing control and intervention on identical initial
weights per agent. This turned out to pair the *entire* pre-freeze phase, not just initialization:
`runEpisode`'s `policyRng`/`worldModelRngs` are already `seed`-derived and `freezeConfig` never
gates which action gets chosen or which RNG value gets drawn, only whether `update()`/training
runs — so once init was the only remaining source of divergence, fixing it made control's and
intervention's pre-freeze `EpisodeStepRecord`s (actions, rewards, observations, world-model
losses) bit-identical for a given seed. Confirmed directly, not assumed: this run's
`assertPreFreezeParity` diagnostic checks every pre-freeze record field-by-field and is `true` for
all 3 seeds (recorded in each manifest's `findings[0].preFreezeParityCheck`, `identical: true`).

| seed | control slope | intervention slope | diffMean (intervention − control) |
| --- | --- | --- | --- |
| 1001 | +0.006150 | +0.007996 | −0.0936 |
| 1002 | +0.070358 | +0.069888 | +0.0036 |
| 1003 | −0.001425 | −0.001425 | +0.0000 |

**Result: still not a clean gate (b) pass, but the picture changed** (`self_checked, medium-high
confidence`). `diffMean` magnitudes collapsed from 2026-08-12's `[−0.3022, +0.1066, +0.1839]` to
`[−0.0936, +0.0036, +0.0000]` — roughly in line with @SakkarinKt's PR #45 review estimate that a
control-vs-control pairing under the old unseeded init would show noise of "the same magnitude as
the reported −0.30/+0.11/+0.18," i.e. most of 2026-08-12's signal *was* init noise, not a
freeze effect, as the review predicted. What's left after removing that noise is small and still
mixed-sign, not confirming a freeze effect either. Seed 1003's `diffMean` is exactly `0.0000`,
not just small — intervention and control produced bit-identical `postFreezeLossSeries` for that
seed, meaning agent 1's post-freeze actions never diverged between "kept training" and "frozen"
for that particular trajectory (self_checked, high confidence, read directly off the equal
telemetry). This is consistent with, and firmer evidence for, the 2026-08-12 hypothesis this
update doesn't newly test: 37 pre-freeze steps may be too few for `QLearningPolicy`'s Q-table to
reach states post-freeze where continuing to train vs. not would actually pick a different action —
if post-freeze exploration keeps landing on already-converged or already-explored keys, "frozen"
and "training" are behaviorally the same regardless of how clean the measurement is. Distinguishing
that from "the mechanism doesn't work at all" still needs the longer-pre-freeze-horizon follow-up
the 2026-08-12 update proposed (not attempted in this run — one increment).

**2026-08-20 update**: processing PR #46's review (@SakkarinKt, 2026-08-13 merge comment, follow-up
1 of 2 — "Record a per-seed count of post-freeze steps where the two conditions' actions differ,
next to `preFreezeParityCheck`: that's the instrument's power metric... land the post-freeze
divergence counter first, so the longer horizon arrives with a power read"). Added
`postFreezeActionDivergenceCount` (`src/experiment/metrics.ts`): per seed, counts post-freeze
steps (aligned by steps-since-freeze index, same convention as `driftAttributableError`) where
control's and intervention's joint actions differ. 4 new tests
(`test/experiment/metrics.test.ts`). `npm test`: 104/104 (100 prior + 4 new). `npm run
typecheck:ratchet`: in-repo count unchanged at 44, exit 0.

Re-ran the identical 3-seed paired-init validation —
`experiments/2026-08-20-post-freeze-action-divergence/run.ts`, same `SEEDS`/`FREEZE_STEP`/
`FROZEN_AGENT_INDEX`/`HORIZON`/configs as 2026-08-13's run — adding only this new measurement.
`diffMean` reproduced 2026-08-13's numbers bit-for-bit (`[−0.0936, +0.0036, +0.0000]`), confirming
the paired setup is still fully deterministic on this commit; `assertPreFreezeParity` again
`identical: true` for all 3 seeds.

| seed | diffMean | post-freeze steps with differing actions |
| --- | --- | --- |
| 1001 | −0.0936 | 22 / 38 |
| 1002 | +0.0036 | 22 / 38 |
| 1003 | +0.0000 | 23 / 38 |

**Result: the power read changes the story** (`self_checked, high confidence` on the counts
themselves — they're a direct tally over committed telemetry, not inferred; `medium confidence` on
the interpretation below). Seed 1003's exactly-zero `diffMean` is **not** because the two
conditions' post-freeze actions never diverged — the earlier 2026-08-13 standup's "agent 1's
actions never diverged... for that trajectory" was a mistaken inference from the loss series alone
(flagged at the time as read directly off equal telemetry, but "equal loss" does not imply "equal
actions," and this run shows it didn't hold). All three seeds show substantial post-freeze action
divergence — 58–61% of steps — which rules out this proposal's leading hypothesis from the
2026-08-12/2026-08-13 updates ("37 pre-freeze steps is too little training, so post-freeze
exploration keeps landing on already-converged/unseen keys and the two conditions rarely if ever
pick different actions"): they picked different actions often. The instrument has power at this
horizon; the frozen agent's world-model *loss* just isn't tracking that divergence into a
consistent, rising, cross-seed signal. That reopens rather than closes the question the earlier
hypothesis was trying to explain, and is a "Decisions needed" item in today's stand-up rather than
something resolved here — worth the human's read before deciding what the multi-episode
training-then-freeze follow-up (PR #46's "Next") should actually be designed to test now that
"underpowered horizon" looks less likely than the earlier updates assumed.

Not attempted in this run (PR #46's follow-up 2, explicitly "neither blocking"): the manifest
`gitCommit` provenance-pointer gap (points to the commit this run started from, which won't
contain this run's own `run.ts` — a pre-existing limitation of writing the manifest before the
run's own commit exists, not introduced or worsened here).

Full detail, all findings, and the raw per-seed manifests + telemetry:
`artifacts/2026-08-20-post-freeze-action-divergence/`.

Full detail and the raw per-seed manifests: `artifacts/2026-08-13-paired-init-instrument-validation/`.
