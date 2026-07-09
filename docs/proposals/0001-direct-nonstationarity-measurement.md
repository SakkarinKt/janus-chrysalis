# Proposal 0001: Directly measuring co-learning non-stationarity across world-model sharing topologies

- **Status**: draft — full 4-arm sweep design; **L2 promotion approved for a scoped Arm-A
  instrument-validation milestone** (PR #7 review, @SakkarinKt, 2026-07-08 — gate revised two-sided,
  see "L2 promotion request" below). Everything else in this document stays L1 (design-only) until
  that milestone is validated.
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
