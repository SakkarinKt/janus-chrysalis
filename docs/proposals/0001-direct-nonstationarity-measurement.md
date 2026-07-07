# Proposal 0001: Directly measuring co-learning non-stationarity across world-model sharing topologies

- **Status**: draft
- **Author**: Claude (loop run 4, 2026-07-07) · **Reviewed by**: (pending human review)

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

**Training budget per run**: target ≤200K environment steps per arm per seed — small enough for a
laptop-CPU overnight run at this environment's scale (rough estimate; to be calibrated once the
environment and backbone are implemented, since no compute-scale figures could be confirmed from
any paper in the map — see each note's "Compute scale" section).

**The measurement intervention** (the actual novel contribution, distinct from the 4 architecture
arms): partway through training, freeze one agent's policy while the other continues training.
Track the frozen agent's world-model one-step (and n-step) prediction error on newly collected
transitions over the following steps. A genuine co-learning non-stationarity signal should show
this error *rising* as the still-training partner's policy drifts away from what the frozen agent's
world model was fit to — versus a control condition where both agents are frozen (prediction error
should stay flat, isolating ordinary evaluation noise from drift-attributable error).

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
