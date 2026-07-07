# Gap analysis — world models in multi-agent RL

Status: draft (run 4, 2026-07-07). Synthesizes `notes/lit-map.md` (10/10 core papers) into a
structured gap analysis, per `loop/GOAL.md` Phase 1 priority 3. Feeds `docs/proposals/0001-*`.
All claims below are `self_checked` (inherited confidence from the underlying paper notes — see
`notes/papers/<key>.md` for per-claim verification status) unless marked otherwise.

## 1. The design space, revised

The lit map's original single "sharing topology" axis undersells the space. Reading all 10 notes
together (see `notes/lit-map.md` gap-observations log, 2026-07-05/06 entries) surfaces at least
**three separable dimensions**, not one:

1. **World-model sharing topology** — where the *dynamics model itself* sits relative to agents.
   Points found in the map: fully independent (unfilled — the strawman), per-agent + peer comms
   (`codreamer-2024`), per-agent + centralized aggregation (`marie-2024`, `gawm-2025`), per-agent +
   train-time-only bi-level sharing (`mabl-2024`), shared/joint parameters (`matwm-2025`,
   `mmsa-2026`).
2. **Policy-execution scope** — independent of (1). `mmsa-2026` and `matwm-2025` both pair a
   shared/joint *world model* with fully decentralized, local-observation-only *execution* — CTDE
   for the world model does not imply CTDE for the policy, and vice versa. `[medium, self_checked]`
   — surfaced explicitly by MMSA's primary-source pass (`notes/papers/mmsa-2026.md`), not yet
   tested as an independent variable anywhere in the map.
3. **Non-stationarity mitigation mechanism** — orthogonal to both of the above. Four structurally
   distinct mechanisms are documented: architecture-level aggregation (`marie-2024`, `gawm-2025`),
   bi-level train-time-only latent sharing (`mabl-2024`), training-procedure-level recency-weighted
   replay (`matwm-2025`), and latent teammate/ToM inference (`dreaming-of-others-2026`, proposal-
   only, no empirical results). A communication-*content* sub-choice sits inside the peer-comms
   topology point specifically: raw latent state (`codreamer-2024`) vs. a compressed self-intention
   / plan (`plans-not-percepts-2025`) vs. an inferred model of the teammate with no channel at all
   (`dreaming-of-others-2026`).

None of the 10 papers varies more than one of these dimensions at a time while holding the others
fixed on a shared codebase — each introduces one new architecture and compares to *prior papers'*
reported numbers (different codebases, environments, and likely hyperparameter budgets per
`notes/lit-map.md` gap observation, 2026-07-04).

## 2. The core empirical gap

**[low→medium, self_checked, strengthening with each paper added — now n=8 empirical multi-agent
papers]** Every empirical multi-agent paper in the map (`mamba-2022`, `mabl-2024`, `codreamer-2024`,
`marie-2024`, `gawm-2025`, `matwm-2025`, `mmsa-2026`, and implicitly the baselines they compare
against) evaluates non-stationarity's *effects* — final return, sample efficiency — never
non-stationarity *itself*. No paper reports a direct, isolated measurement such as: world-model
prediction error over training, attributable specifically to a co-learning partner's policy shift,
with everything else held fixed. `marie-2024` and `matwm-2025` name non-stationarity explicitly as
a design target; `gawm-2025` frames a related but distinct problem (world-model training
instability, not agent-policy non-stationarity — see its note). None of them measure it directly.

This is the gap `docs/proposals/0001-direct-nonstationarity-measurement.md` targets: a method for
isolating and directly measuring the "moving target" effect, rather than inferring its presence
from downstream return curves.

## 3. The secondary empirical gap

**[low, self_checked]** No controlled, single-codebase sweep exists across sharing-topology points
holding environment, world-model backbone, and compute budget fixed. This is the natural
laptop-scale, JS/TS-native empirical contribution distinct from proposing a new (6th/7th)
architecture — cheap relative to a novel-architecture paper, and directly actionable at L1→L2
without new dependencies beyond what ADR-0002 will settle (RSSM vs. Transformer vs. SSM backbone;
see `notes/papers/drama-2024.md` for the SSM data point).

## 4. Design inputs held below citation-for-quantitative-claims bar

Two 2026-07-06 additions are usable as *qualitative* design inputs only, not as quantitative
reference points, per the standing `self_checked`/`WebFetch`-403 protocol (see `notes/lit-map.md`
tooling caveat) — this resolves PR #4's second open "Decisions needed" item (see today's stand-up):

- `drama-2024` (Drama) — single-agent SSM/Mamba backbone, laptop-scale (7M params) — informs
  ADR-0002's dynamics-backbone question, not the MARL sharing-topology proposal directly.
- `plans-not-percepts-2025` (Communicating Plans, Not Percepts) — informs the comms-content
  sub-axis framing (§1.3) qualitatively; no verified quantitative results exist to cite.

Neither is load-bearing for `0001`'s success/kill criteria below.

## 5. What this unlocks

`docs/proposals/0001-direct-nonstationarity-measurement.md` is the first draft proposal, targeting
gap §2 (direct measurement) with gap §3 (controlled sweep) as its natural methodology. A second
proposal candidate — not drafted yet — would target the policy-execution-scope dimension (§1.2) as
an independent variable, using `mmsa-2026`/`matwm-2025` as the existing-work anchor; flagged in
"Tomorrow" for a future increment rather than drafted now, to keep this increment to one bounded
proposal per `loop/GOAL.md`'s one-increment rule.
