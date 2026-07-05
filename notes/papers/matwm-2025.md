# MATWM — Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning

- **arXiv**: [2506.18537](https://arxiv.org/abs/2506.18537) (Jun 2025)
- **Venue**: not confirmed this pass (OpenReview submission surfaced separately as "Action-Conditioned Transformers for Decentralized Multi-Agent World Models" — possibly an earlier or concurrent review-track version of the same work; not confirmed as identical, flagged for follow-up rather than asserted)
- **Authors**: Azad Deihim et al. (full author list not confirmed this pass — ResearchGate profile confirms Deihim as an author; full list needs a primary-source read)
- **Correction context**: this is the paper `marie-2024.md` previously flagged as an earlier session's mis-citation for MARIE itself (arXiv 2406.15836). That correction is re-confirmed this session: 2506.18537 is MATWM, a later paper that explicitly builds on and positions against MARIE.
- **Verification note**: WebFetch still 403s on arXiv this session (persistent, per PR #1 human reply — not re-tested every run). Cross-checked across arXiv abstract, `arxiv.org/html` full text mirror, themoonlight.io review, ResearchGate listing, and an OpenReview forum page for a same/adjacent-titled submission (≥4 independent sources). `self_checked`, confidence capped at `medium` below abstract level.

## Claims

- **[high, self_checked]** MATWM builds directly on STORM (a single-agent Transformer world model), inheriting its architectural choices: transformer-based sequence modeling, KL balance + free bits, percentile return normalization, two-hot symlog reward targets, and discrete latent representations. This is the clearest documented lineage from a *single-agent* world-model architecture into the MARL setting found in the lit map so far — directly relevant to ADR-0002's dynamics-backbone question.
- **[high, self_checked]** Structurally: decentralized imagination (each agent imagines its own rollouts) + a semi-centralized critic + an explicit teammate-prediction module — a fourth distinct point on the shared-vs-per-agent axis, closest to CoDreamer's per-agent-plus-explicit-modeling-of-others but via a learned predictor module rather than GNN message-passing.
- **[high, self_checked]** Directly targets non-stationarity via a *prioritized replay mechanism* that biases world-model training toward recent experience, so the world model tracks teammates' evolving policies rather than fitting a stale mixture-of-policies distribution. This is a training-procedure intervention (data selection), structurally different from MARIE's architecture-level aggregation or MABL's bi-level-latent split — a third distinct *mechanism* for addressing non-stationarity, not just a third architecture.
- **[medium, self_checked]** Teammate-prediction module conditions on the focal agent's own latent-state history to anticipate others' actions — implicit coordination without explicit inter-agent communication or a centralized world state, positioned as a lighter-weight alternative to MARIE's centralized aggregation.
- **[medium, self_checked]** Reported near-optimal performance in as few as 50K environment interactions, evaluated across SMAC, PettingZoo, and MeltingPot — broadest benchmark spread of any paper in the lit map so far; not yet checked against the primary results tables.

## Method summary (best available from abstract + secondary sources)

Per-agent Transformer world models (STORM-derived) generate imagined rollouts independently (decentralized imagination). A semi-centralized critic evaluates joint value from these per-agent rollouts. A lightweight teammate-prediction head, conditioned only on the focal agent's own latent history, predicts other agents' actions — giving each agent an implicit model of its teammates without requiring communication or a shared world state. Non-stationarity is addressed orthogonally to the architecture: a prioritized replay buffer keeps the world model's training distribution weighted toward recent transitions, so it doesn't overfit to an outdated mixture of teammates' past policies.

## Compute scale

Not confirmed from primary text this session. "Near-optimal in as few as 50K environment interactions" is a sample-efficiency claim, not a compute-cost figure — parameter count, wall-clock, and hardware not yet found.

## Relevance to janus-chrysalis

The replay-prioritization mechanism is the first *training-procedure-level* (rather than architecture-level) non-stationarity intervention in the lit map, and it's factorizable: it could in principle be applied on top of any of the other four architectures (MAMBA, MABL, CoDreamer, MARIE, GAWM) as an independent variable. That makes it a strong candidate for the controlled sweep's second axis (architecture × replay-recency-weighting), rather than just one more point on the sharing-topology axis alone. Also the clearest STORM-lineage single-agent-to-MARL adaptation found so far, directly informing ADR-0002 (Transformer-based world model as an alternative dynamics backbone to RSSM, alongside the SSM angle already flagged via Drama, arXiv 2410.08893).

## BibTeX

```bibtex
@article{deihim2025matwm,
  title   = {Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning},
  author  = {Deihim, Azad and others},
  journal = {arXiv preprint arXiv:2506.18537},
  year    = {2025},
  eprint  = {2506.18537},
  archivePrefix = {arXiv},
  note    = {Full author list not yet confirmed from primary source}
}
```
