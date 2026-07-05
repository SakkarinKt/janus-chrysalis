# MATWM — Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning

- **arXiv**: [2506.18537](https://arxiv.org/abs/2506.18537) (Jun 2025)
- **Venue**: not confirmed (arXiv:2506.18537, submitted 2025-06-23, cs.LG — no venue/proceedings listed in the arXiv record itself). OpenReview submission surfaced separately as "Action-Conditioned Transformers for Decentralized Multi-Agent World Models" — possibly an earlier or concurrent review-track version of the same work; not confirmed as identical, still flagged for follow-up rather than asserted (not re-checked this pass).
- **Authors**: **Azad Deihim, Eduardo Alonso, Dimitra Apostolopoulou** (City St George's, University of London & Oxford Institute for Energy Studies) — confirmed from primary source, resolving the earlier "et al." uncertainty.
- **Correction context**: this is the paper `marie-2024.md` previously flagged as an earlier session's mis-citation for MARIE itself (arXiv 2406.15836). That correction is re-confirmed this session: 2506.18537 is MATWM, a later paper that explicitly builds on and positions against MARIE.
- **Verification note**: Primary-source pass done 2026-07-05 in a human interactive session (WebFetch worked directly against `arxiv.org/abs/2506.18537` and the `arxiv.org/html` mirror). Claims below promoted to `verified` where confirmed; one claim (the "per-agent" framing of the world model) is corrected below — the primary text describes a **shared** world model, not independent per-agent models.

## Claims

- **[high, verified]** MATWM builds directly on STORM (a single-agent Transformer world model), inheriting its architectural choices: transformer-based sequence modeling with discrete latents via a Categorical-VAE, KL balance + free bits, percentile return normalization, and two-hot symlog reward targets. This is the clearest documented lineage from a *single-agent* world-model architecture into the MARL setting found in the lit map so far — directly relevant to ADR-0002's dynamics-backbone question.
- **[high, verified — corrected]** Structurally: a **shared** world model (not independent per-agent models as the previous pass's secondary sources implied) computes only the focal agent's trajectory from its own observations — so imagination is decentralized *in execution* (each agent runs the shared model on its own local view) but the model's parameters are shared/pooled across agents, not per-agent. Plus a semi-centralized critic (focal agent's local state + predicted teammate actions, no direct access to others' info during imagination) and an explicit teammate-prediction module. This still lands as its own point on the shared-vs-per-agent axis, but "parameter-shared, decentralized execution" is a materially different topology than "fully independent per-agent world models" — worth being precise about when this gets placed on the lit-map's axis table.
- **[high, verified]** Directly targets non-stationarity via a *prioritized replay mechanism* that biases world-model training toward recent experience, so the world model tracks teammates' evolving policies rather than fitting a stale mixture-of-policies distribution — confirmed in the primary abstract ("incorporates prioritized replay to handle non-stationarity by training on recent experiences"). This is a training-procedure intervention (data selection), structurally different from MARIE's architecture-level aggregation or MABL's bi-level-latent split — a third distinct *mechanism* for addressing non-stationarity, not just a third architecture.
- **[high, verified]** Teammate-prediction module conditions on the focal agent's own latent-state history to anticipate others' actions ("agents do not interact with one another directly; instead, they interact with imagined versions of other agents whose behavior is captured ... by the teammate predictor") — implicit coordination without explicit inter-agent communication or a centralized world state, positioned as a lighter-weight alternative to MARIE's centralized aggregation.
- **[high, verified]** Reported near-optimal performance in as few as 50K environment interactions, evaluated across SMAC (12 maps), PettingZoo Butterfly, and MeltingPot — broadest benchmark spread of any paper in the lit map so far, and the paper claims to be the first multi-agent world model supporting image-based observations.
- **[medium, verified]** Ablations confirm teammate prediction, prioritized experience replay, and action scaling each contribute meaningfully to performance (per the paper's own ablation study — specific numbers not extracted this pass).

## Method summary (best available from abstract + secondary sources)

A shared-parameter Transformer world model (STORM-derived) computes each agent's imagined rollout from that agent's own observations — decentralized in execution, shared in parameters. A semi-centralized critic evaluates joint value using the focal agent's local state plus predicted teammate actions. A lightweight teammate-prediction head, conditioned only on the focal agent's own latent history, predicts other agents' actions — giving each agent an implicit model of its teammates without requiring communication or a shared world state. Non-stationarity is addressed orthogonally to the architecture: a prioritized replay buffer keeps the world model's training distribution weighted toward recent transitions, so it doesn't overfit to an outdated mixture of teammates' past policies.

## Compute scale

Not confirmed from primary text this pass either. "Near-optimal in as few as 50K environment interactions" is a sample-efficiency claim, not a compute-cost figure — parameter count, wall-clock, and hardware still not found.

## Relevance to janus-chrysalis

The replay-prioritization mechanism is the first *training-procedure-level* (rather than architecture-level) non-stationarity intervention in the lit map, and it's factorizable: it could in principle be applied on top of any of the other four architectures (MAMBA, MABL, CoDreamer, MARIE, GAWM) as an independent variable. That makes it a strong candidate for the controlled sweep's second axis (architecture × replay-recency-weighting), rather than just one more point on the sharing-topology axis alone. Also the clearest STORM-lineage single-agent-to-MARL adaptation found so far, directly informing ADR-0002 (Transformer-based world model as an alternative dynamics backbone to RSSM, alongside the SSM angle already flagged via Drama, arXiv 2410.08893).

## BibTeX

```bibtex
@article{deihim2025matwm,
  title   = {Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning},
  author  = {Deihim, Azad and Alonso, Eduardo and Apostolopoulou, Dimitra},
  journal = {arXiv preprint arXiv:2506.18537},
  year    = {2025},
  eprint  = {2506.18537},
  archivePrefix = {arXiv}
}
```
