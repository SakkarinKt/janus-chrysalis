# Dreaming of Others — Latent Teammate Modeling in World Models for Multi-Agent Reinforcement Learning

- **arXiv**: [2605.31361](https://arxiv.org/abs/2605.31361) (May 2026)
- **Venue**: Poster, 2026 World Modeling Workshop — an explicitly 5-page conceptual/workshop paper (2 figures), not a full conference paper.
- **Authors**: Tomas Leroy-Stone (single author; no institutional affiliation listed on arXiv)
- **Status note**: Held out of the lit map pending a primary-source check (human's PR #1 reply: "add it only after a primary-source check; cautious epistemics over coverage") — its single-author listing and very recent date raised hallucination/mis-citation risk under search-only verification. **Primary-source pass done 2026-07-05** (human interactive session, WebFetch direct against arXiv) confirms the paper is real and its content matches the loop's earlier search-based description. Added now per the human's own stated criterion.
- **Verification note**: All claims below read directly from the arXiv HTML full text (not secondary sources) — tagged `verified`. `self_checked` co-authorship protocol still applies (single-reviewer read, not independently cross-checked by a second pass).

## Claims

- **[high, verified]** **This is a position/proposal paper, not an empirical one.** The authors state directly: "This paper is a proposal intended to catalyze discussion in the world-models community; we do not report empirical results." No experiments, results tables, or benchmark numbers exist — it's an architecture + training-objective design plus a *proposed* (not run) evaluation protocol. This is a materially different status than every other paper currently in the lit map, all of which report actual SMAC/PettingZoo/MeltingPot/etc. results.
- **[high, verified]** RSSM latent is factorized as z_t = [z_t^env, z_t^team]: the observation decoder reconstructs from z_t^env only, while a separate teammate-policy decoder maps z_t^team to a predicted distribution over the partner's next action. A Theory-of-Mind (ToM) head infers teammate latents from partial behavioral histories. During imagination, actor and critic condition on both the hidden state and the teammate latents.
- **[high, verified]** Explicitly requires no shared imagination, no centralized policy, and no explicit inter-agent communication — positioned as a fourth mechanism (alongside MARIE's architecture-level aggregation, MABL's bi-level latent split, and MATWM's replay prioritization) for non-stationarity mitigation, this one via *latent teammate identification* rather than architecture pooling or data selection.
- **[high, verified]** Non-stationarity framing: "As teammates update their policies, the effective environment observed by each agent changes." The paper's thesis is that modeling teammates as structured, learnable latent processes — rather than treating their behavioral variation as noise — reduces this non-stationarity and enables rapid adaptation to new partners. Closest in spirit to MARIE/MATWM's agent-policy-non-stationarity framing (as opposed to GAWM's world-model-training-instability framing).
- **[medium, verified]** Training objective: a calibrated cross-entropy loss supervises the teammate-policy decoder, plus a KL regularization term for temporal consistency of the inferred teammate latent — a concrete, checkable design choice, though untested (no ablations exist since there are no experiments).
- **[medium, verified]** Proposes (but does not run) evaluation on Multi-Agent Particle Environments, Overcooked-AI (for zero-shot coordination), and Melting Pot (for robustness across social contexts), using JaxMARL/BenchMARL tooling — a candidate environment/tooling reference for janus-chrysalis's own planned sweep, independent of this paper's own (nonexistent) results.

## Method summary (from primary text)

Each agent's Dreamer-style RSSM latent state is split into two components: an environment component (used to reconstruct/predict observations and rewards, same as a standard world model) and a teammate component (used to predict a partner's next-action distribution via a dedicated decoder). A Theory-of-Mind head infers the teammate latent from partial behavioral histories — it watches what a partner has done and infers a latent summary of their "character, intent, and predicted actions." The actor and critic, during imagination, condition on both the environment hidden state and the inferred teammate latents, letting the agent imagine rollouts against a model of its specific partner rather than an undifferentiated environment. No agent-to-agent communication, shared world model, or centralized policy is required — everything needed about teammates is inferred from observed behavior alone.

## Compute scale

Not applicable — no experiments are run in this paper (proposal/position paper only).

## Relevance to janus-chrysalis

Gives the non-stationarity/opponent-modeling seed area its most direct hit yet: a fourth structurally distinct mitigation mechanism (latent teammate identification via a ToM head) alongside MARIE's aggregation, MABL's bi-level split, and MATWM's replay prioritization — and the clearest match to the mission's own framing of "non-stationarity from co-learning agents." But **its lack of any empirical validation means it can't anchor the planned controlled sweep the way the other papers can** — it's a design idea to potentially implement and test as part of the sweep (a candidate 4th non-stationarity mechanism to combine with the sharing-topology axis), not a baseline with reported numbers to compare against. Treat as inspiration/design input for the gap analysis, not as an empirical reference point.

## BibTeX

```bibtex
@inproceedings{leroystone2026dreaming,
  title     = {Dreaming of Others: Latent Teammate Modeling in World Models for Multi-Agent Reinforcement Learning},
  author    = {Leroy-Stone, Tomas},
  booktitle = {2026 World Modeling Workshop},
  year      = {2026},
  eprint    = {2605.31361},
  archivePrefix = {arXiv},
  note      = {Poster; conceptual/proposal paper, no empirical results}
}
```
