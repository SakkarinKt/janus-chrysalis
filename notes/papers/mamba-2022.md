# MAMBA — Scalable Multi-Agent Model-Based Reinforcement Learning

- **arXiv**: [2205.15023](https://arxiv.org/abs/2205.15023) (v1: May 2022)
- **Venue**: AAMAS 2022 (21st Intl. Conf. on Autonomous Agents and Multiagent Systems), also ACM DL 10.5555/3535850.3535894
- **Authors**: Vladimir Egorov, Alexei Shpilman
- **Code**: [jbr-ai-labs/mamba](https://github.com/jbr-ai-labs/mamba) (official)
- **Verification note**: WebFetch was unavailable in this session (proxy returned 403 on all URLs, including non-arXiv test targets — looks like an environment issue, not arXiv-specific). Claims below are cross-checked across 3+ independent search hits (arXiv listing, ACM DL, official GitHub repo, IFAAMAS proceedings PDF listing) rather than a single full-text read. Tag accordingly: `self_checked`, confidence capped at `medium` for anything below the abstract level.

## Claims

- **[high, self_checked]** MAMBA extends a Dreamer-style (DreamerV2-lineage) recurrent world model to cooperative MARL, replacing Dreamer's Reinforce-style actor update with MAPPO/PPO-style policy updates. Corroborated by GOAL.md's own framing and consistent across all search hits.
- **[high, self_checked]** Centralized training / decentralized execution: agents train a shared or jointly-conditioned world model using centralized information, but each agent carries its own world-model instance for execution, coordinated via communication rather than a single shared global-state world model. This is the "one world model per agent + comms" design that GOAL.md's shared-vs-per-agent axis directly targets.
- **[high, self_checked]** Core sample-efficiency claim: MAMBA reduces environment interactions by up to an order of magnitude vs. model-free MARL baselines on SMAC and Flatland, while matching or beating their asymptotic performance.
- **[medium, self_checked]** MAMBA has become the standard model-based-MARL baseline that later work (MABL, MARIE/MATWM, GAWM, MAG) positions itself against — this is inferred from consistent "improves over MAMBA" framing in multiple independently-surfaced follow-up papers, not from reading MAMBA's own text.
- **[low, self_checked]** Known weakness cited by follow-up work (not verified against MAMBA's own text): local per-agent model prediction errors compound over multi-step imagined rollouts, motivating later corrections (e.g., MAG). Flagged as a claim *about* MAMBA sourced from other papers' framing — needs a direct-source check before it goes in a proposal.

## Method summary (best available from abstract + secondary sources)

World model: per-agent (or per-agent-conditioned) RSSM-style latent dynamics model, Dreamer-lineage, trained centrally on joint trajectories. Policy learning happens entirely in imagined/latent rollouts (no real-env interaction during the "dreaming" phase), with MAPPO-style clipped policy-gradient updates instead of Dreamer's REINFORCE actor. Evaluated on SMAC (StarCraft Multi-Agent Challenge) and Flatland.

## Compute scale

Not directly confirmed from primary text this session (see verification note). SMAC + Flatland at AAMAS-2022-era scale are laptop/single-GPU-plausible per-run budgets (consistent with other MAMBA-era MARL papers), but the exact step counts/wall-clock/hardware were not extracted — **do not cite a specific number without a follow-up primary-source read.**

## Relevance to janus-chrysalis

Direct precedent for the shared-vs-per-agent world-model axis in our mission statement, and the reference point every later paper in this map defines itself against. Should anchor the "baseline" arm of any gap analysis or proposal touching model-based MARL.

## BibTeX

```bibtex
@inproceedings{egorov2022mamba,
  title     = {Scalable Multi-Agent Model-Based Reinforcement Learning},
  author    = {Egorov, Vladimir and Shpilman, Alexei},
  booktitle = {Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2022},
  eprint    = {2205.15023},
  archivePrefix = {arXiv}
}
```
