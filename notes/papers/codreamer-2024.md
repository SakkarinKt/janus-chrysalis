# CoDreamer — Communication-Based Decentralised World Models

- **arXiv**: [2406.13600](https://arxiv.org/abs/2406.13600) (Jun 2024)
- **Venue**: OpenReview submission (forum id `f2bgGy7Af7`); also presented at CoCoMARL 2024 workshop
- **Authors**: Edan Toledo, Amanda Prorok
- **Verification note**: same WebFetch outage as `mamba-2022.md`. Cross-checked across arXiv abstract listing, OpenReview forum page, and CoCoMARL 2024 accepted-papers list. `self_checked`, confidence capped at `medium` below abstract level.

## Claims

- **[high, self_checked]** CoDreamer extends Dreamer to multi-agent settings using a *two-level* Graph Neural Network (GNN) communication system — one communication channel inside the learned world models, a separate one inside each agent's policy — rather than a single shared/centralized model.
- **[high, self_checked]** Positioning: explicitly framed against a "naive" per-agent independent Dreamer (no communication at all), which the authors argue has strictly less expressive power. CoDreamer is the strongest *fully decentralized-but-communicating* point in the design space we're mapping, as opposed to MAMBA/MABL's more centralized-training framings.
- **[medium, self_checked]** Claimed to outperform baseline methods "across various multi-agent environments" — plural, unspecified which in the abstract; needs primary-source read to know if these overlap with SMAC/Flatland/MAMuJoCo (the MAMBA/MABL benchmarks) or are disjoint, which matters for whether cross-paper comparisons are even meaningful.

## Method summary (best available from abstract + secondary sources)

Dreamer-style RSSM per agent, but each agent's world model receives *communicated* latent information from neighbors via a GNN before/during its own dynamics rollout — i.e., the world model is decentralized (one instance per agent, no single global latent) but not independent (agents exchange messages that shape each other's world-model state). A second, separate GNN communication pass happens in the policy/actor networks. Two clearly separable communication mechanisms is the paper's main architectural claim to novelty over prior multi-agent Dreamer variants.

## Compute scale

Not confirmed from primary text this session — flagged for follow-up read.

## Relevance to janus-chrysalis

This is the cleanest "per-agent + explicit communication" reference point for our shared-vs-per-agent axis — a middle ground between MAMBA (per-agent world model, centralized training signal) and a genuinely independent multi-Dreamer baseline (no coordination at all, presumably worst-case non-stationarity). A useful three-way axis for a gap-analysis table: (1) single shared world model, (2) per-agent + communication (CoDreamer-style), (3) per-agent, fully independent (no communication — the non-stationarity stress case).

## BibTeX

```bibtex
@article{toledo2024codreamer,
  title   = {CoDreamer: Communication-Based Decentralised World Models},
  author  = {Toledo, Edan and Prorok, Amanda},
  year    = {2024},
  eprint  = {2406.13600},
  archivePrefix = {arXiv}
}
```
