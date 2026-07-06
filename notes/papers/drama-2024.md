# Drama — Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient

- **arXiv**: [2410.08893](https://arxiv.org/abs/2410.08893) (Oct 2024)
- **Venue**: ICLR 2025 (poster) — confirmed via `proceedings.iclr.cc` and `iclr.cc/virtual/2025/poster/30818`, not just a preprint.
- **Authors**: Wenlong Wang, Ivana Dusparic, Yucheng Shi, Ke Zhang, Vinny Cahill
- **Scope note**: **Single-agent**, not multi-agent. Added per `loop/GOAL.md` priority 4 (JS ML stack / ADR-0002 notes) and the "not yet followed up" list, not per the sharing-topology axis below — it doesn't get an axis point.
- **Verification note**: `self_checked`, not `verified`. `WebFetch` returned HTTP 403 for both `arxiv.org/abs/2410.08893` and `arxiv.org/html/2410.08893` this session (same persistent autonomous-loop constraint as prior runs — see `notes/lit-map.md` tooling caveat). Claims below are cross-checked across ≥3 independent secondary sources (ICLR proceedings/slides, OpenReview, emergentmind, aimodels.fyi, liner.com, paperswithcode) that agree on the specific numbers cited, but none are read from primary PDF/HTML text directly. Confidence capped at `medium`.

## Claims

- **[high, self_checked]** Drama replaces the Transformer backbone used by contemporary world models (e.g. STORM, IRIS, TWM — the same lineage MATWM builds on) with **Mamba / Mamba-2**, a selective state-space model (SSM), to get O(n) memory/compute in sequence length instead of the O(n²) of self-attention, while still capturing long-range dependencies better than plain RNNs (vanishing-gradient framing). This is the direct empirical precedent for the "SSM vs. RSSM dynamics backbone" question ADR-0002 needs to answer.
- **[high, self_checked]** World model: a VAE encodes raw frames into a latent `z_t`; `z_t` and action `a_t` are fed into stacked Mamba blocks as the recurrent/sequence backbone, replacing GRU-style RSSM recurrence or Transformer self-attention with SSM state updates.
- **[high, self_checked]** Reports a **7M-parameter** world model — the smallest parameter count of any world model cited in this map so far (MAMBA/MABL/MARIE/GAWM/MATWM don't state comparably small figures) — and states it is "accessible and trainable on off-the-shelf hardware, such as a standard laptop," matching this project's laptop-scale constraint more explicitly than any other paper in the map.
- **[medium, self_checked]** Reports a normalized mean score of **~105%** on Atari100k using the Mamba-2 variant, cited as competitive with IRIS (105%) and ahead of TWM (96%) — consistent across 3 independent secondary sources (emergentmind, aimodels.fyi, liner.com), but not confirmed against a primary results table this pass.
- **[medium, self_checked]** Introduces **Dynamic Frequency-based Sampling (DFS)**: adaptively re-weights which past transitions get replayed for world-model training, prioritizing transitions the world model has already learned well over ones it hasn't — framed as mitigating a *policy-quality* failure mode (bad early-training world models produce bad imagined rollouts), not a *non-stationarity* failure mode. Distinct from MATWM's recency-weighted replay, which targets non-stationarity from co-learning agents specifically — worth keeping these two "prioritized/reweighted replay" ideas conceptually separate rather than treating DFS as a second data point for MATWM's mechanism. Secondary sources report DFS raises the normalized score from 80% (uniform sampling) to 105% — a large ablation delta, but not primary-source-confirmed.

## Method summary (from secondary sources, cross-checked)

A CNN/VAE encoder compresses each observed frame to a latent `z_t`. A stack of Mamba (SSM) blocks takes the sequence of `(z_t, a_t)` pairs and predicts forward dynamics — replacing both the GRU-based recurrence of classic RSSMs (Dreamer-family) and the self-attention of Transformer world models (STORM/IRIS/TWM/MATWM) with a linear-time selective state-space update. DFS reweights the replay buffer so that training samples transitions the world model currently predicts well, rather than uniformly, to reduce the impact of an inaccurate world model early in training.

## Compute scale

**7M-parameter world model**, evaluated on Atari100k (100K environment steps per game, the standard low-sample-budget Atari benchmark), explicitly reported as trainable on a single laptop. This is the single most directly laptop-scale-relevant compute datapoint in the map so far — every other paper here targets SMAC/MAMujoco-class multi-agent benchmarks with unstated or clearly larger compute footprints.

## Relevance to janus-chrysalis

Doesn't sit on the sharing-topology axis (single-agent), but is the concrete empirical anchor for **ADR-0002**'s SSM-vs-RSSM question: a Mamba/SSM-based world model, laptop-trainable, with a published parameter count an order of magnitude or more below the map's multi-agent papers. If ADR-0002 leans toward an SSM dynamics backbone for custom-autograd feasibility in a JS/TS stack, Drama is the reference architecture to adapt — and its DFS mechanism is a candidate replay-shaping ablation independent of (and conceptually distinct from) MATWM's non-stationarity-motivated replay reweighting.

## BibTeX

```bibtex
@inproceedings{wang2025drama,
  title     = {Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient},
  author    = {Wang, Wenlong and Dusparic, Ivana and Shi, Yucheng and Zhang, Ke and Cahill, Vinny},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2410.08893},
  archivePrefix = {arXiv}
}
```
