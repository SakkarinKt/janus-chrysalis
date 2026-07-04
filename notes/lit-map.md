# Literature map — world models in multi-agent RL

Status: seeded (4 core papers). Target before gap analysis / proposal drafting: ≥10 core papers (per `loop/GOAL.md`).

**Tooling caveat (2026-07-04 run):** `WebFetch` returned HTTP 403 for every URL attempted this session, including a plain `https://example.com` control fetch — this looks like an environment/proxy issue, not arXiv-specific blocking (see loop stand-up report for the day). All claims below were built from `WebSearch` result snippets, cross-checked across ≥2 independent sources per paper (arXiv listing + official GitHub/OpenReview/ACM-DL/proceedings-PDF page), rather than a full-text primary read. Every claim is tagged `self_checked` per the co-authorship protocol, and confidence is deliberately capped at `medium` for anything past the abstract/framing level — see each paper note's "Verification note" for specifics. **Do not promote any of these to `verified` until a primary-source pass is done once WebFetch is working.**

## Design axis this map is organized around

The mission statement's central question is where a world model sits relative to the agents that co-learn in the same environment:

1. **Fully shared** — one world model, joint state/observations, all agents' policies condition on it.
2. **Per-agent + centralized aggregation** — separate world model per agent, but a mechanism (attention, mixing, shared upper-level latent) explicitly pools information to counter non-stationarity. → **MARIE**
3. **Per-agent + bi-level (train-time-only sharing)** — global info baked into an upper-level latent used only during training; execution uses a lower level that's already been informed by it, so no runtime sharing is needed at all. → **MABL**
4. **Per-agent + explicit peer-to-peer communication** — no central aggregator; agents exchange messages (GNN-style) that shape each other's world-model state directly. → **CoDreamer**
5. **Per-agent + comms via imagined rollouts, PPO-style policy update** — the original model-based-MARL reference point everything else is measured against. → **MAMBA**
6. *(Not yet mapped)* **Fully independent, no sharing at all** — the non-stationarity stress case / naive baseline every other paper implicitly argues against. Need to find or construct this as an ablation, not a citation — it's usually the strawman other papers compare to, rarely a paper in its own right.

## Papers (4/10)

| Key | Title | Year | Venue | Axis point | Note |
|---|---|---|---|---|---|
| `mamba-2022` | Scalable Multi-Agent Model-Based RL | 2022 | AAMAS | per-agent + comms, PPO policy update | [notes/papers/mamba-2022.md](papers/mamba-2022.md) |
| `mabl-2024` | MABL: Bi-Level Latent-Variable World Model | 2023/2024 | AAMAS | per-agent + train-time-only global latent | [notes/papers/mabl-2024.md](papers/mabl-2024.md) |
| `codreamer-2024` | CoDreamer: Communication-Based Decentralised World Models | 2024 | OpenReview / CoCoMARL ws. | per-agent + GNN peer comms | [notes/papers/codreamer-2024.md](papers/codreamer-2024.md) |
| `marie-2024` | Decentralized Transformers w/ Centralized Aggregation (MARIE) | 2024/2025 | TMLR | per-agent + centralized aggregation, explicit non-stationarity framing | [notes/papers/marie-2024.md](papers/marie-2024.md) |

## Preliminary gap observations (self_checked, low confidence — 4 papers is too few to generalize; recorded now so they aren't lost, to be re-checked as the map grows)

- **[low, self_checked]** All four papers evaluate primarily on final-return / sample-efficiency curves. None of the abstracts/framings surfaced so far report a *direct, isolated* measurement of non-stationarity itself (e.g., world-model prediction error over training as a co-learning partner's policy shifts, holding everything else fixed). If this holds up as the map grows, it's a candidate gap: existing work treats non-stationarity as a problem to architect *around*, not a quantity to *measure*.
- **[low, self_checked]** No paper in this batch runs the same architecture across a shared-vs-per-agent sweep on one fixed environment/task — each paper introduces one new architecture and compares it to prior papers' reported baselines (different codebases, possibly different hyperparameter budgets). A controlled, single-codebase sweep across the 5-point axis above, on one laptop-scale env, would be a novel (if modest) empirical contribution distinct from proposing a 6th architecture.
- **[low, self_checked]** MARIE is the only paper of the four whose abstract explicitly names non-stationarity as a first-class design target rather than an implicit motivation — worth re-reading first, in full, once WebFetch is available.

## Not yet followed up (candidates for next lit-map increment)

- **"Dreaming of Others: Latent Teammate Modeling in World Models for MARL"**, arXiv 2605.31361 (May 2026) — surfaced by search as directly on-topic (factorizes RSSM latent into environment vs. teammate components, learns a Theory-of-Mind head, explicitly targets non-stationarity from partner policy shifts). **Not added as a full paper note yet**: single-author listing and very recent date make it a higher hallucination/mis-citation risk than the other four given this session's search-only verification; needs a primary-source check (arXiv ID + abstract confirmed against the actual PDF) before a note is written. Good first candidate once WebFetch is back.
- GAWM: Global-Aware World Model for MARL, arXiv 2501.10116 — surfaced, not yet read.
- Multi-Agent Model-Based RL with Joint State-Action Learned Embeddings, arXiv 2602.12520 — surfaced, not yet read.
- MATWM (Deihim et al.), "Transformer World Model for Sample Efficient MARL" — the paper an earlier search conflated with MARIE (see correction note in `marie-2024.md`); positions itself as MARIE's successor, worth reading alongside MARIE for continuity.
- Drama: Mamba(-SSM, unrelated to MAMBA-MARL)-Enabled Model-Based RL, arXiv 2410.08893 — single-agent, but relevant to ADR-0002 (SSM vs. RSSM dynamics backbone).

## Non-stationarity / opponent-modeling-in-latent-space seed area

Still thin — MARIE's centralized-aggregation framing and the not-yet-verified "Dreaming of Others" paper are the two most direct hits. This seed area should get priority in the next lit-map increment, since it's the mission's second named axis and currently has the weakest coverage.
