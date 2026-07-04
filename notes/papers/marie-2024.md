# MARIE — Decentralized Transformers with Centralized Aggregation (Multi-Agent auto-Regressive Imagination for Efficient learning)

- **arXiv**: [2406.15836](https://arxiv.org/abs/2406.15836) (Jun 2024; TMLR-accepted version published ~May 2025)
- **Venue**: Transactions on Machine Learning Research (TMLR)
- **Authors**: Yang Zhang, Chenjia Bai, Bin Zhao, Junchi Yan, Xiu Li, Xuelong Li
- **Code**: [breez3young/MARIE](https://github.com/breez3young/MARIE) (official)
- **Correction note**: an earlier, less-specific search initially mis-surfaced this as "arXiv 2506.18537", which is actually a *different, later* paper — Deihim et al.'s "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning" (MATWM) — that explicitly *positions itself against* MARIE as prior work. Re-searched by exact title to pin down the correct primary source; this is why the increment budget went to 4 papers rather than more.
- **Verification note**: same WebFetch outage as `mamba-2022.md`. Cross-checked across arXiv abstract, official GitHub repo, OpenReview forum, and themoonlight.io literature-review summary. `self_checked`, confidence capped at `medium` below abstract level.

## Claims

- **[high, self_checked]** MARIE is presented as the first Transformer-based world model for multi-agent RL (as opposed to the RSSM/recurrent-latent lineage of MAMBA/MABL/CoDreamer).
- **[high, self_checked]** Explicitly names the two failure modes it's designed around: (1) *scalability* — centralized world-model architectures (like a single joint-state world model) don't scale gracefully with agent count; (2) *non-stationarity* — fully decentralized/independent world models suffer from inter-agent non-stationarity since each agent's dynamics depend on the others' changing policies. This is the clearest direct statement of the exact tension named in our mission statement ("non-stationarity from co-learning agents") found in this batch of papers.
- **[high, self_checked]** Its architectural answer: decentralized Transformer world models per agent (avoids the scalability problem) with a centralized *aggregation* mechanism (mitigates non-stationarity without going fully centralized) — i.e., a third point on the shared-vs-per-agent spectrum, structurally distinct from both MAMBA's per-agent+comms and MABL's bi-level-latent approach.
- **[medium, self_checked]** Reported to substantially improve over MBVD and marginally improve over MAMBA on a wide range of SMAC maps (per a later paper's — MATWM's — framing of MARIE as a baseline; not yet verified against MARIE's own results tables).

## Method summary (best available from abstract + secondary sources)

Each agent has its own Transformer-based (sequence-model, likely token/patch-based rather than RSSM) world model — decentralized, so parameter count and compute scale per agent rather than growing with the joint state space. A centralized aggregation step (mechanism not yet confirmed from primary text — candidate designs include a shared latent bottleneck, cross-attention over agents' tokens, or a mixing network akin to QMIX-style value factorization, but this is *speculation*, not a verified claim) combines information across agents to counteract the non-stationarity that pure decentralization would otherwise induce.

## Compute scale

Not confirmed from primary text this session — flagged for follow-up read of the OpenReview PDF or GitHub repo's README/config files (likely to state SMAC map list, replay-buffer size, and training steps — GitHub repos for this line of work typically do).

## Relevance to janus-chrysalis

The single most relevant paper found this session for the mission's specific framing (shared vs. per-agent world models *and* non-stationarity from co-learning agents, together, in one paper). Should be the primary related-work anchor for any research-proposal draft — our contribution needs to be clearly differentiated from "decentralized-with-centralized-aggregation," e.g. by testing the aggregation mechanism's actual causal effect on non-stationarity (ablate it) rather than only its effect on final return, which is the typical evaluation MARIE-style papers report.

## BibTeX

```bibtex
@article{zhang2024marie,
  title   = {Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models},
  author  = {Zhang, Yang and Bai, Chenjia and Zhao, Bin and Yan, Junchi and Li, Xiu and Li, Xuelong},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  eprint  = {2406.15836},
  archivePrefix = {arXiv}
}
```
