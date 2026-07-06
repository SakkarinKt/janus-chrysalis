# Communicating Plans, Not Percepts — Scalable Multi-Agent Coordination with Embodied World Models

- **arXiv**: [2508.02912](https://arxiv.org/abs/2508.02912) (Aug 2025)
- **Venue**: NeurIPS 2025 Workshop: Scaling Environments for Agents (SEA); also presented at the NeurIPS 2025 Workshops "Embodied World Models for Decision Making" (EWM) and "Optimization for Machine Learning" (OPT). Workshop-tier, not a main-track paper — noting this explicitly, same epistemic treatment as CoDreamer's workshop venue in this map.
- **Authors**: Brennen A. Hill, Mant Koh En Wei, Thangavel Jishnuanandh
- **Verification note**: `self_checked`, not `verified`. `WebFetch` returned HTTP 403 for `arxiv.org/abs/2508.02912` this session (same persistent autonomous-loop constraint noted elsewhere in this map). Claims are cross-checked across independent sources (arXiv abstract listing, NeurIPS workshop program, author's personal site) that agree on authorship, venue, and the two-strategy framing, but no specific quantitative results were recoverable from search snippets alone — flagged below rather than guessed at.

## Claims

- **[high, self_checked]** Frames a central design choice for multi-agent communication as engineered-vs-learned: **Learned Direct Communication (LDC)**, an end-to-end learned communication protocol (emergent-communication style), vs. **Intention Communication**, an engineered pipeline with two components — an **Imagined Trajectory Generation Module (ITGM)**, a compact learned world model combined with the agent's own policy to forecast its own short-term future states, and a **Message Generation Network (MGN)** that compresses that imagined trajectory into a transmitted message.
- **[medium, self_checked]** Reports that the engineered, world-model-based Intention Communication strategy shows better performance, sample efficiency, and scalability than LDC as task/environment complexity increases — direction of the result is consistent across sources, but exact numbers were not recoverable from search snippets this pass; needs a primary-source read before citing any specific figure.
- **[medium, self_checked]** Evaluated on a goal-directed, cooperative task-allocation problem in a grid-world environment, with complexity scaled up across experiments (specific complexity axis — e.g. grid size, agent count — not confirmed from secondary sources).
- **[medium, self_checked]** Architecturally distinct from CoDreamer (this map's other explicit-communication entry): CoDreamer's agents exchange raw latent state via GNN message-passing every step; here, each agent's ITGM first *imagines its own future rollout* using its own world model + policy, then the MGN compresses that imagined plan into a message. The content being communicated is a compressed **self-intention/plan**, not raw perceptual/latent state and not a prediction *about* a teammate — worth keeping distinct from "opponent modeling" (this paper models and broadcasts the sender's own future, not a receiver's model of others).

## Method summary (from secondary sources, cross-checked)

Each agent runs its own world model to imagine a short-horizon rollout of its own likely future trajectory conditioned on its current policy (the ITGM). A separate learned module (the MGN) compresses this imagined trajectory into a fixed-size message broadcast to teammates, who condition their own policies on the received messages. This is compared against a baseline where agents instead learn an end-to-end communication protocol from scratch with no engineered structure (LDC). The paper's thesis is that giving agents a world model and using it specifically to structure *what* gets communicated (a plan, not raw state or a raw learned code) is what drives the reported scalability/efficiency advantage — an explicit test of "engineered inductive bias via world models" vs. "let the network figure out what to say."

## Compute scale

Not recoverable from secondary sources this pass — grid-world scale suggests laptop-feasible, but no explicit parameter count, hardware, or wall-clock figures surfaced in search. Flagged as an open item for a primary-source read.

## Relevance to janus-chrysalis

Adds a **content dimension** to the lit map's sharing-topology axis that wasn't previously distinguished: axis point 4 (explicit peer-to-peer communication) has so far only had CoDreamer (raw latent-state GNN messages) as an example. This paper shows communication content itself is a design axis independent of topology — broadcasting an imagined self-plan (this paper) vs. raw state (CoDreamer) vs. a predicted teammate-action distribution (Dreaming of Others' ToM head, though that one doesn't communicate at all — it infers from observed behavior). Useful to fold into the gap-analysis draft as a "what gets shared" sub-axis alongside "who shares with whom." Not being treated as a non-stationarity-mechanism paper (unlike MARIE/GAWM/MATWM/Dreaming of Others) — its focus is protocol design under a fixed cooperative task, not co-learning-induced non-stationarity specifically.

## BibTeX

```bibtex
@inproceedings{hill2025communicating,
  title     = {Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models},
  author    = {Hill, Brennen A. and Koh En Wei, Mant and Jishnuanandh, Thangavel},
  booktitle = {NeurIPS 2025 Workshop: Scaling Environments for Agents (SEA)},
  year      = {2025},
  eprint    = {2508.02912},
  archivePrefix = {arXiv}
}
```
