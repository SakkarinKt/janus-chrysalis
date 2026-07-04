# janus-chrysalis

**Open research on world models in multi-agent RL, trained entirely in JavaScript/TypeScript — co-authored by a human and Claude, in public.**

> *Janus*: the two-faced god — multiple agents, and two authors looking at the same problem.
> *Chrysalis*: this repo is a learning vehicle as much as a research project.

## What this is

An open-source research project with one primary goal: **novel empirical findings** about world models in multi-agent reinforcement learning — for example, when a shared world model beats per-agent models, and how learned dynamics degrade under the non-stationarity created by other learning agents. The framework that emerges is a means, not the end.

Three things make it unusual:

1. **Browser-native stack.** Training and inference run in JS/TS (Node for training, WebGPU in the browser for the demo). MARL world models in JavaScript is a nearly empty niche; the final artifact includes an interactive demo where the trained world model runs client-side.
2. **Human + Claude co-authorship with real gates.** Every irreversible decision is an ADR the human approves. Core algorithms get an explainer before implementation. Some modules are human-written with Claude reviewing (role flip). No claim enters the writeup without an adversarial review in a fresh session. See [CONTRIBUTING.md](CONTRIBUTING.md).
3. **A daily autonomous loop.** A scheduled Claude agent runs once a day in the cloud, reads [`loop/GOAL.md`](loop/GOAL.md), does one bounded increment, and opens a PR with a stand-up report ([`reports/standup/`](reports/standup/)). The loop itself is one of the experiments — "Loop Engineering."

## Status

**Phase 1 — Autoresearch** (literature sweep → gap analysis → research proposals → question selection at Gate G1).

See [PLAN.html](PLAN.html) for the full roadmap, phases, and gates.

## Repo map

| Path | What lives here |
|---|---|
| `PLAN.html` | The roadmap, refreshed at each phase gate |
| `loop/GOAL.md` | Standing goal + boundaries for the daily autonomous loop |
| `reports/standup/` | One stand-up report per daily loop run |
| `docs/adr/` | Architecture Decision Records (the signoff surface) |
| `docs/explainers/` | Pre-implementation algorithm notes |
| `docs/proposals/` | Research proposals (Phase 1 output, Gate G1 artifact) |
| `notes/` | Literature map + per-paper notes (claims + confidence + bibtex) |
| `experiments/` | Run configs |
| `artifacts/` | Run outputs (gitignored except manifests + summary CSVs) |
| `src/`, `test/` | The TS world-model stack (from Phase 2) |
| `LEARNING.md` | Weekly journal from both authors |

## Lineage

Design patterns (not code) are ported from two prior projects by the same authors: a verified single-agent Dreamer-v1-style RSSM (gradient-separation invariant, KL free-bits, telemetry schema — reused here as executable tests) and an agent-orchestration cookbook whose "RL research lab" conventions (critique rubric, handoff contracts, run manifests, background-loop pattern) are this project's working process.

## License

[MIT](LICENSE)
