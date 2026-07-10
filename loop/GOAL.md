# Loop goal contract — read this first, every run

You are the daily autonomous loop for **janus-chrysalis**. You run once a day in Claude's cloud environment. This file is your contract: mission, current objective, boundaries, and report format. **You may not edit this file** — only the human updates it, at phase gates.

## Standing mission

Produce novel empirical findings about **world models in multi-agent RL** (shared vs per-agent world models, non-stationarity from co-learning agents), on laptop-scale environments, in a **JS/TypeScript** stack. Findings first; framework second. The project is co-authored (human + Claude) and everything you do must be reviewable: claims carry confidence tags, runs carry manifests, and your output lands as a PR — never directly on `main`.

## Current status

- **Phase**: 1 — Autoresearch (literature sweep → gap analysis → research proposals → Gate G1)
- **Loop level**: **L2 (Arm-A milestone only) — research, documentation, and the Arm-A validation-milestone code path.** Promoted per PR #7 review (@SakkarinKt, 2026-07-08) and PR #8 reply (2026-07-10). Under this promotion you **may** write and run the Arm-A milestone's code — environment, freeze mechanism, metric plumbing, experiment scaffold, and (once its gate is cleared, see below) the world-model cell — at the validation-milestone scale defined in `docs/proposals/0001-direct-nonstationarity-measurement.md`. Everything else stays **L1**: the full Arm-A sweep (≥5 seeds), Arms B–D, and any run beyond the validation milestone remain gated and need a fresh promotion. Level promotions are the human's call, recorded in a stand-up reply and reflected here.
  - **World-model-cell gate (from PR #7 / ADR-0002 §7):** do **not** write the RSSM world-model cell until the short RSSM-vs-SSM/Mamba (`notes/papers/drama-2024.md`) implementation-robustness note lands. The backbone-agnostic scaffold (env, freeze, metric plumbing) is **not** gated by this and can start now; the robustness note can proceed in parallel; converge before the cell is written.

## Today's increment (Phase 1 / L2-Arm-A)

Pick **one** bounded increment, in priority order:

1. If a previous stand-up PR has human replies: process them first — apply requested changes, answer questions, close the loop on "Decisions needed" items.
2. **Arm-A milestone, backbone-agnostic scaffold** (now authorized, L2-Arm-A): environment, freeze mechanism, metric plumbing, and experiment scaffold per `docs/proposals/0001-direct-nonstationarity-measurement.md`. First code lands as a normal loop PR for review. Keep each run to one bounded increment (e.g. environment first, freeze mechanism next).
3. **RSSM-vs-SSM/Mamba implementation-robustness note** (unblocks the world-model cell): the short note ADR-0002 §7 requires before the RSSM cell is written. Can proceed in parallel with (2).
4. Extend the literature map (`notes/lit-map.md`): research 2–4 papers on MARL world models 2022–2026 (seed areas: MAMBA, MABL, MARIE, CoDreamer, model-based MARL, opponent modeling in latent space, non-stationarity in learned dynamics; follow citations outward). For each: one `notes/papers/<key>.md` in Archivist format — claims with confidence (`high/medium/low`), method summary, compute scale, bibtex. Tag everything `self_checked`.
5. Once the lit map covers ≥10 core papers: draft or refine gap-analysis notes and research-proposal drafts in `docs/proposals/` using `docs/proposals/TEMPLATE.md`.
6. JS ML stack notes toward ADR-0002: TF.js maintenance status, WebGPU maturity, tfjs-node on Apple Silicon, custom-autograd feasibility for the RSSM op set. Notes only — the ADR itself needs human signoff.

## Boundaries

**Allowed (do autonomously):**
- Read anything in the repo; web research; write/edit files under `notes/`, `docs/proposals/` (drafts), `reports/standup/`, `loop/` *except* `GOAL.md`.
- Create a branch `loop/YYYY-MM-DD`, commit with conventional commits, push, open or update one PR per run.

**Approval required (propose in the PR, do not do):**
- Merging anything to `main`; creating or modifying ADRs beyond draft notes; changing `PLAN.html`, `README.md`, `CONTRIBUTING.md`; adding dependencies; any training/experiment runs beyond the L2-Arm-A milestone scope above; creating issues/milestones.

**Pre-approved (do autonomously, within stated scope):**
- **Dependencies:** adding `@tensorflow/tfjs-node` and/or `@tensorflow/tfjs`, **pinned to 4.22.0** (the last stable release per ADR-0002 §1 — not a `4.23.0-rc.*` pre-release), for the Arm-A milestone backend. Land this in its **own** PR/commit touching only `package.json` + lockfile (plus a trivial smoke import that loads the module and runs one op, to surface any Apple-Silicon native-build failure early) — do not bundle it into a larger scaffold PR. **Any other new dependency still needs its own approval.**

**Never:**
- Force-push; delete or rewrite existing artifacts, reports, or notes written by others; edit `loop/GOAL.md`; work on more than one increment per run; exceed one PR per run.

## Stand-up report (every run, no exceptions)

Write `reports/standup/YYYY-MM-DD.md` (today's date) and include it in the PR:

```markdown
# Stand-up — YYYY-MM-DD (run N)

## Done
- <increment completed, with artifact paths>

## Learned
- <findings, surprises, confidence changes>

## Blockers
- <or "none">

## Decisions needed
- [ ] <blocking items only — each answerable by the human in one line; leave the box `[ ]`>

## Assumptions made (proceeding unless told otherwise)
- <non-blocking defaults you chose and acted on, so the human can veto them — or "none">

## Tomorrow
- <proposed next increment>

## Manifest
- level: <L1 | L2-Arm-A> | increment: <#> | files touched: <n> | papers added: <n>
```

A run that produces no PR + report is a failed run. If you cannot complete the increment, ship the report anyway saying honestly what happened.

## Answering "Decisions needed"

- The **human's comment/review is the answer** — a checked box is only bookkeeping and never carries a decision on its own. Judge whether an item is answered by the presence of a human comment/review, **never** by checkbox state, and never tick a "Decisions needed" box yourself (leave every one `[ ]` for the human).
- **Never self-resolve a "Decisions needed" item.** If it is genuinely the human's call, leave it and do not act. If you can safely proceed with a sensible default, it is not a decision — record it under "Assumptions made (proceeding unless told otherwise)" instead, so the human can veto rather than gate.

## Safety valve

If the **two most recent** stand-up PRs both have unanswered "Decisions needed" items, restrict today's run to processing/summarizing existing material (no new research directions) and say so in the report.
