# Notes toward ADR-0002: JS ML stack / world-model backbone choice

Status: research notes only, **not a decision**. ADR-0002 itself needs human signoff per
`loop/GOAL.md` boundaries — this file collects evidence to make that signoff easier, and directly
unblocks proposal `0001`'s scoped L2 milestone, which needs one backbone named before any code is
written (see `docs/proposals/0001-direct-nonstationarity-measurement.md`, "L2 promotion request").

All claims below are `self_checked`. Where noted "primary-source", the claim comes from a
`WebFetch` of the source itself (not a `WebSearch` snippet) — see the tooling note in §5, which
corrects a standing assumption from `notes/lit-map.md`.

## 1. TF.js maintenance status

**[high, self_checked, primary-source]** `tensorflow/tfjs`'s last tagged release is **v4.22.0,
2024-10-21** — no stable release in the ~20 months since, as of this note (2026-07-08). Confirmed
via `GET https://registry.npmjs.org/@tensorflow/tfjs-node`: `dist-tags.latest` is still `4.22.0`,
though a `4.23.0-rc.0` pre-release tag exists, meaning some release work is in progress but unshipped.

**[high, self_checked, primary-source]** Commit activity has not stopped but has slowed sharply.
`github.com/tensorflow/tfjs/commits/master` (fetched directly): commits are dense through
2025-04-23, then a large gap, then isolated commits (one dated 2025-05-29, one 2025-06-06, then a
gap of ~10 months to the next visible commit 2026-04-06). The repo's own front page (fetched
directly) reports **246 open issues and 340 open PRs** against that commit cadence — a ratio
consistent with a reduced-maintainer-bandwidth project rather than an actively-triaged one, though
no explicit deprecation or archival notice exists anywhere in the README or issues search.

**Read on this**: TF.js is not dead/archived, but is in a low-cadence maintenance mode — patches
land, no stable releases in over a year, and open PR/issue backlog is large relative to merge
rate. For a project on a multi-year research horizon, this is a real risk factor: don't assume
new ops, bug fixes, or WebGPU gradient gaps (§2) will land on any predictable timeline.

## 2. WebGPU backend maturity

**[medium, self_checked]** The WebGPU backend (`tfjs-backend-webgpu`) is real and, per its own
package docs, delivers meaningfully faster **inference** than the WebGL backend (~3x cited in
multiple secondary sources, not independently re-benchmarked here). But the backend's own
maintainers describe it as still missing ops needed for **gradient computation** ("a decent number
of ops that we are missing in WebGPU that are needed for gradient computation" — from the backend's
own README, via `WebSearch` snippet, not independently primary-sourced this session).

**[medium, self_checked, primary-source]** Concretely: `tensorflow/tfjs` issue **#8590** ("[WebGPU]
Gradients are always zero for RNN models"), open since 2025-09-13, assigned to three maintainers,
unresolved as of this note. Fetched directly: the reproduction is a `simpleRNN` model, forward pass
correct, backward pass yields all-zero gradients (no learning), and — importantly — the issue
reporter narrows it to a specific GPU vendor (Pixel 10 Pro / Img-Tec), noting other WebGPU-capable
devices don't reproduce it. So this is evidence of **immaturity in the recurrent-gradient path on
WebGPU**, not a blanket "WebGPU can't train RNNs" claim — but for an RSSM (which is fundamentally
recurrent — GRU-style deterministic state update every step), it's a concrete red flag: the exact
op class our world model depends on most has a live, unresolved, backend-specific gradient bug
report. Given §1's release cadence, there's no guarantee of a near-term fix.

**Implication for the RSSM op set**: WebGPU is plausibly fine as an **inference/rollout**
accelerator later, but training an RSSM on the WebGPU backend today carries real, demonstrated risk
of silently-broken gradients on some hardware, which is a hard failure mode for training a
non-stationarity signal that already needs a clean, trustworthy prediction-error measurement
(directly load-bearing for `0001`'s kill criterion #1). **CPU (tfjs-node) or WebGL remain the
safer default for the training path at least through the Arm-A milestone.**

## 3. tfjs-node on Apple Silicon

**[medium, self_checked]** `@tensorflow/tfjs-node` supports `darwin-arm64` (M1/M2/M3 use the same
architecture) via native bindings, but installation has a history of friction: multiple open/closed
GitHub issues (`#6436`, `#8228`, `#4514`, found via `WebSearch`, not individually primary-sourced
this session) describe `node-pre-gyp install --fallback-to-build` failures on recent macOS versions,
Rosetta-related mis-detection (terminal running under x86 emulation silently pulling the wrong
binary), and missing prebuilt binaries for newer Node/ABI combinations forcing a from-source build.
Community workarounds exist (explicitly using an arm64-native Node install, disabling Rosetta for
the terminal app) but there is no confirmed "just works" story as of this note.

**Read on this**: not a blocker, but budget setup friction — if the human's laptop is Apple
Silicon, confirm `node -p process.arch` reports `arm64` (not `x64` under Rosetta) before relying on
`tfjs-node`'s native bindings, and expect to possibly need a from-source build depending on the
installed Node version. Worth a smoke-test as part of Arm-A milestone setup, not before.

## 4. Custom-autograd feasibility for the RSSM op set

**[medium, self_checked]** `tf.customGrad()` is a real, documented, stable TF.js API (`tensorflow.org/js/guide/custom_ops_kernels_gradients`,
existence confirmed via multiple independent `WebSearch` results though the guide page itself
403'd on `WebFetch` — see §5) for defining forward value + custom backward gradient in one function,
independent of the built-in gradient registry. This is the mechanism the RSSM op set would lean on
for:

- **Straight-through gradients** for the categorical/discrete latent (if the backbone uses a
  discrete stochastic state, DreamerV3-style) — standard `customGrad` use case, well precedented in
  TF (Python) via `tf.custom_gradient`, and the JS API is a direct analog.
- **KL-balancing** between prior and posterior latent distributions — this doesn't need a custom
  gradient per se (it's differentiable through standard ops: a stop-gradient split-KL is expressible
  with `tf.customGrad` or even just two ordinary KL terms with different `tf.stopGradient` placement)
  but does need `stopGradient` support, which TF.js has natively.
- **The GRU-style deterministic recurrent state update** — standard differentiable ops
  (`tf.layers.gru`/custom cell via matmul+activation), no custom gradient needed unless a
  non-standard gating variant is used.

**No evidence found either way** on whether anyone has built a full RSSM (Dreamer-style) in TF.js
specifically — no port or reproduction turned up in search. This proposal would likely be a novel
implementation, not an adaptation of existing JS code. That raises the effort estimate for the
Arm-A milestone (proposal `0001`) beyond "wire up an existing library" to "implement an RSSM cell
from primitives," but doesn't change the *feasibility* verdict: the primitives needed (custom
gradient, stop-gradient, standard recurrent ops) all exist and are stable in TF.js's core (CPU/
WebGL/tfjs-node) backends. Feasibility risk is concentrated in the *backend choice* (§2), not in
whether TF.js's autograd system can express an RSSM at all.

## 5. Tooling correction (relevant beyond this note)

**[high, self_checked, primary-source]** `WebFetch` succeeded this session on `github.com` (repo
pages, commits, releases, issues) and `registry.npmjs.org` (raw JSON API), but still returned
HTTP 403 on `tensorflow.org` and `npmjs.com` (the human-facing package page, as opposed to the
registry API). This refines the standing assumption in `notes/lit-map.md`'s tooling caveat, which
(per PR #2's human reply) treats "`WebFetch` returns 403" as a blanket persistent constraint —
it is evidently **domain-specific** (some domains blocked, others not), not a global proxy failure.
Not changing the lit-map protocol this run (that's out of this increment's scope, and arXiv itself
was not re-tested this session), but flagging it: future runs researching non-arXiv sources should
try `WebFetch` on the specific domain before assuming 403, since `github.com` and
`registry.npmjs.org` are now confirmed reachable.

## 6. Where this leaves the backbone choice

Not a decision (ADR-0002 needs human signoff), but the evidence above supports a **working
default recommendation** for proposal `0001`'s Arm-A milestone specifically: **RSSM implemented
from TF.js primitives (custom cell + `tf.customGrad` for the discrete-latent straight-through
path), trained on the `tfjs-node` (CPU) or WebGL backend, not WebGPU.** This sidesteps §2's
unresolved recurrent-gradient risk entirely for the milestone that most needs a trustworthy
measurement, at the cost of slower training than WebGPU could in principle offer — an acceptable
trade at Arm-A's laptop/CPU scale (`0001` already budgets for CPU-only runs). WebGPU remains worth
revisiting once §2's issue class is resolved or once inference-only (not training) throughput
matters more. The SSM/Mamba data point from `notes/papers/drama-2024.md` (7M-param laptop-trainable
backbone) stays a live alternative worth comparing against RSSM once ADR-0002 is drafted in full —
this note does not attempt that comparison; it only covers the *stack* (TF.js/WebGPU/tfjs-node)
questions `loop/GOAL.md` priority 4 named, not the *architecture* (RSSM vs. SSM) question, which
remains open.
