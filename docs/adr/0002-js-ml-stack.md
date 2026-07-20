# ADR-0002: JS/TS ML stack — TF.js 4.22.0, tfjs-node CPU training, RSSM from primitives

- **Status**: accepted (approved by @SakkarinKt via review + merge of the Gate-G1 PR, 2026-07-20 —
  consolidating the piecewise approvals from PR #7, 2026-07-08, and PR #19, 2026-07-19)
- **Deciders**: @SakkarinKt (signoff), Claude (research + drafting)
- **Evidence**: `notes/adr-0002-js-ml-stack.md` §1–§7 and
  `notes/rssm-vs-ssm-implementation-robustness.md` (claims there are `self_checked`; the human
  review of this PR is the verifying review)

## Decision

1. **Framework: TensorFlow.js, pinned exactly to `4.22.0`** — the last stable release (2024-10-21).
   TF.js is in low-cadence maintenance mode (notes §1); we accept that upstream fixes or new ops may
   not land on any predictable timeline. The `@tensorflow/tfjs-node` pin already landed 2026-07-15
   under `loop/GOAL.md`'s pre-approved carve-out.
2. **Training backend: `tfjs-node` CPU (WebGL acceptable); WebGPU is banned for the training path.**
   An asymmetric bet (approved in PR #7): near-zero cost to avoid WebGPU at this scale, versus
   catastrophic cost if issue #8590's silent zero-gradient RNN bug corrupts the drift-error
   measurement proposal `0001` depends on. WebGPU stays open for inference/demo use later.
3. **Architecture: RSSM composed from TF.js primitives (`tf.layers.gru` + custom heads), not
   SSM/Mamba.** TF.js has no parallel/associative-scan primitive, so Mamba would mean hand-building
   a novel numerical kernel with no reference implementation — a categorically larger risk than
   composing mature ops (`notes/rssm-vs-ssm-implementation-robustness.md`). SSM/Mamba stays a
   documented future comparison, not a Phase-2 path.
4. **Straight-through gradients via `tf.customGrad`, not `tf.stopGradient`** — `tf.stopGradient`
   does not exist as a public function in `tfjs`/`tfjs-core`/`tfjs-node` 4.22.0;
   execution-confirmed while implementing the stochastic latent (PR #20), correcting the earlier
   search-based notes. Every custom-gradient path ships a finite-difference gradient-check test in
   the same increment as the code.
5. **The Phase-2 week-3 spike carries the hard kill criterion**: if end-to-end gradient correctness
   (vs. finite differences) or usable steps/sec fails on this stack, fall back to a small custom
   autograd behind the same thin tensor-op boundary. Tripping it is a human decision point recorded
   as a superseding ADR, not a unilateral loop pivot.
6. **Canonical training environments: Linux x64 (loop sandbox + CI).** `tfjs-node` ships no darwin
   prebuilt addon at the napi-v8 tier for any version ever published (exhaustive GCS bucket probe,
   notes §3). Apple Silicon is deprioritized as a loop concern per the PR #19 review; any local
   darwin path (from-source build, or a darwin-only pure-JS fallback) is the human's to drive
   out-of-band, and the fallback would need a short amendment to this ADR.

## Consequences

- `src/model/rssm.ts` continues on RSSM + `tf.customGrad`; no loop increments go to Apple Silicon
  or to WebGPU training.
- Stack claims get verified by running the pinned package, not by documentation search — the
  `tf.stopGradient` reversal (PR #20) is the standing lesson.
- If decision 5's kill criterion fires, the custom autograd swaps in without touching model code
  and this ADR is superseded by a numbered successor.
