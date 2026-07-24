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

**2026-07-18 update [high, human-reported, PR #17 review]**: no longer a "history of friction"
concern — confirmed broken on the pinned version. `@tensorflow/tfjs-node@4.22.0` fails to install
on Apple Silicon: 404 on the prebuilt `napi-v8/4.22.0` `darwin-arm64` binary, and no working
source fallback (per @SakkarinKt's PR #17 review, run on real Apple Silicon hardware — this
project's sandboxed sessions are x64 Linux and cannot reproduce or independently verify the 404
directly). This is a harder finding than the community-issue survey above: it means the *exact
pinned version this repo depends on* (§7's `4.22.0` pin) is currently unusable on `darwin-arm64`,
not just historically friction-prone on some past versions. Flagged by the reviewer as "worth a
future increment to pin/patch or document a workaround before the cell work depends on running it
on-device" — not actioned in this note; see the 2026-07-18 stand-up for scoping notes on why a fix
attempt was deferred rather than started blind in an environment that can't reproduce the failure.

**2026-07-19 update [high, self_checked, primary-source]**: PR #18's review named a concrete next
step — "a dependency PR bumping to a tfjs-node version that ships a darwin-arm64 prebuilt —
findable by probing the same pre-built-binary/napi-v{N}/{version}/CPU-darwin-\* path across
versions." Ran that probe this session (network reachable to `storage.googleapis.com` from this
sandbox; verified against the GCS JSON list API, not just individual HEAD requests, so this is an
exhaustive listing rather than a sample):

- Read `@tensorflow/tfjs-node`'s own install machinery (`node_modules/@tensorflow/tfjs-node/scripts/{deps-constants,get-addon-name,install}.js`
  after a fresh `npm ci`) rather than guessing the URL shape: the native addon filename never
  encodes CPU architecture (`CPU-darwin-{version}.tar.gz`, identical for Intel and Apple Silicon
  Macs) — only `libtensorflow` (the C shared library, downloaded separately via a hardcoded
  special-case URL in `install.js`) has an arm64-specific path. So "darwin-arm64 binary" in the
  PR #17 finding is shorthand for a `darwin`-wide addon, not an arm64-only gap.
- Confirmed this sandbox's `process.versions.napi` is `10` (Node v22.22.2), and that `npm ci` here
  resolved to `lib/napi-v8` — i.e. node-pre-gyp picks the *highest declared `napi_versions` entry
  ≤ the runtime's napi version*, not the runtime version itself. `@tensorflow/tfjs-node`'s
  `package.json` has declared `napi_versions: [3,4,5,6,7,8]` for years (checked `3.6.1` through
  `4.22.0` — identical array every time), so on any Node ≥ napi-8 runtime (this repo's
  `engines.node` pin, `>=22.18.0 <23`, is comfortably in that range) the addon request always
  targets `napi-v8`, regardless of which `tfjs-node` version is pinned.
- Queried the bucket's public list API (`storage.googleapis.com/storage/v1/b/tf-builds/o?prefix=...`)
  for `pre-built-binary/napi-v8/`: **90 objects total, 88 linux, 1 windows, 0 darwin — across every
  `tfjs-node` version ever built at that napi tier**, not just `4.22.0`. Cross-checked `napi-v3`
  through `napi-v7`: darwin addons exist there (e.g. `napi-v7/3.6.1/CPU-darwin-3.6.1.tar.gz`), but
  **`3.6.1` (pinned to the long-superseded `@tensorflow/tfjs@3.6.0`, 2021) is the newest version
  with *any* darwin addon anywhere in the bucket** — every version from `3.7.0` onward through the
  current `4.23.0-rc.0` has zero darwin coverage at its declared napi tier.

**Conclusion: the "bump the version" fix PR #18 suggested does not exist.** This isn't a
`4.22.0`-specific regression — darwin native-addon prebuilts were dropped from the project's build
pipeline project-wide once it moved past napi-v7 (mid-2021) and never came back, for Intel and
Apple Silicon alike. Pinning to an older version with darwin coverage (`3.6.1`) isn't viable either:
this repo's Node requirement (§7, `engines.node >=22.18.0`) reports napi 10, so node-pre-gyp would
still request `napi-v8` for that old version too, and `napi-v8` has no darwin build regardless of
which `tfjs-node` release declares it. The only paths left, neither actionable from this sandbox:
(a) get `node-pre-gyp install --fallback-to-build` (a from-source compile via `node-gyp` against
the arm64 `libtensorflow` that *does* download fine) working on the human's actual hardware — this
is the path PR #17's review already reported failing, so "why" is the open question, not "whether
to bump a version"; or (b) switch the Apple Silicon path specifically to the pure-JS
`@tensorflow/tfjs` package (no native addon, WebGL/CPU-JS backend) as a platform-specific fallback
— a real architecture change to §7's backend decision, needing its own approval, not something
this note resolves. Recorded as a "Decisions needed" item in the 2026-07-19 stand-up rather than
picked unilaterally.

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
from TF.js primitives (`tf.layers.gru` for the deterministic recurrence + `tf.stopGradient` for the
discrete-latent straight-through path), trained on the `tfjs-node` (CPU) or WebGL backend, not
WebGPU.** This sidesteps §2's
unresolved recurrent-gradient risk entirely for the milestone that most needs a trustworthy
measurement, at the cost of slower training than WebGPU could in principle offer — an acceptable
trade at Arm-A's laptop/CPU scale (`0001` already budgets for CPU-only runs). WebGPU remains worth
revisiting once §2's issue class is resolved or once inference-only (not training) throughput
matters more. The SSM/Mamba data point from `notes/papers/drama-2024.md` (7M-param laptop-trainable
backbone) stays a live alternative worth comparing against RSSM once ADR-0002 is drafted in full —
this note does not attempt that comparison; it only covers the *stack* (TF.js/WebGPU/tfjs-node)
questions `loop/GOAL.md` priority 4 named, not the *architecture* (RSSM vs. SSM) question, which
remains open.

## 7. Decision record — approved to build against (not full ADR-0002 signoff)

Per PR #7 review (@SakkarinKt, 2026-07-08), split into two parts:

- **Backend: CPU (`tfjs-node`) or WebGL, not WebGPU — approved unconditionally** for the training
  path. Rationale given: an asymmetric bet — near-zero cost to avoid WebGPU at this scale, versus
  catastrophic cost if issue #8590's silent zero-gradient bug (§2) corrupts the drift-attributable
  error measurement `docs/proposals/0001-direct-nonstationarity-measurement.md` depends on. WebGPU
  stays live for later inference/rollout use, not training.
- **Architecture (RSSM from TF.js primitives, §6): approved as a *fixture* for the Arm-A milestone,
  not as the ADR-0002 decision itself.** One gate attached: **do not write the world-model cell**
  until a short RSSM-vs-SSM/Mamba (`notes/papers/drama-2024.md`) implementation-robustness note
  lands — the concern is that a subtle gradient bug in a hand-rolled recurrent cell is the top
  threat to the measurement's trustworthiness, the same risk class as §2's WebGPU finding. Per the
  review: the backbone-agnostic parts of the Arm-A milestone (environment, freeze mechanism, metric
  plumbing, experiment scaffold) can start now; the robustness note can proceed in parallel;
  converge before the world-model cell itself is written. Does not block starting the milestone's
  non-backbone work.

**2026-07-14: gate cleared.** The RSSM-vs-SSM/Mamba implementation-robustness note landed —
`notes/rssm-vs-ssm-implementation-robustness.md`. Verdict: proceed with RSSM from TF.js primitives
for the Arm-A milestone's world-model cell (SSM/Mamba's core primitive, a parallel/associative scan,
doesn't exist anywhere in TF.js, a categorically larger risk than composing RSSM from mature,
existing ops). Recommends a finite-difference gradient-check test ship with the cell itself. This is
a `self_checked` recommendation, not a human signoff — flagged as an "Assumption made" in the
2026-07-14 stand-up rather than gating further, per the note's own closing caveat.

**2026-07-15: dependency landed.** `@tensorflow/tfjs-node` added to `package.json`, pinned exactly to
`4.22.0` per `loop/GOAL.md`'s pre-approved carve-out (own commit, `package.json` + lockfile +
`test/smoke/tfjs-node.test.ts`). Smoke-tested successfully on this session's Linux x64 sandbox
(native binding loads, `tf.getBackend()` reports `"tensorflow"` — the CPU backend, matching §7's
backend decision; one tensor op runs and returns the correct result). **Not yet confirmed**: the
Apple Silicon (`darwin-arm64`) install-friction risk flagged in §3 above — this session's environment
is x64 Linux, so that risk remains unverified until run on the human's actual machine. `npm audit`
reports 3 high-severity advisories, all transitively via `@mapbox/node-pre-gyp`'s pinned `tar`
version (used only at install time to fetch the prebuilt native binary, not at runtime) — consistent
with §1's low-maintenance-cadence finding; `npm audit fix --force` would downgrade
`@tensorflow/tfjs-node` to `0.1.11`, clearly wrong, so left as-is and flagged rather than acted on.

**2026-07-17: cell struct/forward-pass landed; §6 wording corrected.** The RSSM cell's
deterministic-recurrence sub-increment landed — `src/model/rssm.ts`, see
`docs/explainers/0003-rssm-world-model-cell.md`. §6's decision-record phrase ("custom cell +
`tf.customGrad` for the discrete-latent straight-through path") is corrected above to
`tf.stopGradient`, per PR #14's review and matching `notes/rssm-vs-ssm-implementation-robustness.md`
§4's finding that the straight-through path needs only `stopGradient`, not a custom gradient — the
stochastic latent/STE code that phrase actually describes is still unwritten (next sub-increment),
so this is a documentation correction to the standing recommendation, not a record of new code
using either mechanism.

**2026-07-18: Apple Silicon install risk upgraded from "friction-prone" to "confirmed broken."**
Per PR #17's review, `@tensorflow/tfjs-node@4.22.0` (the exact version pinned in `package.json`)
404s on its prebuilt `darwin-arm64` binary with no working source fallback — see §3's updated
entry. This doesn't affect the Arm-A milestone's progress in this Linux x64 sandbox, but it is a
known blocker for running any of this code on the human's actual (assumed Apple Silicon) machine.
Not fixed this run — no `darwin-arm64` environment is available here to develop or test a
pin/patch/workaround against, and guessing at one blind (e.g. bumping to an untested newer
version, which would violate §7's `4.22.0` pin rationale) risks a worse outcome than leaving it
documented and open. Flagged for the human in the 2026-07-18 stand-up, since attempting a fix
without a way to verify it is a judgment call this loop shouldn't make unilaterally.

**2026-07-19: version-bump path ruled out by data, not just deferred.** Per PR #18's review
suggestion, probed the `tf-builds` GCS bucket's prebuilt-binary listing exhaustively (see §3's
2026-07-19 entry for method and full evidence) rather than guessing at a fix. Finding: no
`tfjs-node` version — past, current, or the `4.23.0-rc.0` pre-release — has a working darwin addon
at the `napi-v8` tier this repo's Node pin always resolves to; darwin support was dropped
project-wide after `3.6.1` (2021) and never restored, for both Intel and Apple Silicon. A version
bump is not a candidate fix. The remaining two paths (from-source build on real hardware, or a
platform-specific pure-JS `@tensorflow/tfjs` fallback) are recorded as a "Decisions needed" item
in the 2026-07-19 stand-up rather than chosen here — the first needs the human's own machine to
debug, the second is a backend-decision change this note isn't scoped to make.

**2026-07-20: §4/§6's `tf.stopGradient` claim doesn't hold up — corrected to `tf.customGrad`, and
the stochastic latent landed.** Per PR #19's review ("resume the RSSM cell increment: prior/posterior
heads, straight-through categorical sampling, gradient-check test"), that sub-increment landed —
`src/model/rssm.ts`'s `RSSMCell.prior()`/`.posterior()` heads and `straightThroughEstimator()`, tested
in `test/model/rssm.test.ts` (44/44 passing). While implementing it: **`tf.stopGradient` does not
exist as a public function on `@tensorflow/tfjs-node@4.22.0`, `@tensorflow/tfjs-core@4.22.0`, or
`@tensorflow/tfjs@4.22.0`** — confirmed directly (`typeof tf.stopGradient === "undefined"` on all
three, and calling it throws `tf.stopGradient is not a function`), not by a documentation/search
claim. This contradicts §4's "a native, stable TF.js primitive... confirmed present" and the
2026-07-17 entry that corrected §6's wording *to* `tf.stopGradient` — both claims trace back to
`WebSearch`-only verification (§4 says the guide page itself 403'd on `WebFetch`), which evidently
wasn't sufficient here; this is the first point in the ADR-0002 research thread where a claim was
checked by actually running the pinned dependency's code rather than searching about it, and it
didn't hold. §4's and §6's `tf.stopGradient` language is superseded by this entry, not edited in
place, to keep the correction visible rather than silently rewriting history.

**The fix**: `tf.customGrad()` (§4 already correctly identifies this as "a real, documented, stable
TF.js API" and "the mechanism the RSSM op set would lean on for... straight-through gradients" as
one of two options) is what `straightThroughEstimator()` actually uses — confirmed present and
working on all three packages. Implementation note beyond the stopGradient swap: naive
finite-difference gradient-checking cannot validate a straight-through estimator against its *own*
forward pass, since that forward value (the hard sample) is piecewise-constant in the logits by
construction — `(f(x+e) - f(x-e))/2e` is identically zero regardless of correctness. The
finite-difference check in `test/model/rssm.test.ts` instead targets `softmax(logits)` directly
(what the estimator's backward pass is defined to reproduce), with a companion test asserting the
naive check *is* identically zero, so the reason for the indirection is pinned down rather than
just asserted in a comment. `notes/rssm-vs-ssm-implementation-robustness.md` §5's gradient-check
recommendation is satisfied by this test, but its "needs only `tf.stopGradient`" framing (§4 of that
note) inherits the same correction.

## 8. Week-3 stack-validation spike (2026-07-21): multi-step BPTT crashes, likely the ADR-0002 decision-5 hard kill criterion tripping

**[high, self_checked, execution-confirmed]** Extending the standalone `straightThroughEstimator`
gradient-check (§7's 2026-07-20 entry) to a full training-step — `RSSMCell.step()` chained with
`.prior()` across **two or more** timesteps, differentiated via `tf.variableGrads` — throws
`Argument tensors passed to stack must be a Tensor[] or TensorLike[]` from inside `tfjs-layers`'
own RNN backward-pass code, not this repo's. A **single** timestep (`step()` + `.prior()` once, no
chaining) differentiates correctly and passes its finite-difference check
(`test/model/rssm.test.ts`); the crash appears exactly at ≥2 chained steps.

**Root-caused, not just observed**: built minimal repros (outside this repo's code, plain
`tf.layers.gruCell` + `tf.layers.rnn`) to isolate the trigger. A *control* case — the same RNN
layer instance called twice in one differentiation trace, with the second call's hidden-state
input linked to the first only via `initialState` — differentiates fine. A *treatment* case —
identical setup, except the second call's **feature input** also derives from the first call's
output (exactly `RSSMCell.step()`'s `[onehot(action), z_{t-1}]` pattern, since `z_{t-1}` comes from
`.prior()` applied to the previous step's hidden state) — reproduces the crash. So this isn't an
unlucky test shape: **any** real multi-step differentiable rollout through this cell hits it, by
construction of what the RSSM recurrence needs to do.

**[high, self_checked, primary-source]** This matches a long-standing, apparently still-open
upstream tfjs-layers bug, not something new to this pin: `tensorflow/tfjs#1529` (opened 2019,
tfjs 1.0.1 — training an LSTM on a **single-timestep** sequence throws this exact error; a
maintainer-assigned issue with a reported workaround of using ≥2 timesteps *per `.apply()` call*)
and `tensorflow/tfjs#3550` (tfjs 2.0.0, same error, `model.fit()` on an LSTM stack). Fetched both
issues directly (not just search snippets, per this note's own primary-source standard).
Confirmed via direct execution that the identical error string still reproduces on this project's
pinned `4.22.0`, seven years after #1529 was filed — this is not a "will presumably get fixed
soon" situation.

**Why the known workaround (≥2 timesteps per `.apply()` call) doesn't directly apply here**: that
workaround presumes the whole sequence is knowable *before* calling the RNN layer once. RSSM's
recurrence isn't: `z_{t-1}` (the input needed for step `t`) is itself computed *from* step `t-1`'s
hidden state via the prior/posterior dense head, so the sequence can't be assembled up front and
handed to a single multi-timestep `.apply()` call the way an ordinary externally-driven RNN can.

**Read against ADR-0002 decision 5**: "if end-to-end gradient correctness (vs. finite differences)
... fails on this stack ... tripping it is a human decision point ... not a unilateral loop pivot."
This run's finding is exactly that failure — end-to-end (multi-step) gradient correctness doesn't
just have precision issues, it crashes outright. Per `loop/GOAL.md`'s explicit instruction for this
case, no custom-autograd fallback was attempted here; this is raised as a "Decisions needed" item
in the 2026-07-21 stand-up instead. `test/model/rssm.test.ts` now has a test that pins the crash
down (`assert.throws`, not `assert.ok`) so a future tfjs upgrade or workaround silently "fixing" it
is caught, not missed.

**Steps/sec** (the spike's other half, `experiments/2026-07-21-week3-stack-spike/`): measured what
*is* currently computable at Arm-A dims (h=256, z=32 as an 8×4 categorical split — that split isn't
specified anywhere in the repo, picked as a plausible DreamerV3-style scale-down; flagged as an
assumption), batch 16, tfjs-node CPU, 200 timed iterations after 20 warmup: **forward-only
imagination rollout ≈372 steps/sec; a single differentiable training step (no BPTT) ≈101
steps/sec.** Both are comfortably fast enough for proposal `0001`'s ≤200K-steps/arm/seed,
overnight-CPU-run budget considered alone — but that budget assumed multi-step BPTT training would
work at all, which this entry's finding puts in question independent of raw throughput.

## 9. Week-3 spike, continued (2026-07-22): the kill criterion did not fire — `cell.call()` bypass fixes multi-step BPTT

**[high, self_checked, execution-confirmed]** Processing PR #22's review ("take option (b) first —
spike calling the GRU cell's step fn directly, bypassing the `tf.layers.rnn` wrapper... Timebox
it"): **it works.** `RSSMCell.step()` now calls `this.cell.call([recurrentInput, prevDeterministic],
{training: false})` directly — the same primitive `RNN.call()`'s own `step` closure calls internally
(`tfjs-layers/dist/layers/recurrent.js`, `RNN.call()`) — instead of wrapping the cell in
`tf.layers.rnn` and calling `.apply()` on a length-1 sequence. Chained across 2, 3, and 4 timesteps
in an ad-hoc spike script, then committed as a real test
(`test/model/rssm.test.ts`'s "chaining step()+prior()... matches finite differences"): no crash, and
the GRU cell's own `kernel`/`recurrent_kernel`/`bias` gradients match finite differences at every
step count tried.

**Root cause, precisely**: read `tfjs-layers/dist/layers/recurrent.js`'s `rnn()` helper (the function
`RNN.call()` delegates to) directly. It calls `tfc.unstack(inputs)` on the time axis
**unconditionally** — even for a length-1 sequence — and, when `needPerStepOutputs` is true (i.e.
`returnSequences: true`; not our case, but the same function), `tfc.stack()`s the per-step outputs
back together. §8's minimal repros already showed the crash needs a hidden-output-into-next-feature-
input dependency across `.apply()` calls; this session's reading of the actual source narrows it
further — the unstack/stack bookkeeping inside `rnn()` is the only plausible place a
`tf.stack`-shaped error could originate from tfjs-layers' own backward pass, and it runs on every
`RNN.call()`, sequence length notwithstanding. `GRUCell.call()` itself (`tfjs-layers/dist/layers/
recurrent.js`, class `GRUCell`) is `tf.split`/`K.dot`/activation calls only — no stack/unstack
anywhere — which is exactly why calling it directly sidesteps the bug rather than working around it.

**This supersedes §8's "likely tripped" read of ADR-0002 decision 5.** The kill criterion is: "if
end-to-end gradient correctness ... fails on this stack ... fall back to a small custom autograd."
It didn't fail — a stack-primitive-level bypass fixes it cleanly, with no new gradient math, per the
reviewer's own instruction to try this before considering the custom-autograd fallback (a). Per
PR #22's review, (a) stays not-invoked.

**One implementation wrinkle**: `GRUCell.build(inputShape)` needs the input shape as a single
`Shape` (`getExactlyOneShape` inside `GRUCell.build()`), not the `[featureShape, stateShape]` pair
`Layer.apply()` would infer if `cell.apply([input, h])` were called directly instead of
`cell.call()` — confirmed by hitting `ValueError: Expected exactly 1 Shape; got 2` in the spike
script before switching to explicitly calling `cell.build([null, recurrentInputWidth])` once
(lazily, on `step()`'s first call, mirroring how `RNN.build()` itself builds the cell before ever
calling `.call()`) and then using `cell.call()` — never `cell.apply()` — for every subsequent step.

**Caveat, not yet resolved**: `test/model/rssm.test.ts`'s multi-step finite-difference check only
validates the GRU cell's own weights (`kernel`/`recurrent_kernel`/`bias`), not `priorDense`'s —
when `prior()` is called with a `fixedHard` override (needed to make a multi-step forward pass
deterministic across the repeated calls finite-differencing requires), the sampled `z_t`'s forward
value is that fixed constant regardless of `priorDense`'s weights, so `priorDense`'s finite-
difference gradient is genuinely zero here — the same STE trap §7's 2026-07-20 entry documents,
now surfacing at the RSSMCell level instead of the standalone estimator level. Not a defect in
today's fix; just a gap in what this particular test construction can validate. `priorDense`'s own
gradient correctness is separately covered by the single-step test (which reads out `probs` instead
of chaining through `sample`).

The `tf.layers.rnn`-wrapper bug itself is still real and still upstream (tfjs-layers hasn't shipped
a fix for `tensorflow/tfjs#1529`/`#3550`) — `test/model/rssm.test.ts` keeps a standalone regression
test pinning it directly against raw `tf.layers.gruCell`/`tf.layers.rnn`, independent of `RSSMCell`,
so it stays caught if any future code in this repo reaches for the wrapper again.

## 10. Week-3 spike, closed out (2026-07-23, corrected same day per PR #24 review): multi-step BPTT throughput is measured, and it's usable — but "usable" against proposal 0001's budget is replay-ratio-dependent, not fixed yet

**[medium, self_checked, execution-confirmed]** §9 fixed multi-step gradient *correctness*; the
other half of ADR-0002 decision 5's kill criterion — "usable steps/sec" — was still an open number,
since `experiments/2026-07-21-week3-stack-spike/benchmark.ts` could only measure forward-rollout and
single-step gradient throughput while multi-step BPTT crashed. That benchmark now also sweeps
truncated-BPTT chain lengths {2, 4, 8, 16, 32} (picked to span "shorter than a plausible training
chunk" to "longer than one" — no chunk length is specified anywhere in the repo; same status as the
file's pre-existing `BATCH = 16` assumption) at Arm-A dims (h=256, 8×4 categorical latent), batch 16,
on this run's container (`tfjs-node` CPU backend).

**Correction (same day, caught in @SakkarinKt's PR #24 review):** the version of this entry first
committed today reported "env-steps/sec = chain length × gradient-steps/sec" and checked that
directly against proposal `0001`'s ≤200K-env-steps-per-arm-per-seed budget. That's wrong — it
silently assumes every one of the 16 batch rows in a gradient step is a distinct, never-replayed
environment step (replay ratio 1), which is neither stated nor a realistic training setup. The
benchmark measures `modelTimestepsPerSec = chainLength × BATCH × gradient-steps/sec` — how many
(batch row, timestep) pairs the stack actually differentiates per second — and that's a genuinely
stack-throughput number, not an environment-throughput one. Converting it to environment-steps/sec
requires a **replay ratio** (how many times each collected transition is replayed through a gradient
step, on average), which isn't fixed anywhere in this repo yet.

Confidence is `medium`, not `high`, for reasons beyond the correction itself: (1) only 30 timed
iterations per chain length (5 warmup) — an order-of-magnitude read, not a precision benchmark, given
each iteration's cost scales with chain length; (2) the single-step figure has moved across every run
of this script so far (~101/s on 2026-07-21, ~57/s and then ~90/s across two 2026-07-23 runs, all
different containers) with no change to the forward/gradient arithmetic between those dates —
container-to-container CPU variance, not a code regression. Treat every absolute number here as
this-container-this-run, not a portable hardware spec.

Raw result (`experiments/2026-07-21-week3-stack-spike/summary.json`, freshest run):

| chain length | gradient-steps/sec | modelTimestepsPerSec (chain length × batch × gradient-steps/sec) |
| --- | --- | --- |
| 2 | 58.2 | 1,862 |
| 4 | 26.1 | 1,670 |
| 8 | 14.0 | 1,789 |
| 16 | 6.7 | 1,709 |
| 32 | 3.7 | 1,900 |

`modelTimestepsPerSec` stays roughly flat across chain lengths (~1,700–1,900 this run; ~1,200 on the
initial same-day run before the container variance noted above) — expected, since total compute is
~linear in chain length and wrapper/dispatch overhead is a small, roughly constant fraction of it.
**On raw stack throughput, the kill criterion does not fire**: ~1,200–1,900 (batch row, timestep)
pairs differentiated per second is not a slow stack by any reasonable reading.

Whether that's "usable" *against proposal `0001`'s specific overnight-run budget* depends on the
replay ratio, which isn't decided: environment-steps/sec = modelTimestepsPerSec ÷ replay ratio, and
200,000 environment steps takes 200,000 ÷ (modelTimestepsPerSec ÷ replay ratio) seconds. Using this
run's ~1,800/sec as a round figure:

| replay ratio | env-steps/sec | time for 200K env steps |
| --- | --- | --- |
| 16 | ~113 | ~30 min |
| 512 | ~3.5 | ~16 h |

Both endpoints are plausible training configurations in the literature (low-replay on-policy-ish
setups vs. high-replay-ratio sample-efficient setups), and they bracket "comfortably inside an
overnight run" and "right at the edge of one" respectively. **This is a note, not an ADR edit**:
still `self_checked`, awaiting the same human-signoff treatment §9's finding got via the PR #23
review before any of it is folded into the formal ADR-0002 text — and per the correction above, the
right formal-ADR wording should say "stack throughput is usable; the overnight-budget check is
contingent on a replay ratio not yet chosen," not "the budget check passes."

Caveat, unchanged from the original entry: this is throughput on synthetic fixed actions/targets with
no real environment stepping, data loading, replay-buffer sampling, or loss computation beyond the
toy `probs`-weighted sum used here — it bounds the RSSM forward/backward cost specifically, not the
full training loop's throughput once the replay buffer (human's G2 module) and the real losses
(priority 3) are wired in. Revisit once those land, and once a replay ratio is chosen.
