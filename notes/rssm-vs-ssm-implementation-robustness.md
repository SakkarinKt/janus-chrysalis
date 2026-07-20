# RSSM vs. SSM/Mamba implementation-robustness note

Status: **short note**, satisfies the gate `notes/adr-0002-js-ml-stack.md` §7 sets on the world-model
cell ("do not write the RSSM world-model cell until the short RSSM-vs-SSM/Mamba
(`notes/papers/drama-2024.md`) implementation-robustness note lands"). Not an ADR — ADR-0002 itself
still needs human signoff. This note only answers the narrower question the gate asks: given we have
to *hand-implement* either backbone in TF.js (§4 of `adr-0002-js-ml-stack.md` found no existing port
of either), which one carries less risk of a silent, undetected gradient/correctness bug corrupting
the freeze-intervention prediction-error measurement that proposal `0001`'s Arm-A milestone depends
on?

All claims below are `self_checked` via `WebSearch` this session (`WebFetch` was not attempted this
run — see `notes/adr-0002-js-ml-stack.md` §5 for the domain-specific-403 finding from a prior run;
not re-tested here since the sources needed were reachable via search snippets and GitHub search
results directly).

## 1. No parallel/associative scan primitive exists in TF.js

**[high, self_checked]** Mamba's forward pass is not just "an RNN with extra gating" — its
efficiency claim rests on computing the recurrence via a **parallel (associative) scan** rather than
a sequential loop. `tfp.math.scan_associative` (TensorFlow Probability) and `jax.lax.associative_scan`
(JAX) exist; Keras 3 has `keras.ops.associative_scan`. Searching specifically for a TF.js equivalent
turned up nothing — no `tf.scanAssociative`/`tf.associativeScan` in `tfjs-core`'s API surface, and no
third-party TF.js package providing one. This means an SSM/Mamba backbone in this stack would need
either (a) a hand-built parallel scan op (a novel, numerically-sensitive kernel with no TF.js
reference to check against), or (b) falling back to a sequential `tf.scan`-style step-by-step loop,
which is implementable with ordinary ops but forfeits the architecture's own efficiency rationale —
at that point it stops being "the Mamba architecture" in any meaningful sense and just becomes another
custom recurrent cell, with none of Drama's laptop-trainability evidence (`notes/papers/drama-2024.md`)
still applying (that evidence was measured against the real parallel-scan implementation, not a
sequential fallback).

## 2. Mamba's own reference scan is reported as non-trivial even by its authors' community

**[medium, self_checked]** `state-spaces/mamba` issue #936 ("Difficulty understanding selective scan
in Mamba implementation") shows third parties struggling to follow the reference selective-scan
implementation. Secondary write-ups (Goomba Lab / Tri Dao's own Mamba-2 blog series) describe the
original Mamba-1 scan as sufficiently complex that Mamba-2's main design goal was a *simpler*
algorithm; they also note the original scan's CUDA kernel was limited to small state expansion
(N=16) because it doesn't use tensor cores — i.e., the "reference" implementation is itself a
hardware-specific, hand-tuned kernel, not a portable algorithm description. That is the artifact a
from-scratch TF.js port would be reimplementing blind, with no existing JS/TF.js code to diff
against and no upstream maintainers who've hardened a JS version.

## 3. RSSM's recurrence maps onto a mature, standard TF.js op

**[medium, self_checked]** DreamerV3-style RSSM's deterministic path is GRU-style recurrence
(LayerNorm GRU, per multiple independent PyTorch/JAX reimplementations —
`danijar/dreamerv3` (reference JAX), `NM512/dreamerv3-torch`, `Eclectic-Sheep/sheeprl`,
`A-SHOJAEI/dreamerv3-robotic-control`). TF.js ships `tf.layers.gru` as a standard, documented layer.
Targeted search for GRU-specific gradient bugs in `tensorflow/tfjs` (as opposed to
`tensorflow/tensorflow`, a different, Python-only repo not relevant to this stack) found none beyond
the WebGPU-specific issue #8590 already covered in `notes/adr-0002-js-ml-stack.md` §2 — and that
issue is already excluded from the training path by §7's backend decision (CPU/`tfjs-node` or WebGL,
not WebGPU). No CPU/WebGL-specific GRU gradient bug turned up in this session's search. This doesn't
prove `tf.layers.gru` is bug-free on CPU/WebGL — absence of a found issue is not absence of the
issue — but it's a materially different evidence picture than "no primitive exists at all" (§1).

## 4. The categorical-latent straight-through path needs no custom gradient at all

**[medium, self_checked]** RSSM's stochastic path (categorical latent, DreamerV3-style) needs a
straight-through estimator: forward pass uses a hard (discrete) sample, backward pass uses the soft
(Gumbel-softmax-style) gradient. The standard implementation trick —
`hard + stopGradient(soft - hard)` (forward-evaluates to `hard` since the stop-gradient term is a
no-op on the forward value; backward gradient flows only through `soft`) — needs only `tf.stopGradient`,
which is a native, stable TF.js primitive (confirmed present in `notes/adr-0002-js-ml-stack.md` §4).
This narrows the genuinely novel, hand-verified custom-gradient surface for an RSSM cell to
essentially nothing beyond composing existing stable ops — contrary to this note's own working
assumption going in (framed in `adr-0002-js-ml-stack.md` §7 as "a subtle gradient bug in a hand-rolled
recurrent cell," which turns out to be less about custom-gradient code and more about correctly
composing standard ops).

## 5. Verification method, backbone-agnostic

**[medium, self_checked]** TF.js has no first-class numerical-gradient-check utility built into
`tfjs-core` equivalent to Python TF's `tf.test.compute_gradient`. But `tf.grad()`/`tf.grads()`
analytic gradients can still be checked against a small hand-written finite-difference helper in the
test suite (perturb one input dimension by `epsilon`, compare `(f(x+e) - f(x-e)) / 2e` against the
analytic gradient at that dimension) — ordinary test code, no new dependency. This applies to
whichever backbone gets built and should be part of the world-model cell's test coverage regardless
of the verdict below; it is the concrete artifact that turns "we didn't find a bug via literature
search" into "we tested for one directly."

## Verdict

**[medium, self_checked]** The two backbones are not symmetric in implementation-robustness risk for
this stack. RSSM's risk is concentrated in composing mature, standard ops (`tf.layers.gru`,
`tf.stopGradient`) that TF.js already ships and tests — the remaining risk is ordinary
implementation-bug risk, mitigable by §5's gradient-check harness. SSM/Mamba's risk starts one level
lower: the core primitive (parallel/associative scan) doesn't exist in TF.js at all, so the milestone
would be building and trusting a novel numerical kernel with no upstream TF.js reference and no
existing JS port to check against, *before* any gradient-correctness question is even reachable — and
a sequential fallback forfeits the architecture's own evidence base (Drama's laptop-scale results were
measured against the real parallel-scan implementation).

This supports proceeding with `notes/adr-0002-js-ml-stack.md` §7's working default — **RSSM
implemented from TF.js primitives** — for the Arm-A milestone's world-model cell, satisfying this
note's gate. Recommendation for whoever writes that cell: include a finite-difference gradient-check
test (§5) for the custom-composed straight-through path as part of that increment's test coverage,
not as a follow-up. SSM/Mamba stays a documented future comparison (per
`notes/adr-0002-js-ml-stack.md` §6) once either TF.js gains an associative-scan primitive upstream, or
building one is scoped as its own deliberate increment — not bundled into the milestone's critical
path.

**Not decided by this note** (needs human judgment if surfaced): whether this verdict is strong
enough to actually unblock cell-writing, or whether the human wants a closer look (e.g. an actual
finite-difference smoke test against a toy GRU + straight-through composition) before treating the
gate as cleared. Flagged as an "Assumption made" in this run's stand-up rather than a "Decision
needed," since the note's own conclusion is a bounded default the human can veto.

## Correction (2026-07-20): §4's "needs only `tf.stopGradient`" doesn't hold for this pinned version

**[high, self_checked]** §4 above states the straight-through path "needs only `tf.stopGradient`,
which is a native, stable TF.js primitive." Writing `src/model/rssm.ts`'s
`straightThroughEstimator()` against the actual pinned `@tensorflow/tfjs-node@4.22.0` found this
false: `tf.stopGradient` is not a function on that package (nor `tfjs-core`/`tfjs`, same version) —
confirmed by running it, not by search. `tf.customGrad` (available and confirmed working) is the
actual mechanism used; see `notes/adr-0002-js-ml-stack.md`'s 2026-07-20 entry for the full
correction and implementation notes. §4's core conclusion — the straight-through path needs no
*novel numerical kernel*, just composing existing stable ops — still holds; only the specific op
name was wrong.
