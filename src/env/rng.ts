/**
 * Minimal seedable PRNG (mulberry32) so environment rollouts are
 * reproducible per-seed without an external dependency.
 */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Returns a float in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Returns an integer in [0, maxExclusive). */
  nextInt(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }
}

/**
 * Derives a distinct 32-bit seed from a base `seed` and a small integer
 * `salt` (e.g. an agent index), for callers that need several independent
 * `Rng` streams from one episode seed instead of one shared stream — e.g.
 * `runEpisode` (src/experiment/freeze.ts) giving each agent's `WorldModel`
 * its own stream, decoupled from the policy-action stream (PR #39 review
 * follow-up 2: a shared `Rng` interleaves world-model sampling into the
 * policy's draw sequence, so changing a model-size config that changes how
 * many draws a `WorldModel.step()` call consumes shifts every later policy
 * draw too — a model-size ablation would not be holding the environment
 * trajectory constant).
 *
 * Adds a large odd constant scaled by `salt + 1`, then runs MurmurHash3's
 * 32-bit finalizer (`fmix32`, a bijection on uint32) over the sum. The
 * non-linear finalizer is load-bearing: callers *nest* derivations
 * (experiment `deriveSeed(seed, agent)` → `WorldModel` salt 0–3 →
 * `RSSMCell` salt 0–3), and the pre-2026-09-23 plain-XOR version was
 * commutative and self-inverse under nesting — `f(f(s, a), b) === f(f(s, b), a)`
 * and `f(f(s, a), a) === s` — so agent 0's GRU input kernel shared a seed
 * with agent 1's recurrent kernel (and vice versa), and agent 0's RSSM seed
 * collapsed back to the raw experiment seed (session audit N2,
 * `reports/quality/2026-09-23-session-audit.md`). Still not a cryptographic
 * hash — none is needed, only distinct, uncorrelated-in-practice seeds for a
 * handful of small salts. Changing this changed every seeded init/sampling
 * stream from this commit on; earlier manifests stay reproducible at the
 * `gitCommit` they record.
 */
export function deriveSeed(seed: number, salt: number): number {
  let h = (seed + Math.imul(0x9e3779b9, salt + 1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}
