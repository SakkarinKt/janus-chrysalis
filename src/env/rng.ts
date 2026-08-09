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
 * XORs in a large odd constant scaled by `salt + 1` — mulberry32 (`Rng`'s
 * generator) has no known short-cycle or correlation issue across arbitrary
 * 32-bit seeds, and this project only ever calls this with a handful of
 * small, distinct salts (one per agent index), not a security or
 * statistical-quality requirement — so a simple, deterministic, injective-
 * enough-in-practice mix is sufficient; no cryptographic hash is needed.
 */
export function deriveSeed(seed: number, salt: number): number {
  return (seed ^ Math.imul(0x9e3779b9, salt + 1)) >>> 0;
}
