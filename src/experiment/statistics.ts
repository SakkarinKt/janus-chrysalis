/**
 * Exact binomial significance tests under the p=0.5 null (a "sign test" —
 * used across analysis scripts to ask "is this split of positive/negative
 * or same/different outcomes more extreme than a fair coin flip would
 * produce"). First duplicated in `experiments/2026-08-27-high-visibility-preregistered/run.ts`
 * and `experiments/2026-08-27-power-aware-radius-analysis/analyze.ts` — both
 * outside `tsconfig.json`'s `include`, so this was, until now, the only code
 * computing the p-values published in `docs/proposals/0001` that CI never
 * checked (PR #58 review, PR #60 review). Extracted verbatim; behavior is
 * unchanged.
 */

/** One-sided binomial P(X >= k) for X ~ Binomial(n, 0.5). */
export function binomialUpperTail(k: number, n: number): number {
  let total = 0;
  let coefficient = 1;
  for (let i = 0; i <= n; i++) {
    if (i > 0) coefficient = (coefficient * (n - i + 1)) / i;
    if (i >= k) total += coefficient;
  }
  return total / 2 ** n;
}

/** Two-sided exact binomial p-value for X ~ Binomial(n, 0.5): 2 * min(P(X>=k), P(X<=k)), capped at 1. */
export function binomialTwoSided(k: number, n: number): number {
  const upper = binomialUpperTail(k, n);
  const lower = binomialUpperTail(n - k, n); // P(X<=k) = P(X>=n-k) by symmetry at p=0.5, valid at k=0 too (P(X<=0) = P(X>=n) = 1/2^n)
  return Math.min(1, 2 * Math.min(upper, lower));
}
