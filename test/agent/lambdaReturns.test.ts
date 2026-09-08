import { test } from "node:test";
import assert from "node:assert/strict";
import { computeLambdaReturns } from "../../src/agent/lambdaReturns.ts";

// Gate G2 spec skeleton (loop/GOAL.md priority 6, docs/explainers/0012-replay-buffer-lambda-returns-spec.md).
// computeLambdaReturns has no implementation yet — the human implements it under Gate G2's
// role-flip, with Claude reviewing. The one test below is real (not `{ todo }`): it pins the
// current, intentional throwing behavior. Every other test describes a boundary property the
// backward-recursion formula in docs/explainers/0012 implies, marked `{ todo }` so `npm test`
// stays green while nothing is implemented — flip each to a real assertion as part of implementing it.

test("computeLambdaReturns is an intentional Gate G2 stub — calling it throws", () => {
  assert.throws(
    () =>
      computeLambdaReturns({
        rewards: [1],
        values: [0],
        continues: [1],
        bootstrapValue: 0,
        gamma: 0.99,
        lambda: 0.95,
      }),
    /Gate G2 spec-only skeleton/,
  );
});

test("lambda: 0 reduces to the one-step TD target: rewards[t] + gamma*continues[t]*V_(t+1)", {
  todo: "Gate G2 — see docs/explainers/0012's recurrence",
});

test("lambda: 1 reduces to the full discounted Monte-Carlo return gated by continues", {
  todo: "Gate G2 — see docs/explainers/0012's recurrence",
});

test("continues[t] === 0 severs bootstrapping beyond step t (R_t reduces to rewards[t] alone)", {
  todo: "Gate G2 — matches the continue head's done ? 0 : 1 target convention, docs/explainers/0011",
});

test("output length equals rewards.length", { todo: "Gate G2 — see docs/explainers/0012" });

test("mismatched rewards/values/continues lengths throw rather than silently truncating", {
  todo: "Gate G2 — matches src/experiment/metrics.ts's length-check convention (e.g. driftAttributableError)",
});
