import { test } from "node:test";
import assert from "node:assert/strict";
import { ReplayBuffer } from "../../src/agent/replayBuffer.ts";

// Gate G2 spec skeleton (loop/GOAL.md priority 6, docs/explainers/0012-replay-buffer-lambda-returns-spec.md).
// ReplayBuffer has no implementation yet — the human implements it under Gate G2's role-flip, with
// Claude reviewing. The one test below is real (not `{ todo }`): it pins the current, intentional
// throwing behavior, so this file can't silently start passing on an accidental no-op stub. Every
// other test describes a behavioral property the implementation needs to satisfy, marked `{ todo }`
// so `npm test` stays green while nothing is implemented — flip each to a real assertion (against
// `ReplayBuffer`'s actual behavior, not this comment) as part of implementing it.

test("ReplayBuffer is an intentional Gate G2 stub — constructing it throws", () => {
  assert.throws(() => new ReplayBuffer({ capacity: 10, sampling: "uniform" }), /Gate G2 spec-only skeleton/);
});

test("add() then size() reflects the number of entries added, up to capacity", { todo: "Gate G2 — see docs/explainers/0012" });

test("at capacity, add() evicts an entry per the implementation's documented eviction policy", {
  todo: "Gate G2 — see docs/explainers/0012's 'Open design questions' #1 (oldest-first proposed)",
});

test("sample() only ever returns entries that were add()-ed", { todo: "Gate G2 — see docs/explainers/0012" });

test("sample() is driven by the passed-in rng, not internal state — same seed reproduces the same batch", {
  todo: "Gate G2 — mirrors test/agent/policy.test.ts's RandomPolicy rng-determinism convention",
});

test("sampling: 'uniform' draws each entry with roughly equal frequency over many samples", {
  todo: "Gate G2 — see docs/explainers/0012",
});

test("sampling: 'recency-weighted' draws more-recently-added entries more often than older ones", {
  todo: "Gate G2 — see docs/explainers/0012's 'Open design questions' #2 (recencyHalfLife)",
});

test("constructing with sampling: 'recency-weighted' and no recencyHalfLife throws at construction", {
  todo: "Gate G2 — see ReplayBufferConfig's doc comment (src/agent/replayBuffer.ts)",
});
