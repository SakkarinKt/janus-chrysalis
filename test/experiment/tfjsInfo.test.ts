import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { tfjsInfo } from "../../src/experiment/tfjsInfo.ts";

// Needs the tfjs-node native binding, so this file runs in CI and in the loop's sandbox but not
// on a machine where tfjs-node is not installed.

test("tfjsInfo: reports the tfjs-node version pinned in package.json and a non-empty backend", () => {
  const pinned = (JSON.parse(readFileSync("package.json", "utf8")) as { dependencies: Record<string, string> }).dependencies[
    "@tensorflow/tfjs-node"
  ];
  const info = tfjsInfo();
  assert.equal(info.version, pinned);
  assert.ok(info.backend.length > 0);
});
