/**
 * Child process of the experiment harness. Not meant to be run by hand: `scripts/exp.ts` and
 * `scripts/repro.ts` spawn it (via `src/experiment/harness.ts`) with one argument, the path of a
 * request file, so that the card's `run.ts` executes in its own process. That is what lets the
 * harness enforce a wall-time budget (it kills this process) and stay free of tfjs itself.
 *
 * A thrown error here means "no result": the harness records nothing.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import type { CardRun } from "../src/experiment/card.ts";
import type { RunnerRequest, RunnerResponse } from "../src/experiment/harness.ts";
import type { TfjsInfo } from "../src/experiment/telemetry.ts";

const requestPath = process.argv[2];
if (requestPath === undefined) {
  console.error("exp-runner: expected the path of a request file. Use `npm run exp` instead.");
  process.exit(2);
}
const request = JSON.parse(readFileSync(requestPath, "utf8")) as RunnerRequest;

const cardModule = (await import(pathToFileURL(join(request.cardDir, "run.ts")).href)) as { run?: unknown };
if (typeof cardModule.run !== "function") {
  throw new Error(`${join(request.cardDir, "run.ts")} must export a function named \`run\``);
}
const result = await (cardModule.run as CardRun)({
  card: request.card,
  seeds: request.seeds,
  outDir: request.outDir,
  dependencies: request.dependencies,
  log: (line) => console.log(line),
});

// Asked after the run so a card that never touches tfjs does not pay to load it first. Where
// tfjs is not installed at all, such a card still runs and the manifest records null.
let tfjs: TfjsInfo | null = null;
try {
  const { tfjsInfo } = await import("../src/experiment/tfjsInfo.ts");
  tfjs = tfjsInfo();
} catch {
  tfjs = null;
}

const response: RunnerResponse = { result, tfjs };
writeFileSync(request.resultPath, JSON.stringify(response));
