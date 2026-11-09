import { test } from "node:test";
import assert from "node:assert/strict";
import { CardError, findCard, listCards, parseCard } from "../../src/experiment/card.ts";
import { makeFixtureRepo, validCard } from "./harnessFixture.ts";
import { join } from "node:path";

/** The problems `parseCard` reports for `raw`, or `[]` if it accepts it. */
function problems(raw: unknown, dirName = "0001-fixture"): string[] {
  try {
    parseCard(raw, dirName);
    return [];
  } catch (error) {
    assert.ok(error instanceof CardError, `expected a CardError, got ${String(error)}`);
    return error.problems;
  }
}

test("parseCard: accepts a complete card and returns it unchanged", () => {
  const card = validCard();
  assert.deepEqual(parseCard(structuredClone(card), "0001-fixture"), card);
});

test("parseCard: the id must be four digits and equal the directory prefix", () => {
  assert.match(problems(validCard({ id: "1" })).join("\n"), /id must be four digits/);
  assert.match(problems(validCard({ id: "0002" })).join("\n"), /does not match the directory prefix "0001"/);
  assert.match(problems(validCard(), "TEMPLATE").join("\n"), /directory name must look like/);
  assert.match(problems(validCard(), "0001_Bad_Name").join("\n"), /directory name must look like/);
});

test("parseCard: reports every problem at once, not just the first", () => {
  const found = problems({ ...validCard(), title: "", kind: "exploratory", seeds: [], budget: { maxWallMinutes: 0 } });
  assert.equal(found.length, 4, found.join("\n"));
});

test("parseCard: rejects each malformed field", () => {
  const cases: Array<[Record<string, unknown>, RegExp]> = [
    [{ hypothesis: "  " }, /hypothesis must be a non-empty string/],
    [{ kind: "exploratory" }, /kind must be one of confirmatory, calibration, repro/],
    [{ config: [] }, /config must be an object/],
    [{ seeds: [1, 1] }, /seeds must not repeat/],
    [{ seeds: [1.5] }, /seeds must be non-negative integers/],
    [{ seeds: [-1] }, /seeds must be non-negative integers/],
    [{ secondaryMetrics: [""] }, /secondaryMetrics/],
    [{ decisionRule: { text: "", thresholds: {} } }, /decisionRule\.text/],
    [{ decisionRule: { text: "x", thresholds: { a: "1" } } }, /decisionRule\.thresholds/],
    [{ dependsOn: [{ card: "1", verdict: "pass" }] }, /dependsOn\[0\]\.card must be a four-digit card id/],
    [{ dependsOn: [{ card: "0001", verdict: "pass" }] }, /is this card itself/],
    [{ dependsOn: [{ card: "0000", verdict: "ok" }] }, /dependsOn\[0\]\.verdict/],
    [{ budget: {} }, /budget\.maxWallMinutes must be a positive number/],
    [{ supersedes: "0001" }, /supersedes is this card itself/],
    [{ supersedes: 7 }, /supersedes must be null or a four-digit card id/],
    [{ notAField: true }, /unknown field "notAField"/],
  ];
  for (const [override, expected] of cases) {
    assert.match(problems({ ...validCard(), ...override }).join("\n"), expected, JSON.stringify(override));
  }
});

test("parseCard: a repro card must say what it reproduces", () => {
  assert.match(problems(validCard({ kind: "repro" })).join("\n"), /config\.reproOf/);
  assert.deepEqual(problems(validCard({ kind: "repro", config: { reproOf: "legacy:some-run" } })), []);
});

test("listCards: only NNNN-slug directories, sorted by id; TEMPLATE, proposed and stray files are ignored", () => {
  const repo = makeFixtureRepo();
  try {
    repo.writeCard("0002-second", {});
    repo.writeCard("0001-first", {});
    repo.write("experiments/cards/TEMPLATE/card.json", "{ this is not even valid JSON");
    repo.write("experiments/cards/proposed/0009-idea/card.json", "{}");
    repo.write("experiments/cards/README.md", "# cards");

    const cards = listCards(join(repo.root, "experiments", "cards"));
    assert.deepEqual(cards.map((loaded) => loaded.dirName), ["0001-first", "0002-second"]);
    assert.equal(findCard(cards, "0002").dirName, "0002-second");
    assert.equal(findCard(cards, "0001-first").card.id, "0001");
    assert.throws(() => findCard(cards, "0003"), /no card "0003" \(known cards: 0001-first, 0002-second\)/);
  } finally {
    repo.dispose();
  }
});

test("listCards: a card's hash changes when its run.ts changes, and not when another card does", () => {
  const repo = makeFixtureRepo();
  try {
    repo.writeCard("0001-first", {});
    repo.writeCard("0002-second", {});
    const cardsDir = join(repo.root, "experiments", "cards");
    const [firstBefore, secondBefore] = listCards(cardsDir).map((loaded) => loaded.sha256);

    repo.write("experiments/cards/0001-first/run.ts", "export function run() { return null; }\n");
    const [firstAfter, secondAfter] = listCards(cardsDir).map((loaded) => loaded.sha256);
    assert.notEqual(firstAfter, firstBefore);
    assert.equal(secondAfter, secondBefore);
  } finally {
    repo.dispose();
  }
});

test("listCards: refuses a card without run.ts, invalid JSON, and two cards sharing an id", () => {
  const repo = makeFixtureRepo();
  const cardsDir = join(repo.root, "experiments", "cards");
  try {
    repo.write("experiments/cards/0001-no-run/card.json", JSON.stringify(validCard()));
    assert.throws(() => listCards(cardsDir), /run\.ts is missing/);
    repo.write("experiments/cards/0001-no-run/run.ts", "export function run() {}\n");
    assert.equal(listCards(cardsDir).length, 1);

    repo.writeCard("0001-duplicate", {});
    assert.throws(() => listCards(cardsDir), /id 0001 is already used by 0001-duplicate/);

    repo.write("experiments/cards/0001-duplicate/card.json", "{ nope");
    assert.throws(() => listCards(cardsDir), /card\.json is not valid JSON/);
  } finally {
    repo.dispose();
  }
});

test("listCards: a missing cards directory is an empty list", () => {
  assert.deepEqual(listCards("/nonexistent/experiments/cards"), []);
});
