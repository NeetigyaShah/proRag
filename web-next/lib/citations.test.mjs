// node --test --experimental-strip-types lib/citations.test.mjs  (from web-next/)
//
// The client marker regex must accept every form prorag/chat/citations.py
// normalizes, because the server streams RAW tokens and only normalizes after
// the stream ends. If these drift apart, a citation renders as dead plain text
// while streaming and as a clickable pill after reload.
import assert from "node:assert/strict";
import test from "node:test";

import { isCitationMarker as isMarker, splitOnCitations as split } from "./citations.ts";

// One case per _DEVIATION_PATTERNS entry in citations.py, plus the canonical.
const SERVER_FORMS = ["[S1]", "(S1)", "[s1]", "[source 1]", "[SOURCE 12]", "[1]"];

test("every form the server normalizes is a clickable marker", () => {
  for (const form of SERVER_FORMS) {
    assert.ok(isMarker(form), `${form} should render as a pill, not plain text`);
  }
});

test("marker number is extracted from any form", () => {
  const n = (s) => Number(s.match(/\d+/)[0]);
  assert.equal(n("[S1]"), 1);
  assert.equal(n("(S7)"), 7);
  assert.equal(n("[source 12]"), 12);
  assert.equal(n("[3]"), 3);
});

test("prose is not mistaken for a marker", () => {
  for (const s of ["plain text", "[Sx]", "[]", "a [bracket] word", "cost was $12"]) {
    assert.ok(!isMarker(s), `${s} must stay plain text`);
  }
});

test("splitting keeps surrounding prose intact and drops empties", () => {
  const out = split("Drills are monthly [S1] and inspections quarterly (S2), per [source 3].");
  assert.deepEqual(out, [
    "Drills are monthly ",
    "[S1]",
    " and inspections quarterly ",
    "(S2)",
    ", per ",
    "[source 3]",
    ".",
  ]);
  assert.ok(!out.includes(""), "empty strings would render pointless <span/>s");
});

test("adjacent markers both survive the split", () => {
  assert.deepEqual(split("claim [S1][S2] end"), ["claim ", "[S1]", "[S2]", " end"]);
});

test("a partial marker mid-stream stays plain text", () => {
  // Tokens arrive a few chars at a time; "[S1" must not render half a pill.
  assert.deepEqual(split("the answer [S1"), ["the answer [S1"]);
});

test("the /g regex is not shared statefully across calls", () => {
  // A module-level /g object carries lastIndex between uses; identical input
  // must give identical output every time.
  const a = split("x [S1] y");
  const b = split("x [S1] y");
  assert.deepEqual(a, b);
});
