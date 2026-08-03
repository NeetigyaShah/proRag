// node --test --experimental-strip-types lib/fx-settings.test.mjs  (from web-next/)
//
// fxDuration is the pacing contract shared by the monster modal and the radar
// drawer: every animation scales from it, so the speed math must be exact.
import assert from "node:assert/strict";
import test from "node:test";

import { fxDuration, isFxSpeed } from "./fx-settings.ts";

test("fxDuration: 1x keeps the base duration", () => {
  assert.equal(fxDuration(1, 900), 900);
});

test("fxDuration: faster speeds shorten proportionally", () => {
  assert.equal(fxDuration(2, 900), 450);
  assert.equal(fxDuration(3, 900), 300);
  assert.equal(fxDuration(5, 900), 180);
});

test("fxDuration: slower speeds lengthen proportionally", () => {
  assert.equal(fxDuration(0.5, 900), 1800);
});

test("fxDuration: instant collapses to zero", () => {
  assert.equal(fxDuration("instant", 900), 0);
});

test("isFxSpeed: accepts the six presets only", () => {
  for (const v of [0.5, 1, 2, 3, 5, "instant"]) assert.equal(isFxSpeed(v), true, `${v}`);
  for (const v of [0, 4, 10, "fast", "1", null, undefined]) assert.equal(isFxSpeed(v), false, `${v}`);
});
