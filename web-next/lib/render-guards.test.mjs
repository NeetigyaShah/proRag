// node --test lib/render-guards.test.mjs   (run from web-next/)
//
// JSX renders a falsy NUMBER as a text node — `{n && <X/>}` prints a bare "0"
// when n is 0, unlike undefined/null/false which render nothing. source-chips
// guards page with `!= null` for exactly that reason. These pin the guard and
// the memo-identity assumption behind message-list.

import assert from "node:assert/strict";
import test from "node:test";

/** What React actually puts in the DOM for `{cond && <span/>}`. */
const renderShortCircuit = (cond) => {
  const out = cond && "<span/>";
  return out === false || out === null || out === undefined ? "" : String(out);
};

/** The guard source-chips.tsx now uses. */
const renderNullCheck = (page) => (page != null ? "<span/>" : "");

test("truthiness guard leaks a stray 0 into the DOM", () => {
  assert.equal(renderShortCircuit(0), "0", "this is the bug the != null guard fixes");
});

test("!= null guard renders page 0 as a real page label, not a stray digit", () => {
  assert.equal(renderNullCheck(0), "<span/>");
  assert.equal(renderNullCheck(1), "<span/>");
});

test("!= null guard still hides a missing page", () => {
  assert.equal(renderNullCheck(undefined), "");
  assert.equal(renderNullCheck(null), "");
});

// --- memo identity ----------------------------------------------------------
// memo() only skips a re-render when props are shallow-equal. page.tsx's
// patch() returns the SAME object for non-streaming messages, which is what
// makes memo worth adding; `?? []` inline would break it for sources/citedNs.

const patch = (messages, id, fn) => messages.map((m) => (m.id === id ? fn(m) : m));

test("patch preserves object identity for untouched messages", () => {
  const before = [{ id: "a", content: "hi" }, { id: "b", content: "" }];
  const after = patch(before, "b", (m) => ({ ...m, content: m.content + "x" }));

  assert.equal(after[0], before[0], "untouched message must keep identity or memo is useless");
  assert.notEqual(after[1], before[1], "the streaming message must be a new object");
});

test("a shared empty-array constant keeps identity across renders", () => {
  const NO_SOURCES = [];
  const a = undefined ?? NO_SOURCES;
  const b = undefined ?? NO_SOURCES;
  assert.equal(a, b, "same reference -> memo skips");
  assert.notEqual([], [], "inline `?? []` allocates fresh each render -> memo always misses");
});
