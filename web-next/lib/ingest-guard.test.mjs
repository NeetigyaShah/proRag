// node --test lib/ingest-guard.test.mjs   (run from web-next/)
// The concurrent-upload guard from app/page.tsx, isolated. The component wires
// ingest() to BOTH the Upload button and the drop handler, so a guard on the
// button alone does not hold.
import assert from "node:assert/strict";
import test from "node:test";

/** Mirrors ingest() in app/page.tsx: ref guard, try/catch/finally release. */
function makeIngest(post) {
  const ingestingRef = { current: false };
  const calls = [];
  const results = [];
  const ingest = async (file) => {
    if (ingestingRef.current) return;
    ingestingRef.current = true;
    calls.push(file);
    try {
      await post(file);
      results.push(`Added ${file}`);
    } catch {
      results.push(`Could not ingest ${file}`); // the real one swallows too
    } finally {
      ingestingRef.current = false;
    }
  };
  return { ingest, calls, results, ingestingRef };
}

/** An upload held open until release() — stands in for a slow POST /ingest. */
function pending() {
  let release;
  const post = () => new Promise((r) => (release = r));
  return { post, release: () => release() };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test("a drop during an in-flight upload is ignored", async () => {
  const { post, release } = pending();
  const { ingest, calls } = makeIngest(post);
  const first = ingest("a.pdf"); // via the Upload button
  await ingest("b.pdf"); // via drag-and-drop, mid-ingest
  assert.deepEqual(calls, ["a.pdf"]);
  release();
  await first;
});

test("two calls in the SAME tick collapse to one", async () => {
  // Why a ref and not the `ingesting` state: setState is async, so both calls
  // in one tick would read the stale value and both proceed.
  const { post, release } = pending();
  const { ingest, calls } = makeIngest(post);
  const both = Promise.all([ingest("a.pdf"), ingest("b.pdf")]);
  await tick();
  assert.deepEqual(calls, ["a.pdf"]);
  release();
  await both;
});

test("the next upload works once the first finishes", async () => {
  const { ingest, calls } = makeIngest(async () => {});
  await ingest("a.pdf");
  await ingest("b.pdf");
  assert.deepEqual(calls, ["a.pdf", "b.pdf"]);
});

test("a failed upload still releases the guard", async () => {
  // Without the finally, one network blip would wedge uploads until reload.
  const { ingest, calls, results, ingestingRef } = makeIngest(async () => {
    throw new Error("network down");
  });
  await ingest("a.pdf");
  assert.equal(ingestingRef.current, false, "guard must not stick after a failure");
  await ingest("b.pdf");
  assert.deepEqual(calls, ["a.pdf", "b.pdf"]);
  assert.deepEqual(results, ["Could not ingest a.pdf", "Could not ingest b.pdf"]);
});
