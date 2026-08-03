// node --test --experimental-strip-types lib/markdown.test.mjs  (from web-next/)
//
// The parser runs on every streamed token (~600 times per answer), so block
// boundaries must be stable mid-stream: an unclosed ``` at EOF is still a code
// block, and a partial table separator must not swallow the paragraph.
import assert from "node:assert/strict";
import test from "node:test";

import { parseBlocks as parse, tokenizeInline as inline } from "./markdown.ts";

test("inline: bold, italic and inline code are tokenized", () => {
  assert.deepEqual(inline("a **bold** and *italic* with `code` end"), [
    { type: "text", content: "a " },
    { type: "bold", content: "bold" },
    { type: "text", content: " and " },
    { type: "italic", content: "italic" },
    { type: "text", content: " with " },
    { type: "code", content: "code" },
    { type: "text", content: " end" },
  ]);
});

test("inline: ** binds before * so bold is never split", () => {
  const out = inline("**bold**");
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], { type: "bold", content: "bold" });
});

test("inline: unmatched asterisks stay plain text", () => {
  assert.deepEqual(inline("a lone * asterisk"), [
    { type: "text", content: "a lone * asterisk" },
  ]);
  assert.deepEqual(inline("2*3"), [{ type: "text", content: "2*3" }]);
});

test("paragraphs join wrapped lines and split on blank lines", () => {
  const blocks = parse("line one\nline two\n\nsecond para");
  assert.equal(blocks.length, 2);
  assert.deepEqual(blocks[0], { type: "paragraph", lines: ["line one", "line two"] });
  assert.deepEqual(blocks[1], { type: "paragraph", lines: ["second para"] });
});

test("fenced code blocks carry their language label", () => {
  const blocks = parse("```python\nx = 1\nprint(x)\n```");
  assert.equal(blocks.length, 1);
  assert.deepEqual(blocks[0], { type: "code", lang: "python", code: "x = 1\nprint(x)" });
});

test("an unclosed fence at EOF still parses as code", () => {
  // Streaming: the closing ``` arrives last; until then the block must not
  // flash back into paragraph prose.
  const blocks = parse("```js\nconst a = 1;");
  assert.deepEqual(blocks, [{ type: "code", lang: "js", code: "const a = 1;" }]);
});

test("unordered and ordered lists group consecutive items", () => {
  assert.deepEqual(parse("- one\n- two\n\nafter"), [
    { type: "list", ordered: false, items: ["one", "two"] },
    { type: "paragraph", lines: ["after"] },
  ]);
  assert.deepEqual(parse("1. first\n2. second"), [
    { type: "list", ordered: true, items: ["first", "second"] },
  ]);
});

test("tables parse header and rows, dropping the separator", () => {
  const blocks = parse("| Metric | Value |\n| --- | --- |\n| Cost | $10 |\n| Time | 2h |");
  assert.equal(blocks.length, 1);
  assert.deepEqual(blocks[0], {
    type: "table",
    header: ["Metric", "Value"],
    rows: [
      ["Cost", "$10"],
      ["Time", "2h"],
    ],
  });
});

test("headings carry their level", () => {
  const blocks = parse("## Results\n\nProse");
  assert.equal(blocks.length, 2);
  assert.deepEqual(blocks[0], { type: "heading", level: 2, text: "Results" });
});

test("a lone pipe line is a paragraph, not a broken table", () => {
  // No separator row follows — the table lookahead must reject it.
  assert.deepEqual(parse("| just a pipe"), [{ type: "paragraph", lines: ["| just a pipe"] }]);
});
