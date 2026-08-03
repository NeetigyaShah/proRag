// Lightweight block/inline markdown parsers for streamed answer text.
//
// Kept in a .ts module rather than beside the component so it stays testable
// (node --experimental-strip-types can't load .tsx). The typewriter re-runs
// the parse ~600 times per answer, so this stays regex-scaled, no dependencies.

export type InlineToken =
  | { type: "text"; content: string }
  | { type: "code"; content: string }
  | { type: "bold"; content: string }
  | { type: "italic"; content: string };

// Alternation order matters: ** before *, or bold content would match *.
const INLINE_RE = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*)/g;

export function tokenizeInline(text: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = INLINE_RE.exec(text))) {
    if (m.index > last) tokens.push({ type: "text", content: text.slice(last, m.index) });
    const tok = m[0];
    if (tok.startsWith("`")) tokens.push({ type: "code", content: tok.slice(1, -1) });
    else if (tok.startsWith("**")) tokens.push({ type: "bold", content: tok.slice(2, -2) });
    else tokens.push({ type: "italic", content: tok.slice(1, -1) });
    last = m.index + tok.length;
  }
  if (last < text.length) tokens.push({ type: "text", content: text.slice(last) });
  return tokens;
}

export type Block =
  | { type: "paragraph"; lines: string[] }
  | { type: "code"; lang: string; code: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "table"; header: string[]; rows: string[][] }
  | { type: "heading"; level: number; text: string };

const FENCE_RE = /^```([\w+-]*)\s*$/;
const CLOSE_FENCE_RE = /^```\s*$/;
const HEADING_RE = /^(#{1,3})\s+(.*)$/;
const LIST_ITEM_RE = /^([-*]|\d+\.)\s+(.*)$/;
const TABLE_ROW_RE = /^\|/;

const isTableSeparator = (line: string | undefined) =>
  !!line && /^\|?[\s:|-]*-\s*[\s:|-]*\|?$/.test(line.trim());

export function parseBlocks(text: string): Block[] {
  const lines = text.split(/\n/);
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // Fenced code block (unclosed fence at EOF still renders as code, so a
    // stream interrupted mid-block doesn't flash raw ``` at the reader).
    const fence = trimmed.match(FENCE_RE);
    if (fence) {
      const lang = fence[1] || "code";
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !CLOSE_FENCE_RE.test(lines[i].trim())) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++; // consume the closing fence
      blocks.push({ type: "code", lang, code: codeLines.join("\n") });
      continue;
    }

    // Heading
    const heading = trimmed.match(HEADING_RE);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      i++;
      continue;
    }

    // Table: | a | b | followed by a |---|---| separator row
    if (TABLE_ROW_RE.test(trimmed) && isTableSeparator(lines[i + 1])) {
      const rawRows: string[][] = [];
      while (i < lines.length && TABLE_ROW_RE.test(lines[i].trim())) {
        rawRows.push(
          lines[i]
            .trim()
            .replace(/^\||\|$/g, "")
            .split("|")
            .map((c) => c.trim()),
        );
        i++;
      }
      const header = rawRows[0] ?? [];
      const rows = rawRows.slice(2); // drop the separator row
      blocks.push({ type: "table", header, rows });
      continue;
    }

    // List (consecutive items of the same kind)
    const listMatch = trimmed.match(LIST_ITEM_RE);
    if (listMatch) {
      const ordered = /\d/.test(listMatch[1]);
      const items: string[] = [];
      while (i < lines.length) {
        const m = lines[i].trim().match(LIST_ITEM_RE);
        if (!m) break;
        items.push(m[2]);
        i++;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }

    // Paragraph: accumulate until a blank line or another block start. A `|`
    // line only breaks the paragraph when it actually starts a table — the
    // table branch requires a separator row on the NEXT line, so a lone pipe
    // (or a pipe mid-prose) must stay inside the paragraph, or the loop
    // breaks without advancing and spins forever.
    const para: string[] = [];
    while (i < lines.length) {
      const l = lines[i].trim();
      if (!l) break;
      const startsTable = TABLE_ROW_RE.test(l) && isTableSeparator(lines[i + 1]);
      if (FENCE_RE.test(l) || HEADING_RE.test(l) || startsTable || LIST_ITEM_RE.test(l)) break;
      para.push(lines[i]);
      i++;
    }
    blocks.push({ type: "paragraph", lines: para });
  }

  return blocks;
}
