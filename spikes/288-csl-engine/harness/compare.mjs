// Formats the 30 references with every style in both engines and compares them.
// usage: node compare.mjs <work> <pkgDir> <outDir>
// <work> is the folder setup.sh prepared (styles/, locales/, citeproc.js); <pkgDir> is build.sh's output.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { loadCSL, render as citeprocRender } from "./citeproc-run.mjs";

const [work, pkgDir, outDir] = process.argv.slice(2);
// The fixtures moved to the unit tests, which run them through every shipped style (#289).
const fixtures = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../tests/fixtures/citation-styles",
);
const require = createRequire(import.meta.url);
const { CslEngine } = require(path.join(pkgDir, "node", "csl_spike.js"));

const STYLES = [
  ["apa", "APA 7th", "apa.csl"],
  ["mla", "MLA 9th", "modern-language-association.csl"],
  ["chicago-notes", "Chicago 18th, notes and bibliography", "chicago-notes-bibliography.csl"],
  ["chicago-ad", "Chicago 18th, author-date", "chicago-author-date.csl"],
  ["ieee", "IEEE", "ieee.csl"],
  ["vancouver", "Vancouver (NLM Citing Medicine)", "nlm-citation-sequence.csl"],
  ["harvard", "Harvard (Cite Them Right 12th)", "harvard-cite-them-right.csl"],
  ["gbt2015-num", "GB/T 7714—2015 numeric", "china-national-standard-gb-t-7714-2015-numeric.csl"],
  ["gbt2015-ad", "GB/T 7714—2015 author-date", "china-national-standard-gb-t-7714-2015-author-date.csl"],
  ["gbt2025-num", "GB/T 7714—2025 numeric", "china-national-standard-gb-t-7714-2025-numeric.csl"],
  ["gbt2025-ad", "GB/T 7714—2025 author-date", "china-national-standard-gb-t-7714-2025-author-date.csl"],
];

const stylesDir = path.join(work, "styles");
const localesDir = path.join(work, "locales");
const refsJson = fs.readFileSync(path.join(fixtures, "references.json"), "utf8");
const citesJson = fs.readFileSync(path.join(fixtures, "citations.json"), "utf8");
const refs = JSON.parse(refsJson);
const cites = JSON.parse(citesJson);
const localeXml = ["en-US", "en-GB", "zh-CN"].map((l) =>
  fs.readFileSync(path.join(localesDir, `locales-${l}.xml`), "utf8"),
);
const CSL = loadCSL(path.join(work, "citeproc.js"));

const ENTITIES = { "&amp;": "&", "&#38;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&#x27;": "'", "&nbsp;": " ", "&#160;": " " };
const decode = (s) => s.replace(/&(amp|lt|gt|quot|nbsp|#38|#39|#x27|#160);/g, (m) => ENTITIES[m]);

// Markup reduced to what a reader sees: italic, bold, small caps, super- and subscript.
function canonicalHtml(html) {
  let s = html;
  s = s.replace(/<span style="font-style: ?italic;?">/g, "<i>");
  s = s.replace(/<span style="font-weight: ?bold;?">/g, "<b>");
  s = s.replace(/<span style="font-variant: ?small-caps;?">/g, "<sc>");
  s = s.replace(/<span style="vertical-align: ?super;?">/g, "<sup>");
  s = s.replace(/<span style="vertical-align: ?sub;?">/g, "<sub>");
  s = s.replace(/<span style="font-style: ?normal;?">/g, "<n>");
  // Close tags: map every </span> to the tag it opened, using a stack.
  const out = [];
  const stack = [];
  const re = /<(\/?)([a-z0-9]+)([^>]*)>|[^<]+/g;
  let m;
  while ((m = re.exec(s))) {
    if (!m[2]) {
      out.push(m[0]);
      continue;
    }
    const [_, close, tag] = m;
    const keep = ["i", "b", "sc", "sup", "sub", "n"];
    if (!close) {
      stack.push(tag);
      if (keep.includes(tag)) out.push(`<${tag}>`);
      if (tag === "em") out.push("<i>");
      if (tag === "strong") out.push("<b>");
    } else {
      const open = stack.pop();
      if (keep.includes(open)) out.push(`</${open}>`);
      if (open === "em") out.push("</i>");
      if (open === "strong") out.push("</b>");
    }
  }
  // Adjacent runs with the same formatting read the same: <sup>[</sup><sup>1</sup> is <sup>[1</sup>.
  return decode(out.join(""))
    .replace(/<\/(i|b|sc|sup|sub|n)><\1>/g, "")
    .replace(/<(i|b|sc|sup|sub|n)><\/\1>/g, "");
}

const nfc = (s) => s.normalize("NFC");
const squash = (s) => s.replace(/\s+/g, " ").trim();

function compareStrings(a, b) {
  if (nfc(a.trim()) === nfc(b.trim())) return "same";
  if (squash(nfc(a)) === squash(nfc(b))) return "whitespace";
  return "differs";
}

// Token diff (LCS on words) to list a difference concretely.
function wordDiff(a, b) {
  const A = a.split(/(\s+)/).filter((t) => t.trim());
  const B = b.split(/(\s+)/).filter((t) => t.trim());
  const dp = Array.from({ length: A.length + 1 }, () => new Array(B.length + 1).fill(0));
  for (let i = A.length - 1; i >= 0; i--)
    for (let j = B.length - 1; j >= 0; j--)
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const parts = [];
  let i = 0;
  let j = 0;
  let del = [];
  let ins = [];
  const flush = () => {
    if (del.length || ins.length) parts.push(`[-${del.join(" ")}-]{+${ins.join(" ")}+}`);
    del = [];
    ins = [];
  };
  while (i < A.length || j < B.length) {
    if (i < A.length && j < B.length && A[i] === B[j]) {
      flush();
      i++;
      j++;
    } else if (j >= B.length || (i < A.length && dp[i + 1][j] >= dp[i][j + 1])) del.push(A[i++]);
    else ins.push(B[j++]);
  }
  flush();
  return parts.join(" … ");
}

function hayEntry(e) {
  return e.firstField ? `${e.firstField} ${e.text}` : e.text;
}

function cpEntryHtml(text) {
  // citeproc-js puts the citation number in a left-margin div; join it with a space.
  return text.replace(/<div class="csl-left-margin">(.*?)<\/div>/, "$1 ");
}

const results = [];
for (const [key, name, file] of STYLES) {
  const styleXml = fs.readFileSync(path.join(stylesDir, file), "utf8");
  const res = { key, name, file, text: {}, html: {} };
  for (const format of ["text", "html"]) {
    const cp = citeprocRender(CSL, styleXml, refs, cites, format, localesDir);
    let hy;
    try {
      const engine = new CslEngine(styleXml, localeXml);
      hy = JSON.parse(engine.render(refsJson, citesJson, format));
      engine.free();
    } catch (e) {
      res[format].error = String(e);
      continue;
    }
    const prep = format === "html" ? (s) => squash(canonicalHtml(s)) : (s) => s;
    const citations = cp.citations.map((c, i) => {
      const a = prep(c);
      const b = prep(hy.citations[i] ?? "");
      const verdict = compareStrings(a, b);
      return { i, verdict, citeproc: a, hayagriva: b, diff: verdict === "differs" ? wordDiff(squash(a), squash(b)) : "" };
    });
    const hyById = new Map(hy.bibliography.map((e) => [e.id, e]));
    const bibliography = cp.bibliography.map((e) => {
      const h = hyById.get(e.id);
      const a = prep(format === "html" ? cpEntryHtml(e.text) : e.text);
      const b = h ? prep(hayEntry(h)) : "(missing)";
      const verdict = compareStrings(a, b);
      return { id: e.id, verdict, citeproc: a, hayagriva: b, diff: verdict === "differs" ? wordDiff(squash(a), squash(b)) : "" };
    });
    const sameOrder =
      cp.bibliography.map((e) => e.id).join() === hy.bibliography.map((e) => e.id).join();
    res[format] = {
      citations,
      bibliography,
      sameOrder,
      citeprocOrder: cp.bibliography.map((e) => e.id),
      hayagrivaOrder: hy.bibliography.map((e) => e.id),
      hangingIndent: [cp.hangingIndent, hy.hangingIndent],
    };
  }
  results.push(res);
}

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, "results.json"), JSON.stringify(results, null, 1));

const count = (arr, v) => arr.filter((x) => x.verdict === v).length;
const lines = ["| Style | Citations same (of 34) | Bibliography entries same (of 30) | Bibliography order | Formatting (HTML) differences beyond text |", "|---|---|---|---|---|"];
for (const r of results) {
  if (r.text.error) {
    lines.push(`| ${r.name} | error: ${r.text.error} | | | |`);
    continue;
  }
  const t = r.text;
  const h = r.html;
  const fmtOnly =
    h.citations.filter((c, i) => c.verdict === "differs" && t.citations[i].verdict !== "differs").length +
    h.bibliography.filter((c, i) => c.verdict === "differs" && t.bibliography[i].verdict !== "differs").length;
  lines.push(
    `| ${r.name} | ${count(t.citations, "same")} (+${count(t.citations, "whitespace")} whitespace only) | ${count(t.bibliography, "same")} (+${count(t.bibliography, "whitespace")} whitespace only) | ${t.sameOrder ? "same" : "differs"} | ${fmtOnly} |`,
  );
}
fs.writeFileSync(path.join(outDir, "summary.md"), `${lines.join("\n")}\n`);

const detail = [];
for (const r of results) {
  if (r.text.error) continue;
  detail.push(`\n## ${r.name} (${r.file})\n`);
  for (const c of r.text.citations.filter((c) => c.verdict === "differs"))
    detail.push(`- cite ${c.i + 1}: ${c.diff}\n    - citeproc-js: ${c.citeproc}\n    - Hayagriva:   ${c.hayagriva}`);
  for (const c of r.text.bibliography.filter((c) => c.verdict === "differs"))
    detail.push(`- bib ${c.id}: ${c.diff}\n    - citeproc-js: ${squash(c.citeproc)}\n    - Hayagriva:   ${squash(c.hayagriva)}`);
  if (!r.text.sameOrder) detail.push(`- order: citeproc-js ${r.text.citeprocOrder.join(", ")}\n  - Hayagriva ${r.text.hayagrivaOrder.join(", ")}`);
  const h = r.html;
  h.citations.forEach((c, i) => {
    if (c.verdict === "differs" && r.text.citations[i].verdict !== "differs")
      detail.push(`- html cite ${i + 1}: ${c.diff}`);
  });
  h.bibliography.forEach((c, i) => {
    if (c.verdict === "differs" && r.text.bibliography[i].verdict !== "differs")
      detail.push(`- html bib ${c.id}: ${c.diff}`);
  });
}
fs.writeFileSync(path.join(outDir, "details.md"), detail.join("\n"));
console.log(lines.join("\n"));
