// Runs bench.mjs in fresh processes (so "cold" is really cold) and reports medians.
// usage: node bench-all.mjs <work> <pkgDir> <runsPerStyle> [--citeproc]
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const [work, pkgDir, runsArg, flag] = process.argv.slice(2);
const runs = Number(runsArg);
const here = path.dirname(fileURLToPath(import.meta.url));
const STYLES = [
  "apa.csl",
  "modern-language-association.csl",
  "chicago-notes-bibliography.csl",
  "chicago-author-date.csl",
  "ieee.csl",
  "nlm-citation-sequence.csl",
  "harvard-cite-them-right.csl",
  "china-national-standard-gb-t-7714-2015-numeric.csl",
  "china-national-standard-gb-t-7714-2015-author-date.csl",
  "china-national-standard-gb-t-7714-2025-numeric.csl",
  "china-national-standard-gb-t-7714-2025-author-date.csl",
];
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};
const engines = flag === "--citeproc" ? ["citeproc"] : ["hayagriva"];
for (const engine of engines) {
  for (const style of STYLES) {
    const samples = [];
    for (let i = 0; i < runs; i++) {
      const out = execFileSync(
        process.execPath,
        [path.join(here, "bench.mjs"), work, engine, style, pkgDir].filter(Boolean),
        { encoding: "utf8" },
      );
      samples.push(JSON.parse(out));
    }
    const keys = Object.keys(samples[0]).filter((k) => typeof samples[0][k] === "number");
    const row = { engine, style };
    for (const k of keys) row[k] = Math.round(median(samples.map((s) => s[k])) * 10) / 10;
    console.log(JSON.stringify(row));
  }
}
