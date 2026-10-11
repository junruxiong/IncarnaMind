// Times formatting the 30 references (34 citations and the bibliography) in one process:
// Hayagriva returns formatted runs (what the app would use), citeproc-js returns HTML.
// usage: node bench.mjs <work> <hayagriva|citeproc> <styleFile> [pkgDir]
// Prints one JSON line: cold timings (first call in a fresh process) and warm timings.
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const [work, engineName, styleFile, pkgDir] = process.argv.slice(2);
const fixtures = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const t0 = performance.now();
const read = (p) => fs.readFileSync(path.join(work, p), "utf8");
const styleXml = read(`styles/${styleFile}`);
const refsJson = fs.readFileSync(path.join(fixtures, "references.json"), "utf8");
const citesJson = fs.readFileSync(path.join(fixtures, "citations.json"), "utf8");
const WARM_RUNS = 50;
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

if (engineName === "hayagriva") {
  // The web build, as a renderer or worker would load it: bytes in, no Node-specific glue.
  const glue = await import(path.join(pkgDir, "web", "csl_spike.js"));
  const bytes = fs.readFileSync(path.join(pkgDir, "web", "csl_spike_bg.wasm"));
  const locales = ["en-US", "en-GB", "zh-CN"].map((l) => read(`locales/locales-${l}.xml`));
  const t1 = performance.now();
  const wasm = glue.initSync({ module: bytes });
  const t2 = performance.now();
  const engine = new glue.CslEngine(styleXml, locales);
  const t3 = performance.now();
  engine.render(refsJson, citesJson, "runs");
  const t4 = performance.now();
  const warm = [];
  for (let i = 0; i < WARM_RUNS; i++) {
    const a = performance.now();
    engine.render(refsJson, citesJson, "runs");
    warm.push(performance.now() - a);
  }
  // A fresh engine per render (parsing the style each time), warm module.
  const warmWithParse = [];
  for (let i = 0; i < 20; i++) {
    const a = performance.now();
    const e = new glue.CslEngine(styleXml, locales);
    e.render(refsJson, citesJson, "runs");
    e.free();
    warmWithParse.push(performance.now() - a);
  }
  console.log(
    JSON.stringify({
      engine: "hayagriva",
      style: styleFile,
      loadGlue: t1 - t0,
      instantiate: t2 - t1,
      parseStyle: t3 - t2,
      firstRender: t4 - t3,
      coldTotal: t4 - t0,
      warmRender: median(warm),
      warmParseAndRender: median(warmWithParse),
      wasmMemoryMB: wasm.memory.buffer.byteLength / 2 ** 20,
    }),
  );
} else {
  const { loadCSL, render } = await import("./citeproc-run.mjs");
  const refs = JSON.parse(refsJson);
  const cites = JSON.parse(citesJson);
  const localesDir = path.join(work, "locales");
  const t1 = performance.now();
  const CSL = loadCSL(path.join(work, "citeproc.js"));
  const t2 = performance.now();
  render(CSL, styleXml, refs, cites, "html", localesDir);
  const t3 = performance.now();
  const warm = [];
  for (let i = 0; i < WARM_RUNS; i++) {
    const a = performance.now();
    render(CSL, styleXml, refs, cites, "html", localesDir);
    warm.push(performance.now() - a);
  }
  console.log(
    JSON.stringify({
      engine: "citeproc-js",
      style: styleFile,
      loadEngine: t2 - t1,
      firstRender: t3 - t2,
      coldTotal: t3 - t0,
      // citeproc-js parses the style inside every render here (new CSL.Engine each time).
      warmParseAndRender: median(warm),
    }),
  );
}
