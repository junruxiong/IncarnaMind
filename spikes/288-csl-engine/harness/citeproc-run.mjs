// Runs the citeproc.js that Zotero ships (the stand-in for Zotero's output),
// configured the way Zotero's Style.getCiteProc configures it.
// usage: node citeproc-run.mjs <citeproc.js> <style.csl> <references.json> <citations.json> <html|text> <localesDir>
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export function loadCSL(citeprocPath) {
  const src = fs.readFileSync(citeprocPath, "utf8");
  // citeproc.js declares a global `CSL` and exports nothing.
  return new Function(`${src}\nreturn CSL;`)();
}

export function makeEngine(CSL, styleXml, items, localesDir, format) {
  const byId = new Map(items.map((i) => [i.id, i]));
  const sys = {
    retrieveLocale: (lang) => {
      const file = path.join(localesDir, `locales-${lang}.xml`);
      return fs.existsSync(file) ? fs.readFileSync(file, "utf8") : false;
    },
    retrieveItem: (id) => byId.get(id),
  };
  const shortId = (styleXml.match(/<id>\s*http:\/\/www\.zotero\.org\/styles\/([^<\s]+)/) || [])[1];
  // Zotero turns on uppercase subtitles for APA and a few others.
  if (shortId && /^apa($|-)|^academy-of-management($|-)|^(freshwater-science)/.test(shortId)) {
    sys.uppercase_subtitles = true;
  }
  // Zotero passes the app's locale and lets the style's default-locale win.
  const engine = new CSL.Engine(sys, styleXml, "en-US", false);
  engine.setOutputFormat(format);
  engine.opt.development_extensions.wrap_url_and_doi = true;
  engine.opt.development_extensions.parse_names = false;
  return engine;
}

export function render(CSL, styleXml, items, citations, format, localesDir) {
  const engine = makeEngine(CSL, styleXml, items, localesDir, format);
  const isNote = engine.opt.xclass === "note";
  const cits = citations.map((c, i) => ({
    citationID: `c${i}`,
    citationItems: c.items.map((it) => ({ id: it.id, locator: it.locator, label: it.label })),
    properties: { noteIndex: isNote ? c.noteIndex : 0 },
  }));
  const rebuilt = engine.rebuildProcessorState(cits, format);
  const out = { citations: rebuilt.map((r) => r[2]), bibliography: [] };
  const bib = engine.makeBibliography();
  if (bib) {
    const [meta, entries] = bib;
    out.hangingIndent = !!meta.hangingindent;
    out.bibliography = meta.entry_ids.map((ids, i) => ({ id: String(ids[0]), text: entries[i] }));
  }
  return out;
}

// Run as a script (paths compared as paths, so a folder name with a space still matches).
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const [citeproc, style, refs, cites, format, localesDir] = process.argv.slice(2);
  const CSL = loadCSL(citeproc);
  const out = render(
    CSL,
    fs.readFileSync(style, "utf8"),
    JSON.parse(fs.readFileSync(refs, "utf8")),
    JSON.parse(fs.readFileSync(cites, "utf8")),
    format,
    localesDir,
  );
  console.log(JSON.stringify(out, null, 1));
}
