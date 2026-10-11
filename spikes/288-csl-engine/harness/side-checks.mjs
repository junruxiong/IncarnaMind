// Checks outside the 30-reference table, each run in both engines.
// usage: node side-checks.mjs <work> <pkgDir>
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { loadCSL, render as citeprocRender } from "./citeproc-run.mjs";

const [work, pkgDir] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const { CslEngine } = require(path.join(pkgDir, "node", "csl_spike.js"));
const fixtures = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const refs = JSON.parse(fs.readFileSync(path.join(fixtures, "references.json"), "utf8"));
const CSL = loadCSL(path.join(work, "citeproc.js"));
const localesDir = path.join(work, "locales");
const locales = ["en-US", "en-GB", "zh-CN"].map((l) =>
  fs.readFileSync(path.join(localesDir, `locales-${l}.xml`), "utf8"),
);
const style = (f) => fs.readFileSync(path.join(work, "styles", f), "utf8");

function both(styleXml, items, cites, format = "text", hayLocale) {
  const cp = citeprocRender(CSL, styleXml, items, cites, format, localesDir);
  let hy;
  try {
    const e = new CslEngine(styleXml, locales);
    hy = JSON.parse(e.render(JSON.stringify(items), JSON.stringify(cites), format, hayLocale));
  } catch (err) {
    hy = { error: String(err) };
  }
  return { cp, hy };
}
const show = (title, { cp, hy }) => {
  console.log(`\n### ${title}`);
  console.log("citeproc-js citations:", JSON.stringify(cp.citations));
  for (const b of cp.bibliography) console.log("citeproc-js:", b.text.trim());
  if (hy.error) return console.log("Hayagriva: error:", hy.error);
  console.log("Hayagriva citations:  ", JSON.stringify(hy.citations));
  for (const b of hy.bibliography) console.log("Hayagriva:  ", [b.firstField, typeof b.text === "string" ? b.text : JSON.stringify(b.text)].filter(Boolean).join(" "));
};
const ref = (id) => refs.find((r) => r.id === id);
const one = (ids) => ids.map((id) => ({ items: [{ id }] }));

// 1. Year suffixes follow the bibliography's order (CSL 1.0.2, disambiguate-add-year-suffix),
//    whatever order the works are cited in.
show(
  "APA: year suffixes when the later title is cited first",
  both(style("apa.csl"), [ref("en-oecd-2019a"), ref("en-oecd-2019b")], one(["en-oecd-2019b", "en-oecd-2019a"])),
);

// 2. A conference paper in edited proceedings, English and Chinese.
const paper = { ...ref("zh-jia-2011"), id: "en-paper", language: "en", title: "A paper", "container-title": "Proceedings of X", editor: [{ family: "Smith", given: "John" }], author: [{ family: "Doe", given: "Jane" }] };
show("APA: conference paper with an editor", both(style("apa.csl"), [paper, ref("zh-jia-2011")], one(["en-paper", "zh-jia-2011"])));

// 3. Rich text in fields, as Zotero exports it in CSL JSON.
const rich = { id: "rich", type: "article-journal", language: "en", title: "Growth of <i>Escherichia coli</i> at 37 °C and the CO<sub>2</sub> effect", author: [{ family: "Smith", given: "Ann" }], "container-title": "Microbiology", issued: { "date-parts": [[2020]] } };
show("Chicago author-date: rich text in a title", both(style("chicago-author-date.csl"), [rich], one(["rich"])));

// 4. Field text that looks like markup must stay text.
const hostile = { id: "hostile", type: "article-journal", title: "Cats & dogs <img src=x onerror=alert(1)>", author: [{ family: "Smith", given: "A" }], "container-title": "J & K", issued: { "date-parts": [[2020]] } };
show("APA, HTML output: field text that looks like markup", both(style("apa.csl"), [hostile], one(["hostile"]), "html"));
{
  const e = new CslEngine(style("apa.csl"), locales);
  const out = JSON.parse(e.render(JSON.stringify([hostile]), JSON.stringify(one(["hostile"])), "runs"));
  console.log("\n### APA, the binding's runs output: same input");
  console.log("Hayagriva:  ", JSON.stringify(out.bibliography[0].text));
}

// 5. The CSL-M layout the GB/T 7714—2015 styles ship commented out ("et al." for English entries).
const gb = style("china-national-standard-gb-t-7714-2015-numeric.csl");
const gbM = gb.replace(/<!-- (<layout suffix="\." locale="en">[\s\S]*?<\/layout>)\s*-->/, "$1");
if (gbM === gb) throw new Error("CSL-M layout not found");
const english = [ref("en-jumper-2021"), ref("en-kuhn-2012"), ref("zh-liu-2016")];
show("GB/T 7714—2015 numeric with its CSL-M layout uncommented", both(gbM, english, one(english.map((r) => r.id))));

// 6. The same result without CSL-M: render twice and keep each entry in its own language.
{
  const e = new CslEngine(gb, locales);
  const items = JSON.stringify(english);
  const cites = JSON.stringify(one(english.map((r) => r.id)));
  const zh = JSON.parse(e.render(items, cites, "text"));
  const en = JSON.parse(e.render(items, cites, "text", "en-US"));
  const enById = new Map(en.bibliography.map((b) => [b.id, b]));
  console.log("\n### GB/T 7714—2015 numeric, Hayagriva rendered twice, entry picked by its language");
  for (const b of zh.bibliography) {
    const lang = english.find((r) => r.id === b.id).language;
    const pick = lang.startsWith("zh") ? b : enById.get(b.id);
    console.log("Hayagriva:  ", pick.firstField, pick.text);
  }
}
