/**
 * `npm run eval:pptx` (README.md): the PowerPoint patch spike over the pinned
 * corpus. Each deck is saved with no edit, then with text and notes edited on
 * three slides, and, with an embedded chart, with the chart's data changed;
 * a new deck is made from the template. What it checks here is what needs no
 * Office: bytes, parts and the chart's two copies of its data. CI adds the
 * schema check and LibreOffice (.github/workflows/pptx-spike.yml).
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { test } from "vitest";
import {
  child,
  descendants,
  elements,
  local,
  parseXml,
} from "../../src/core/documents/formats/xml";
import { openPackage, type ZipArchive } from "../../src/core/documents/formats/zip";
import {
  applySplices,
  endOf,
  type Splice,
  startOf,
} from "../../src/core/exports/packagePatch/text";
import {
  chartAgreement,
  chartSeries,
  setChartValues,
  workbookPart,
} from "../../src/core/exports/pptxPatch/charts";
import { Deck, PRESENTATION_RELATIONSHIPS } from "../../src/core/exports/pptxPatch/deck";
import { setNotes } from "../../src/core/exports/pptxPatch/notes";
import { setShapeText, tagShape } from "../../src/core/exports/pptxPatch/shapes";
import { addSlide } from "../../src/core/exports/pptxPatch/slides";
import { formulaCells, Workbook } from "../../src/core/exports/pptxPatch/workbook";
import { evalCache } from "../lib/cache";
import { createLog } from "../lib/log";
import { CORPUS, type CorpusDeck } from "./corpus";

const ROOT = resolve(__dirname, "../..");

/** python-pptx's default template (MIT), made in PowerPoint: one master, eleven layouts, no slides. */
const TEMPLATE = {
  url: "https://raw.githubusercontent.com/scanny/python-pptx/278b47b1dedd5b46ee84c286e77cdfb0bf4594be/src/pptx/templates/default.pptx",
  sha256: "e10cc9e120961f6bd4074a373c9c80d2a06c497157e8f4972977b7bea83a8f34",
};

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

async function fetchPinned(url: string, expected: string, cache: string): Promise<Uint8Array> {
  const file = join(cache, `${expected}.pptx`);
  if (existsSync(file)) {
    const kept = new Uint8Array(readFileSync(file));
    if (sha256(kept) === expected) return kept;
  }
  const response = await fetch(url, { signal: AbortSignal.timeout(120_000) });
  if (!response.ok) throw new Error(`${url} answered ${response.status}.`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const got = sha256(bytes);
  if (got !== expected) throw new Error(`${url} has changed: SHA-256 ${got}, pinned ${expected}.`);
  await writeFile(file, bytes);
  return bytes;
}

const sameRaw = (a: ZipArchive, b: ZipArchive, name: string) => {
  const x = a.raw(name);
  const y = b.raw(name);
  return (
    x !== undefined &&
    y !== undefined &&
    x.crc32 === y.crc32 &&
    x.size === y.size &&
    x.method === y.method &&
    Buffer.from(x.data).equals(Buffer.from(y.data))
  );
};

/** The parts of the original whose bytes differ in the copy, and those the copy lost. */
function changedParts(original: ZipArchive, copy: ZipArchive): string[] {
  return original.names().filter((name) => !copy.has(name) || !sameRaw(original, copy, name));
}

const CATEGORIES: [string, RegExp][] = [
  ["masters", /^ppt\/slideMasters\//],
  ["layouts", /^ppt\/slideLayouts\//],
  ["themes", /^ppt\/theme\//],
  ["fonts", /^ppt\/fonts\//],
  ["smartArt", /^ppt\/diagrams\//],
  ["comments", /^ppt\/(comments|commentAuthors)/],
  ["media", /^ppt\/media\//],
  ["embeddings", /^ppt\/embeddings\//],
  ["other slides", /^ppt\/slides\//],
];

/** On an edited part, the text that changed lies inside these elements and nowhere else. */
function changedOnlyInside(
  before: string,
  after: string,
  regions: readonly { from: number; to: number }[],
) {
  let prefix = 0;
  while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix])
    prefix++;
  let suffix = 0;
  while (
    suffix < before.length - prefix &&
    suffix < after.length - prefix &&
    before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) {
    suffix++;
  }
  const from = prefix;
  const to = before.length - suffix;
  return from >= to || regions.some((region) => region.from <= from && to <= region.to);
}

const EDIT = {
  en: "Revenue grew 4.2% in the third quarter, checked against the source documents.",
  zh: "第三季度收入增长 4.2%，已与原始文件核对。",
};
const NOTE = {
  en: "Speaker note written by the patch spike.",
  zh: "这条演讲者备注由补丁试验写入。",
};

interface DeckResult {
  id: string;
  name: string;
  sha256: string;
  origin: string;
  savedBy: string;
  has: string[];
  opened: boolean;
  error: string | null;
  slides: number;
  noEdit: { identical: boolean; differing: string[] } | null;
  edited: {
    slides: number[];
    shapes: number;
    notesWritten: number;
    notesSlidesMade: number;
    notesMasterMade: boolean;
    refused: string[];
    changedParts: string[];
    untouchedKept: Record<string, { parts: number; kept: number }>;
    editedSlidesOnlyInShapes: boolean;
    timingKept: boolean;
    reread: boolean;
    output: string;
  } | null;
  chart: {
    part: string;
    before: { points: number; agreeing: number };
    after: { points: number; agreeing: number; differences: string[] };
    output: string;
    error: string | null;
  } | null;
}

async function editThreeSlides(deck: Deck, original: Uint8Array, cjk: boolean) {
  const slides = await deck.slides();
  const chosen: { number: number; part: string; shape: string; paragraphs: number }[] = [];
  for (const slide of slides) {
    if (chosen.length === 3) break;
    const shapes = await deck.textShapes(slide.part);
    const shape = shapes.find(
      (each) =>
        !["dt", "ftr", "sldNum"].includes(each.placeholder?.type ?? "") &&
        each.paragraphs.some((paragraph) => paragraph.trim() !== ""),
    );
    if (shape) {
      chosen.push({
        number: slide.number,
        part: slide.part,
        shape: shape.id,
        paragraphs: shape.paragraphs.length,
      });
    }
  }
  const before = new Map<
    string,
    { text: string; regions: { from: number; to: number }[]; timing: string | null }
  >();
  const refused: string[] = [];
  let notesWritten = 0;
  let notesSlidesMade = 0;
  const hadNotesMaster = (await deck.pkg.relationshipsOf(await deck.presentationPart())).ofType(
    PRESENTATION_RELATIONSHIPS.notesMaster,
  );
  const lang = cjk ? "zh" : "en";
  for (const each of chosen) {
    const part = await deck.pkg.part(each.part);
    if (!part) continue;
    const shape = descendants(part.root, "p:sp").find((sp) =>
      elements(sp).some(
        (nv) => local(nv.name).startsWith("nv") && child(nv, "p:cNvPr")?.attrs.id === each.shape,
      ),
    );
    const txBody = shape && child(shape, "p:txBody");
    const timing = child(part.root, "p:timing");
    before.set(each.part, {
      text: part.text,
      regions: txBody ? [{ from: startOf(txBody), to: endOf(txBody) }] : [],
      timing: timing ? part.text.slice(startOf(timing), endOf(timing)) : null,
    });
    const shapes = await deck.textShapes(each.part);
    const current = shapes.find((shapeText) => shapeText.id === each.shape);
    const paragraphs = [EDIT[lang], ...(current?.paragraphs.slice(1) ?? [])];
    try {
      await setShapeText(deck, each.part, each.shape, paragraphs);
    } catch (error) {
      refused.push(
        `slide ${each.number}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const hadNotes = (await deck.slides()).find((slide) => slide.part === each.part)?.notes;
    try {
      await setNotes(deck, each.part, [NOTE[lang]]);
      notesWritten++;
      if (!hadNotes) notesSlidesMade++;
    } catch (error) {
      refused.push(
        `notes of slide ${each.number}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  const saved = await deck.save();
  const originalZip = openPackage(original);
  const savedZip = openPackage(saved);
  const changed = changedParts(originalZip, savedZip);
  const editedParts = new Set(chosen.map((each) => each.part));
  const untouchedKept: Record<string, { parts: number; kept: number }> = {};
  for (const [category, pattern] of CATEGORIES) {
    const parts = originalZip
      .names()
      .filter((name) => pattern.test(name) && !editedParts.has(name) && !name.includes("/_rels/"));
    untouchedKept[category] = {
      parts: parts.length,
      kept: parts.filter((name) => sameRaw(originalZip, savedZip, name)).length,
    };
  }
  let onlyInShapes = true;
  let timingKept = true;
  for (const [part, was] of before) {
    const now = await savedZip.readText(part);
    if (now === undefined) continue;
    if (!changedOnlyInside(was.text, now, was.regions)) onlyInShapes = false;
    const timing = child(parseXml(now), "p:timing");
    const nowTiming = timing ? now.slice(startOf(timing), endOf(timing)) : null;
    if (nowTiming !== was.timing) timingKept = false;
  }
  const reread = Deck.open(saved);
  const texts = await Promise.all(chosen.map((each) => reread.textShapes(each.part)));
  return {
    saved,
    result: {
      slides: chosen.map((each) => each.number),
      shapes: chosen.length,
      notesWritten,
      notesSlidesMade,
      notesMasterMade: !hadNotesMaster && notesSlidesMade > 0,
      refused,
      changedParts: changed,
      untouchedKept,
      editedSlidesOnlyInShapes: onlyInShapes,
      timingKept,
      reread: texts.every((shapes) => shapes.some((shape) => shape.paragraphs[0] === EDIT[lang])),
    },
  };
}

async function editChart(deck: Deck): Promise<{
  part: string;
  saved: Uint8Array | null;
  result: Omit<NonNullable<DeckResult["chart"]>, "output">;
} | null> {
  const charts = deck.pkg.zip.names().filter((name) => /^ppt\/charts\/chart\d+\.xml$/.test(name));
  for (const part of charts) {
    if (!(await workbookPart(deck, part))) continue;
    const series = await chartSeries(deck, part);
    if (!series.some((each) => each.formula && each.points.length > 0)) continue;
    let before = { points: 0, agreeing: 0 };
    try {
      const agreement = await chartAgreement(deck, part);
      before = { points: agreement.points, agreeing: agreement.agreeing };
      const values = series.map((each) => {
        const out: number[] = [];
        for (const point of each.points)
          out[point.idx] = Math.round(((point.value ?? 0) * 1.1 + 1) * 100) / 100;
        return out;
      });
      await setChartValues(deck, part, values);
      const saved = await deck.save();
      const after = await chartAgreement(Deck.open(saved), part);
      return {
        part,
        saved,
        result: {
          part,
          before,
          after: {
            points: after.points,
            agreeing: after.agreeing,
            differences: after.differences.slice(0, 5),
          },
          error: null,
        },
      };
    } catch (error) {
      return {
        part,
        saved: null,
        result: {
          part,
          before,
          after: { points: 0, agreeing: 0, differences: [] },
          error: error instanceof Error ? error.message : String(error),
        },
      };
    }
  }
  return null;
}

/** The template made 16:9 and given East Asian theme fonts, as a patch of it. */
async function bundledTemplate(bytes: Uint8Array): Promise<Uint8Array> {
  const deck = Deck.open(bytes);
  const presentation = await deck.pkg.part("ppt/presentation.xml");
  if (!presentation) throw new Error("The template has no presentation part.");
  const size = child(presentation.root, "p:sldSz");
  if (size) {
    deck.pkg.set(
      "ppt/presentation.xml",
      applySplices(presentation.text, [
        { from: startOf(size), to: endOf(size), text: '<p:sldSz cx="12192000" cy="6858000"/>' },
      ]),
    );
  }
  const widen = 12192000 / 9144000;
  for (const name of deck.pkg.zip
    .names()
    .filter((each) => /^ppt\/(slideMasters|slideLayouts)\/[^/]+\.xml$/.test(each))) {
    const part = await deck.pkg.part(name);
    if (!part) continue;
    const splices: Splice[] = [];
    for (const element of [
      ...descendants(part.root, "a:off"),
      ...descendants(part.root, "a:ext"),
    ]) {
      const key = local(element.name) === "off" ? "x" : "cx";
      const value = element.attrs[key];
      if (value === undefined) continue;
      const tag = part.text.slice(startOf(element), endOf(element));
      splices.push({
        from: startOf(element),
        to: endOf(element),
        text: tag.replace(`${key}="${value}"`, `${key}="${Math.round(Number(value) * widen)}"`),
      });
    }
    deck.pkg.set(name, applySplices(part.text, splices));
  }
  const theme = await deck.pkg.part("ppt/theme/theme1.xml");
  if (theme) {
    const splices: Splice[] = [];
    for (const [font, typeface] of [
      ["majorFont", "等线 Light"],
      ["minorFont", "等线"],
    ] as const) {
      const scheme = descendants(theme.root, `a:${font}`)[0];
      const ea = scheme && child(scheme, "a:ea");
      if (ea)
        splices.push({ from: startOf(ea), to: endOf(ea), text: `<a:ea typeface="${typeface}"/>` });
    }
    deck.pkg.set("ppt/theme/theme1.xml", applySplices(theme.text, splices));
  }
  return deck.save();
}

async function newDeckFromTemplate(template: Uint8Array): Promise<Uint8Array> {
  const deck = Deck.open(template);
  const layouts = await deck.layouts();
  const byType = (type: string) =>
    layouts.find((each) => each.type === type)?.part ?? layouts[0]?.part;
  const title = byType("title");
  const content = byType("obj");
  if (!title || !content) throw new Error("The template has no layouts.");
  const first = await addSlide(deck, title, {
    title: "Northwind Retail: Q3 review",
    body: ["Prepared from the source documents"],
  });
  const second = await addSlide(deck, content, {
    title: "What changed",
    body: [
      "Revenue grew 4.2% in the quarter",
      "Gross margin fell to 38.6%",
      "Net debt to EBITDA stood at 3.1 times",
    ],
  });
  const third = await addSlide(deck, content, {
    title: "要点",
    body: ["第三季度收入增长 4.2%", "毛利率降至 38.6%", "净债务与 EBITDA 之比为 3.1 倍"],
    language: "zh-CN",
  });
  await setNotes(deck, first, ["Opening: say where each figure comes from."]);
  await setNotes(deck, second, ["Each figure is cited in the notes of the source Mind."]);
  await setNotes(deck, third, ["中文页：数字与英文页一致。"]);
  return deck.save();
}

/** A deck with a Citation tag on its chart's frame and a note on the chart's first value cell. */
async function citationSurvival(
  bytes: Uint8Array,
): Promise<{ saved: Uint8Array; slide: string; frame: string; cell: string } | null> {
  const deck = Deck.open(bytes);
  for (const slide of await deck.slides()) {
    const own = await deck.pkg.relationshipsOf(slide.part);
    const chart = own.existing.find((each) => each.type === PRESENTATION_RELATIONSHIPS.chart);
    if (!chart) continue;
    const embedded = await workbookPart(deck, chart.target);
    const series = (await chartSeries(deck, chart.target)).find((each) => each.formula);
    if (!embedded || !series?.formula) continue;
    const part = await deck.pkg.part(slide.part);
    const frame =
      part &&
      descendants(part.root, "p:graphicFrame").find((each) =>
        descendants(each, "c:chart").some((ref) =>
          Object.entries(ref.attrs).some(
            ([key, value]) => key.endsWith(":id") && value === chart.id,
          ),
        ),
      );
    const frameId = frame && descendants(frame, "p:cNvPr")[0]?.attrs.id;
    if (!frameId) continue;
    await tagShape(deck, slide.part, frameId, { INCARNAMIND_CITATION: "spike-citation-1" });
    const { sheet, cells } = formulaCells(series.formula);
    const cell = cells[0];
    const workbookBytes = await deck.pkg.bytes(embedded);
    if (!cell || !workbookBytes) continue;
    const workbook = Workbook.open(workbookBytes);
    await workbook.addNote(sheet, cell, "IncarnaMind", "Source: spike-citation-1");
    deck.pkg.set(embedded, await workbook.save());
    return {
      saved: await deck.save(),
      slide: slide.part,
      frame: frameId,
      cell: `${sheet}!${cell}`,
    };
  }
  return null;
}

test("PowerPoint patch spike", async () => {
  const log = createLog();
  const cache = join(evalCache(), "pptx-corpus");
  await mkdir(cache, { recursive: true });
  const folder = join(
    ROOT,
    "eval",
    "results",
    `pptx-${new Date().toISOString().replace(/[:.]/g, "-")}`,
  );
  const outputs = join(folder, "outputs");
  await mkdir(outputs, { recursive: true });
  const results: DeckResult[] = [];
  for (const entry of CORPUS) {
    log(`${entry.id} ${entry.name}`);
    results.push(await runDeck(entry, cache, outputs));
  }
  const template = await bundledTemplate(await fetchPinned(TEMPLATE.url, TEMPLATE.sha256, cache));
  await writeFile(join(outputs, "template.pptx"), template);
  await writeFile(join(outputs, "new-from-template.pptx"), await newDeckFromTemplate(template));
  let survival: Awaited<ReturnType<typeof citationSurvival>> = null;
  for (const entry of CORPUS.filter((each) =>
    each.has.some((feature) => feature.startsWith("workbooks")),
  )) {
    try {
      survival = await citationSurvival(await fetchPinned(entry.url, entry.sha256, cache));
    } catch (error) {
      log(
        `${entry.id}: no Citation tag and note (${error instanceof Error ? error.message : String(error)})`,
      );
    }
    if (survival) {
      await writeFile(join(outputs, "citation-survival.pptx"), survival.saved);
      log(`Citation tag and note: ${entry.id}`);
      break;
    }
  }
  await writeFile(
    join(folder, "results.json"),
    JSON.stringify({ results, survival: survival && { ...survival, saved: undefined } }, null, 2),
  );
  log(`Results in ${folder}`);
});

async function runDeck(entry: CorpusDeck, cache: string, outputs: string): Promise<DeckResult> {
  const result: DeckResult = {
    id: entry.id,
    name: entry.name,
    sha256: entry.sha256,
    origin: entry.origin,
    savedBy: entry.savedBy,
    has: entry.has,
    opened: false,
    error: null,
    slides: 0,
    noEdit: null,
    edited: null,
    chart: null,
  };
  try {
    const original = await fetchPinned(entry.url, entry.sha256, cache);
    const untouched = Deck.open(original);
    result.slides = (await untouched.slides()).length;
    result.opened = true;
    const resaved = await untouched.save();
    const differing = changedParts(openPackage(original), openPackage(resaved));
    result.noEdit = { identical: differing.length === 0, differing };
    const cjk = entry.has.includes("cjk");
    const edited = await editThreeSlides(Deck.open(original), original, cjk);
    const output = `${entry.id}-edited.pptx`;
    await writeFile(join(outputs, output), edited.saved);
    result.edited = { ...edited.result, output };
    const chart = await editChart(Deck.open(original));
    if (chart) {
      const chartOutput = `${entry.id}-chart.pptx`;
      if (chart.saved) await writeFile(join(outputs, chartOutput), chart.saved);
      result.chart = { ...chart.result, output: chart.saved ? chartOutput : "" };
    }
  } catch (error) {
    result.error = error instanceof Error ? error.message : String(error);
  }
  return result;
}
