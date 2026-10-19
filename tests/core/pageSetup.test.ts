/**
 * The pages of a Mind (#803, part 1 of Page view), through the core's public interface and over the
 * Word corpus: each section's size, orientation and margins, its headers and footers with their page
 * fields, as the Word file has them; a native Mind's paper and margins by region; and none of it in
 * the Mind's content.
 */
import { readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { JSONContent } from "@tiptap/core";
import { DOMParser, type Element } from "@xmldom/xmldom";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import { readManifest, WORD_DIR } from "../../eval/word/corpus";
import type { PageFurniture, PageFurnitureRun, PageSection, PageSetup } from "../../src/core";
import { openWordPackage } from "../../src/core/documents/formats/docxPackage";
import { readPageSetup } from "../../src/core/documents/formats/docxPages";
import { createTempDataFolder, startCore } from "../helpers/core";
import { part, unzip } from "../helpers/docx";
import { connectToMind } from "../helpers/mindClient";
import { readMind } from "../helpers/minds";
import { field, paragraph, run, wordFile } from "../helpers/word";

const REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

const plain = (furniture: PageFurniture | null): string | null =>
  furniture === null
    ? null
    : furniture.paragraphs
        .map((each) =>
          each.runs
            .map((piece) => (piece.kind === "text" ? piece.text : `{${piece.field}}`))
            .join(""),
        )
        .join("\n");

/** A Word file with three sections that differ in what a section can differ in. */
function threeSections(): Buffer {
  const footer = (instructions: string) =>
    `<w:p><w:pPr><w:jc w:val="center"/></w:pPr>${run("Page ")}${field("PAGE", "1")}${run(" of ")}<w:fldSimple w:instr=" ${instructions} "><w:r><w:t>9</w:t></w:r></w:fldSimple></w:p>`;
  return wordFile({
    body: [
      paragraph("Chapter one", {
        sectPr:
          '<w:sectPr><w:headerReference w:type="default" r:id="rIdH1"/><w:headerReference w:type="first" r:id="rIdH2"/><w:footerReference w:type="default" r:id="rIdF1"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1021" w:bottom="1247" w:left="1304" w:header="567" w:footer="624" w:gutter="0"/><w:titlePg/></w:sectPr>',
      }),
      paragraph("Chapter two", {
        sectPr:
          '<w:sectPr><w:type w:val="oddPage"/><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="360"/><w:pgNumType w:start="1" w:fmt="lowerRoman"/></w:sectPr>',
      }),
      paragraph("Chapter three"),
    ].join(""),
    headers: {
      rIdH1: `<w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>Annual report</w:t></w:r></w:p>`,
      rIdH2: `<w:p>${run("Cover")}</w:p>`,
    },
    footers: {
      rIdF1: footer("NUMPAGES"),
      rIdF2: `<w:p>${field("SECTIONPAGES \\* ROMAN", "iv")}</w:p>`,
    },
    sectPr:
      '<w:footerReference w:type="default" r:id="rIdF2"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" w:header="708" w:footer="708"/>',
    rels: `<Relationship Id="rIdSettings" Type="${REL}/settings" Target="settings.xml"/>`,
    parts: [
      {
        name: "word/settings.xml",
        data: '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:evenAndOddHeaders/></w:settings>',
      },
    ],
  });
}

async function importedPageSetup(bytes: Buffer, region?: string) {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, region === undefined ? {} : { systemRegion: () => region });
  const path = join(dataDir, "Report.docx");
  await writeFile(path, bytes);
  const result = await core.importWordFile({ path, origin: "own" });
  if (!result.ok) throw new Error(`Refused: ${result.refusal}`);
  const client = await connectToMind(core, result.mind.id);
  await new Promise((resolve) => setTimeout(resolve, 200));
  return { core, mindId: result.mind.id, client };
}

describe("The pages of a Mind imported from Word", () => {
  // protects=each section's size, orientation, margins, first-page rule, page numbering and its headers and footers, with their page fields, reach the window as the file has them, and a section that names none has the section before's; fails_when=a section's size or margin is the default, a field is read as text, or an inherited footer is missing; no_claim=how pages are laid out from this, which is part 2.
  test("every section comes through as the file has it, a section that names no header or footer inheriting the one before's", async () => {
    const { core, mindId } = await importedPageSetup(threeSections());

    const setup = await core.getPageSetup(mindId);

    expect(setup.source).toBe("word");
    expect(setup.oddAndEven).toBe(true);
    const [first, second, third] = setup.sections as [PageSection, PageSection, PageSection];
    expect(setup.sections).toHaveLength(3);

    expect(first).toMatchObject({
      breakPlace: "w:sectPr#0",
      start: "nextPage",
      width: 11906,
      height: 16838,
      orientation: "portrait",
      margins: {
        top: 1134,
        right: 1021,
        bottom: 1247,
        left: 1304,
        gutter: 0,
        header: 567,
        footer: 624,
      },
      differentFirstPage: true,
      pageNumbers: { start: null, format: "decimal" },
    });
    expect(plain(first.headers.default)).toBe("Annual report");
    expect(first.headers.default?.paragraphs[0]).toMatchObject({
      align: "right",
      runs: [{ kind: "text", text: "Annual report", bold: true }],
    });
    expect(plain(first.headers.first)).toBe("Cover");
    expect(first.headers.even).toBeNull();
    const pages: PageFurnitureRun[] = [
      { kind: "text", text: "Page " },
      { kind: "field", field: "PAGE", format: null },
      { kind: "text", text: " of " },
      { kind: "field", field: "NUMPAGES", format: null },
    ];
    expect(first.footers.default?.paragraphs).toEqual([{ align: "center", runs: pages }]);

    expect(second).toMatchObject({
      breakPlace: "w:sectPr#1",
      start: "oddPage",
      width: 15840,
      height: 12240,
      orientation: "landscape",
      margins: { gutter: 360 },
      differentFirstPage: false,
      pageNumbers: { start: 1, format: "lowerRoman" },
    });
    expect(second.headers).toEqual(first.headers);
    expect(second.footers).toEqual(first.footers);

    expect(third).toMatchObject({ breakPlace: null, width: 12240, height: 15840 });
    expect(third.margins).toMatchObject({ left: 1800, right: 1800, header: 708, footer: 708 });
    expect(third.footers.default?.paragraphs).toEqual([
      { align: "left", runs: [{ kind: "field", field: "SECTIONPAGES", format: "upperRoman" }] },
    ]);
    expect(third.headers).toEqual(first.headers);
  });

  // protects=the place a section carries is the place of the section break the Mind holds, so Page view can end the section there; fails_when=the numbers drift from the importer's; no_claim=none beyond protects.
  test("a section's place is the place of its break in the Mind", async () => {
    const { core, mindId, client } = await importedPageSetup(threeSections());
    const places: string[] = [];
    const visit = (node: JSONContent) => {
      if (node.attrs?.kind === "sectionBreak" && typeof node.attrs.source === "string") {
        places.push(node.attrs.source);
      }
      for (const each of node.content ?? []) visit(each);
    };
    visit(readMind(client).toJSON());
    const setup = await core.getPageSetup(mindId);
    const ends = setup.sections.flatMap((each) => each.breakPlace ?? []);
    expect(places.length).toBeGreaterThan(0);
    expect(ends).toEqual(places);
  });

  // protects=pages are layout only: asking for them changes nothing in the Mind; fails_when=the Mind's document changes; no_claim=none beyond protects.
  test("asking for the pages leaves the Mind's content as it was", async () => {
    const { core, mindId, client } = await importedPageSetup(threeSections());
    const before = Y.encodeStateAsUpdate(client.doc);

    await core.getPageSetup(mindId);
    await core.getPageSetup(mindId);

    expect(Y.encodeStateAsUpdate(client.doc)).toEqual(before);
  });
});

describe("The pages of a native Mind", () => {
  // protects=a Mind that wasn't imported takes the native Word export's paper by region and its margins: A4 where the region isn't one that prints on Letter; fails_when=Letter in Europe or A4 in the US, or margins other than the export's; no_claim=the export itself, which its own tests cover.
  test.each([
    ["GB", "a4", 11906, 16838],
    ["US", "letter", 12240, 15840],
  ] as const)(
    "in %s it is %s, with one inch margins, in one section",
    async (region, _paper, width, height) => {
      const dataDir = await createTempDataFolder();
      const core = startCore(dataDir, { systemRegion: () => region });
      const mind = await core.createMind({ title: "Notes" });

      const setup: PageSetup = await core.getPageSetup(mind.id);

      expect(setup.source).toBe("native");
      expect(setup.oddAndEven).toBe(false);
      expect(setup.sections).toEqual([
        {
          breakPlace: null,
          start: "nextPage",
          width,
          height,
          orientation: "portrait",
          margins: {
            top: 1440,
            right: 1440,
            bottom: 1440,
            left: 1440,
            gutter: 0,
            header: 720,
            footer: 720,
          },
          differentFirstPage: false,
          pageNumbers: { start: null, format: "decimal" },
          headers: { default: null, first: null, even: null },
          footers: { default: null, first: null, even: null },
        },
      ]);
    },
  );

  test("a Mind that doesn't exist has no pages", async () => {
    const core = startCore(await createTempDataFolder());
    await expect(core.getPageSetup("no-such-mind")).rejects.toThrow();
  });
});

/** What the file says of a section, read with another parser than the one under test. */
function expectedSections(xml: string) {
  const document = new DOMParser().parseFromString(xml.replace(/^\uFEFF/, ""), "text/xml");
  const body = document.getElementsByTagName("w:body")[0] as Element;
  const sectPrs: Element[] = [];
  for (let node = body.firstChild; node; node = node.nextSibling) {
    if (node.nodeType !== 1) continue;
    const element = node as Element;
    if (element.tagName === "w:p") {
      const pPr = Array.from(element.childNodes).find(
        (each) => (each as Element).tagName === "w:pPr",
      );
      const found = pPr
        ? Array.from(pPr.childNodes).find((each) => (each as Element).tagName === "w:sectPr")
        : undefined;
      if (found) sectPrs.push(found as Element);
    }
  }
  const last = Array.from(body.childNodes).find((each) => (each as Element).tagName === "w:sectPr");
  if (last) sectPrs.push(last as Element);
  return sectPrs.map((sectPr) => {
    const own = (name: string) =>
      Array.from(sectPr.childNodes).filter(
        (each) => (each as Element).tagName === name,
      ) as Element[];
    const number = (element: Element | undefined, attribute: string, fallback: number) => {
      const value = element?.getAttribute(attribute);
      return value ? Number(value) : fallback;
    };
    const size = own("w:pgSz")[0];
    const margins = own("w:pgMar")[0];
    const titlePg = own("w:titlePg")[0];
    return {
      width: number(size, "w:w", 12240),
      height: number(size, "w:h", 15840),
      margins: {
        top: number(margins, "w:top", 1440),
        left: number(margins, "w:left", 1440),
        header: number(margins, "w:header", 720),
        footer: number(margins, "w:footer", 720),
      },
      differentFirstPage:
        !!titlePg && !["0", "false", "off"].includes(titlePg.getAttribute("w:val") ?? "1"),
      headerReferences: own("w:headerReference").map((each) => each.getAttribute("r:id")),
      footerReferences: own("w:footerReference").map((each) => each.getAttribute("r:id")),
    };
  });
}

const fieldsIn = (furniture: PageFurniture | null) =>
  (furniture?.paragraphs ?? []).flatMap((each) =>
    each.runs.flatMap((piece) => (piece.kind === "field" ? [piece.field] : [])),
  );

/** The fields a part's XML names, counted from its text with a pattern. */
function fieldsNamedIn(xml: string): string[] {
  const named = [
    ...xml.matchAll(/<w:instrText[^>]*>([^<]*)<\/w:instrText>/g),
    ...xml.matchAll(/w:instr="([^"]*)"/g),
  ].map((each) => each[1] ?? "");
  return named.flatMap((instruction) => {
    const first = instruction.trim().split(/\s+/)[0]?.toUpperCase() ?? "";
    return ["PAGE", "NUMPAGES", "SECTIONPAGES"].includes(first) ? [first] : [];
  });
}

describe("The Word corpus's pages", () => {
  const files = readManifest().map((entry) => entry.file);

  // protects=for every file of the corpus, the sections, sizes, margins, first-page rules and the header and footer parts each section names come through as the file has them, and the page fields in each are counted as its text names them; fails_when=a section is missed or has another size or margin, a part named by a section is missing, or a page field is read as text; no_claim=files outside the corpus.
  test.each(files)(
    "%s: every section's size, margins and furniture are the file's",
    async (file) => {
      const bytes = readFileSync(join(WORD_DIR, file));
      const pkg = await openWordPackage(bytes);
      const setup = await readPageSetup(pkg);
      const files = unzip(bytes);
      const expected = expectedSections(part(files, pkg.part));

      expect(setup.sections).toHaveLength(expected.length);
      const relationships = new Map(
        [...part(files, "word/_rels/document.xml.rels").matchAll(/<Relationship\b[^>]*>/g)].map(
          (each) => {
            const tag = each[0];
            return [
              /Id="([^"]*)"/.exec(tag)?.[1] ?? "",
              /Target="([^"]*)"/.exec(tag)?.[1] ?? "",
            ] as const;
          },
        ),
      );
      for (const [index, want] of expected.entries()) {
        const got = setup.sections[index] as PageSection;
        expect(got.width, `section ${index + 1} width`).toBe(want.width);
        expect(got.height, `section ${index + 1} height`).toBe(want.height);
        expect(got.margins.top).toBe(want.margins.top);
        expect(got.margins.left).toBe(want.margins.left);
        expect(got.margins.header).toBe(want.margins.header);
        expect(got.margins.footer).toBe(want.margins.footer);
        expect(got.differentFirstPage).toBe(want.differentFirstPage);
        for (const [kind, references, resolved] of [
          ["header", want.headerReferences, got.headers],
          ["footer", want.footerReferences, got.footers],
        ] as const) {
          const targets = references.flatMap((id) => {
            const target = relationships.get(id ?? "");
            return target ? [`word/${target.replace(/^\/?(word\/)?/, "")}`] : [];
          });
          // A section that names parts has them: their fields are the ones its default, first and even hold.
          if (targets.length === 0) continue;
          const heldFields = [resolved.default, resolved.first, resolved.even]
            .flatMap(fieldsIn)
            .sort();
          const namedFields = targets
            .flatMap((target) => (files.has(target) ? fieldsNamedIn(part(files, target)) : []))
            .sort();
          // The section may also hold a part it inherited, which adds its fields.
          for (const name of namedFields) {
            expect(heldFields, `${kind} fields of section ${index + 1}`).toContain(name);
          }
        }
      }
    },
    60_000,
  );
});
