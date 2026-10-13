/**
 * The round-trip check for the Word corpus (#105): a file imported as a Mind
 * (#104) and exported back into itself.
 *
 * - Unchanged, the export is the file byte for byte, every entry.
 * - After a scripted edit (a word changed in the first paragraph that has
 *   one, and a paragraph with a Citation added after its Block), the edit is
 *   in the file, the Citation is a footnote, every entry but those an edit
 *   needs is byte for byte the file's, every other paragraph and table of the
 *   body is as it was, the package passes the checks Word's repair prompt is
 *   about, nothing is reported as lost, and the file imports again with the edit.
 */
import * as Y from "yjs";
import { createElement } from "../../src/core/answers/blocks";
import { MIND_CONTENT_FIELD, type WordRefusalReason } from "../../src/core/api";
import { readWordDocument, readWordStructure } from "../../src/core/documents/formats/docxImport";
import {
  openWordPackage,
  reviewOf,
  WordRefusedError,
} from "../../src/core/documents/formats/docxPackage";
import { child, elements, parseXml, walk } from "../../src/core/documents/formats/xml";
import { openPackage } from "../../src/core/documents/formats/zip";
import { exportLabels } from "../../src/core/exports/labels";
import { exportWordPatch } from "../../src/core/exports/wordPatch";
import { recordSources } from "../../src/core/wordImports";
import type { NodeJSON } from "../../src/shared/blockJson";
import { citationStyle } from "../../src/shared/citationStyles";
import {
  recordWordImportProgress,
  startWordImport,
  wordSourceOf,
} from "../../src/shared/wordFiles";
import { inspectPackage } from "./corpus";

const REFUSED: ReadonlySet<WordRefusalReason> = new Set([
  "legacy",
  "macros",
  "encrypted",
  "strict",
]);

/** The footnote the added Citation becomes, as the export names its source. */
const SOURCE = "Round-trip check source, p. 1";
const ADDED = "Added by the round-trip check.";
const EDITED = "Roundtripped";

export interface ExportCheck {
  ok: boolean;
  detail: string;
  /** The export with the scripted edit, for opening in LibreOffice. */
  exported?: Uint8Array;
}

/** A Mind as the import writes one: the Blocks, where each came from, and the file. */
async function mindOf(bytes: Uint8Array): Promise<Y.Doc> {
  const pkg = await openWordPackage(bytes);
  const structure = await readWordStructure(pkg, await reviewOf(pkg));
  const doc = new Y.Doc();
  const blocks = doc.getXmlFragment(MIND_CONTENT_FIELD);
  doc.transact(() => {
    startWordImport(
      doc,
      { name: "corpus.docx", sha256: "", size: bytes.byteLength, part: pkg.part },
      "own",
      structure.report,
      structure.blocks.length,
    );
    const created = structure.blocks.map(createElement);
    blocks.insert(0, created);
    for (const element of created) recordSources(doc, element, structure);
    recordWordImportProgress(doc, structure.blocks.length, structure.blocks.length);
  });
  return doc;
}

async function exportOf(bytes: Uint8Array, doc: Y.Doc) {
  return exportWordPatch({
    bytes,
    blocks: doc.getXmlFragment(MIND_CONTENT_FIELD),
    readOptions: {
      includeQuestions: false,
      footnote: () => ({ source: SOURCE, unverified: false }),
      trimBeforeCitation: true,
      trimBeforeFootnote: true,
    },
    labels: exportLabels("en"),
    style: citationStyle("numbered"),
    bibliography: false,
    bibliographyHeading: "References",
    sources: { language: "en", documents: [] },
    formatCitations: () => Promise.reject(new Error("Numbered needs no formatting.")),
    withPictures: async (blocks) => blocks,
  });
}

/** Each entry of a package, by name. */
async function entriesOf(bytes: Uint8Array): Promise<Map<string, Uint8Array>> {
  const zip = openPackage(bytes);
  const entries = new Map<string, Uint8Array>();
  for (const name of zip.names()) entries.set(name, (await zip.read(name)) ?? new Uint8Array());
  return entries;
}

const same = (a: Uint8Array | undefined, b: Uint8Array | undefined) =>
  a !== undefined && b !== undefined && Buffer.from(a).equals(Buffer.from(b));

/**
 * The scripted edit: a word of the first imported paragraph that has one
 * changed, and a paragraph with a Citation added after the Block it is in.
 * Returns the place of the paragraph changed, or null if none has a word.
 */
function edit(doc: Y.Doc): string | null {
  const blocks = doc.getXmlFragment(MIND_CONTENT_FIELD);
  let changed: string | null = null;
  let after = blocks.length;
  const visit = (element: Y.XmlElement, top: number): boolean => {
    if (element.nodeName === "paragraph" || element.nodeName === "heading") {
      const id = element.getAttribute("id");
      const source = typeof id === "string" ? wordSourceOf(doc, id) : null;
      for (const text of element.toArray()) {
        if (!(text instanceof Y.XmlText) || !source) continue;
        const word = /[\p{L}]{4,}|[\p{Script=Han}]{2,}/u.exec(text.toString());
        if (!word) continue;
        text.delete(word.index, word[0].length);
        text.insert(word.index, EDITED);
        changed = source.n;
        after = top + 1;
        return true;
      }
    }
    return element.toArray().some((each) => each instanceof Y.XmlElement && visit(each, top));
  };
  doc.transact(() => {
    blocks.toArray().some((each, index) => each instanceof Y.XmlElement && visit(each, index));
    const paragraph = new Y.XmlElement("paragraph");
    const citation = new Y.XmlElement("citation");
    for (const [key, value] of Object.entries({
      passageId: "round-trip",
      documentId: "round-trip",
      documentName: "Round-trip check source",
      quote: "A quote.",
      check: "found",
    })) {
      citation.setAttribute(key, value);
    }
    paragraph.insert(0, [new Y.XmlText(`${ADDED} `), citation]);
    blocks.insert(after, [paragraph]);
  });
  return changed;
}

/** The top-level body elements of a main part, as written, and where the element at `place` is. */
async function bodyPieces(bytes: Uint8Array, place: string | null) {
  const pkg = await openWordPackage(bytes);
  const document = await readWordDocument(pkg, await reviewOf(pkg));
  const body = child(document.root, "w:body");
  const edited = place ? document.places.get(place) : undefined;
  return (body ? elements(body) : []).map((element) => ({
    place: document.places.placeOf(element) ?? "",
    xml: document.text.slice(element.start ?? 0, element.end ?? 0),
    holdsEdit:
      edited !== undefined &&
      (element.start ?? 0) <= (edited.start ?? 0) &&
      (edited.end ?? 0) <= (element.end ?? 0),
  }));
}

function* textsOf(node: NodeJSON): Generator<string> {
  if (node.text) yield node.text;
  for (const each of node.content ?? []) yield* textsOf(each);
}

export async function checkExport(bytes: Uint8Array): Promise<ExportCheck> {
  let doc: Y.Doc;
  try {
    doc = await mindOf(bytes);
  } catch (error) {
    if (error instanceof WordRefusedError && REFUSED.has(error.reason)) {
      return { ok: true, detail: `refused, as it should be: ${error.reason}` };
    }
    return {
      ok: false,
      detail: `didn't import: ${error instanceof Error ? error.message : error}`,
    };
  }
  const original = await entriesOf(bytes);
  const pkg = await openWordPackage(bytes);

  try {
    const unchanged = await exportOf(bytes, doc);
    const differ = [...(await entriesOf(unchanged.data))].filter(
      ([name, data]) => !same(data, original.get(name)),
    );
    if (differ.length > 0) {
      return {
        ok: false,
        detail: `unchanged, the export still differs in ${differ.map(([name]) => name).join(", ")}`,
      };
    }

    const edited = edit(doc);
    const result = await exportOf(bytes, doc);
    const exported = await entriesOf(result.data);
    // Word's repair prompt: nothing the file didn't already have.
    const before = new Set((await inspectPackage(bytes)).problems);
    const problems = (await inspectPackage(result.data)).problems.filter(
      (each) => !before.has(each),
    );
    if (problems.length > 0)
      return {
        ok: false,
        detail: `the export: ${problems.slice(0, 2).join("; ")}`,
        exported: result.data,
      };
    if (result.report.notKept.length > 0) {
      return {
        ok: false,
        detail: `reported as not kept: ${result.report.notKept.map((each) => each.kind).join(", ")}`,
        exported: result.data,
      };
    }

    // Only the parts the edit needs are written.
    const allowed = new Set([pkg.part, "[Content_Types].xml"]);
    for (const name of exported.keys()) {
      if (/^word\/(footnotes\d*|styles)\.xml$|^word\/_rels\//.test(name)) allowed.add(name);
    }
    const touched = [...exported.keys()].filter(
      (name) => !same(exported.get(name), original.get(name)),
    );
    const unexpected = touched.filter((name) => !allowed.has(name));
    if (unexpected.length > 0)
      return { ok: false, detail: `also wrote ${unexpected.join(", ")}`, exported: result.data };
    const missing = [...original.keys()].filter((name) => !exported.has(name));
    if (missing.length > 0)
      return { ok: false, detail: `left out ${missing.join(", ")}`, exported: result.data };

    // Every other paragraph and table of the body is as it was, in order, and so is everything
    // before the body: the root's namespaces, mc:Ignorable among them, are never pruned.
    const main = new TextDecoder().decode(exported.get(pkg.part));
    const head = (xml: string) => xml.slice(0, xml.search(/<(?:[\w.-]+:)?body[\s>]/));
    if (head(main) !== head(new TextDecoder().decode(original.get(pkg.part)))) {
      return { ok: false, detail: "the document's start changed", exported: result.data };
    }
    let at = 0;
    let kept = 0;
    for (const piece of await bodyPieces(bytes, edited)) {
      if (piece.holdsEdit) continue;
      const found = main.indexOf(piece.xml, at);
      if (found < 0) {
        return { ok: false, detail: `${piece.place} isn't as it was`, exported: result.data };
      }
      at = found + piece.xml.length;
      kept++;
    }

    // The edit is there, and the Citation is a footnote.
    // The text of every w:t, without recursion: a table can nest thousands deep.
    const shown = [...walk(parseXml(main))]
      .filter((element) => element.name === "w:t")
      .map((element) => element.children.filter((part) => typeof part === "string").join(""))
      .join("");
    if (edited && !shown.includes(EDITED))
      return { ok: false, detail: "the changed word isn't in the file", exported: result.data };
    if (!shown.includes(ADDED))
      return { ok: false, detail: "the added paragraph isn't in the file", exported: result.data };
    const footnotes = [...exported].find(([name]) => /^word\/footnotes\d*\.xml$/.test(name));
    if (!footnotes || !new TextDecoder().decode(footnotes[1]).includes(SOURCE)) {
      return { ok: false, detail: "the Citation isn't a footnote", exported: result.data };
    }
    const again = await readWordStructure(
      await openWordPackage(result.data),
      await reviewOf(await openWordPackage(result.data)),
    );
    const text = JSON.stringify(again.blocks.map((block) => [...textsOf(block)].join("")));
    if (!text.includes(ADDED) || (edited && !text.includes(EDITED))) {
      return {
        ok: false,
        detail: "the export doesn't import with the edit",
        exported: result.data,
      };
    }
    return {
      ok: true,
      detail: `${edited ? "a word changed, " : ""}a paragraph and a footnote added; ${touched.length} parts written, ${kept} body elements as they were`,
      exported: result.data,
    };
  } catch (error) {
    return {
      ok: false,
      detail: `export failed: ${error instanceof Error ? error.message : error}`,
    };
  } finally {
    doc.destroy();
  }
}

/**
 * Edits of every kind the export writes, on one file: a paragraph deleted, a
 * word made bold, a link added, a heading's level changed, a list item and a
 * table row added, a paragraph moved, and a new list and table at the end.
 * The export must pass the package checks, lose nothing but what was deleted,
 * and import again with the new text.
 */
export async function checkEdits(bytes: Uint8Array): Promise<ExportCheck> {
  let doc: Y.Doc;
  try {
    doc = await mindOf(bytes);
  } catch (error) {
    if (error instanceof WordRefusedError && REFUSED.has(error.reason)) {
      return { ok: true, detail: `refused, as it should be: ${error.reason}` };
    }
    return {
      ok: false,
      detail: `didn't import: ${error instanceof Error ? error.message : error}`,
    };
  }
  try {
    const blocks = doc.getXmlFragment(MIND_CONTENT_FIELD);
    const all = (kind: string) => {
      const found: Y.XmlElement[] = [];
      const visit = (element: Y.XmlElement) => {
        if (element.nodeName === kind) found.push(element);
        for (const each of element.toArray()) if (each instanceof Y.XmlElement) visit(each);
      };
      for (const each of blocks.toArray()) if (each instanceof Y.XmlElement) visit(each);
      return found;
    };
    const texts = (element: Y.XmlElement) =>
      element.toArray().filter((each): each is Y.XmlText => each instanceof Y.XmlText);
    doc.transact(() => {
      const top = () =>
        blocks.toArray().filter((each): each is Y.XmlElement => each instanceof Y.XmlElement);
      const paragraphs = top().filter((each) => each.nodeName === "paragraph");
      if (paragraphs.length >= 3) blocks.delete(top().indexOf(paragraphs[1] as Y.XmlElement), 1);
      const bolded = all("paragraph").find((each) => texts(each).some((text) => text.length > 3));
      const text = bolded && texts(bolded).find((each) => each.length > 3);
      text?.format(0, 3, { bold: true });
      const linked = all("paragraph").at(-1);
      const end = linked && texts(linked).at(-1);
      end?.insert(end.length, " (see the site)", {
        link: { href: "https://example.com/round-trip" },
      });
      const heading = all("heading")[0];
      if (heading)
        heading.setAttribute(
          "level",
          String((Number(heading.getAttribute("level")) % 6) + 1) as never,
        );
      const item = all("listItem")[0];
      if (item?.parent instanceof Y.XmlElement) {
        const list = item.parent;
        list.insert(list.toArray().indexOf(item) + 1, [
          createElement({
            type: "listItem",
            content: [{ type: "paragraph", content: [{ type: "text", text: "A new item." }] }],
          }),
        ]);
      }
      const table = all("table")[0];
      const firstRow = table?.toArray()[0];
      if (table && firstRow instanceof Y.XmlElement) {
        const cells = firstRow.toArray().length;
        table.push([
          createElement({
            type: "tableRow",
            content: Array.from({ length: cells }, () => ({
              type: "tableCell",
              content: [{ type: "paragraph", content: [{ type: "text", text: "A new row." }] }],
            })),
          }),
        ]);
      }
      const last = top()
        .filter((each) => each.nodeName === "paragraph")
        .at(-1);
      if (last && top().length > 2) {
        const json = last.toJSON();
        void json;
        const at = top().indexOf(last);
        const moved = last.clone();
        blocks.delete(at, 1);
        blocks.insert(0, [moved]);
      }
      blocks.push([
        createElement({
          type: "bulletList",
          content: [
            {
              type: "listItem",
              content: [{ type: "paragraph", content: [{ type: "text", text: "A new list." }] }],
            },
          ],
        }),
        createElement({
          type: "table",
          content: [
            {
              type: "tableRow",
              content: [
                {
                  type: "tableHeader",
                  content: [{ type: "paragraph", content: [{ type: "text", text: "New" }] }],
                },
                {
                  type: "tableHeader",
                  content: [{ type: "paragraph", content: [{ type: "text", text: "Table" }] }],
                },
              ],
            },
          ],
        }),
      ]);
    });
    const result = await exportOf(bytes, doc);
    const before = new Set((await inspectPackage(bytes)).problems);
    const problems = (await inspectPackage(result.data)).problems.filter(
      (each) => !before.has(each),
    );
    if (problems.length > 0)
      return {
        ok: false,
        detail: `the export: ${problems.slice(0, 2).join("; ")}`,
        exported: result.data,
      };
    const lost = result.report.notKept.filter(
      (each) => each.kind !== "comment" && each.kind !== "change",
    );
    if (lost.length > 0)
      return {
        ok: false,
        detail: `reported as not kept: ${lost.map((each) => `${each.kind} "${each.text}"`).join(", ")}`,
        exported: result.data,
      };
    const again = await readWordStructure(
      await openWordPackage(result.data),
      await reviewOf(await openWordPackage(result.data)),
    );
    const text = JSON.stringify(again.blocks.map((block) => [...textsOf(block)].join("")));
    for (const expected of ["A new list.", "New", "Table"]) {
      if (!text.includes(expected))
        return { ok: false, detail: `"${expected}" doesn't import again`, exported: result.data };
    }
    return { ok: true, detail: "edits of every kind written", exported: result.data };
  } catch (error) {
    return {
      ok: false,
      detail: `export failed: ${error instanceof Error ? (error.stack ?? error.message) : error}`,
    };
  } finally {
    doc.destroy();
  }
}
