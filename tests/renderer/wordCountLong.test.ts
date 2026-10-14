/**
 * The count of a long Mind (#812): after an edit only the Blocks it changed
 * are counted again, a first count can go on a slice at a time, and Block by
 * Block the counts are the same as counting the Mind's text in one go, on the
 * whole Word corpus, in every basis.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { getSchema } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { EditorState } from "@tiptap/pm/state";
import { describe, expect, test } from "vitest";
import { readManifest, WORD_DIR } from "../../eval/word/corpus";
import { readWordStructure } from "../../src/core/documents/formats/docxImport";
import {
  openWordPackage,
  reviewOf,
  WordRefusedError,
} from "../../src/core/documents/formats/docxPackage";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import {
  countableText,
  createMindCounter,
  type MindCount,
} from "../../src/renderer/src/editor/wordCountText";
import { COUNT_BASES, countFor, countText } from "../../src/shared/wordCount";
import { longContract } from "../helpers/longContract";

const schema = getSchema(noteExtensions());

/** The Mind a Word file imports as; null for a file the import refuses on purpose. */
async function mindOf(bytes: Uint8Array): Promise<ProseMirrorNode | null> {
  try {
    const pkg = await openWordPackage(bytes);
    const { blocks } = await readWordStructure(pkg, await reviewOf(pkg));
    return schema.nodeFromJSON({ type: "doc", content: blocks });
  } catch (error) {
    if (error instanceof WordRefusedError) return null;
    throw error;
  }
}

/** The count as it was before #812: the Mind's text, and the selection's, counted in one go. */
function inOneGo(doc: ProseMirrorNode, selection: { from: number; to: number }): MindCount {
  const selected =
    selection.from === selection.to ? null : countText(countableText(doc, selection));
  return {
    total: countText(countableText(doc)),
    selection: selected && selected.characters > 0 ? selected : null,
  };
}

const NONE = { from: 0, to: 0 };

describe("the 304-page contract", () => {
  test("after one character is typed, only the Block it went in is counted again, and the count is exact", async () => {
    const doc = await mindOf(longContract());
    if (!doc) throw new Error("The contract was refused.");
    expect(doc.childCount).toBeGreaterThan(2800);
    const counter = createMindCounter();
    expect(counter.count(doc, NONE)).toEqual(inOneGo(doc, NONE));
    expect(counter.blocksRead()).toBe(doc.childCount);

    // A letter typed at the start of the middle Block, on page 150 or so.
    let at = 0;
    doc.forEach((_block, offset, index) => {
      if (index === Math.floor(doc.childCount / 2)) at = offset + 1;
    });
    const state = EditorState.create({ doc });
    const typed = state.apply(state.tr.insertText("x", at)).doc;
    expect(counter.count(typed, NONE)).toEqual(inOneGo(typed, NONE));
    expect(counter.blocksRead()).toBe(1);

    // Selecting everything reads nothing again.
    const all = { from: 0, to: typed.content.size };
    expect(counter.count(typed, all)).toEqual(inOneGo(typed, all));
    expect(counter.blocksRead()).toBe(0);
  });

  test("a first count stopped after each slice goes on where it stopped, and ends exact", async () => {
    const doc = await mindOf(longContract());
    if (!doc) throw new Error("The contract was refused.");
    const counter = createMindCounter();
    const slice = 500;
    let calls = 0;
    let counted: MindCount | null = null;
    while (!counted) {
      calls++;
      let read = 0;
      counted = counter.count(doc, NONE, () => ++read >= slice);
      expect(counter.blocksRead()).toBeLessThanOrEqual(slice);
    }
    expect(calls).toBe(Math.ceil(doc.childCount / slice));
    expect(counted).toEqual(inOneGo(doc, NONE));
  });
});

const corpus = readManifest().filter((entry) => entry.file.endsWith(".docx"));

describe("the Word corpus", () => {
  test.each(corpus.map((entry) => [entry.file, entry.language] as const))(
    "%s (%s): Block by Block, every basis counts as the text in one go, for the whole and a selection",
    async (file) => {
      const doc = await mindOf(readFileSync(join(WORD_DIR, file)));
      if (!doc) return;
      const size = doc.content.size;
      const selection = { from: Math.floor(size / 3), to: Math.floor((size * 2) / 3) };
      const counted = createMindCounter().count(doc, selection);
      const expected = inOneGo(doc, selection);
      expect(counted).toEqual(expected);
      for (const basis of COUNT_BASES) {
        expect(counted && countFor(counted.total, basis)).toBe(countFor(expected.total, basis));
      }
    },
    60_000,
  );

  test("the corpus has English and Chinese writing to count", async () => {
    const counted = async (language: string) => {
      const entry = corpus.find((each) => each.language === language);
      if (!entry) throw new Error(`No ${language} file in the corpus.`);
      const doc = await mindOf(readFileSync(join(WORD_DIR, entry.file)));
      return doc ? createMindCounter().count(doc, NONE)?.total : undefined;
    };
    expect((await counted("en"))?.words).toBeGreaterThan(0);
    expect((await counted("zh"))?.cjk).toBeGreaterThan(0);
  });
});
