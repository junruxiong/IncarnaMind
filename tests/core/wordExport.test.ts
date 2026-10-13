/**
 * Exporting a Mind imported from a Word file back into that file (#105),
 * through the core's public interface: what the Mind didn't change is
 * byte for byte what the file had, edits are written into the paragraphs
 * they were made in with their formatting kept, new Citations become
 * footnotes, and what can't be kept is said before anything is saved.
 */
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { Core, Document } from "../../src/core";
import { wordFileOf, wordImportProgressOf } from "../../src/shared/wordFiles";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { unzip } from "../helpers/docx";
import { connectToMind } from "../helpers/mindClient";
import { editMind } from "../helpers/minds";
import { commented, NUMBERING, paragraph, run, table, wordFile } from "../helpers/word";

async function importFile(core: Core, bytes: Buffer, origin: "own" | "someone-else" = "own") {
  const dataDir = await createTempDataFolder();
  const path = join(dataDir, "Memo.docx");
  await writeFile(path, bytes);
  const result = await core.importWordFile({ path, origin });
  if (!result.ok) throw new Error(`Refused: ${result.refusal}`);
  const client = await connectToMind(core, result.mind.id);
  await new Promise<void>((resolve) => {
    const check = () => {
      if (wordFileOf(client.doc) && wordImportProgressOf(client.doc) === null) {
        client.doc.off("update", check);
        resolve();
      }
    };
    client.doc.on("update", check);
    check();
  });
  return { mindId: result.mind.id, client };
}

async function exported(core: Core, mindId: string) {
  const result = await core.exportMind(mindId, { format: "docx" });
  return { files: unzip(result.data), word: result.word };
}

const text = (value: string): JSONContent => ({ type: "text", text: value });

function changed(before: Map<string, Buffer>, after: Map<string, Buffer>): string[] {
  const names = new Set([...before.keys(), ...after.keys()]);
  return [...names].filter((name) => !before.get(name)?.equals(after.get(name) ?? Buffer.alloc(0)));
}

describe("a Mind imported from Word, exported back into its file", () => {
  test("with nothing changed, every entry is byte for byte the file's", async () => {
    const core = startCore(await createTempDataFolder());
    const bytes = wordFile({
      body:
        paragraph("Heading", { style: "Heading1" }) +
        paragraph(`${run("Plain and ")}${run("bold", { b: true })}`) +
        paragraph("First item", { numId: 1 }) +
        table([
          ["A", "B"],
          ["C", "D"],
        ]),
      numbering: NUMBERING,
    });
    const { mindId } = await importFile(core, bytes);
    const { files, word } = await exported(core, mindId);
    expect(changed(unzip(bytes), files)).toEqual([]);
    expect(word).toEqual({ notKept: [], unknownParts: [], keptFurniture: [] });
  });

  test("a word changed is written into its own paragraph: the rest of the part stays as it was", async () => {
    const core = startCore(await createTempDataFolder());
    const first = paragraph(
      `${run("The rent is ", { font: "Georgia" })}${run("due", { b: true, font: "Georgia" })}${run(" monthly.", { font: "Georgia" })}`,
      { paraId: "1A2B3C4D" },
    );
    const second = paragraph("Unchanged paragraph.", { paraId: "5E6F7A8B" });
    const bytes = wordFile({ body: first + second });
    const { mindId, client } = await importFile(core, bytes);
    editMind(client, (blocks) => {
      const [paragraphNode, ...rest] = blocks;
      const content = (paragraphNode?.content ?? []).map((node) =>
        node.text === " monthly." ? { ...node, text: " every month." } : node,
      );
      return [{ ...paragraphNode, content }, ...rest] as JSONContent[];
    });
    await client.settled();
    const { files } = await exported(core, mindId);
    expect(changed(unzip(bytes), files)).toEqual(["word/document.xml"]);
    const document = files.get("word/document.xml")?.toString("utf8") ?? "";
    expect(document).toContain(second);
    expect(document).toContain('<w:p w14:paraId="1A2B3C4D">');
    expect(document).toContain("every month.");
    expect(document).toContain(
      '<w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/></w:rPr><w:t xml:space="preserve"> every month.</w:t>',
    );
    expect(document).toContain(`${run("due", { b: true, font: "Georgia" })}`);
  });

  test("a new paragraph with a Citation becomes a paragraph after the one before it, the Citation a footnote", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const sources = await createTempDataFolder();
    const path = await writeSourceFile(sources, "Tides.txt", "Spring tides happen at new moon.");
    const [tides] = (await addAndProcess(core, [path])) as Document[];
    const bytes = wordFile({ body: paragraph("Intro.") + paragraph("Outro.") });
    const { mindId, client } = await importFile(core, bytes);
    editMind(client, (blocks) => [
      blocks[0] as JSONContent,
      {
        type: "paragraph",
        content: [
          text("Tides are strongest at new moon. "),
          {
            type: "citation",
            attrs: {
              passageId: "p1",
              documentId: tides?.id,
              documentName: "Tides",
              contentHash: tides?.contentHash,
              quote: "Spring tides happen at new moon.",
              check: "found",
            },
          },
        ],
      },
      blocks[1] as JSONContent,
    ]);
    await client.settled();
    const { files } = await exported(core, mindId);
    const document = files.get("word/document.xml")?.toString("utf8") ?? "";
    expect(document).toMatch(
      /Intro\.<\/w:t><\/w:r><\/w:p><w:p>.*Tides are strongest at new moon\.<\/w:t><\/w:r><w:r><w:rPr><w:rStyle w:val="FootnoteReference"\/><\/w:rPr><w:footnoteReference w:id="1"\/><\/w:r><\/w:p><w:p>/,
    );
    const footnotes = files.get("word/footnotes.xml")?.toString("utf8") ?? "";
    expect(footnotes).toContain("Tides");
    const styles = files.get("word/styles.xml")?.toString("utf8") ?? "";
    expect(styles).toContain('w:styleId="FootnoteText"');
    expect(styles).toContain('w:styleId="FootnoteReference"');
  });

  test("someone's comment on a paragraph the User deleted is said before saving", async () => {
    const core = startCore(await createTempDataFolder());
    const { inline, comment } = commented(
      5,
      "Lars Eklund",
      "Check duty rates first",
      run("Duty adds 12%."),
    );
    const bytes = wordFile({ body: paragraph("Keep.") + paragraph(inline), comments: comment });
    const { mindId, client } = await importFile(core, bytes, "someone-else");
    editMind(client, (blocks) => blocks.slice(0, 1));
    await client.settled();
    const review = await core.saveMindExport(mindId, { format: "docx" });
    expect(review).toMatchObject({
      status: "review",
      word: {
        notKept: [{ kind: "comment", author: "Lars Eklund", text: "Check duty rates first" }],
      },
    });
  });
});
