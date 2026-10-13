/**
 * What a Word export says for review (#107), through the core's public
 * interface: changes still waiting go in as IncarnaMind's tracked changes,
 * AI text and Citations whose check failed get IncarnaMind's comments, each
 * as its option says; and the tracked changes the writers share
 * (src/core/exports/wordReview.ts) mark runs, paragraphs and rows as Word does.
 */
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { Core, Document, ExportMindOptions } from "../../src/core";
import { WordIds } from "../../src/core/exports/wordml";
import { trackBlocks, trackInline } from "../../src/core/exports/wordReview";
import { recordProposalRequest } from "../../src/shared/proposals";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { acceptedParagraphs, checkWellFormed, unzip } from "../helpers/docx";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { resolveChanges, writeMind } from "../helpers/minds";

const ASKED = "2026-10-12T14:30:00.000Z";

async function mindWith(
  core: Core,
  blocks: JSONContent[],
): Promise<{ id: string; client: MindClient }> {
  const mind = await core.createMind({ title: "" });
  const client = await connectToMind(core, mind.id);
  recordProposalRequest(client.doc, "r1", {
    source: "edit-with-ai",
    providerId: "anthropic",
    modelId: "claude",
    createdAt: ASKED,
  });
  writeMind(client, blocks);
  await client.settled();
  return { id: mind.id, client };
}

const text = (value: string, ...marks: JSONContent["marks"][]): JSONContent => {
  const all = marks.flat().filter((mark): mark is NonNullable<typeof mark> => mark !== undefined);
  return { type: "text", text: value, ...(all.length > 0 && { marks: all }) };
};
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });
const adds = (id: string) => [{ type: "proposalInsert", attrs: { id, request: "r1" } }];
const removes = (id: string) => [{ type: "proposalDelete", attrs: { id, request: "r1" } }];
const ai = [{ type: "generated" }];

async function word(core: Core, mindId: string, options: Partial<ExportMindOptions> = {}) {
  const { data } = await core.exportMind(mindId, { format: "docx", ...options });
  const files = unzip(data);
  const part = (name: string) => files.get(name)?.toString("utf8") ?? "";
  for (const [name, content] of files) {
    if (name.endsWith(".xml") || name.endsWith(".rels")) checkWellFormed(content.toString("utf8"));
  }
  return { files, document: part("word/document.xml"), part };
}

async function tides(core: Core): Promise<Document> {
  const sources = await createTempDataFolder();
  const path = await writeSourceFile(sources, "Tides.txt", "Spring tides happen at new moon.");
  const [document] = await addAndProcess(core, [path]);
  return document as Document;
}

const citation = (document: Document, check: string, quote: string): JSONContent => ({
  type: "citation",
  attrs: {
    passageId: "p1",
    documentId: document.id,
    documentName: "Tides",
    contentHash: document.contentHash,
    pageFrom: 1,
    pageTo: 1,
    quote,
    check,
  },
});

describe("changes still waiting", () => {
  test("a replacement waiting to be accepted is IncarnaMind's deletion then insertion, dated when it was asked for", async () => {
    const core = startCore(await createTempDataFolder());
    const { id } = await mindWith(core, [
      paragraph(
        text("Rent is due "),
        text("monthly", removes("c1")),
        text("weekly", adds("c1")),
        text("."),
      ),
    ]);

    const { document } = await word(core, id);

    expect(document).toContain(
      '<w:del w:id="0" w:author="IncarnaMind" w:date="2026-10-12T14:30:00Z"><w:r><w:delText xml:space="preserve">monthly</w:delText></w:r></w:del><w:ins w:id="1" w:author="IncarnaMind" w:date="2026-10-12T14:30:00Z"><w:r><w:t xml:space="preserve">weekly</w:t></w:r></w:ins>',
    );
    expect(acceptedParagraphs(document)).toEqual(["Rent is due weekly."]);
  });

  test("a paragraph a change adds or removes is marked whole: its runs and its paragraph mark", async () => {
    const core = startCore(await createTempDataFolder());
    const added = {
      type: "paragraph",
      attrs: { proposal: { id: "c2", request: "r1", kind: "insert" } },
    };
    const removed = {
      type: "paragraph",
      attrs: { proposal: { id: "c3", request: "r1", kind: "delete" } },
    };
    const { id } = await mindWith(core, [
      paragraph(text("Kept.")),
      { ...removed, content: [text("Gone.")] },
      { ...added, content: [text("New.")] },
      paragraph(text("Last.")),
    ]);

    const { document } = await word(core, id);

    expect(document).toMatch(
      /<w:p><w:pPr><w:rPr><w:del w:id="\d+" w:author="IncarnaMind" w:date="2026-10-12T14:30:00Z"\/><\/w:rPr><\/w:pPr><w:del [^>]*><w:r><w:delText xml:space="preserve">Gone.<\/w:delText><\/w:r><\/w:del><\/w:p>/,
    );
    expect(document).toMatch(
      /<w:p><w:pPr><w:rPr><w:ins w:id="\d+" w:author="IncarnaMind"[^>]*\/><\/w:rPr><\/w:pPr><w:ins [^>]*><w:r><w:t xml:space="preserve">New.<\/w:t><\/w:r><\/w:ins><\/w:p>/,
    );
    expect(acceptedParagraphs(document)).toEqual(["Kept.", "New.", "Last."]);
  });

  test("accepting every change in Word gives the text accepting every change in IncarnaMind does", async () => {
    const core = startCore(await createTempDataFolder());
    const { id, client } = await mindWith(core, [
      paragraph(
        text("Spring tides are "),
        text("weakest", removes("c1")),
        text("strongest", adds("c1")),
        text(" at new moon."),
      ),
      {
        type: "paragraph",
        attrs: { proposal: { id: "c2", request: "r1", kind: "delete" } },
        content: [text("A sentence the change removes.")],
      },
      {
        type: "bulletList",
        content: [
          { type: "listItem", content: [paragraph(text("One"))] },
          {
            type: "listItem",
            attrs: { proposal: { id: "c3", request: "r1", kind: "insert" } },
            content: [paragraph(text("Two, added"))],
          },
        ],
      },
      paragraph(text("Neap tides come between."), text(" Twice a month.", adds("c4"))),
    ]);
    const shown = acceptedParagraphs((await word(core, id)).document);

    await resolveChanges(client, "accept");
    const accepted = acceptedParagraphs((await word(core, id)).document);

    expect(shown).toEqual(accepted);
    expect(accepted).toEqual([
      "Spring tides are strongest at new moon.",
      "One",
      "Two, added",
      "Neap tides come between. Twice a month.",
    ]);
  });

  test("a change out of date, which can only be rejected, goes in as the Mind reads it", async () => {
    const core = startCore(await createTempDataFolder());
    const stale = { id: "c9", request: "r1", target: "not the text it removes" };
    const { id } = await mindWith(core, [
      paragraph(
        text("Rent is due "),
        text("monthly", [{ type: "proposalDelete", attrs: stale }]),
        text("weekly", [{ type: "proposalInsert", attrs: stale }]),
        text("."),
      ),
    ]);

    const { document } = await word(core, id);

    expect(document).not.toMatch(/<w:(ins|del) /);
    expect(acceptedParagraphs(document)).toEqual(["Rent is due monthly."]);
  });
});

describe("IncarnaMind's comments", () => {
  test("AI text gets one comment a passage, across its paragraphs, when asked; none otherwise", async () => {
    const core = startCore(await createTempDataFolder());
    const { id } = await mindWith(core, [
      paragraph(text("My own line.")),
      paragraph(text("The AI's first paragraph.", ai)),
      paragraph(text("And its second.", ai)),
      paragraph(text("Mine again, "), text("with AI words", ai), text(" inside.")),
    ]);

    const marked = await word(core, id, { markAiText: true });
    const plain = await word(core, id, { markAiText: false });

    const comments = marked.part("word/comments.xml");
    expect([...comments.matchAll(/<w:comment /g)]).toHaveLength(2);
    expect(comments).toContain('w:author="IncarnaMind"');
    expect(comments).toContain("Written by IncarnaMind");
    // The first passage starts in one paragraph and ends in the next.
    const paragraphs = marked.document.match(/<w:p>[\s\S]*?<\/w:p>/g) ?? [];
    expect(paragraphs[1]).toContain("<w:commentRangeStart w:id=");
    expect(paragraphs[1]).not.toContain("<w:commentRangeEnd");
    expect(paragraphs[2]).toContain("<w:commentRangeEnd w:id=");
    expect(paragraphs[2]).toContain(
      '<w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference',
    );
    expect(marked.part("word/_rels/document.xml.rels")).toContain('Target="comments.xml"');
    expect(marked.part("[Content_Types].xml")).toContain('PartName="/word/comments.xml"');
    expect(marked.part("word/styles.xml")).toContain('w:styleId="CommentText"');

    expect(plain.files.has("word/comments.xml")).toBe(false);
    expect(plain.document).not.toContain("commentRange");
    expect(plain.part("word/styles.xml")).not.toContain("CommentText");
  });

  test("a Citation whose check failed gets a comment saying what failed; one whose quote was found gets none", async () => {
    const core = startCore(await createTempDataFolder());
    const document = await tides(core);
    const { id } = await mindWith(core, [
      paragraph(
        text("Spring tides happen at new moon."),
        citation(document, "found", "Spring tides happen at new moon."),
      ),
      paragraph(
        text("Neap tides are the highest."),
        citation(document, "not-found", "Neap tides are the highest."),
      ),
    ]);

    const commented = await word(core, id, { commentFailedChecks: true });
    const plain = await word(core, id, { commentFailedChecks: false });

    const comments = commented.part("word/comments.xml");
    expect([...comments.matchAll(/<w:comment /g)]).toHaveLength(1);
    expect(comments).toContain("Quote not found in Tides, p. 1: “Neap tides are the highest.”");
    const [, second] = commented.document.match(/<w:p>[\s\S]*?<\/w:p>/g) ?? [];
    expect(second).toMatch(
      /<w:commentRangeStart w:id="(\d+)"\/><w:r><w:rPr><w:rStyle w:val="FootnoteReference"\/><\/w:rPr><w:footnoteReference w:id="2"\/><\/w:r><w:commentRangeEnd w:id="\1"\/>/,
    );
    expect(plain.files.has("word/comments.xml")).toBe(false);
  });

  test("without any option, a Mind with no change waiting is the file it was before #107", async () => {
    const core = startCore(await createTempDataFolder());
    const { id } = await mindWith(core, [paragraph(text("AI words", ai)), paragraph(text("Mine"))]);

    const { files } = await word(core, id);

    expect(files.has("word/comments.xml")).toBe(false);
    expect([...files.keys()].sort()).toEqual(
      [
        "[Content_Types].xml",
        "_rels/.rels",
        "docProps/core.xml",
        "word/_rels/document.xml.rels",
        "word/document.xml",
        "word/footnotes.xml",
        "word/numbering.xml",
        "word/settings.xml",
        "word/styles.xml",
      ].sort(),
    );
  });
});

describe("tracked changes as Word writes them", () => {
  const by = (kind: "insert" | "delete") => ({ kind, author: "Priya Raman", date: ASKED });

  test("runs go inside the change; a link's go inside the link; marks between runs stay where they are", () => {
    const xml =
      '<w:r><w:t>One</w:t></w:r><w:bookmarkStart w:id="9" w:name="b"/><w:hyperlink r:id="rId3"><w:r><w:t>two</w:t></w:r></w:hyperlink>';

    expect(trackInline(xml, by("insert"), new WordIds())).toBe(
      '<w:ins w:id="0" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"><w:r><w:t>One</w:t></w:r></w:ins><w:bookmarkStart w:id="9" w:name="b"/><w:hyperlink r:id="rId3"><w:ins w:id="1" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"><w:r><w:t>two</w:t></w:r></w:ins></w:hyperlink>',
    );
  });

  test("deleted text is w:delText, a field's code w:delInstrText; deleting what someone inserted marks it inside their insertion", () => {
    const xml =
      '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:ins w:id="5" w:author="Lars"><w:r><w:t>theirs</w:t></w:r></w:ins><w:del w:id="6" w:author="Lars"><w:r><w:delText>gone</w:delText></w:r></w:del>';

    expect(trackInline(xml, by("delete"), new WordIds())).toBe(
      '<w:del w:id="0" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"><w:r><w:delInstrText xml:space="preserve"> PAGE </w:delInstrText></w:r></w:del><w:ins w:id="5" w:author="Lars"><w:del w:id="1" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"><w:r><w:delText>theirs</w:delText></w:r></w:del></w:ins><w:del w:id="6" w:author="Lars"><w:r><w:delText>gone</w:delText></w:r></w:del>',
    );
  });

  test("a paragraph's mark goes first in its run properties, and a row's change last in its properties", () => {
    const ids = new WordIds();
    const blocks =
      '<w:p><w:pPr><w:pStyle w:val="Body"/><w:rPr><w:b/></w:rPr></w:pPr><w:r><w:t>A</w:t></w:r></w:p>' +
      "<w:tbl><w:tblPr/><w:tr><w:trPr><w:cantSplit/></w:trPr><w:tc><w:p/></w:tc></w:tr></w:tbl>";

    const deleted = trackBlocks(blocks, by("delete"), ids);

    expect(deleted).toBe(
      '<w:p><w:pPr><w:pStyle w:val="Body"/><w:rPr><w:del w:id="0" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"/><w:b/></w:rPr></w:pPr><w:del w:id="1" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"><w:r><w:delText>A</w:delText></w:r></w:del></w:p>' +
        '<w:tbl><w:tblPr/><w:tr><w:trPr><w:cantSplit/><w:del w:id="2" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"/></w:trPr><w:tc><w:p><w:pPr><w:rPr><w:del w:id="3" w:author="Priya Raman" w:date="2026-10-12T14:30:00Z"/></w:rPr></w:pPr></w:p></w:tc></w:tr></w:tbl>',
    );
  });
});
