/**
 * Someone else's tracked changes in an imported Word file (#108), through
 * the core's public interface: each insertion and deletion of text, and each
 * paragraph added or removed whole, is a change to accept or reject with its
 * author and date, as the AI's are; the export writes the ones accepted or
 * rejected without a trace and the rest back as they were. What can't be
 * accepted or rejected here is kept as is, and listed.
 */
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { Core } from "../../src/core";
import {
  aiUseOfAccepted,
  outOfDateProposals,
  pendingProposals,
  proposalRequestOf,
} from "../../src/shared/proposals";
import { wordReportOf } from "../../src/shared/wordFiles";
import { createTempDataFolder, startCore } from "../helpers/core";
import { acceptedParagraphs, checkWellFormed, unzip } from "../helpers/docx";
import type { MindClient } from "../helpers/mindClient";
import { editMind, readMind, resolveChanges, shownWithChanges } from "../helpers/minds";
import { field, NUMBERING, paragraph, run, wordFile } from "../helpers/word";
import { importWord } from "../helpers/wordImports";

const PRIYA = 'w:author="Priya Nair" w:date="2026-10-02T09:00:00Z"';
const LARS = 'w:author="Lars Berg" w:date="2026-10-05T16:20:00Z"';

const ins = (id: number, who: string, text: string) =>
  `<w:ins w:id="${id}" ${who}>${run(text)}</w:ins>`;
const del = (id: number, who: string, text: string) =>
  `<w:del w:id="${id}" ${who}><w:r><w:delText xml:space="preserve">${text}</w:delText></w:r></w:del>`;
/** A paragraph someone added or removed whole: its mark and its words. */
const wholeParagraph = (kind: "ins" | "del", id: number, who: string, text: string, pPr = "") =>
  `<w:p><w:pPr>${pPr}<w:rPr><w:${kind} w:id="${id}" ${who}/></w:rPr></w:pPr>${kind === "ins" ? ins(id + 1, who, text) : del(id + 1, who, text)}</w:p>`;

/** A lease clause with Priya's and Lars's changes: words, a replacement, paragraphs, a list item. */
const REDLINE = [
  paragraph(
    `${run("The Tenant pays rent")}${ins(1, PRIYA, " monthly")}${run(" in advance")}${del(2, PRIYA, " by cheque")}${run(".")}`,
  ),
  paragraph(
    `${run("Notice is ")}${del(3, LARS, "one month")}${ins(4, LARS, "three months")}${run(".")}`,
  ),
  wholeParagraph("ins", 5, LARS, "The Landlord insures the building."),
  wholeParagraph("del", 7, PRIYA, "Pets are not allowed."),
  paragraph("Repairs:", {}),
  paragraph("Roof", { numId: 1 }),
  wholeParagraph(
    "ins",
    9,
    PRIYA,
    "Boiler",
    '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>',
  ),
].join("");

async function imported(body: string): Promise<{
  core: Core;
  mindId: string;
  client: MindClient;
  bytes: Buffer;
}> {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir);
  const bytes = wordFile({ body, numbering: NUMBERING });
  return { core, bytes, ...(await importWord(core, dataDir, bytes)) };
}

async function exported(core: Core, mindId: string, trackChanges = true) {
  const result = await core.exportMind(mindId, { format: "docx", trackChanges });
  const files = unzip(result.data);
  for (const [name, content] of files) {
    if (name.endsWith(".xml") || name.endsWith(".rels")) checkWellFormed(content.toString("utf8"));
  }
  return { files, document: files.get("word/document.xml")?.toString("utf8") ?? "" };
}

/** Who made each change waiting in the Mind, in order: its author and date, as the file says. */
function changesOf(client: MindClient) {
  return pendingProposals(client.blocks).map((change) => {
    const made = proposalRequestOf(client.doc, change.request);
    return made?.source === "word-change" ? `${made.author} ${made.date}` : null;
  });
}

/** What changes still waiting do to each list item as a whole: "insert", "delete" or null. */
function listItemChanges(client: MindClient): (string | null)[] {
  const kinds: (string | null)[] = [];
  readMind(client).descendants((node) => {
    if (node.type.name === "listItem") kinds.push(node.attrs.proposal?.kind ?? null);
    return true;
  });
  return kinds;
}

const byAuthor = (client: MindClient, author: string) => (id: string) => {
  const change = pendingProposals(client.blocks).find((each) => each.id === id);
  const made = change ? proposalRequestOf(client.doc, change.request) : null;
  return made?.source === "word-change" && made.author === author;
};

describe("someone else's tracked changes", () => {
  test("come in as changes to accept or reject, each with its author and date: words, a replacement, paragraphs and a list item added or removed whole", async () => {
    const { core, mindId, client, bytes } = await imported(REDLINE);

    expect(shownWithChanges(client)).toEqual([
      "The Tenant pays rent{+ monthly+} in advance[- by cheque-].",
      "Notice is [-one month-]{+three months+}.",
      "+ The Landlord insures the building.",
      "- Pets are not allowed.",
      "Repairs:",
      "Roof",
      "Boiler",
    ]);
    expect(listItemChanges(client)).toEqual([null, "insert"]);
    expect(changesOf(client)).toEqual([
      "Priya Nair 2026-10-02T09:00:00Z",
      "Priya Nair 2026-10-02T09:00:00Z",
      "Lars Berg 2026-10-05T16:20:00Z",
      "Lars Berg 2026-10-05T16:20:00Z",
      "Priya Nair 2026-10-02T09:00:00Z",
      "Priya Nair 2026-10-02T09:00:00Z",
    ]);
    expect(outOfDateProposals(client.blocks).size).toBe(0);

    // Left as they are, they go back as they were: the document isn't written again.
    const { document } = await exported(core, mindId);
    expect(document).toBe(unzip(bytes).get("word/document.xml")?.toString("utf8"));
  });

  test("accepted and rejected one by one and by author, they leave no trace, and the rest keep their author and date", async () => {
    const { core, mindId, client } = await imported(REDLINE);
    const [monthly, cheque] = pendingProposals(client.blocks).map((change) => change.id);

    await resolveChanges(client, "accept", (id) => id === monthly);
    await resolveChanges(client, "reject", (id) => id === cheque);
    await resolveChanges(client, "reject", byAuthor(client, "Lars Berg"));

    expect(shownWithChanges(client)).toEqual([
      "The Tenant pays rent monthly in advance by cheque.",
      "Notice is one month.",
      "- Pets are not allowed.",
      "Repairs:",
      "Roof",
      "Boiler",
    ]);
    const { document } = await exported(core, mindId);
    expect(document).toContain(
      `${run("The Tenant pays rent")}${run(" monthly")}${run(" in advance")}<w:r><w:t xml:space="preserve"> by cheque</w:t></w:r>${run(".")}`,
    );
    expect(document).toContain(`${run("Notice is ")}${run("one month")}${run(".")}`);
    expect(document).not.toContain("Lars Berg");
    expect(document).not.toContain("insures");
    // Those still waiting are as they were in the file.
    expect(document).toContain(wholeParagraph("del", 7, PRIYA, "Pets are not allowed."));
    expect(document).toContain(
      wholeParagraph(
        "ins",
        9,
        PRIYA,
        "Boiler",
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>',
      ),
    );
    // Resolving a change isn't an edit of the User's: nothing is tracked as theirs.
    expect(document).not.toMatch(/w:author="(?!Priya Nair)/);

    // Accepting the rest in Word gives what accepting them here does.
    await resolveChanges(client, "accept");
    expect(acceptedParagraphs(document)).toEqual(
      shownWithChanges(client).filter((line) => line !== ""),
    );
  });

  test("a paragraph whose removal is rejected, or whose addition is accepted, keeps no mark of the change", async () => {
    const { core, mindId, client } = await imported(REDLINE);

    await resolveChanges(client, "reject", byAuthor(client, "Priya Nair"));
    await resolveChanges(client, "accept", byAuthor(client, "Lars Berg"));

    const { document } = await exported(core, mindId);
    expect(document).toContain(
      `<w:p>${run("The Landlord insures the building.")}</w:p><w:p><w:r><w:t xml:space="preserve">Pets are not allowed.</w:t></w:r></w:p>`,
    );
    expect(document).not.toContain("<w:ins");
    expect(document).not.toContain("<w:del");
    expect(document).not.toContain("Boiler");
    expect(acceptedParagraphs(document)).toEqual([
      "The Tenant pays rent in advance by cheque.",
      "Notice is three months.",
      "The Landlord insures the building.",
      "Pets are not allowed.",
      "Repairs:",
      "Roof",
    ]);
  });

  test("accepting one in a paragraph the User edited follows the rule for changes out of date: an edit to its own words keeps it waiting", async () => {
    const { core, mindId, client } = await imported(REDLINE);

    // An edit elsewhere in the paragraph leaves the change as it was; one inside its words doesn't.
    // What the User types is theirs, without the change's mark, as the editor makes it.
    editMind(client, (blocks) =>
      blocks.map((block, index) => {
        if (index !== 0) return block;
        const content = (block.content ?? []).flatMap((node): JSONContent[] =>
          node.text === " in advance"
            ? [{ ...node, text: " in arrears" }]
            : node.text === " by cheque"
              ? [
                  { ...node, text: " by " },
                  { type: "text", text: "cash " },
                  { ...node, text: "cheque" },
                ]
              : [node],
        );
        return { ...block, content };
      }),
    );
    await client.settled();
    const [monthly, cheque] = pendingProposals(client.blocks).map((change) => change.id);
    expect([...outOfDateProposals(client.blocks)]).toEqual([cheque]);

    const accepted = await resolveChanges(
      client,
      "accept",
      (id) => id === monthly || id === cheque,
    );

    expect(accepted.map((each) => each.id)).toEqual([monthly]);
    expect(shownWithChanges(client)[0]).toBe(
      "The Tenant pays rent monthly in arrears[- by -]cash [-cheque-].",
    );
    // Out of date, it is still Priya's deletion in the file, around the User's insertion.
    const { document } = await exported(core, mindId);
    expect(document).toMatch(
      /<w:del w:id="2" w:author="Priya Nair" w:date="2026-10-02T09:00:00Z"><w:r><w:delText xml:space="preserve"> by <\/w:delText><\/w:r><\/w:del><w:ins w:id="\d+" w:author="(?!Priya Nair)[^"]+"[^>]*><w:r><w:t xml:space="preserve">cash <\/w:t><\/w:r><\/w:ins><w:del w:id="(?!2")\d+" w:author="Priya Nair" w:date="2026-10-02T09:00:00Z"><w:r><w:delText xml:space="preserve">cheque<\/w:delText>/,
    );
  });

  test("what can't be accepted or rejected here is kept as is, listed by kind, and goes back as it was", async () => {
    const format = `<w:r><w:rPr><w:b/><w:rPrChange w:id="20" ${PRIYA}><w:rPr/></w:rPrChange></w:rPr><w:t>Bold now.</w:t></w:r>`;
    const move = `<w:moveFromRangeStart w:id="21" w:name="move1" ${PRIYA}/><w:moveFrom w:id="22" ${PRIYA}>${run("Moved.")}</w:moveFrom><w:moveFromRangeEnd w:id="21"/>`;
    const moved = `<w:moveToRangeStart w:id="23" w:name="move1" ${PRIYA}/><w:moveTo w:id="24" ${PRIYA}>${run("Moved.")}</w:moveTo><w:moveToRangeEnd w:id="23"/>`;
    const withField = `<w:ins w:id="25" ${PRIYA}>${field("PAGE", "3")}</w:ins>`;
    const mark = `<w:p><w:pPr><w:rPr><w:ins w:id="26" ${PRIYA}/></w:rPr></w:pPr>${run("Joined, ")}</w:p>`;
    const row = `<w:tbl><w:tblPr/><w:tblGrid><w:gridCol w:w="2000"/></w:tblGrid><w:tr><w:trPr><w:ins w:id="27" ${PRIYA}/></w:trPr><w:tc>${paragraph("New row")}</w:tc></w:tr></w:tbl>`;
    const { core, mindId, client } = await imported(
      [
        paragraph(`${format}${move}`),
        paragraph(`${run("Page ")}${withField}${moved}`),
        mark,
        paragraph("then this."),
        row,
        paragraph("End."),
      ].join(""),
    );

    expect(wordReportOf(client.doc)?.keptChanges).toEqual({
      format: 1,
      move: 1,
      content: 1,
      paragraphMark: 1,
      row: 1,
    });
    editMind(client, (blocks) =>
      blocks.map((block) =>
        block.content?.[0]?.text === "End."
          ? { ...block, content: [{ type: "text", text: "The end." }] }
          : block,
      ),
    );
    await client.settled();
    const { document } = await exported(core, mindId);
    for (const kept of [format, move, moved, withField, mark, row])
      expect(document).toContain(kept);
  });

  test("accepting someone's change isn't recorded as the AI's", async () => {
    const { client } = await imported(REDLINE);

    const accepted = await resolveChanges(client, "accept");

    expect(accepted.length).toBe(6);
    expect(aiUseOfAccepted(client.doc, "mind", accepted, "2026-10-13T09:00:00Z")).toEqual([]);
  });
});
