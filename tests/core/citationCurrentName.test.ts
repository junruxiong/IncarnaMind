/**
 * A Citation shows its Document's current name (#532): an export, which the core writes, says
 * the name a Document has now, and the stored name only once the Document is deleted. The
 * stored attribute is never rewritten, and a rename changes nothing in any Mind.
 */
import type { JSONContent } from "@tiptap/core";
import { expect, test } from "vitest";
import * as Y from "yjs";
import { MIND_SETTINGS_FIELD } from "../../src/core";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { writeMind } from "../helpers/minds";

const exported = async (core: ReturnType<typeof startCore>, mindId: string) =>
  new TextDecoder().decode((await core.exportMind(mindId, { format: "markdown" })).data);

test("a renamed Document is named as it is now in an export; deleted, as it was cited; the Mind never changes", async () => {
  const dataDir = await createTempDataFolder();
  const sources = await createTempDataFolder();
  const core = startCore(dataDir, { now: manualClock().now });
  const [ferry] = await addAndProcess(core, [
    await writeSourceFile(sources, "Ferry.txt", "The ferry leaves at nine."),
  ]);
  if (!ferry) throw new Error("No Document.");
  const mind = await core.createMind({ title: "Crossing" });
  const client = await connectToMind(core, mind.id);
  const citation: JSONContent = {
    type: "citation",
    attrs: {
      passageId: "passage-1",
      documentId: ferry.id,
      documentName: ferry.name,
      contentHash: ferry.contentHash,
      location: { kind: "page", from: 2, to: 2 },
      quote: "The ferry leaves at nine.",
      check: "found",
    },
  };
  writeMind(client, [
    { type: "paragraph", content: [{ type: "text", text: "It leaves early " }, citation] },
  ]);
  client.doc.getMap(MIND_SETTINGS_FIELD).set("bibliography", "shown");
  await client.settled();
  expect(ferry.name).toBe("Ferry");

  const before = {
    blocks: client.blocks.toJSON(),
    vector: Y.encodeStateVector(client.doc),
    minds: await core.listMinds(),
  };
  const storedName = () => {
    const names: unknown[] = [];
    const visit = (node: Y.XmlFragment | Y.XmlElement | Y.XmlText) => {
      if (node instanceof Y.XmlElement && node.nodeName === "citation") {
        names.push(node.getAttribute("documentName"));
      }
      if (node instanceof Y.XmlText) return;
      for (const child of node.toArray()) {
        if (child instanceof Y.XmlElement || child instanceof Y.XmlText) visit(child);
      }
    };
    visit(client.blocks);
    return names;
  };
  expect(storedName()).toEqual(["Ferry"]);
  expect(await exported(core, mind.id)).toContain("Ferry, p. 2");

  await core.renameDocument(ferry.id, "Ferry timetable");
  const renamed = await exported(core, mind.id);
  expect(renamed).toContain("Ferry timetable, p. 2");
  expect(renamed).toContain("Ferry timetable. Cited: p. 2");
  expect(renamed).not.toContain("Ferry, p. 2");
  expect(renamed).not.toContain("Ferry. Cited");

  // Nothing in the Mind changed: not its Blocks, not what this client heard, not what a new one reads.
  const reopened = await connectToMind(core, mind.id);
  expect(reopened.blocks.toJSON()).toBe(before.blocks);
  expect(Y.encodeStateVector(reopened.doc)).toEqual(before.vector);
  expect(Y.encodeStateVector(client.doc)).toEqual(before.vector);
  expect(await core.listMinds()).toEqual(before.minds);
  expect(storedName()).toEqual(["Ferry"]);

  await core.deleteDocument(ferry.id);
  const deleted = await exported(core, mind.id);
  expect(deleted).toContain("Ferry, p. 2");
  expect(deleted).not.toContain("Ferry timetable");
  expect(storedName()).toEqual(["Ferry"]);
  await core.close();
});
