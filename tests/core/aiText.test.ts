import type { JSONContent } from "@tiptap/core";
import { describe, expect, onTestFinished, test } from "vitest";
import {
  CONTENT_SCHEMA_VERSION,
  CONTENT_SCHEMA_VERSION_KEY,
  type Core,
  GENERATED_MARK,
  MIND_SETTINGS_FIELD,
} from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { readMind, resolveChanges, shownWithAiText, writeMind } from "../helpers/minds";

/*
 * AI text (#95), as the core keeps it: what the AI writes into a Mind is
 * marked, Minds written before stay as they were, and nothing read out of a
 * Mind (exports, Question context) shows the mark.
 */

const AI = { type: GENERATED_MARK };
type MarkJSON = { type: string; attrs?: Record<string, unknown> };
const text = (value: string, ...marks: MarkJSON[]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks.length > 0 && { marks }),
});
const paragraph = (...content: JSONContent[]): JSONContent => ({ type: "paragraph", content });

async function exported(core: Core, mindId: string): Promise<string> {
  const file = await core.exportMind(mindId, { format: "markdown" });
  return new TextDecoder().decode(file.data);
}

async function setUp() {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir);
  onTestFinished(() => core.close());
  return { core, dataDir };
}

describe("AI text in the core", () => {
  test("Add to note puts AI text in, at once or as a new Mind's changes that stay AI text once accepted", async () => {
    const { core } = await setUp();
    const blocks = [
      paragraph(text("Spring tides "), text("twice", { type: "bold" }), text(" a month.")),
    ];

    const mind = await core.createMind({ title: "Tides" });
    await core.addToNote({ mindId: mind.id, blocks, origin: null, model: null });
    const client = await connectToMind(core, mind.id);
    const note = readMind(client);
    expect(shownWithAiText(note.child(note.childCount - 1))).toBe("«Spring tides twice a month.»");
    expect(client.doc.getMap(MIND_SETTINGS_FIELD).get(CONTENT_SCHEMA_VERSION_KEY)).toBe(
      CONTENT_SCHEMA_VERSION,
    );
    expect(CONTENT_SCHEMA_VERSION).toBeGreaterThanOrEqual(11);

    const fresh = await core.createMind({ title: "From an Answer" });
    await core.addToNote({ mindId: fresh.id, blocks, origin: null, model: null, proposed: true });
    const other = await connectToMind(core, fresh.id);
    await resolveChanges(other, "accept");
    expect(shownWithAiText(readMind(other).child(0))).toBe("«Spring tides twice a month.»");
  });

  test("a Mind written before reads as it was, its Answer still the AI's as a whole", async () => {
    const { core, dataDir } = await setUp();
    const mind = await core.createMind({ title: "Tides" });
    const writer = await connectToMind(core, mind.id);
    const older: JSONContent[] = [
      paragraph(text("My note.")),
      { type: "question", content: [text("What moves the tides?")] },
      { type: "answer", content: [paragraph(text("The Moon pulls the sea."))] },
    ];
    writeMind(writer, older);
    writer.doc.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, 7);
    await writer.settled();
    const before = readMind(writer).toJSON();
    const markdown = await exported(core, mind.id);
    core.close();

    const again = startCore(dataDir);
    onTestFinished(() => again.close());
    const reader = await connectToMind(again, mind.id);
    expect(readMind(reader).toJSON()).toEqual(before);
    expect(readMind(reader).child(2).attrs.generatedMarks).toBeNull();
    expect(await exported(again, mind.id)).toBe(markdown);
  });

  test("the Markdown export is the same with AI text as without", async () => {
    const { core } = await setUp();
    const content = (...marks: MarkJSON[]) => [
      { type: "heading", attrs: { level: 2 }, content: [text("Tides", ...marks)] },
      paragraph(
        text("Spring tides ", ...marks),
        text("twice", ...marks, { type: "bold" }),
        text(" a month, see ", ...marks),
        text("NOAA", ...marks, { type: "link", attrs: { href: "https://noaa.gov" } }),
        text(".", ...marks),
      ),
      {
        type: "bulletList",
        content: [{ type: "listItem", content: [paragraph(text("High water", ...marks))] }],
      },
    ];
    const plain = await core.createMind({ title: "Tides" });
    const plainClient = await connectToMind(core, plain.id);
    writeMind(plainClient, content());
    await plainClient.settled();
    const marked = await core.createMind({ title: "Tides" });
    const client = await connectToMind(core, marked.id);
    writeMind(client, content(AI));
    await client.settled();

    expect(shownWithAiText(readMind(client).child(1))).toBe(
      "«Spring tides twice a month, see NOAA.»",
    );
    const markdown = await exported(core, marked.id);
    expect(markdown).toBe(await exported(core, plain.id));
    expect(markdown).toContain("Spring tides **twice** a month, see [NOAA](https://noaa.gov).");
  });
});
