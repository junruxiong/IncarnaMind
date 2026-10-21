import { describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import {
  CONTENT_SCHEMA_VERSION,
  CONTENT_SCHEMA_VERSION_KEY,
  MIND_CONTENT_FIELD,
  MIND_SETTINGS_FIELD,
  MindReadOnlyError,
  WORD_TAB_MARK,
} from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";

/* The version before tab stops: this core with the newest content schema it understands set back to 22. */
vi.mock("../../src/core/api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  CONTENT_SCHEMA_VERSION: 22,
}));

const TABS = 23;

describe("a version from before tab stops", () => {
  test("opens a Mind with tab marks read-only, so its editor can't strip them", async () => {
    expect(CONTENT_SCHEMA_VERSION).toBe(22);
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind({ title: "CV" });
    const written = new Y.Doc();
    const paragraph = new Y.XmlElement("paragraph");
    paragraph.setAttribute("id", "block-1");
    const text = new Y.XmlText();
    text.insert(0, "School");
    text.insert(6, "\t", { [WORD_TAB_MARK]: {} });
    text.insert(7, "2026");
    paragraph.insert(0, [text]);
    written.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
    written.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, TABS);
    await core.applyMindUpdate(mind.id, Y.encodeStateAsUpdate(written));

    const opened = await core.openMind(mind.id);
    expect(opened.access).toBe("read-only");
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    const sent: Uint8Array[] = [];
    doc.on("update", (update: Uint8Array) => sent.push(update));
    const line = doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
    const tab = line instanceof Y.XmlElement ? line.get(0) : null;
    if (!(tab instanceof Y.XmlText)) throw new Error("No line.");
    tab.format(6, 1, { [WORD_TAB_MARK]: null });
    const [update] = sent;
    if (!update) throw new Error("Nothing sent.");
    await expect(core.applyMindUpdate(mind.id, update)).rejects.toThrow(MindReadOnlyError);
  });
});
