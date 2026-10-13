import { describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import {
  CLAIM_CHECK_MARK,
  CONTENT_SCHEMA_VERSION,
  CONTENT_SCHEMA_VERSION_KEY,
  MIND_CONTENT_FIELD,
  MIND_SETTINGS_FIELD,
  MindReadOnlyError,
} from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";

/*
 * The version before contradiction watch (#112): its core is this one with
 * the newest content schema it understands set back to 18, which reads the
 * "contradiction-watch" action as Find support.
 */
vi.mock("../../src/core/api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  CONTENT_SCHEMA_VERSION: 18,
}));

/** The content schema version the action came with: what a Mind written since records. */
const WATCH = 19;

const SENTENCE = "Harbours publish the times of high water every year.";

const watchMark = {
  id: "watch-1",
  request: "r1",
  action: "contradiction-watch",
  state: "contradicted",
  reason: null,
  detail: null,
  text: SENTENCE,
  evidence: null,
  says: "The harbour office says every week.",
  scope: { folderId: "folder-1", widened: false },
  versions: [],
  excluded: [],
  checkedAt: "2026-10-13T09:00:00.000Z",
};

function mindWithWatchMark(): Y.Doc {
  const doc = new Y.Doc();
  const paragraph = new Y.XmlElement("paragraph");
  paragraph.setAttribute("id", "block-1");
  const text = new Y.XmlText();
  text.insert(0, SENTENCE, { [CLAIM_CHECK_MARK]: watchMark });
  paragraph.insert(0, [text]);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
  doc.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, WATCH);
  return doc;
}

describe("a version from before contradiction watch", () => {
  test("opens a Mind with a watch's mark read-only, so it can't turn the mark into Find support's", async () => {
    expect(CONTENT_SCHEMA_VERSION).toBe(18);
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind({ title: "Harbour notes" });
    await core.applyMindUpdate(mind.id, Y.encodeStateAsUpdate(mindWithWatchMark()));

    const opened = await core.openMind(mind.id);
    expect(opened.access).toBe("read-only");

    // Its editor, which doesn't know the action, would write the mark back as Find support's: refused.
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    const sent: Uint8Array[] = [];
    doc.on("update", (update: Uint8Array) => sent.push(update));
    const paragraph = doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
    const text = paragraph instanceof Y.XmlElement ? paragraph.get(0) : null;
    if (!(text instanceof Y.XmlText)) throw new Error("No sentence.");
    text.format(0, SENTENCE.length, {
      [CLAIM_CHECK_MARK]: { ...watchMark, action: "find-support" },
    });
    await expect(core.applyMindUpdate(mind.id, sent[0] as Uint8Array)).rejects.toThrow(
      MindReadOnlyError,
    );
  });
});
