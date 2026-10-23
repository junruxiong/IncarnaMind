import { describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import {
  CLAIM_CHECK_MARK,
  CONTENT_SCHEMA_VERSION,
  CONTENT_SCHEMA_VERSION_KEY,
  MIND_CONTENT_FIELD,
  MIND_SETTINGS_FIELD,
} from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";

/*
 * The version before a check in the Mind's chosen scope (#1623): its core is this one with the
 * newest content schema it understands set back to 25, whose card would name the Mind's Folder,
 * or every Document, for a check made in one Document.
 */
vi.mock("../../src/core/api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  CONTENT_SCHEMA_VERSION: 25,
}));

/** The content schema version `chosen` came with: what a Mind written since records. */
const CHOSEN = 26;

const SENTENCE = "The Employee may work from home two days a week.";

describe("a version from before checks in the Mind's chosen scope", () => {
  // protects=a version that doesn't know `chosen` opens a Mind holding it read-only, so it can't misname where the check looked or drop it; fails_when=it opens writable; no_claim=how the newer version shows it.
  test("opens a Mind with a check made in its chosen scope read-only", async () => {
    expect(CONTENT_SCHEMA_VERSION).toBe(25);
    const doc = new Y.Doc();
    const paragraph = new Y.XmlElement("paragraph");
    paragraph.setAttribute("id", "block-1");
    const text = new Y.XmlText();
    text.insert(0, SENTENCE, {
      [CLAIM_CHECK_MARK]: {
        id: "check-1",
        request: "r1",
        action: "check-against-documents",
        state: "supported",
        text: SENTENCE,
        scope: {
          folderId: null,
          widened: false,
          chosen: { folderIds: [], tagIds: [], documentIds: ["agreement"] },
        },
        versions: [],
        excluded: [],
        checkedAt: "2026-10-23T09:00:00.000Z",
      },
    });
    paragraph.insert(0, [text]);
    doc.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
    doc.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, CHOSEN);

    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind({ title: "Summary" });
    await core.applyMindUpdate(mind.id, Y.encodeStateAsUpdate(doc));

    expect((await core.openMind(mind.id)).access).toBe("read-only");
  });
});
