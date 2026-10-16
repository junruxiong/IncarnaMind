import { describe, expect, test } from "vitest";
import type {
  Conversation,
  DocumentGroupAssignment,
  LibraryGroup,
  LibrarySnapshot,
  Mind,
} from "../../src/core/api";
import { chosenScope, composerScope, NO_SCOPE } from "../../src/renderer/src/composer";
import { movedLocally, moveMessage, moveName, undoMoves } from "../../src/renderer/src/moves";
import { scopeChips, scopeCounts } from "../../src/renderer/src/scope";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";
import { hasSearchScope } from "../../src/shared/searchScope";

/**
 * Folders are projects (#111), as the window shows them: the composer's
 * Search scope starts with the Mind's Folder; a deleted Folder's chip keeps
 * its name; a move shows at once, says what it did, and can be undone.
 */

const mind = (id: string, folderId: string | null, title = id): Mind => ({
  id,
  title,
  autoTitle: "",
  kind: "mind",
  createdAt: "",
  updatedAt: "",
  folderId,
});

const group = (id: string, name: string): LibraryGroup => ({
  id,
  name,
  description: "",
  countsOwnFiles: false,
  createdAt: "",
  updatedAt: "",
});

const assignment = (documentId: string, groupId: string | null): DocumentGroupAssignment => ({
  documentId,
  groupId,
  source: "automatic",
  suggested: false,
  status: "classified",
  error: null,
  model: null,
});

const t = (key: MessageKey, params?: MessageParams) => translate("en", key, params);

describe("the composer's Search scope", () => {
  const draft = { text: "", scope: null, skill: null, pastes: [] };

  test("a Mind in a Folder starts with the Folder as its chip; one in none searches everything", () => {
    expect(composerScope(draft, "finance")).toEqual({
      folderIds: ["finance"],
      tagIds: [],
      documentIds: [],
    });
    expect(hasSearchScope(composerScope(draft, null))).toBe(false);
  });

  test("taking the Folder's chip out searches everything, and stays so", () => {
    const removed = chosenScope(NO_SCOPE, "finance");
    expect(removed).toEqual(NO_SCOPE);
    expect(hasSearchScope(composerScope({ ...draft, scope: removed }, "finance"))).toBe(false);
  });

  test("the Mind's own scope follows it to another Folder; a scope the User chose stays", () => {
    // Its own Folder, as it started: kept as the Mind's own (null).
    const own = chosenScope({ folderIds: ["finance"], tagIds: [], documentIds: [] }, "finance");
    expect(own).toBeNull();
    expect(composerScope({ ...draft, scope: own }, "research").folderIds).toEqual(["research"]);
    // A Tag added: the User's own choice, which a move leaves as it is.
    const chosen = chosenScope(
      { folderIds: ["finance"], tagIds: ["urgent"], documentIds: [] },
      "finance",
    );
    expect(composerScope({ ...draft, scope: chosen }, "research")).toEqual({
      folderIds: ["finance"],
      tagIds: ["urgent"],
      documentIds: [],
    });
    // In no Folder, nothing chosen is the Mind's own: it follows the Mind into one.
    expect(chosenScope(NO_SCOPE, null)).toBeNull();
  });
});

describe("a Search scope's chips", () => {
  const library = {
    folders: [],
    groups: [group("live", "Vendor selection")],
    deletedGroups: [group("gone", "Finance")],
    tags: [],
    documents: [],
  };

  test("a deleted Folder keeps its name, struck through; one never known says so", () => {
    expect(
      scopeChips(library, { folderIds: ["live", "gone", "unknown"], tagIds: [], documentIds: [] }),
    ).toEqual([
      {
        kind: "folder",
        id: "live",
        name: "Vendor selection",
        deleted: false,
        documentKind: null,
        count: 0,
      },
      { kind: "folder", id: "gone", name: "Finance", deleted: true, documentKind: null },
      { kind: "folder", id: "unknown", name: null, deleted: true, documentKind: null },
    ]);
  });
});

describe("moving Minds and Documents", () => {
  const library: LibrarySnapshot = {
    groups: [group("finance", "Finance"), group("research", "Research")],
    deletedGroups: [],
    assignments: [assignment("moon", "research")],
    settings: { classifier: { kind: "automatic" }, automatic: false, organizeWith: null },
  };
  const minds = [mind("shortlist", null, "Shortlist"), mind("notes", "research", "Notes")];

  test("shows at once: each Mind in its new Folder, each Document as the User's choice", () => {
    const moved = movedLocally(
      { minds, library },
      { mindIds: ["shortlist"], documentIds: ["moon", "harbour"] },
      "finance",
    );
    expect(moved.minds.map((each) => [each.id, each.folderId])).toEqual([
      ["shortlist", "finance"],
      ["notes", "research"],
    ]);
    expect(
      moved.library?.assignments.map((each) => [each.documentId, each.groupId, each.source]),
    ).toEqual([
      ["moon", "finance", "user"],
      ["harbour", "finance", "user"],
    ]);
  });

  test("undoing moves each Mind and conversation back to where it was, in one move per Folder it came from", () => {
    expect(
      undoMoves({
        folderId: "finance",
        minds: [
          { id: "shortlist", from: null },
          { id: "notes", from: "research" },
          { id: "already", from: "finance" },
        ],
        // Documents go back with `restoreAssignment`, each with who chose its Folder.
        documents: [{ id: "moon", from: "research", previous: null }],
        conversations: [
          { id: "asked", from: null },
          { id: "here", from: "finance" },
        ],
      }),
    ).toEqual([
      { folderId: null, items: { mindIds: ["shortlist"], conversationIds: ["asked"] } },
      { folderId: "research", items: { mindIds: ["notes"] } },
    ]);
  });

  test("the line says what moved where", () => {
    const names = { minds, documents: [] };
    expect(moveMessage(t, { mindIds: ["shortlist"] }, "Finance", names)).toBe(
      "Moved “Shortlist” to Finance",
    );
    expect(moveMessage(t, { documentIds: ["a", "b", "c"] }, "Finance", names)).toBe(
      "Moved 3 Documents to Finance",
    );
    expect(moveMessage(t, { mindIds: ["shortlist", "notes"] }, "Not in a Folder", names)).toBe(
      "Moved 2 Minds to Not in a Folder",
    );
    expect(moveMessage(t, { mindIds: ["shortlist"], documentIds: ["a"] }, "Finance", names)).toBe(
      "Moved 2 items to Finance",
    );
  });

  test("the Edit menu names the move: one by its name, several by how many of what", () => {
    const names = { minds, documents: [] };
    expect(moveName(t, { mindIds: ["shortlist"] }, names)).toBe("Move “Shortlist”");
    expect(moveName(t, { documentIds: ["a", "b", "c"] }, names)).toBe("Move 3 Documents");
    expect(moveName(t, { mindIds: ["shortlist"], documentIds: ["a"] }, names)).toBe("Move 2 items");
    expect(
      moveName(
        (key, params) => translate("zh-CN", key, params),
        { mindIds: ["shortlist", "notes"] },
        names,
      ),
    ).toBe("移动 2 个 Mind");
  });

  test("the line names a conversation moved, or counts them", () => {
    const conversation = (id: string, title: string, autoTitle: string): Conversation => ({
      id,
      title,
      autoTitle,
      folderId: null,
      askedFrom: [],
      createdAt: "",
      updatedAt: "",
    });
    const names = {
      minds,
      documents: [],
      conversations: [
        conversation("heat", "", "Why do parks cool at night"),
        conversation("tides", "Tides", "When are spring tides"),
        conversation("blank", "", ""),
      ],
    };
    expect(moveMessage(t, { conversationIds: ["heat"] }, "Finance", names)).toBe(
      "Moved “Why do parks cool at night” to Finance",
    );
    expect(moveMessage(t, { conversationIds: ["tides"] }, "Finance", names)).toBe(
      "Moved “Tides” to Finance",
    );
    expect(moveMessage(t, { conversationIds: ["blank"] }, "Finance", names)).toBe(
      "Moved “New chat” to Finance",
    );
    expect(moveMessage(t, { conversationIds: ["heat", "tides"] }, "Finance", names)).toBe(
      "Moved 2 conversations to Finance",
    );
    expect(
      moveMessage(
        (key, params) => translate("zh-CN", key, params),
        { conversationIds: ["heat", "tides"] },
        "财务",
        names,
      ),
    ).toBe("已将 2 个对话移动到“财务”");
  });
});

describe("how many Documents each Folder and Tag holds (#764)", () => {
  const document = (id: string, over: Record<string, unknown> = {}) =>
    ({
      id,
      name: id,
      kind: "pdf",
      status: "ready",
      fileStatus: "available",
      folderId: null,
      tags: [],
      ...over,
    }) as never;

  test("counts the searchable Documents of an in-app Folder, a Linked folder with its sub-Folders, and a Tag", () => {
    const folder = (id: string, parentId: string | null) => ({ id, name: id, parentId }) as never;
    const counts = scopeCounts({
      folders: [folder("root", null), folder("deep", "root"), folder("quiet", null)],
      groups: [group("filed", "Filed"), group("empty", "Empty")],
      tags: [{ id: "t1", name: "Tide" }] as never,
      documents: [
        document("a", { folderId: "root", tags: [{ tagId: "t1" }] }),
        document("b", { folderId: "deep" }),
        document("scan", { folderId: "deep", status: "no-text" }),
        document("gone", { folderId: "root", fileStatus: "missing" }),
        document("c"),
      ],
      assignments: [
        { documentId: "c", groupId: "filed" },
        { documentId: "scan", groupId: "filed" },
      ] as never,
    });
    expect(counts.get("root")).toBe(2);
    expect(counts.get("deep")).toBe(1);
    expect(counts.get("quiet")).toBe(0);
    expect(counts.get("filed")).toBe(1);
    expect(counts.get("empty")).toBe(0);
    expect(counts.get("t1")).toBe(1);
  });
});
