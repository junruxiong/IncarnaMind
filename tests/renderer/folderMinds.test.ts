import { describe, expect, test } from "vitest";
import type {
  DocumentGroupAssignment,
  LibraryGroup,
  LibrarySnapshot,
  Mind,
} from "../../src/core/api";
import { chosenScope, composerScope, NO_SCOPE } from "../../src/renderer/src/composer";
import { movedLocally, moveMessage, undoMoves } from "../../src/renderer/src/moves";
import { scopeChips } from "../../src/renderer/src/scope";
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
  kind: "mind",
  createdAt: "",
  updatedAt: "",
  folderId,
});

const group = (id: string, name: string): LibraryGroup => ({
  id,
  name,
  description: "",
  createdAt: "",
  updatedAt: "",
});

const assignment = (documentId: string, groupId: string | null): DocumentGroupAssignment => ({
  documentId,
  groupId,
  source: "automatic",
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
      { kind: "folder", id: "live", name: "Vendor selection", deleted: false, documentKind: null },
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
    settings: { classifier: null, automatic: false },
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

  test("undoing moves each back to where it was, in one move per Folder it came from", () => {
    expect(
      undoMoves({
        folderId: "finance",
        minds: [
          { id: "shortlist", from: null },
          { id: "already", from: "finance" },
        ],
        documents: [
          { id: "moon", from: "research" },
          { id: "harbour", from: null },
        ],
      }),
    ).toEqual([
      { folderId: null, items: { mindIds: ["shortlist"], documentIds: ["harbour"] } },
      { folderId: "research", items: { mindIds: [], documentIds: ["moon"] } },
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
});
