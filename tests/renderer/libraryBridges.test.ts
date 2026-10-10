import { describe, expect, test } from "vitest";
import type { Mind } from "../../src/core/api";
import { bridgeFrom, mindToAskIn } from "../../src/renderer/src/libraryBridges";
import { scopeAttributesOf, searchScopeOf } from "../../src/shared/searchScope";

const folder = { kind: "folder", id: "folder-research", name: "Research" } as const;

describe("what a Question from the Library searches", () => {
  test("a Folder as it is: the Folder itself, so Documents added to it later count too", () => {
    expect(bridgeFrom(folder, ["a", "b", "c"], false)).toEqual({
      kind: "folder",
      title: "Research",
      count: 3,
      scope: { folderIds: ["folder-research"], tagIds: [], documentIds: [] },
    });
  });

  test("a filtered Folder: exactly the Documents shown, and how many", () => {
    expect(bridgeFrom(folder, ["b", "c"], true)).toEqual({
      kind: "documents",
      title: "Research",
      count: 2,
      scope: { folderIds: [], tagIds: [], documentIds: ["b", "c"] },
    });
  });

  test("Unsorted is no Folder: its Documents, as shown", () => {
    expect(bridgeFrom({ kind: "unsorted", name: "Unsorted" }, ["x"], false)).toEqual({
      kind: "documents",
      title: "Unsorted",
      count: 1,
      scope: { folderIds: [], tagIds: [], documentIds: ["x"] },
    });
  });

  test("all Documents: nothing to narrow unless filtered, then the Documents shown", () => {
    const all = { kind: "all", name: "All Documents" } as const;
    expect(bridgeFrom(all, ["a", "b"], false)).toBeNull();
    expect(bridgeFrom(all, ["a"], true)?.scope.documentIds).toEqual(["a"]);
  });

  test("nothing shown: nothing to ask about", () => {
    expect(bridgeFrom(folder, [], false)).toBeNull();
    expect(bridgeFrom(folder, [], true)).toBeNull();
  });

  test("the scope is stored on the Question as the editor and the core read it", () => {
    const filtered = bridgeFrom(folder, ["b", "c"], true);
    if (!filtered) throw new Error("No bridge.");
    const attributes = scopeAttributesOf(filtered.scope);
    expect(attributes).toEqual({
      scopeFolderIds: null,
      scopeTagIds: null,
      scopeDocumentIds: ["b", "c"],
    });
    expect(searchScopeOf(attributes)).toEqual(filtered.scope);
  });
});

describe("the Mind a Question from the Library goes in", () => {
  const mind = (id: string): Mind => ({
    id,
    title: id,
    kind: "mind",
    createdAt: "",
    updatedAt: "",
    folderId: null,
  });
  // Most recently edited first, as the sidebar lists them.
  const minds = [mind("edited"), mind("older"), mind("example")];

  test("the Mind last shown, which the Library covered", () => {
    expect(mindToAskIn({ minds, openMindId: "older", exampleMindId: "example" })).toBe("older");
  });

  test("with no Mind open, the one edited most recently", () => {
    expect(mindToAskIn({ minds, openMindId: null, exampleMindId: "example" })).toBe("edited");
  });

  test("never the example Mind, nor a Mind deleted meanwhile", () => {
    expect(mindToAskIn({ minds, openMindId: "example", exampleMindId: "example" })).toBe("edited");
    expect(mindToAskIn({ minds, openMindId: "gone", exampleMindId: null })).toBe("edited");
    expect(
      mindToAskIn({ minds: [mind("example")], openMindId: "example", exampleMindId: "example" }),
    ).toBeNull();
  });

  test("no Mind yet: none, so a new one is made", () => {
    expect(mindToAskIn({ minds: [], openMindId: null, exampleMindId: null })).toBeNull();
  });
});
