import { beforeAll, describe, expect, test } from "vitest";
import {
  buildMenu,
  type MenuEntry,
  menuIds,
  registerMenu,
} from "../../src/renderer/src/commands/menus";
import {
  type Command,
  type CommandContext,
  commands,
} from "../../src/renderer/src/commands/registry";
import type { ObjectKind, ObjectRef } from "../../src/renderer/src/objects";

/*
 * Every object has one menu, in one order (docs/agents/interaction.md, rule
 * 5): the registry puts a feature's commands in the rule's order, leaves out
 * what doesn't apply and disables, with a reason, what applies but can't run
 * now. That every kind has a menu, and its order, is in conformance.test.ts.
 */

const focusOn = (kind: ObjectKind, here?: CommandContext["here"]): CommandContext => ({
  selection: [],
  focus: { kind, id: "one" },
  surface: "sidebar",
  ...(here && { here }),
});

const idsIn = (entries: MenuEntry[]) =>
  entries.map((entry) => (entry === "rule" ? "—" : entry.command.id));

beforeAll(async () => {
  // The commands reach the store, which reads the bridge the preload exposes.
  const bridge = new Proxy({}, { get: () => () => () => undefined });
  (globalThis as unknown as { window: unknown }).window = {
    incarnamind: bridge,
    incarnamindFiles: bridge,
  };
  await import("../../src/renderer/src/commands/app");
  const { useAppStore } = await import("../../src/renderer/src/store");
  useAppStore.setState({
    minds: [{ id: "one" } as never],
    documents: [
      { id: "one", path: "/Contracts/MSA.pdf", web: null, fileStatus: "available" } as never,
    ],
    linkedFolders: [{ id: "one", path: "/Contracts", status: "idle" } as never],
  });
});

describe("the menus", () => {
  test("a Mind's is Open, Open to the side, Outline, Export as… and Export options…, History, Make this a Skill, Move to and New Folder with this Mind, Rename, then Delete after a rule", () => {
    const here = {
      "mind.citationStyle": () => undefined,
      "mind.moveTo": () => undefined,
      "mind.rename": () => undefined,
    };
    expect(idsIn(buildMenu("mind", focusOn("mind", here)))).toEqual([
      "mind.open",
      "mind.openSide",
      "—",
      "mind.outline",
      "mind.export",
      "mind.exportAs.pdf",
      "mind.exportAs.source-pack",
      "mind.exportAs.html",
      "mind.exportAs.markdown",
      "mind.exportOptions",
      "mind.history",
      "mind.citationStyle",
      "skill.fromMind",
      "—",
      "mind.moveTo",
      "folder.newWith",
      "—",
      "mind.rename",
      "—",
      "mind.delete",
    ]);
    // Rename and Move to start in the row, so a place without one doesn't offer them.
    expect(idsIn(buildMenu("mind", focusOn("mind")))).toEqual([
      "mind.open",
      "mind.openSide",
      "—",
      "mind.outline",
      "mind.export",
      "mind.exportAs.pdf",
      "mind.exportAs.source-pack",
      "mind.exportAs.html",
      "mind.exportAs.markdown",
      "mind.exportOptions",
      "mind.history",
      "skill.fromMind",
      "—",
      "folder.newWith",
      "—",
      "mind.delete",
    ]);
  });

  test("a Document's file group is Open in default app, Show the file, Copy path, then Get info", () => {
    const here = { "document.rename": () => undefined };
    const ids = idsIn(buildMenu("document", focusOn("document", here)));
    const start = ids.indexOf("document.openExternally");
    expect(ids.slice(start, ids.indexOf("—", start))).toEqual([
      "document.openExternally",
      "document.showInFolder",
      "file.copyPath",
      "file.info",
    ]);
    expect(ids.at(-1)).toBe("document.delete");
  });

  test("a Linked folder's has Copy path and Get info with its file items", () => {
    const ids = idsIn(buildMenu("linkedFolder", focusOn("linkedFolder")));
    expect(ids).toEqual(
      expect.arrayContaining([
        "linkedFolder.reveal",
        "file.copyPath",
        "file.info",
        "linkedFolder.locate",
      ]),
    );
    expect(ids.indexOf("file.copyPath")).toBe(ids.indexOf("linkedFolder.reveal") + 1);
  });

  test("a feature adds its items to a kind's menu, and they take their group's place", () => {
    const registered: Command[] = [
      {
        id: "test.late",
        label: "mind.open",
        group: "move",
        appliesTo: ["conversation"],
        reversible: true,
        run: () => undefined,
      },
      {
        id: "test.early",
        label: "mind.open",
        group: "open",
        appliesTo: ["conversation"],
        reversible: true,
        run: () => undefined,
      },
      {
        id: "test.last",
        label: "mind.open",
        group: "danger",
        appliesTo: ["conversation"],
        reversible: { confirm: "deleted.confirm.foot" },
        run: () => undefined,
      },
    ];
    for (const command of registered) commands.register(command);
    // Registered in any order, from different places.
    registerMenu("conversation", ["test.last", "test.late"]);
    registerMenu("conversation", ["test.early", "test.late"]);
    expect(menuIds("conversation")).toEqual(["test.last", "test.late", "test.early"]);
    expect(idsIn(buildMenu("conversation", focusOn("conversation")))).toEqual([
      "test.early",
      "—",
      "test.late",
      "—",
      "test.last",
    ]);
  });

  test("leave out a command that doesn't apply, and disable one that can't run now, with its reason", () => {
    const base = {
      label: "mind.open",
      group: "main",
      reversible: true,
      run: () => undefined,
    } as const;
    commands.register({ ...base, id: "test.notHere", appliesTo: ["folder"], when: () => false });
    commands.register({ ...base, id: "test.wrongKind", appliesTo: ["document"] });
    commands.register({
      ...base,
      id: "test.notNow",
      appliesTo: ["folder"],
      enabled: () => ({ reason: "documents.file.missingReason" }),
    });
    commands.register({ ...base, id: "test.fine", appliesTo: ["folder"] });
    registerMenu("folder", ["test.notHere", "test.wrongKind", "test.notNow", "test.fine"]);
    const entries = buildMenu("folder", focusOn("folder"));
    const items = entries.flatMap((entry) => (entry === "rule" ? [] : [entry]));
    const byId = Object.fromEntries(items.map((item) => [item.command.id, item.enabled]));
    expect(byId["test.notHere"]).toBeUndefined();
    expect(byId["test.wrongKind"]).toBeUndefined();
    expect(byId["test.notNow"]).toEqual({ reason: "documents.file.missingReason" });
    expect(byId["test.fine"]).toBe(true);
  });

  test("act on the selection when it holds the row, whose count a label can say", () => {
    const selection: ObjectRef[] = [
      { kind: "document", id: "a" },
      { kind: "document", id: "b" },
      { kind: "document", id: "c" },
    ];
    commands.register({
      id: "test.count",
      label: (ctx, t) => `${t("deleted.restore")} ${ctx.selection.length}`,
      group: "danger",
      appliesTo: ["document"],
      reversible: true,
      run: () => undefined,
    });
    registerMenu("document", ["test.count"]);
    const ctx: CommandContext = { selection, focus: selection[0] ?? null, surface: "library" };
    const item = buildMenu("document", ctx).find(
      (entry) => entry !== "rule" && entry.command.id === "test.count",
    );
    expect(item).toBeDefined();
    // And a selection with something that isn't a Document is no menu for Documents.
    const mixed: CommandContext = { ...ctx, selection: [...selection, { kind: "mind", id: "m" }] };
    expect(
      buildMenu("document", mixed).some((e) => e !== "rule" && e.command.id === "test.count"),
    ).toBe(false);
  });
});
