import { beforeAll, describe, expect, test } from "vitest";
import { buildMenu, type MenuEntry } from "../../src/renderer/src/commands/menus";
import {
  type CommandContext,
  commands,
  labelOf,
  menuLabelOf,
} from "../../src/renderer/src/commands/registry";
import type { ObjectRef } from "../../src/renderer/src/objects";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

/*
 * The selection bar and the right-click menu offer the same commands for the
 * selection, each saying how many (docs/agents/interaction.md, rule 6).
 */

const doc = (id: string): ObjectRef => ({ kind: "document", id });
const mind = (id: string): ObjectRef => ({ kind: "mind", id });
const selectionOf = (selection: ObjectRef[], surface: CommandContext["surface"] = "library") =>
  ({ selection, focus: selection.at(-1) ?? null, surface }) satisfies CommandContext;

const idsIn = (entries: MenuEntry[]) =>
  entries.flatMap((entry) => (entry === "rule" ? [] : [entry.command.id]));

const english = (key: MessageKey, params?: MessageParams) => translate("en", key, params);
const chinese = (key: MessageKey, params?: MessageParams) => translate("zh-CN", key, params);

beforeAll(async () => {
  const bridge = new Proxy({}, { get: () => () => () => undefined });
  (globalThis as unknown as { window: unknown }).window = {
    incarnamind: bridge,
    incarnamindFiles: bridge,
  };
  await import("../../src/renderer/src/commands/app");
  const { useAppStore } = await import("../../src/renderer/src/store");
  useAppStore.setState({
    minds: ["m", "n"].map((id) => ({ id }) as never),
    documents: ["a", "b", "c"].map(
      (id) =>
        ({
          id,
          name: `Doc ${id}`,
          kind: "pdf",
          status: "ready",
          fileStatus: "available",
          path: `/${id}.pdf`,
        }) as never,
    ),
  });
});

describe("a selection of Documents", () => {
  const ctx = selectionOf([doc("a"), doc("b"), doc("c")]);
  const here = { "document.tags": () => undefined };

  test("is offered what applies to every one, and not what opens or renames one", () => {
    const ids = idsIn(buildMenu("document", { ...ctx, here }));
    expect(ids).toEqual(
      expect.arrayContaining(["document.ask", "document.startMind", "document.tags"]),
    );
    for (const one of [
      "document.open",
      "document.openTab",
      "document.rename",
      "document.openExternally",
      "document.showInFolder",
    ]) {
      expect(ids, one).not.toContain(one);
    }
  });

  test("says how many, in English and in Chinese", () => {
    const entries = buildMenu("document", { ...ctx, here }).flatMap((entry) =>
      entry === "rule" ? [] : [entry],
    );
    const ask = entries.find((entry) => entry.command.id === "document.ask");
    const start = entries.find((entry) => entry.command.id === "document.startMind");
    expect(ask && menuLabelOf(ask.command, ctx, english)).toBe("Ask about these 3 Documents");
    expect(ask && menuLabelOf(ask.command, ctx, chinese)).toBe("就这 3 个文档提问");
    expect(start && labelOf(start.command, ctx, english)).toBe(
      "Start a Mind from these 3 Documents",
    );
    expect(start && labelOf(start.command, ctx, chinese)).toBe("用这 3 个文档新建 Mind");
  });

  test("offers Delete with the count", () => {
    const entries = buildMenu("document", { ...ctx, here }).flatMap((entry) =>
      entry === "rule" ? [] : [entry],
    );
    const remove = entries.find((entry) => entry.command.id === "document.delete");
    expect(remove && menuLabelOf(remove.command, ctx, english)).toBe("Delete 3 Documents");
    expect(remove && menuLabelOf(remove.command, ctx, chinese)).toBe("删除 3 个文档");
  });

  test("a single Document keeps its own words", () => {
    const one = selectionOf([doc("a")]);
    const ask = commands.get("document.ask");
    expect(ask && labelOf(ask, one, english)).toBe("Ask about this Document");
    expect(idsIn(buildMenu("document", one))).toContain("document.open");
  });

  test("asks about all of them, not the first", async () => {
    const { useAppStore } = await import("../../src/renderer/src/store");
    let asked: unknown;
    useAppStore.setState({
      askAbout: ((bridge: unknown) => {
        asked = bridge;
        return undefined;
      }) as never,
    });
    await commands.run("document.ask", ctx);
    expect(asked).toMatchObject({
      kind: "documents",
      count: 3,
      scope: { documentIds: ["a", "b", "c"] },
    });
  });
});

describe("a selection of Minds", () => {
  const ctx = selectionOf([mind("m"), mind("n")], "sidebar");

  test("is offered Move to, and not what opens, exports or deletes one", () => {
    const here = { "mind.moveTo": () => undefined, "mind.rename": () => undefined };
    const ids = idsIn(buildMenu("mind", { ...ctx, here }));
    expect(ids).toEqual(["mind.moveTo"]);
  });
});

describe("a selection of Minds and Documents", () => {
  test("is offered only what applies to every kind selected", () => {
    const ctx = selectionOf([mind("m"), doc("a")], "sidebar");
    expect(idsIn(buildMenu("document", ctx))).toEqual([]);
    expect(idsIn(buildMenu("mind", ctx))).toEqual([]);
  });
});
