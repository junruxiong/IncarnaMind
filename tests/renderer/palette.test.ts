import { beforeAll, describe, expect, test } from "vitest";
import type { Document, Mind, Tag } from "../../src/core/api";
import type * as Palette from "../../src/renderer/src/palette";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

let palette: typeof Palette;

beforeAll(async () => {
  // The commands reach the store, which reads the bridge the preload exposes.
  const bridge = new Proxy({}, { get: () => () => () => undefined });
  (globalThis as unknown as { window: unknown }).window = {
    incarnamind: bridge,
    incarnamindFiles: bridge,
  };
  await import("../../src/renderer/src/commands/app");
  palette = await import("../../src/renderer/src/palette");
});

const en = (key: MessageKey, params?: MessageParams) => translate("en", key, params);
const zh = (key: MessageKey, params?: MessageParams) => translate("zh-CN", key, params);
const AT = "2026-10-13T09:00:00.000Z";

const mind = (id: string, title: string, updatedAt = AT): Mind => ({
  id,
  title,
  autoTitle: "",
  kind: "mind",
  createdAt: AT,
  updatedAt,
  folderId: id === "m-finance" ? "f-finance" : null,
});

const documentNamed = (id: string, name: string, tagIds: string[] = []) =>
  ({ id, name, kind: "pdf", tags: tagIds.map((tagId) => ({ tagId })) }) as unknown as Document;

const things = {
  minds: [
    mind("m-contract", "Contract checklist", "2026-10-12T09:00:00.000Z"),
    mind("m-finance", "财务月报：九月"),
    mind("m-standup", "Stand-up, 6 October", "2026-10-11T09:00:00.000Z"),
  ],
  documents: [
    documentNamed("d-register", "Contracts register 2026", ["t-contract"]),
    documentNamed("d-msa", "Aldermoor MSA", ["t-contract"]),
    documentNamed("d-report", "2026 财务报表"),
  ],
  folders: [
    { id: "f-finance", name: "财务", description: "" },
    { id: "f-contracts", name: "Contracts & agreements", description: "" },
  ] as never,
  assignments: [{ documentId: "d-report", groupId: "f-finance" }] as never,
  tags: [{ id: "t-contract", name: "Contract", colour: "red" } as unknown as Tag],
};

const shown = (groups: Palette.PaletteGroup[]) =>
  groups.map((group) => [group.id, group.items.map((item) => item.name)]);

describe("⌘K's names", () => {
  test("Minds, Documents, then Folders and Tags, each best match first", () => {
    const { words } = palette.readQuery("contract");
    expect(shown(palette.nameGroups(words, things, en))).toEqual([
      ["minds", ["Contract checklist"]],
      ["documents", ["Contracts register 2026"]],
      ["foldersAndTags", ["Contract", "Contracts & agreements"]],
    ]);
    const [, , foldersAndTags] = palette.nameGroups(words, things, en);
    expect(foldersAndTags?.items.map((item) => ("meta" in item ? item.meta : ""))).toEqual([
      "Tag · on 2 Documents",
      "Folder · 0 Documents",
    ]);
  });

  test("Chinese names, in Chinese, with the Folder each is in", () => {
    const groups = palette.nameGroups(palette.readQuery("财务").words, things, zh);
    expect(shown(groups)).toEqual([
      ["minds", ["财务月报：九月"]],
      ["documents", ["2026 财务报表"]],
      ["foldersAndTags", ["财务"]],
    ]);
    expect(groups[1]?.items[0]).toMatchObject({ meta: "PDF · 财务" });
  });

  test("with nothing typed, the Minds edited last", () => {
    expect(shown(palette.recentGroup(things, en))).toEqual([
      ["recent", ["财务月报：九月", "Contract checklist", "Stand-up, 6 October"]],
    ]);
    expect(palette.recentGroup({ ...things, minds: [] }, en)).toEqual([]);
  });
});

describe("⌘K's actions", () => {
  test("are the registry's, found by name or keyword, in English and in Chinese", () => {
    const actions = (query: string, t = en) =>
      shown(palette.actionGroup(palette.readQuery(query).words, palette.PALETTE_CONTEXT, t));
    expect(actions("settings models")).toEqual([["actions", ["Settings: Models"]]]);
    expect(actions("new mind")).toEqual([["actions", ["New Mind"]]]);
    expect(actions("设置 模型", zh)).toEqual([["actions", ["设置：模型"]]]);
    expect(actions("zzqx")).toEqual([]);
  });

  test("with nothing typed, start with what makes something new, and never list ⌘K itself", () => {
    const [actions] = palette.actionGroup([], palette.PALETTE_CONTEXT, en);
    const keys = actions?.items.map((item) => item.key) ?? [];
    expect(keys[0]).toBe("command:mind.new");
    expect(keys.indexOf("command:settings.open")).toBeGreaterThan(
      keys.indexOf("command:webPage.add"),
    );
    expect(keys.at(-1)).toBe("command:deleted.empty");
    expect(keys).not.toContain("command:palette.open");
  });

  test('">" asks for actions only; the words inside are searched for from two letters, or one Chinese character', () => {
    expect(palette.readQuery("> settings")).toEqual({
      text: "settings",
      words: ["settings"],
      actionsOnly: true,
    });
    expect(palette.readQuery(" Contract ").actionsOnly).toBe(false);
    expect(palette.searchesWords("c")).toBe(false);
    expect(palette.searchesWords("co")).toBe(true);
    expect(palette.searchesWords("财")).toBe(true);
  });
});
