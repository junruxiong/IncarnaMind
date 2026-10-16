import { beforeAll, describe, expect, test } from "vitest";
import type { ConversationMatch, Mind } from "../../src/core/api";
import type { FindableKind, FoundObject } from "../../src/renderer/src/finding";
import type { ObjectKind, ObjectRef } from "../../src/renderer/src/objects";
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

const kindOf = (kind: ObjectKind, group: string, title: MessageKey): FindableKind => ({
  kind,
  group: { id: group, title },
  objects: () => [],
  subscribe: () => () => undefined,
  open: () => undefined,
});

const object = (kind: ObjectKind, id: string, name: string, meta = ""): FoundObject => ({
  ref: { kind, id },
  name,
  meta,
  icon: () => null,
});

const mind = (id: string, title: string, updatedAt = AT): Mind => ({
  id,
  title,
  autoTitle: "",
  kind: "mind",
  createdAt: AT,
  updatedAt,
  folderId: null,
});

const minds = [
  mind("m-contract", "Contract checklist", "2026-10-12T09:00:00.000Z"),
  mind("m-finance", "财务月报：九月"),
  mind("m-standup", "Stand-up, 6 October", "2026-10-11T09:00:00.000Z"),
];

function findingsOf(groups: [FindableKind, FoundObject[]][]): Palette.Findings {
  const names = new Map<string, string>();
  for (const [, objects] of groups) {
    for (const each of objects) names.set(`${each.ref.kind}:${each.ref.id}`, each.name);
  }
  return { kinds: groups.map(([kind, objects]) => ({ kind, objects })), names };
}

const found = findingsOf([
  [
    kindOf("mind", "minds", "palette.group.minds"),
    minds.map((each) => object("mind", each.id, each.title, "Mind")),
  ],
  [
    kindOf("conversation", "conversations", "palette.group.conversations"),
    [object("conversation", "c-carriers", "Carrier contracts ending in 2027")],
  ],
  [
    kindOf("document", "documents", "palette.group.documents"),
    [
      object("document", "d-register", "Contracts register 2026", "PDF"),
      object("document", "d-msa", "Aldermoor MSA", "PDF"),
      object("document", "d-report", "2026 财务报表", "PDF · 财务"),
      object("document", "d-agreement", "供应商合同", "Word"),
    ],
  ],
  [
    kindOf("folder", "foldersAndTags", "palette.group.foldersAndTags"),
    [
      object("folder", "f-finance", "财务", "Folder · 1 Document"),
      object("folder", "f-contracts", "Contracts & agreements"),
    ],
  ],
  [
    kindOf("tag", "foldersAndTags", "palette.group.foldersAndTags"),
    [object("tag", "t-contract", "Contract", "Tag · on 2 Documents")],
  ],
]);

const shown = (groups: Palette.PaletteGroup[]) =>
  groups.map((group) => [group.id, group.items.map((item) => item.name)]);
const names = (query: string) => shown(palette.nameGroups(palette.readQuery(query).words, found));

describe("⌘K's names", () => {
  test("each registered kind, in the order registered, kinds sharing a group ranked together", () => {
    expect(names("contract")).toEqual([
      ["minds", ["Contract checklist"]],
      ["conversations", ["Carrier contracts ending in 2027"]],
      ["documents", ["Contracts register 2026"]],
      ["foldersAndTags", ["Contract", "Contracts & agreements"]],
    ]);
    const [, , , foldersAndTags] = palette.nameGroups(["contract"], found);
    expect(foldersAndTags?.title).toBe("palette.group.foldersAndTags");
  });

  test("Chinese names by their characters, their pinyin and its initials, and a name with a typo", () => {
    const finance = [
      ["minds", ["财务月报：九月"]],
      ["documents", ["2026 财务报表"]],
      ["foldersAndTags", ["财务"]],
    ];
    expect(names("财务")).toEqual(finance);
    expect(names("caiwu")).toEqual(finance);
    expect(names("cw")).toEqual(finance);
    expect(names("hetong")).toEqual([["documents", ["供应商合同"]]]);
    expect(names("contarct")).toEqual([
      ["minds", ["Contract checklist"]],
      ["conversations", ["Carrier contracts ending in 2027"]],
      ["documents", ["Contracts register 2026"]],
      ["foldersAndTags", ["Contract", "Contracts & agreements"]],
    ]);
  });

  test("with nothing typed, the Minds edited last", () => {
    expect(shown(palette.recentGroup(minds, found))).toEqual([
      ["recent", ["财务月报：九月", "Contract checklist", "Stand-up, 6 October"]],
    ]);
    expect(palette.recentGroup([], found)).toEqual([]);
  });
});

describe("⌘K's actions", () => {
  const actions = (query: string, language: "en" | "zh-CN" = "en") =>
    shown(
      palette.actionGroup(
        palette.readQuery(query).words,
        { selection: [], focus: null, surface: "palette" },
        language,
        found.names,
      ),
    );

  test("are the registry's, found by their name or keywords in English or in Chinese", () => {
    expect(actions("settings models")).toEqual([["actions", ["Settings: Models"]]]);
    expect(actions("new mind")).toEqual([["actions", ["New Mind"]]]);
    expect(actions("设置 模型")).toEqual([["actions", ["Settings: Models"]]]);
    expect(actions("设置 模型", "zh-CN")).toEqual([["actions", ["设置：模型"]]]);
    expect(actions("settings models", "zh-CN")).toEqual([["actions", ["设置：模型"]]]);
    expect(actions("zzqx")).toEqual([]);
  });

  test("say what a command is for in a line under its name, in the language shown, and most have none", () => {
    const itemOf = (key: string, language: "en" | "zh-CN") =>
      palette
        .actionGroup(
          palette.readQuery("verify").words,
          { selection: [], focus: null, surface: "palette" },
          language,
          found.names,
        )[0]
        ?.items.find((item) => item.key === key);

    expect(itemOf("command:verify.document", "en")).toMatchObject({
      purpose: "Check the claims in a PDF someone else wrote",
    });
    expect(itemOf("command:verify.document", "zh-CN")).toMatchObject({
      purpose: "核对别人写的 PDF 里的说法",
    });
    const [list] = palette.actionGroup(
      [],
      { selection: [], focus: null, surface: "palette" },
      "en",
      found.names,
    );
    expect(list?.items.find((item) => item.key === "command:mind.new")).toMatchObject({
      purpose: "",
    });
  });

  test("act on the selection, saying how many, or on the one thing in front, naming it", () => {
    const [withTwo] = palette.actionGroup(
      ["new", "folder"],
      {
        selection: [
          { kind: "document", id: "d-register" },
          { kind: "document", id: "d-msa" },
        ],
        focus: null,
        surface: "palette",
      },
      "en",
      found.names,
    );
    expect(withTwo?.items.map((item) => item.name)).toContain("New Folder with 2 Documents");
    const one: ObjectRef = { kind: "document", id: "d-msa" };
    const [onOne] = palette.actionGroup(
      [],
      { selection: [], focus: one, surface: "palette" },
      "en",
      found.names,
    );
    expect(onOne?.items.find((item) => item.key === "command:document.delete")?.meta).toBe(
      "Aldermoor MSA",
    );
  });

  test("with nothing typed, start with what makes something new, and never list ⌘K itself", () => {
    const [list] = palette.actionGroup(
      [],
      { selection: [], focus: null, surface: "palette" },
      "en",
      found.names,
    );
    const keys = list?.items.map((item) => item.key) ?? [];
    expect(keys[0]).toBe("command:mind.new");
    expect(keys.indexOf("command:settings.open")).toBeGreaterThan(
      keys.indexOf("command:webPage.add"),
    );
    expect(keys.at(-1)).toBe("command:deleted.empty");
    expect(keys).not.toContain("command:palette.open");
  });
});

describe("what is typed", () => {
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

  test("words in conversations show the first message that holds them, and asking names the conversation", () => {
    const match: ConversationMatch = {
      conversation: {
        id: "c-carriers",
        title: "",
        autoTitle: "Which carriers end in 2027?",
        folderId: null,
        askedFrom: [],
        createdAt: AT,
        updatedAt: AT,
      },
      titleHighlights: [],
      messages: [
        {
          messageId: "a-1",
          kind: "answer",
          excerpt: "Leeds Docks ends its lease on 31 December.",
          highlights: [{ start: 22, end: 27 }],
        },
      ],
    };
    const [group] = palette.conversationTextGroup([match, { ...match, messages: [] }]);
    expect(group?.items).toEqual([
      expect.objectContaining({
        kind: "said",
        conversationId: "c-carriers",
        answerId: "a-1",
        name: "Which carriers end in 2027?",
        excerpt: "Leeds Docks ends its lease on 31 December.",
      }),
    ]);
    expect(shown(palette.askGroup("how long is the warranty", "New chat", en))).toEqual([
      ["ask", ["Ask about “how long is the warranty”"]],
    ]);
    const [ask] = palette.askGroup("保修期多长", "供应商审查", zh);
    expect(ask?.items[0]).toMatchObject({ name: "就“保修期多长”提问", meta: "在“供应商审查”里" });
  });
});
