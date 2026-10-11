import { beforeAll, describe, expect, test } from "vitest";
import { keysForm, keysLabel, keysOn, pressed } from "../../src/renderer/src/commands/keys";
import {
  COMMAND_GROUPS,
  type Command,
  type CommandContext,
  commands,
  labelOf,
} from "../../src/renderer/src/commands/registry";
import { MENU_COMMANDS, type MenuCommandId } from "../../src/shared/commands";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

/*
 * Every command the app has, checked as a reviewer would check rule 4 and
 * rule 7 (docs/agents/interaction.md): named in both languages, placed in a
 * menu group, marked reversible or asking first, reachable by keyboard or
 * ⌘K, and no key bound twice where both commands could act.
 */

const LANGUAGES = ["en", "zh-CN"] as const;
const SYSTEMS = [
  { name: "macOS", mac: true },
  { name: "Windows and Linux", mac: false },
] as const;
const NOWHERE: CommandContext = { selection: [], focus: null, surface: "mind" };

const translator =
  (language: (typeof LANGUAGES)[number]) => (key: MessageKey, params?: MessageParams) =>
    translate(language, key, params);

/** Keys that mean something while the User types: editing, moving the caret, and the note's formatting. */
const TEXT_KEYS = [
  "CmdOrCtrl+A",
  "CmdOrCtrl+C",
  "CmdOrCtrl+V",
  "CmdOrCtrl+X",
  "CmdOrCtrl+Z",
  "CmdOrCtrl+Shift+Z",
  "CmdOrCtrl+Y",
  "CmdOrCtrl+Backspace",
  "CmdOrCtrl+Delete",
  "CmdOrCtrl+Left",
  "CmdOrCtrl+Right",
  "CmdOrCtrl+Up",
  "CmdOrCtrl+Down",
  "Alt+Backspace",
  "Alt+Left",
  "Alt+Right",
  "CmdOrCtrl+B",
  "CmdOrCtrl+I",
  "CmdOrCtrl+Shift+S",
  "CmdOrCtrl+Shift+H",
];

let all: Command[];

beforeAll(async () => {
  // The commands reach the store, which reads the bridge the preload exposes.
  const bridge = new Proxy({}, { get: () => () => () => undefined });
  (globalThis as unknown as { window: unknown }).window = {
    incarnamind: bridge,
    incarnamindFiles: bridge,
  };
  await import("../../src/renderer/src/commands/app");
  all = commands.all();
});

describe("every command", () => {
  test("has a label in English and in Chinese", () => {
    expect(all.length).toBeGreaterThan(0);
    for (const command of all) {
      for (const language of LANGUAGES) {
        const label = labelOf(command, NOWHERE, translator(language));
        expect(label.trim(), `${command.id} in ${language}`).not.toBe("");
        expect(label, `${command.id} in ${language}`).not.toMatch(/\{\w+\}/);
      }
    }
  });

  test("has a menu group and says whether it can be undone", () => {
    for (const command of all) {
      expect(COMMAND_GROUPS, command.id).toContain(command.group);
      expect(
        command.reversible === true || typeof command.reversible.confirm === "string",
        command.id,
      ).toBe(true);
    }
  });

  test("can be found in ⌘K or has a shortcut", () => {
    for (const command of all) {
      expect(command.palette !== false || command.shortcut !== undefined, command.id).toBe(true);
    }
  });

  test("that asks first names what will be lost, in both languages", () => {
    for (const command of all) {
      if (command.reversible === true) continue;
      for (const language of LANGUAGES) {
        const t = translator(language);
        const confirm = t(command.reversible.confirm);
        expect(confirm.trim(), `${command.id} in ${language}`).not.toBe("");
        expect(confirm, `${command.id} in ${language}`).not.toBe(labelOf(command, NOWHERE, t));
      }
    }
  });
});

describe("shortcuts", () => {
  test.each(SYSTEMS)("no keys are bound twice where both commands can act, on $name", ({ mac }) => {
    const bound = all.filter((command) => command.shortcut);
    for (const [index, one] of bound.entries()) {
      for (const other of bound.slice(index + 1)) {
        if (!one.shortcut || !other.shortcut) continue;
        const sameKeys =
          keysForm(keysOn(one.shortcut.keys, mac), mac) ===
          keysForm(keysOn(other.shortcut.keys, mac), mac);
        // Commands for different kinds of object never act on the same focus.
        const overlap =
          !one.appliesTo ||
          !other.appliesTo ||
          one.appliesTo.some((kind) => other.appliesTo?.includes(kind));
        expect(sameKeys && overlap, `${one.id} and ${other.id}`).toBe(false);
      }
    }
  });

  test.each(SYSTEMS)(
    "that act while the User types keep the keys of text to it, on $name",
    ({ mac }) => {
      const text = new Set(TEXT_KEYS.map((keys) => keysForm(keys, mac)));
      for (const command of all) {
        if (!command.shortcut || command.shortcut.list) continue;
        const form = keysForm(keysOn(command.shortcut.keys, mac), mac);
        // A plain key, or one with Shift, types; a function key doesn't.
        expect(/(meta|ctrl|alt)\+|(^|\+)f\d+$/i.test(form), command.id).toBe(true);
        // A key of text is taken only on a Block, where the command says it applies.
        if (command.shortcut.inText) {
          expect(command.appliesTo, command.id).toEqual(["block"]);
          expect(command.when, command.id).toBeDefined();
          expect(command.when?.({ selection: [], focus: null, surface: "mind" }), command.id).toBe(
            false,
          );
          continue;
        }
        expect(text.has(form), command.id).toBe(false);
      }
    },
  );

  test("the application menu's items are commands, with the menu's label and keys", () => {
    for (const id of Object.keys(MENU_COMMANDS) as MenuCommandId[]) {
      const command = commands.get(id);
      expect(command, id).toBeDefined();
      expect(command?.shortcut?.keys).toBe(MENU_COMMANDS[id].keys);
      if (typeof command?.label === "string") expect(command.label).toBe(MENU_COMMANDS[id].label);
    }
  });
});

describe("keys", () => {
  const press = (
    key: string,
    code: string,
    held: Partial<Record<"meta" | "ctrl" | "alt" | "shift", boolean>>,
  ) => ({
    key,
    code,
    metaKey: held.meta ?? false,
    ctrlKey: held.ctrl ?? false,
    altKey: held.alt ?? false,
    shiftKey: held.shift ?? false,
  });

  test("CmdOrCtrl is ⌘ on macOS and Ctrl elsewhere, with exactly the modifiers written", () => {
    expect(pressed("CmdOrCtrl+J", press("j", "KeyJ", { meta: true }), true)).toBe(true);
    expect(pressed("CmdOrCtrl+J", press("j", "KeyJ", { ctrl: true }), true)).toBe(false);
    expect(pressed("CmdOrCtrl+J", press("j", "KeyJ", { ctrl: true }), false)).toBe(true);
    expect(pressed("CmdOrCtrl+J", press("J", "KeyJ", { meta: true, shift: true }), true)).toBe(
      false,
    );
    expect(
      pressed("CmdOrCtrl+Shift+D", press("D", "KeyD", { meta: true, shift: true }), true),
    ).toBe(true);
    expect(pressed("Ctrl+Tab", press("Tab", "Tab", { ctrl: true }), true)).toBe(true);
    expect(pressed("Ctrl+Shift+Tab", press("Tab", "Tab", { ctrl: true, shift: true }), false)).toBe(
      true,
    );
    expect(pressed("CmdOrCtrl+\\", press("\\", "Backslash", { meta: true }), true)).toBe(true);
    expect(pressed("CmdOrCtrl+1", press("1", "Digit1", { ctrl: true }), false)).toBe(true);
  });

  test("letters follow the User's keyboard layout, and the physical key stands in where it types none", () => {
    // AZERTY: Z is where QWERTY has W, so ⌘Z stays the text's undo, and the key marked W closes the tab.
    expect(pressed("CmdOrCtrl+W", press("z", "KeyW", { meta: true }), true)).toBe(false);
    expect(pressed("CmdOrCtrl+W", press("w", "KeyZ", { meta: true }), true)).toBe(true);
    expect(pressed("CmdOrCtrl+1", press("&", "Digit1", { meta: true }), true)).toBe(true);
    // Cyrillic, and a German Mac's key where US layouts have \.
    expect(pressed("CmdOrCtrl+W", press("ц", "KeyW", { ctrl: true }), false)).toBe(true);
    expect(pressed("CmdOrCtrl+\\", press("#", "Backslash", { meta: true }), true)).toBe(true);
  });

  test("a sign typed with AltGr is text, though Windows reports AltGr as Ctrl+Alt", async () => {
    // The keymap reaches the store, which reads the bridge set up above.
    const { runShortcut } = await import("../../src/renderer/src/commands/keymap");
    let ran = 0;
    const stop = commands.register({
      id: "test.altGr",
      label: "menu.newMind",
      group: "main",
      reversible: true,
      shortcut: { keys: "Ctrl+Alt+\\" },
      run: () => {
        ran += 1;
      },
    });
    const globals = globalThis as unknown as { document?: unknown };
    const page = globals.document;
    globals.document = { querySelector: () => null };
    // A German keyboard types \ with AltGr+ß.
    const typed = (altGr: boolean) =>
      ({
        ...press("\\", "Minus", { ctrl: true, alt: true }),
        target: null,
        getModifierState: (key: string) => altGr && key === "AltGraph",
        preventDefault: () => undefined,
      }) as unknown as KeyboardEvent;
    try {
      expect(runShortcut(typed(true), NOWHERE)).toBe(false);
      expect(runShortcut(typed(false), NOWHERE)).toBe(true);
      expect(ran).toBe(1);
    } finally {
      stop();
      globals.document = page;
    }
  });

  test("are written as each system writes them", () => {
    expect(keysLabel("CmdOrCtrl+J", true)).toBe("⌘J");
    expect(keysLabel("CmdOrCtrl+J", false)).toBe("Ctrl+J");
    expect(keysLabel("CmdOrCtrl+\\", true)).toBe("⌘\\");
    expect(keysLabel("CmdOrCtrl+\\", false)).toBe("Ctrl+\\");
    expect(keysLabel("CmdOrCtrl+Shift+Backspace", true)).toBe("⇧⌘⌫");
    expect(keysLabel("CmdOrCtrl+Alt+\\", true)).toBe("⌥⌘\\");
    expect(keysLabel("CmdOrCtrl+Alt+\\", false)).toBe("Ctrl+Alt+\\");
    expect(keysLabel("CmdOrCtrl+Shift+Backspace", false)).toBe("Ctrl+Shift+Backspace");
    expect(keysLabel("Ctrl+Tab", true)).toBe("⌃⇥");
  });

  test("a platform's own keys apply on that platform only", () => {
    const rename = { mac: "Enter", other: "F2" };
    expect(keysOn(rename, true)).toBe("Enter");
    expect(keysOn(rename, false)).toBe("F2");
  });

  test("keys the keymap can't bind are refused when the command is added", () => {
    const base = { group: "main", reversible: true, run: () => undefined } as const;
    expect(() =>
      commands.register({
        ...base,
        id: "test.lower",
        label: "menu.newMind",
        shortcut: { keys: "CmdOrCtrl+n" },
      }),
    ).toThrow();
    expect(() =>
      commands.register({
        ...base,
        id: "test.hyper",
        label: "menu.newMind",
        shortcut: { keys: "Hyper+N" },
      }),
    ).toThrow();
  });
});

describe("the registry", () => {
  const base = { label: "menu.newMind", group: "main", reversible: true } as const;

  test("offers a command for objects only when the selection or focus is of its kinds", async () => {
    let ran: readonly string[] = [];
    const stop = commands.register({
      ...base,
      id: "test.documents",
      appliesTo: ["document"],
      run: (ctx) => {
        ran = ctx.selection.map((ref) => ref.id);
      },
    });
    try {
      const documents: CommandContext = {
        ...NOWHERE,
        selection: [
          { kind: "document", id: "a" },
          { kind: "document", id: "b" },
        ],
      };
      expect(commands.list(NOWHERE).map((command) => command.id)).not.toContain("test.documents");
      expect(commands.list(documents).map((command) => command.id)).toContain("test.documents");
      const mixed = { ...NOWHERE, selection: [...documents.selection, { kind: "mind", id: "m" }] };
      expect(commands.list(mixed as CommandContext).map((command) => command.id)).not.toContain(
        "test.documents",
      );

      await commands.run("test.documents", NOWHERE);
      expect(ran).toEqual([]);
      await commands.run("test.documents", documents);
      expect(ran).toEqual(["a", "b"]);
    } finally {
      stop();
    }
    expect(commands.get("test.documents")).toBeUndefined();
  });

  test("doesn't run a command that isn't enabled, and returns what can undo one that ran", async () => {
    const undo = { message: "Did it", undo: async () => undefined };
    const stops = [
      commands.register({
        ...base,
        id: "test.disabled",
        enabled: () => ({ reason: "menu.newMind" }),
        run: () => undo,
      }),
      commands.register({ ...base, id: "test.undoable", run: async () => undo }),
    ];
    try {
      expect(await commands.run("test.disabled")).toBeUndefined();
      expect(await commands.run("test.undoable")).toBe(undo);
    } finally {
      for (const stop of stops) stop();
    }
  });

  test("refuses two commands with one ID", () => {
    expect(() => commands.register({ ...base, id: "mind.new", run: () => undefined })).toThrow();
  });
});
