import type { MessageKey } from "./i18n";

/** What a command acts on, then what it does: "mind.new", "document.moveTo". */
export type CommandId = `${string}.${string}`;

/**
 * Keys written as an Electron accelerator ("CmdOrCtrl+Shift+M", "Ctrl+Tab",
 * "F2"), so the application menu shows the same keys the window binds.
 * CmdOrCtrl is ⌘ on macOS and Ctrl elsewhere.
 */
export type Keys = string;

/** The same keys on every system, or other keys on macOS. */
export type ShortcutKeys = Keys | { readonly mac: Keys; readonly other: Keys };

/**
 * The application menu's items that are commands. The menu shows each one's
 * label and keys, and a click runs it in the window in front; the window's
 * keymap binds the keys.
 */
export const MENU_COMMANDS = {
  "mind.new": { label: "menu.newMind", keys: "CmdOrCtrl+N" },
  "tab.close": { label: "menu.closeTab", keys: "CmdOrCtrl+W" },
  "settings.open": { label: "menu.settings", keys: "CmdOrCtrl+," },
  "sidebar.toggle": { label: "menu.hideSidebar", keys: "CmdOrCtrl+\\" },
} as const satisfies Record<CommandId, { label: MessageKey; keys: Keys }>;

export type MenuCommandId = keyof typeof MENU_COMMANDS;
