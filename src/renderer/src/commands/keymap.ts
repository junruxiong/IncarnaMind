import { useEffect } from "react";
import type { CommandId } from "../../../shared/commands";
import { files } from "../core";
import { useAppStore } from "../store";
import { typesText } from "../undo";
import { keysLabel, keysOn, pressed } from "./keys";
import {
  type Command,
  type CommandContext,
  commands,
  isEnabled,
  offered,
  SURFACES,
  type Surface,
} from "./registry";

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/** Lists whose items can be selected, where a list's own keys act. */
const SELECTABLE_LIST =
  '[role="listbox"], [role="tree"], [role="treegrid"], [role="grid"], [role="tablist"]';

const inList = (target: EventTarget | null) =>
  target instanceof Element && !typesText(target) && target.closest(SELECTABLE_LIST) !== null;

const isSurface = (value: string | undefined): value is Surface =>
  (SURFACES as readonly (string | undefined)[]).includes(value);

/** The surface an element is on: the nearest `data-surface`, or else the Mind. */
function surfaceOf(target: EventTarget | null): Surface {
  const marked =
    target instanceof Element
      ? target.closest<HTMLElement>("[data-surface]")?.dataset.surface
      : undefined;
  return isSurface(marked) ? marked : "mind";
}

const contextAt = (target: EventTarget | null): CommandContext => ({
  selection: [],
  focus: null,
  surface: surfaceOf(target),
});

const dialogOpen = () => document.querySelector("dialog[open]") !== null;

const report = (error: unknown) => useAppStore.getState().reportError(error);

/**
 * Runs the command this key is the shortcut of, if it is offered and enabled
 * here: true if it ran. Nothing while a dialog is open. With `objectsOnly`,
 * only commands that act on objects are considered.
 */
export function runShortcut(
  event: KeyboardEvent,
  ctx: CommandContext,
  { objectsOnly = false } = {},
): boolean {
  // Windows reports AltGr as Ctrl+Alt: what it types (a German \, a Polish ą) is text.
  if (event.getModifierState("AltGraph")) return false;
  for (const command of commands.all()) {
    const shortcut = command.shortcut;
    if (!shortcut || (objectsOnly && !command.appliesTo)) continue;
    const keys = [
      keysOn(shortcut.keys, isMac),
      ...(!isMac && shortcut.alsoOther ? [shortcut.alsoOther] : []),
    ];
    if (!keys.some((each) => pressed(each, event, isMac))) continue;
    if (shortcut.list && !inList(event.target)) continue;
    if (shortcut.outsideText && typesText(event.target)) continue;
    // Only now, for keys that are a shortcut: the search goes through the whole page.
    if (dialogOpen()) return false;
    if (!offered(command, ctx) || !isEnabled(command, ctx)) continue;
    event.preventDefault();
    commands.run(command.id, ctx).catch(report);
    return true;
  }
  return false;
}

/**
 * The window's keymap, where every command's shortcut is bound, and the
 * application menu's items, which run their commands here.
 */
export function useKeymap(): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!event.defaultPrevented) runShortcut(event, contextAt(event.target));
    };
    window.addEventListener("keydown", onKey);
    const stopMenu = files.onMenuCommand((id) => {
      const focus = document.activeElement;
      // In text, Undo and Redo are the text's own, as their keys are, in a dialog too.
      const textsOwn = commands.get(id)?.shortcut?.outsideText === true && typesText(focus);
      if (textsOwn || !dialogOpen()) commands.run(id, contextAt(focus)).catch(report);
    });
    return () => {
      window.removeEventListener("keydown", onKey);
      stopMenu();
    };
  }, []);
}

/** A command's shortcut as this system writes it, for tooltips and menus. */
export function shortcutLabel(id: CommandId): string {
  const shortcut = commands.get(id)?.shortcut;
  if (!shortcut) throw new Error(`The command ${id} has no shortcut.`);
  return keysLabel(keysOn(shortcut.keys, isMac), isMac);
}

/** The keys a menu shows beside a command: its shortcut, or the keys its row takes, as this system writes them. */
export function menuKeysLabel(command: Command): string | null {
  const keys = command.shortcut?.keys ?? command.rowKeys;
  return keys ? keysLabel(keysOn(keys, isMac), isMac) : null;
}
