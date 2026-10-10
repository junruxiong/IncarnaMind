import type { Keys, ShortcutKeys } from "../../../shared/commands";

interface KeyPress {
  key: string;
  code: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}

/** The key as the accelerator writes it, its physical key (`KeyboardEvent.code`), and the modifiers. */
interface Chord {
  key: string;
  code: string;
  meta: boolean;
  ctrl: boolean;
  alt: boolean;
  shift: boolean;
}

const CODES: Readonly<Record<string, string>> = {
  ",": "Comma",
  ".": "Period",
  "/": "Slash",
  "\\": "Backslash",
  "[": "BracketLeft",
  "]": "BracketRight",
  "-": "Minus",
  "=": "Equal",
  Space: "Space",
  Tab: "Tab",
  Backspace: "Backspace",
  Delete: "Delete",
  Enter: "Enter",
  Escape: "Escape",
  Up: "ArrowUp",
  Down: "ArrowDown",
  Left: "ArrowLeft",
  Right: "ArrowRight",
  Home: "Home",
  End: "End",
};

const MAC_GLYPHS: Readonly<Record<string, string>> = {
  Backspace: "⌫",
  Delete: "⌦",
  Enter: "↩",
  Tab: "⇥",
  Escape: "⎋",
  Up: "↑",
  Down: "↓",
  Left: "←",
  Right: "→",
};

export const keysOn = (shortcut: ShortcutKeys, mac: boolean): Keys =>
  typeof shortcut === "string" ? shortcut : mac ? shortcut.mac : shortcut.other;

const parsed = new Map<string, Chord>();

/** Reads an accelerator, or throws for one the keymap can't bind. */
function chordOf(keys: Keys, mac: boolean): Chord {
  const cacheKey = `${mac}:${keys}`;
  const cached = parsed.get(cacheKey);
  if (cached) return cached;
  const modifiers = keys.split("+");
  const key = modifiers.pop() ?? "";
  const code = /^[A-Z]$/.test(key)
    ? `Key${key}`
    : /^\d$/.test(key)
      ? `Digit${key}`
      : /^F([1-9]|1\d|2[0-4])$/.test(key)
        ? key
        : CODES[key];
  if (!code) throw new Error(`A shortcut names a key the keymap doesn't know: ${keys}`);
  const chord: Chord = { key, code, meta: false, ctrl: false, alt: false, shift: false };
  for (const modifier of modifiers) {
    if (modifier === "CmdOrCtrl") chord[mac ? "meta" : "ctrl"] = true;
    else if (modifier === "Cmd") chord.meta = true;
    else if (modifier === "Ctrl") chord.ctrl = true;
    else if (modifier === "Alt") chord.alt = true;
    else if (modifier === "Shift") chord.shift = true;
    else throw new Error(`A shortcut names a modifier the keymap doesn't know: ${keys}`);
  }
  parsed.set(cacheKey, chord);
  return chord;
}

/** Throws if the keymap can't bind these keys on macOS or elsewhere. */
export function checkKeys(shortcut: ShortcutKeys): void {
  chordOf(keysOn(shortcut, true), true);
  chordOf(keysOn(shortcut, false), false);
}

/** The keys in one form, to compare: two accelerators with the same form are the same keys. */
export function keysForm(keys: Keys, mac: boolean): string {
  const { code, meta, ctrl, alt, shift } = chordOf(keys, mac);
  return [meta && "meta", ctrl && "ctrl", alt && "alt", shift && "shift", code]
    .filter(Boolean)
    .join("+");
}

/**
 * A letter, digit or sign is the key that types it in the User's layout, as
 * browsers match shortcuts: on AZERTY, ⌘Z is pressed where QWERTY has W, and
 * must not close the tab. Where the layout types no Latin letter or digit
 * there (Cyrillic, ⌥ held, a sign on another key), the physical key stands in.
 */
function sameKey(chord: Chord, press: KeyPress): boolean {
  if (chord.key.length === 1 && press.key.length === 1) {
    if (press.key.toUpperCase() === chord.key) return true;
    if (/^[a-z0-9]$/i.test(press.key)) return false;
  }
  return press.code === chord.code;
}

/** Whether a key pressed is these keys, with exactly these modifiers. */
export function pressed(keys: Keys, press: KeyPress, mac: boolean): boolean {
  const chord = chordOf(keys, mac);
  return (
    sameKey(chord, press) &&
    press.metaKey === chord.meta &&
    press.ctrlKey === chord.ctrl &&
    press.altKey === chord.alt &&
    press.shiftKey === chord.shift
  );
}

/** The keys as this system writes them: "⌘⇧⌫" on macOS, "Ctrl+Shift+Backspace" elsewhere. */
export function keysLabel(keys: Keys, mac: boolean): string {
  const { key, meta, ctrl, alt, shift } = chordOf(keys, mac);
  if (mac) {
    const modifiers = `${meta ? "⌘" : ""}${ctrl ? "⌃" : ""}${alt ? "⌥" : ""}${shift ? "⇧" : ""}`;
    return `${modifiers}${MAC_GLYPHS[key] ?? key}`;
  }
  return [ctrl && "Ctrl", alt && "Alt", shift && "Shift", meta && "Win", key]
    .filter(Boolean)
    .join("+");
}
