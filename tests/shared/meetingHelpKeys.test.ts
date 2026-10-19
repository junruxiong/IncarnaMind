import { describe, expect, test } from "vitest";
import {
  defaultHelpKeys,
  HELP_SHORTCUTS,
  helpKeysOf,
  helpShortcutLabel,
  isGlobalKeys,
  isHelpKeys,
  isHelpShortcut,
} from "../../src/shared/meetingHelp";

describe("The shortcuts held during a call", () => {
  test("Pause my mic is ⌃⌥M on macOS and Ctrl+Shift+U elsewhere, where Ctrl+Alt is AltGr on many layouts", () => {
    expect(defaultHelpKeys("mic", true)).toBe("Ctrl+Alt+M");
    expect(defaultHelpKeys("mic", false)).toBe("Ctrl+Shift+U");
    expect(isHelpShortcut("mic")).toBe(true);
    expect(helpShortcutLabel("mic")).toBe("meetings.mic.pause");
  });

  test("every shortcut has keys of its own a person can hold, on both kinds of computer", () => {
    for (const mac of [true, false]) {
      const keys = HELP_SHORTCUTS.map((shortcut) => defaultHelpKeys(shortcut, mac));
      expect(new Set(keys).size).toBe(keys.length);
      for (const each of keys) expect(isGlobalKeys(each), each).toBe(true);
    }
  });

  test("the User's own keys for it are kept, or none at all", () => {
    expect(isHelpKeys({ mic: "Ctrl+Alt+F8" })).toBe(true);
    expect(isHelpKeys({ mic: "" })).toBe(true);
    expect(isHelpKeys({ mic: "Shift+U" })).toBe(false);
    expect(helpKeysOf("mic", { mic: "Ctrl+Alt+F8" }, false)).toBe("Ctrl+Alt+F8");
    expect(helpKeysOf("mic", { mic: "" }, false)).toBeNull();
    expect(helpKeysOf("mic", {}, true)).toBe("Ctrl+Alt+M");
  });
});
