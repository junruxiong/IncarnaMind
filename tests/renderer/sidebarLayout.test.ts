import { describe, expect, test } from "vitest";
import { isNarrow, sidebarIsHidden } from "../../src/renderer/src/sidebarLayout";

describe("the sidebar's visibility", () => {
  test("with the viewer open, a window without room for the sidebar and two readable panes is narrow", () => {
    // 248 + 17 + 720 = 985
    expect(isNarrow(true, 984, 248)).toBe(true);
    expect(isNarrow(true, 985, 248)).toBe(false);
    expect(isNarrow(true, 900, 248)).toBe(true);
    // A wider sidebar needs a wider window.
    expect(isNarrow(true, 1100, 400)).toBe(true);
    expect(isNarrow(false, 900, 248)).toBe(false);
  });

  test("the User's choice hides it in any window", () => {
    expect(sidebarIsHidden({ userHidden: true, narrow: false, forcedShown: false })).toBe(true);
    expect(sidebarIsHidden({ userHidden: true, narrow: true, forcedShown: true })).toBe(true);
  });

  test("a narrow window hides it unless the User showed it since", () => {
    expect(sidebarIsHidden({ userHidden: false, narrow: true, forcedShown: false })).toBe(true);
    expect(sidebarIsHidden({ userHidden: false, narrow: true, forcedShown: true })).toBe(false);
    expect(sidebarIsHidden({ userHidden: false, narrow: false, forcedShown: false })).toBe(false);
  });
});
