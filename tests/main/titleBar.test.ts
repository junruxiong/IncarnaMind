import { describe, expect, it } from "vitest";
import { titleBarOptions } from "../../src/main/titleBar";

/*
 * The window's title bar per platform. Only macOS's runs in the smoke tests
 * (e2e/titleBar.spec.ts), so Windows' and Linux's options are pinned here.
 */
describe("titleBarOptions", () => {
  it("puts macOS's traffic lights in the 44px band, centred, on the sidebar's icon column", () => {
    expect(titleBarOptions("darwin")).toEqual({
      titleBarStyle: "hiddenInset",
      trafficLightPosition: { x: 16, y: 15 },
    });
  });

  it("draws Windows' buttons over the band's right end, as tall as the band and in its colours", () => {
    expect(titleBarOptions("win32")).toEqual({
      titleBarStyle: "hidden",
      titleBarOverlay: { color: "#E6E8EB", symbolColor: "#4A4F57", height: 44 },
    });
  });

  it("keeps Linux's own title bar", () => {
    expect(titleBarOptions("linux")).toEqual({});
  });
});
