/**
 * The window's title bar options per platform (src/shared/titleBar.ts), and
 * telling the page when the window goes full screen, where macOS hides its
 * traffic lights and the page gives their room back.
 */
import type { BrowserWindow, BrowserWindowConstructorOptions } from "electron";
import { FILES_CHANNELS } from "../shared/bridge";
import { titleBarOf } from "../shared/titleBar";

/** Every pane header is 44px (DESIGN.md), and so is the band they make. */
const BAND_HEIGHT = 44;
/** macOS 26's traffic lights are 14px circles (older ones 12px, in a 14px frame). */
const LIGHT_SIZE = 14;

export function titleBarOptions(platform: NodeJS.Platform): BrowserWindowConstructorOptions {
  switch (titleBarOf(platform)) {
    case "inset":
      return {
        titleBarStyle: "hiddenInset",
        // Centred in the band, the first at the sidebar's icon column (x 16).
        trafficLightPosition: { x: 16, y: (BAND_HEIGHT - LIGHT_SIZE) / 2 },
      };
    case "overlay":
      return {
        titleBarStyle: "hidden",
        // Drawn over the band's right end in its colours: `tab-strip`, symbols in `ink-secondary`.
        titleBarOverlay: { color: "#E6E8EB", symbolColor: "#4A4F57", height: BAND_HEIGHT },
      };
    case "native":
      return {};
  }
}

/** Tells the page when the window enters full screen (true) and leaves it (false). */
export function reportFullScreen(window: BrowserWindow): void {
  window.on("enter-full-screen", () => window.webContents.send(FILES_CHANNELS.fullScreen, true));
  window.on("leave-full-screen", () => window.webContents.send(FILES_CHANNELS.fullScreen, false));
}
