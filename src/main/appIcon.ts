import { join } from "node:path";
import { app, type BrowserWindowConstructorOptions, nativeTheme } from "electron";
import linuxIcon from "../../build/icon/png/512x512.png?asset";

// A packaged app takes its icon from its bundle (macOS) or its .exe (Windows), whose small
// sizes are the 16 and 32 px drawings, so only a development build, which runs Electron's own
// binary, is given one here. A Linux window has no icon unless given one, packaged or not.

const iconFolder = () => join(app.getAppPath(), "build", "icon");

export function windowIcon(): Pick<BrowserWindowConstructorOptions, "icon"> {
  if (process.platform === "linux") return { icon: linuxIcon };
  if (process.platform === "win32" && !app.isPackaged) {
    return { icon: join(iconFolder(), "icon.ico") };
  }
  return {};
}

export function showDevelopmentDockIcon(): void {
  const dock = app.dock;
  if (process.platform !== "darwin" || app.isPackaged || !dock) return;
  const show = () =>
    dock.setIcon(
      join(iconFolder(), nativeTheme.shouldUseDarkColors ? "dock-dark.png" : "dock-light.png"),
    );
  show();
  nativeTheme.on("updated", show);
}
