import { Menu, type MenuItemConstructorOptions } from "electron";

/**
 * The application menu: Electron's default one, except that closing the
 * window moves to Shift+⌘W (Shift+Ctrl+W). The Mind pane has tabs, and as in
 * a browser ⌘W (Ctrl+W) closes the tab; the window's page handles that, with
 * ⌘T, ⌘1–⌘9 and Ctrl+Tab (see MindTabs.tsx). The default menu would close
 * the whole window on ⌘W first.
 */
export function installAppMenu(): void {
  const isMac = process.platform === "darwin";
  const closeWindow: MenuItemConstructorOptions = {
    role: "close",
    accelerator: "Shift+CmdOrCtrl+W",
  };
  const template: MenuItemConstructorOptions[] = [
    ...(isMac ? [{ role: "appMenu" } satisfies MenuItemConstructorOptions] : []),
    { role: "fileMenu", submenu: isMac ? [closeWindow] : [closeWindow, { role: "quit" }] },
    { role: "editMenu" },
    { role: "viewMenu" },
    {
      role: "windowMenu",
      submenu: isMac
        ? [{ role: "minimize" }, { role: "zoom" }, { type: "separator" }, { role: "front" }]
        : [{ role: "minimize" }, { role: "zoom" }],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
