import { BrowserWindow, Menu, type MenuItemConstructorOptions } from "electron";
import type { Language } from "../core/language";
import { FILES_CHANNELS, type MenuCommand } from "../shared/bridge";
import { translate } from "../shared/i18n";

/** Asks the window in front to do what a menu item says (see `files.onMenuCommand`). */
function send(command: MenuCommand): void {
  const window = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0];
  if (window && !window.isDestroyed()) window.webContents.send(FILES_CHANNELS.menuCommand, command);
}

/**
 * The application menu, in the interface language: New Mind (⌘N), Close Tab
 * (⌘W) and Settings… (⌘,) with the usual Edit, View and Window items. As in
 * a browser, ⌘W closes the tab and Shift+⌘W the window; the window's page
 * also handles ⌘T, ⌘1–⌘9 and Ctrl+Tab (see MindTabs.tsx). Reload and the
 * developer tools are only in development and test builds. Ctrl stands in
 * for ⌘ on Windows and Linux.
 */
export function installAppMenu(language: Language, { developer }: { developer: boolean }): void {
  installed = { language, developer };
  buildMenu();
}

/** What the menu was last built from, to build it again when the sidebar is hidden or shown. */
let installed: { language: Language; developer: boolean } | null = null;
let sidebarHidden = false;

/** Makes View say "Show Sidebar" while the sidebar is hidden, "Hide Sidebar" otherwise. */
export function setMenuSidebarHidden(hidden: boolean): void {
  if (hidden === sidebarHidden) return;
  sidebarHidden = hidden;
  if (installed) buildMenu();
}

function buildMenu(): void {
  if (!installed) return;
  const { language, developer } = installed;
  const isMac = process.platform === "darwin";
  const t = (key: Parameters<typeof translate>[1]) => translate(language, key);
  const settings: MenuItemConstructorOptions = {
    label: t("menu.settings"),
    accelerator: "CmdOrCtrl+,",
    click: () => send("open-settings"),
  };

  const template: MenuItemConstructorOptions[] = [
    ...(isMac
      ? [
          {
            label: "IncarnaMind",
            submenu: [
              { role: "about", label: t("menu.about") },
              { type: "separator" },
              settings,
              { type: "separator" },
              { role: "services", label: t("menu.services") },
              { type: "separator" },
              { role: "hide", label: t("menu.hide") },
              { role: "hideOthers", label: t("menu.hideOthers") },
              { role: "unhide", label: t("menu.showAll") },
              { type: "separator" },
              { role: "quit", label: t("menu.quit") },
            ],
          } satisfies MenuItemConstructorOptions,
        ]
      : []),
    {
      label: t("menu.file"),
      submenu: [
        { label: t("menu.newMind"), accelerator: "CmdOrCtrl+N", click: () => send("new-mind") },
        { type: "separator" },
        { label: t("menu.closeTab"), accelerator: "CmdOrCtrl+W", click: () => send("close-tab") },
        { role: "close", label: t("menu.closeWindow"), accelerator: "Shift+CmdOrCtrl+W" },
        ...(isMac
          ? []
          : ([
              { type: "separator" },
              settings,
              { type: "separator" },
              { role: "quit", label: t("menu.quit") },
            ] satisfies MenuItemConstructorOptions[])),
      ],
    },
    {
      label: t("menu.edit"),
      submenu: [
        { role: "undo", label: t("menu.undo") },
        { role: "redo", label: t("menu.redo") },
        { type: "separator" },
        { role: "cut", label: t("menu.cut") },
        { role: "copy", label: t("menu.copy") },
        { role: "paste", label: t("menu.paste") },
        { role: "selectAll", label: t("menu.selectAll") },
      ],
    },
    {
      label: t("menu.view"),
      submenu: [
        ...(developer
          ? ([
              { role: "reload", label: t("menu.reload") },
              { role: "forceReload", label: t("menu.forceReload") },
              { role: "toggleDevTools", label: t("menu.devTools") },
              { type: "separator" },
            ] satisfies MenuItemConstructorOptions[])
          : []),
        {
          label: t(sidebarHidden ? "menu.showSidebar" : "menu.hideSidebar"),
          accelerator: "CmdOrCtrl+\\",
          click: () => send("toggle-sidebar"),
        },
        { type: "separator" },
        { role: "resetZoom", label: t("menu.actualSize") },
        { role: "zoomIn", label: t("menu.zoomIn") },
        { role: "zoomOut", label: t("menu.zoomOut") },
        { type: "separator" },
        { role: "togglefullscreen", label: t("menu.fullScreen") },
      ],
    },
    {
      label: t("menu.window"),
      role: "windowMenu",
      submenu: [
        { role: "minimize", label: t("menu.minimize") },
        { role: "zoom", label: t("menu.zoom") },
        ...(isMac
          ? ([
              { type: "separator" },
              { role: "front", label: t("menu.front") },
            ] satisfies MenuItemConstructorOptions[])
          : []),
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
