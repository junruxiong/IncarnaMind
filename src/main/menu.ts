import { BrowserWindow, Menu, type MenuItemConstructorOptions } from "electron";
import type { Language } from "../core/language";
import { FILES_CHANNELS, type MenuCommand } from "../shared/bridge";
import { MENU_COMMANDS } from "../shared/commands";
import { type MessageKey, translate } from "../shared/i18n";

/** Asks the window in front to run a menu item's command (see `files.onMenuCommand`). */
function send(command: MenuCommand): void {
  const window = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0];
  if (window && !window.isDestroyed()) window.webContents.send(FILES_CHANNELS.menuCommand, command);
}

/**
 * The application menu, in the interface language: New Mind (⌘N), Close Tab
 * (⌘W), Reopen Closed Tab (⇧⌘T) and Settings… (⌘,) with the usual Edit, View
 * and Window items. As in a browser, ⌘W closes the tab and Shift+⌘W the
 * window. Reload and the developer tools are only in development and test
 * builds. Ctrl stands in for ⌘ on Windows and Linux.
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
  const t = (key: MessageKey) => translate(language, key);
  const command = (
    id: MenuCommand,
    label: MessageKey = MENU_COMMANDS[id].label,
  ): MenuItemConstructorOptions => ({
    label: t(label),
    accelerator: MENU_COMMANDS[id].keys,
    // The window's keymap binds the keys. macOS ignores this, but there the page has them first.
    registerAccelerator: false,
    click: () => send(id),
  });
  const settings = command("settings.open");

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
        command("mind.new"),
        { type: "separator" },
        command("tab.close"),
        command("tab.reopen"),
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
        { role: "copy", label: t(isMac ? "menu.copy" : "menu.copy.other") },
        { role: "paste", label: t("menu.paste") },
        { role: "selectAll", label: t("menu.selectAll") },
      ],
    },
    {
      label: t(isMac ? "menu.view" : "menu.view.other"),
      submenu: [
        ...(developer
          ? ([
              { role: "reload", label: t("menu.reload") },
              { role: "forceReload", label: t("menu.forceReload") },
              { role: "toggleDevTools", label: t("menu.devTools") },
              { type: "separator" },
            ] satisfies MenuItemConstructorOptions[])
          : []),
        command("sidebar.toggle", sidebarHidden ? "menu.showSidebar" : "menu.hideSidebar"),
        command("tab.split"),
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
