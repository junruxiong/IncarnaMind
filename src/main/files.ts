/**
 * The main process's half of the renderer's file helpers (`FilesBridge`):
 * choosing the Output location and showing a file saved there, showing the data and logs
 * folders, choosing a Skill folder or zip to import, choosing a folder to
 * link, opening a Document's file in another app or showing it (or a Linked
 * folder) in the file manager, and logging the window's uncaught errors. The core saves
 * files, reads the Skill, and checks a Document is live and its
 * file there before it hands the file's path to the shell adapter; only
 * this side touches dialogs.
 */
import { mkdir } from "node:fs/promises";
import {
  app,
  BrowserWindow,
  dialog,
  type IpcMainEvent,
  type IpcMainInvokeEvent,
  ipcMain,
  type OpenDialogOptions,
  shell,
} from "electron";
import type { Core } from "../core";
import { FILES_CHANNELS } from "../shared/bridge";
import { translate } from "../shared/i18n";
import { type FileLogger, logsFolder, logWindowError } from "./log";
import { setMenuSidebarHidden } from "./menu";
import { rememberPick, startFolder } from "./pickerFolders";

/** What "Add Documents" offers; the core decides what it takes (and says what it skipped). */
const DOCUMENT_EXTENSIONS = ["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", "markdown"];

export interface FileActionsOptions {
  /** The data folder, which "Open data folder" shows. */
  dataDir: string;
  /** Whether a call comes from IncarnaMind's own page. */
  trusted(event: IpcMainInvokeEvent | IpcMainEvent): boolean;
  /** Where the window's uncaught errors are logged. */
  logger: FileLogger;
}

export function serveFileActions(
  core: Core,
  { dataDir, trusted, logger }: FileActionsOptions,
): void {
  const refuseUnknown = (event: IpcMainInvokeEvent) => {
    if (!trusted(event)) throw new Error("Refused a call from an unknown page.");
  };

  /** Opens a folder in the system's file manager, or a file in its default app. */
  const openPath = async (path: string) => {
    const error = await shell.openPath(path);
    if (error) throw new Error(error);
  };

  // The core checks it saved the file, then shows it through the shell adapter.
  ipcMain.handle(FILES_CHANNELS.showSavedFile, async (event, path: unknown) => {
    refuseUnknown(event);
    await core.showSavedFile(path as string);
  });

  ipcMain.handle(FILES_CHANNELS.pickOutputLocation, async (event): Promise<string | null> => {
    refuseUnknown(event);
    const [{ language }, { location }] = await Promise.all([
      core.getSettings(),
      core.getOutputSettings(),
    ]);
    const openOptions: OpenDialogOptions = {
      title: translate(language, "outputLocation.pick.title"),
      buttonLabel: translate(language, "outputLocation.pick.button"),
      properties: ["openDirectory", "createDirectory", "promptToCreate"],
      defaultPath: location ?? startFolder(dataDir, "folders", app.getPath("home")),
    };
    const window = BrowserWindow.fromWebContents(event.sender);
    const { canceled, filePaths } = window
      ? await dialog.showOpenDialog(window, openOptions)
      : await dialog.showOpenDialog(openOptions);
    return canceled ? null : (filePaths[0] ?? null);
  });

  ipcMain.handle(FILES_CHANNELS.openDataFolder, async (event) => {
    refuseUnknown(event);
    await openPath(dataDir);
  });

  ipcMain.handle(FILES_CHANNELS.openLogsFolder, async (event) => {
    refuseUnknown(event);
    const folder = logsFolder(dataDir);
    await mkdir(folder, { recursive: true });
    await openPath(folder);
  });

  ipcMain.handle(FILES_CHANNELS.pickSkill, async (event, kind: unknown): Promise<string | null> => {
    refuseUnknown(event);
    const { language } = await core.getSettings();
    const openOptions: OpenDialogOptions =
      kind === "zip"
        ? {
            title: translate(language, "skills.import.pickZip"),
            properties: ["openFile"],
            filters: [
              { name: translate(language, "skills.import.zipFilter"), extensions: ["zip"] },
            ],
          }
        : { title: translate(language, "skills.import.pickFolder"), properties: ["openDirectory"] };
    const window = BrowserWindow.fromWebContents(event.sender);
    const { canceled, filePaths } = window
      ? await dialog.showOpenDialog(window, openOptions)
      : await dialog.showOpenDialog(openOptions);
    return canceled ? null : (filePaths[0] ?? null);
  });

  // The core checks the Document is live and its file there, then opens it through the shell adapter.
  ipcMain.handle(FILES_CHANNELS.openDocumentExternally, async (event, documentId: unknown) => {
    refuseUnknown(event);
    await core.openDocumentInApp(documentId as string);
  });

  ipcMain.handle(FILES_CHANNELS.showDocumentInFolder, async (event, documentId: unknown) => {
    refuseUnknown(event);
    await core.showDocumentInFolder(documentId as string);
  });

  ipcMain.handle(FILES_CHANNELS.pickLinkedFolder, async (event): Promise<string | null> => {
    refuseUnknown(event);
    const { language } = await core.getSettings();
    const openOptions: OpenDialogOptions = {
      title: translate(language, "linkedFolders.pick.title"),
      buttonLabel: translate(language, "linkedFolders.pick.button"),
      properties: ["openDirectory"],
      defaultPath: startFolder(dataDir, "folders", app.getPath("home")),
    };
    const window = BrowserWindow.fromWebContents(event.sender);
    const { canceled, filePaths } = window
      ? await dialog.showOpenDialog(window, openOptions)
      : await dialog.showOpenDialog(openOptions);
    const picked = canceled ? null : (filePaths[0] ?? null);
    if (picked) rememberPick(dataDir, "folders", picked);
    return picked;
  });

  // A sheet on the window that starts in a folder on this computer (see pickerFolders.ts),
  // instead of the web file input, which starts wherever macOS last was, often iCloud.
  ipcMain.handle(FILES_CHANNELS.pickDocuments, async (event): Promise<string[]> => {
    refuseUnknown(event);
    const { language } = await core.getSettings();
    const openOptions: OpenDialogOptions = {
      title: translate(language, "documents.pick.title"),
      properties: ["openFile", "multiSelections"],
      filters: [
        { name: translate(language, "documents.pick.filter"), extensions: DOCUMENT_EXTENSIONS },
      ],
      defaultPath: startFolder(dataDir, "documents", app.getPath("downloads")),
    };
    const window = BrowserWindow.fromWebContents(event.sender);
    const { canceled, filePaths } = window
      ? await dialog.showOpenDialog(window, openOptions)
      : await dialog.showOpenDialog(openOptions);
    if (canceled) return [];
    if (filePaths[0]) rememberPick(dataDir, "documents", filePaths[0]);
    return filePaths;
  });

  // Only a folder the User linked: its path comes from the core's list, never from the page.
  ipcMain.handle(FILES_CHANNELS.showLinkedFolder, async (event, linkedFolderId: unknown) => {
    refuseUnknown(event);
    const linked = (await core.listLinkedFolders()).find((each) => each.id === linkedFolderId);
    if (!linked) throw new Error("There is no such Linked folder.");
    if (linked.status === "unavailable") {
      throw new Error("The Linked folder can't be reached right now.");
    }
    shell.showItemInFolder(linked.path);
  });

  ipcMain.on(FILES_CHANNELS.sidebarHidden, (event, hidden: unknown) => {
    if (trusted(event) && typeof hidden === "boolean") setMenuSidebarHidden(hidden);
  });

  ipcMain.on(FILES_CHANNELS.logError, (event, report: unknown) => {
    if (trusted(event)) logWindowError(logger, report);
  });
}
