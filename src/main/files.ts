/**
 * The main process's half of the renderer's file helpers (`FilesBridge`):
 * saving an exported Mind where the User chooses, showing the data and logs
 * folders, choosing a Skill folder or zip to import, getting a Document's
 * original back (opened in another app, or saved where the User chooses),
 * and logging the window's uncaught errors. The core makes the export's
 * bytes, reads the Skill and copies Documents' files, and only for live
 * Documents; only this side touches dialogs, the shell and paths the User picks.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";
import {
  app,
  BrowserWindow,
  dialog,
  type IpcMainEvent,
  type IpcMainInvokeEvent,
  ipcMain,
  type OpenDialogOptions,
  type SaveDialogOptions,
  shell,
} from "electron";
import type { Core, DocumentKind, ExportFormat, ExportMindOptions } from "../core";
import { FILES_CHANNELS } from "../shared/bridge";
import { type MessageKey, translate } from "../shared/i18n";
import { type FileLogger, logsFolder, logWindowError } from "./log";

const FILE_TYPES: Readonly<Record<ExportFormat, { name: MessageKey; extension: string }>> = {
  docx: { name: "export.dialog.filter.docx", extension: "docx" },
  markdown: { name: "export.dialog.filter.markdown", extension: "md" },
};

const DOCUMENT_FILE_TYPES: Readonly<Record<DocumentKind, MessageKey>> = {
  pdf: "documents.copy.filter.pdf",
  text: "documents.copy.filter.text",
  markdown: "documents.copy.filter.markdown",
};

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

  /** The system save dialog, in front of the window that asked. */
  const showSaveDialog = (event: IpcMainInvokeEvent, options: SaveDialogOptions) => {
    const window = BrowserWindow.fromWebContents(event.sender);
    return window ? dialog.showSaveDialog(window, options) : dialog.showSaveDialog(options);
  };

  /** Opens a folder in the system's file manager, or a file in its default app. */
  const openPath = async (path: string) => {
    const error = await shell.openPath(path);
    if (error) throw new Error(error);
  };

  ipcMain.handle(
    FILES_CHANNELS.saveMindExport,
    async (event, mindId: string, options: ExportMindOptions): Promise<string | null> => {
      refuseUnknown(event);
      // The core checks the Mind and the options, and suggests the file name.
      const { fileName } = await core.previewMindExport(mindId, options);
      const { language } = await core.getSettings();
      const type = FILE_TYPES[options.format];
      const { canceled, filePath } = await showSaveDialog(event, {
        title: translate(language, "export.dialog.title"),
        defaultPath: join(app.getPath("documents"), fileName),
        filters: [{ name: translate(language, type.name), extensions: [type.extension] }],
      });
      if (canceled || !filePath) return null;
      // Exported now, so the file has any edits made while the dialog was open.
      const { data } = await core.exportMind(mindId, options);
      await writeFile(filePath, data);
      return filePath;
    },
  );

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

  // A copy named after the Document, so the app it opens in shows its name, not the stored file's hash.
  ipcMain.handle(FILES_CHANNELS.openDocumentExternally, async (event, documentId: unknown) => {
    refuseUnknown(event);
    await openPath(await core.temporaryDocumentCopy(documentId as string));
  });

  ipcMain.handle(
    FILES_CHANNELS.saveDocumentCopy,
    async (event, documentId: unknown): Promise<string | null> => {
      refuseUnknown(event);
      // The core checks the Document is live, and names the copy.
      const { document, fileName } = await core.documentCopyName(documentId as string);
      const { language } = await core.getSettings();
      const { canceled, filePath } = await showSaveDialog(event, {
        title: translate(language, "documents.copy.dialogTitle"),
        defaultPath: join(app.getPath("documents"), fileName),
        filters: [
          {
            name: translate(language, DOCUMENT_FILE_TYPES[document.kind]),
            extensions: [extname(fileName).slice(1)],
          },
        ],
      });
      if (canceled || !filePath) return null;
      // Checked again: the Document may have been deleted while the dialog was open.
      await core.saveDocumentCopy(document.id, filePath);
      return filePath;
    },
  );

  ipcMain.on(FILES_CHANNELS.logError, (event, report: unknown) => {
    if (trusted(event)) logWindowError(logger, report);
  });
}
