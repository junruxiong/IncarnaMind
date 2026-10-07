/**
 * The main process's half of the renderer's file helpers (`FilesBridge`):
 * saving an exported Mind where the User chooses, showing the data folder,
 * and choosing a Skill folder or zip to import. The core makes the export's
 * bytes and reads the Skill; only this side touches dialogs and disks.
 */
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  app,
  BrowserWindow,
  dialog,
  type IpcMainInvokeEvent,
  ipcMain,
  type OpenDialogOptions,
  shell,
} from "electron";
import type { Core, ExportFormat, ExportMindOptions } from "../core";
import { FILES_CHANNELS } from "../shared/bridge";
import { type MessageKey, translate } from "../shared/i18n";

const FILE_TYPES: Readonly<Record<ExportFormat, { name: MessageKey; extension: string }>> = {
  docx: { name: "export.dialog.filter.docx", extension: "docx" },
  markdown: { name: "export.dialog.filter.markdown", extension: "md" },
};

export interface FileActionsOptions {
  /** The data folder, which "Open data folder" shows. */
  dataDir: string;
  /** Whether a call comes from IncarnaMind's own page. */
  trusted(event: IpcMainInvokeEvent): boolean;
}

export function serveFileActions(core: Core, { dataDir, trusted }: FileActionsOptions): void {
  const refuseUnknown = (event: IpcMainInvokeEvent) => {
    if (!trusted(event)) throw new Error("Refused a call from an unknown page.");
  };

  ipcMain.handle(
    FILES_CHANNELS.saveMindExport,
    async (event, mindId: string, options: ExportMindOptions): Promise<string | null> => {
      refuseUnknown(event);
      // The core checks the Mind and the options, and suggests the file name.
      const { fileName } = await core.previewMindExport(mindId, options);
      const { language } = await core.getSettings();
      const type = FILE_TYPES[options.format];
      const saveOptions = {
        title: translate(language, "export.dialog.title"),
        defaultPath: join(app.getPath("documents"), fileName),
        filters: [{ name: translate(language, type.name), extensions: [type.extension] }],
      };
      const window = BrowserWindow.fromWebContents(event.sender);
      const { canceled, filePath } = window
        ? await dialog.showSaveDialog(window, saveOptions)
        : await dialog.showSaveDialog(saveOptions);
      if (canceled || !filePath) return null;
      // Exported now, so the file has any edits made while the dialog was open.
      const { data } = await core.exportMind(mindId, options);
      await writeFile(filePath, data);
      return filePath;
    },
  );

  ipcMain.handle(FILES_CHANNELS.openDataFolder, async (event) => {
    refuseUnknown(event);
    const error = await shell.openPath(dataDir);
    if (error) throw new Error(error);
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
}
