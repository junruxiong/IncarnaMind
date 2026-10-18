/**
 * Changes to Minds storage refused (#1119), on the main process's side: saving
 * a copy of a Mind as a .docx wherever the User chooses, which another drive
 * can take when this one is full; and a quit that would lose them. Quitting
 * tries to store them once more, and only if that fails asks, in one short
 * sheet: save a copy, quit anyway, or keep IncarnaMind open (the default).
 * Lost typing has no Undo, so this one asks first.
 */
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { app, type BrowserWindow, dialog, type SaveDialogOptions } from "electron";
import type { Core, UnsavedChanges } from "../core";
import { type MessageKey, type MessageParams, translate } from "../shared/i18n";

/** A file name the file systems take: a Mind's title without what they refuse. */
const fileNameOf = (title: string) => title.replace(/[\\/:*?"<>|]+/g, " ").trim() || "Mind";

/** Saves the Mind as a .docx where the User chooses; false when they didn't choose. */
export async function saveMindCopy(
  core: Core,
  window: BrowserWindow | null,
  mindId: string,
): Promise<boolean> {
  const [{ language }, minds] = await Promise.all([core.getSettings(), core.listMinds()]);
  const title = minds.find((mind) => mind.id === mindId)?.title ?? "";
  const options: SaveDialogOptions = {
    title: translate(language, "unsaved.saveCopy.title"),
    defaultPath: join(app.getPath("documents"), `${fileNameOf(title)}.docx`),
    filters: [{ name: "Word", extensions: ["docx"] }],
  };
  const { canceled, filePath } = window
    ? await dialog.showSaveDialog(window, options)
    : await dialog.showSaveDialog(options);
  if (canceled || !filePath) return false;
  const { data } = await core.exportMind(mindId, { format: "docx" });
  await writeFile(filePath, data);
  return true;
}

export function guardQuit(core: Core, focusedWindow: () => BrowserWindow | null): void {
  let unsaved: UnsavedChanges | null = null;
  core.on("minds.unsaved", (state) => {
    unsaved = state;
  });
  let leaving = false;
  const decide = async () => {
    const left = await core.saveChangesAgain();
    if (left) {
      const { language } = await core.getSettings();
      const t = (key: MessageKey, params?: MessageParams) => translate(language, key, params);
      const window = focusedWindow();
      const options = {
        type: "warning" as const,
        message:
          left.count === 1
            ? t("quit.unsaved.title.one")
            : t("quit.unsaved.title", { count: left.count }),
        buttons: [t("unsaved.saveCopy"), t("quit.unsaved.quit"), t("quit.unsaved.keep")],
        defaultId: 2,
        cancelId: 2,
        noLink: true,
      };
      const { response } = window
        ? await dialog.showMessageBox(window, options)
        : await dialog.showMessageBox(options);
      if (response === 2) return;
      if (response === 0) {
        for (const mindId of left.mindIds) {
          if (!(await saveMindCopy(core, window, mindId))) return;
        }
      }
    }
    leaving = true;
    app.quit();
  };
  app.on("before-quit", (event) => {
    if (leaving || !unsaved) return;
    event.preventDefault();
    decide().catch((error: unknown) => {
      console.error("quit: the changes waiting couldn't be dealt with", error);
      leaving = true;
      app.quit();
    });
  });
}
