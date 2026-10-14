import type { OutputSettings } from "../../core/api";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import { core, files } from "./core";
import { folderName } from "./linkedFolders";
import { useAppStore } from "./store";

/**
 * Asks for a folder in the system's open dialog and makes it the Output
 * location (a save waiting to know where goes on there). Resolves with the
 * settings, or null if the User cancelled.
 */
export async function chooseOutputLocation(): Promise<OutputSettings | null> {
  const picked = await files.pickOutputLocation();
  return picked === null ? null : core.setOutputLocation(picked);
}

/** A folder as the lines about saving name it: "Documents › IncarnaMind" for the default, else its own name. */
export const outputFolderLabel = (
  folder: string,
  suggested: string,
  t: (key: MessageKey, params?: MessageParams) => string,
): string => (folder === suggested ? t("outputLocation.defaultName") : folderName(folder));

/** Opens Settings at the Output location, its Change… focused: a first save's Change folder… (D209). */
export const showOutputLocationSetting = (): void =>
  useAppStore.getState().openSettings("general", "output-location");
