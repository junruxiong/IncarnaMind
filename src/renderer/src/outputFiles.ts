import type { OutputSettings } from "../../core/api";
import { core, files } from "./core";

/**
 * Asks for a folder in the system's open dialog and makes it the Output
 * location (a save waiting to know where goes on there). Resolves with the
 * settings, or null if the User cancelled.
 */
export async function chooseOutputLocation(): Promise<OutputSettings | null> {
  const picked = await files.pickOutputLocation();
  return picked === null ? null : core.setOutputLocation(picked);
}
