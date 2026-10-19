/**
 * Turns the Office add-in bridge on for a development run (spike for #1272):
 * `INCARNAMIND_OFFICE_BRIDGE=1` in an unpackaged app. A packaged app never
 * has it, so no release listens for an add-in.
 */
import { join } from "node:path";
import { app, dialog } from "electron";
import type { Core, CoreAdapters } from "../core";

const DEFAULT_PORT = 47361;

export function officeBridgeAdapter(): CoreAdapters["officeBridge"] {
  if (app.isPackaged || process.env.INCARNAMIND_OFFICE_BRIDGE !== "1") return undefined;
  return {
    port: Number(process.env.INCARNAMIND_OFFICE_PORT ?? DEFAULT_PORT),
    staticDir: join(app.getAppPath(), "addins", "office"),
    authorityFile: join(app.getPath("userData"), "office-add-in-authority.pem"),
  };
}

/** Shows a pairing code until the User is done; "New code" replaces the one on screen. */
export async function showPairingCode(
  bridge: NonNullable<Core["officeBridge"]>,
  authorityFile: string,
): Promise<void> {
  for (;;) {
    const { code } = bridge.beginPairing();
    const { response } = await dialog.showMessageBox({
      type: "info",
      message: `Pairing code ${code}`,
      detail: `Type it into the IncarnaMind pane in Word or Excel within two minutes.\n\nThe certificate to trust is ${authorityFile}`,
      buttons: ["New code", "Done"],
      defaultId: 1,
    });
    if (response === 1) return;
  }
}
