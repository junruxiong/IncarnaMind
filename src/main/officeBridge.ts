/**
 * Turns the Office add-in bridge on for a development run (spike for #1272):
 * `INCARNAMIND_OFFICE_BRIDGE=1` in an unpackaged app. A packaged app never
 * has it, so no release listens for an add-in.
 */
import { join } from "node:path";
import { app, type BrowserWindow, dialog } from "electron";
import type { Core, CoreAdapters } from "../core";

const DEFAULT_PORT = 47361;
// manifest.http.xml names this one.
const HTTP_PORT = 47362;

export function officeBridgeAdapter(): CoreAdapters["officeBridge"] {
  if (app.isPackaged || process.env.INCARNAMIND_OFFICE_BRIDGE !== "1") return undefined;
  const scheme = process.env.INCARNAMIND_OFFICE_SCHEME === "http" ? "http" : "https";
  return {
    port: Number(
      process.env.INCARNAMIND_OFFICE_PORT ?? (scheme === "http" ? HTTP_PORT : DEFAULT_PORT),
    ),
    staticDir: join(app.getAppPath(), "addins", "office"),
    authorityFile: join(app.getPath("userData"), "office-add-in-authority.pem"),
    scheme,
  };
}

/**
 * Shows a pairing code until the User is done. "New code" replaces the one on
 * screen; "Disconnect add-ins" ends every pairing, so a pane stops at once.
 */
export async function showPairingCode(
  bridge: NonNullable<Core["officeBridge"]>,
  authorityFile: string | null,
  window: BrowserWindow,
): Promise<void> {
  await bridge.start();
  for (;;) {
    const { code } = bridge.beginPairing();
    // A sheet on the window: a message box with no window is app-modal on macOS and stops the
    // main process's event loop, so the bridge couldn't answer the pane while the code shows.
    const { response } = await dialog.showMessageBox(window, {
      type: "info",
      message: `Pairing code ${code}`,
      detail: `Type it into the IncarnaMind pane in Word or Excel within two minutes.${authorityFile ? `\n\nThe certificate to trust is ${authorityFile}` : ""}`,
      buttons: ["New code", "Disconnect add-ins", "Done"],
      defaultId: 2,
    });
    if (response === 2) return;
    if (response === 1) {
      for (const pairing of await bridge.pairings()) await bridge.unpair(pairing.id);
    }
  }
}
