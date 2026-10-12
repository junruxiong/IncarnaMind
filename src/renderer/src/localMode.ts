import { core } from "./core";
import { translateNow } from "./i18n";
import { useAppStore } from "./store";
import { useUndo } from "./undo";

/** Local mode ("Keep everything on this computer") on or off, in this window's state too. */
async function setLocalMode(enabled: boolean): Promise<void> {
  useAppStore.setState({ embedding: await core.setLocalOnly(enabled) });
}

/**
 * Turns local mode off where a Question was refused, and offers Undo, which
 * turns it on again (and so keeps a cloud model from receiving anything).
 */
export async function turnOffLocalMode(): Promise<void> {
  await setLocalMode(false);
  useUndo.getState().push({
    message: translateNow("localMode.turnedOff"),
    undo: () => setLocalMode(true),
    redo: () => setLocalMode(false),
  });
}
