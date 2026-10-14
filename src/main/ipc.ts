import { type IpcMainInvokeEvent, ipcMain } from "electron";
import { errorForWindow } from "../shared/errorWording";

/**
 * `ipcMain.handle`, sending a failure's wording to the window with its sentence, so the window can
 * say it in the interface language (#627).
 */
export function handle(
  channel: string,
  listener: (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown,
): void {
  ipcMain.handle(channel, async (event, ...args) => {
    try {
      return await listener(event, ...args);
    } catch (error) {
      throw errorForWindow(error);
    }
  });
}
