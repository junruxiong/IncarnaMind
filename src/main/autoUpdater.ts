/**
 * electron-updater, which ./updater imports dynamically: it takes tens of
 * milliseconds to load, and only a packaged app checks for updates, after
 * startup.
 */
export { autoUpdater } from "electron-updater";
