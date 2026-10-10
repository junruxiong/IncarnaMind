/**
 * UI audit: the smallest Electron app, for a baseline of how long Electron
 * itself takes on this machine to show a window with a line of text.
 */
const { app, BrowserWindow } = require("electron");

app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 1280, height: 800, show: false });
  window.once("ready-to-show", () => window.show());
  void window.loadURL(
    "data:text/html,<!doctype html><title>Baseline</title><body style='font:14px sans-serif'><button data-testid='new-mind'>New Mind</button><div contenteditable data-testid='plain-editor' style='width:600px;min-height:200px;font:17px/28px serif;border:1px solid gray'></div>",
  );
});
