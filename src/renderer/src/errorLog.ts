import type { RendererErrorReport } from "../../shared/bridge";
import { files } from "./core";

/** Chromium's notice that layout took two passes: harmless, and frequent. */
const RESIZE_OBSERVER_NOTICE = /^ResizeObserver loop/;

function report(kind: RendererErrorReport["kind"], reason: unknown): void {
  const error = reason instanceof Error ? reason : null;
  const message = error ? error.message : typeof reason === "string" ? reason : "";
  if (RESIZE_OBSERVER_NOTICE.test(message)) return;
  try {
    files.logError({
      kind,
      name: error?.name ?? typeof reason,
      message,
      stack: error?.stack ?? "",
    });
  } catch {
    // The log is a help, never a reason for another error.
  }
}

/**
 * Sends errors nothing in the window caught to the log in the data folder.
 * The main process scrubs them of the User's content before writing them.
 */
export function installErrorLog(): void {
  window.addEventListener("error", (event) => report("error", event.error ?? event.message));
  window.addEventListener("unhandledrejection", (event) => report("rejection", event.reason));
}
