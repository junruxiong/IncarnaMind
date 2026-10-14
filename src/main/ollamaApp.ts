/**
 * The Ollama app on this computer: found where its installer puts it, and opened through
 * the OS. Where there is no such app (Linux runs Ollama as a service, with a command),
 * it is reported as not installed, and the User is offered to get it.
 */
import { posix, win32 } from "node:path";
import type { OllamaApp } from "../core";

export interface OllamaAppEnvironment {
  platform: NodeJS.Platform;
  home: string;
  /** `%LOCALAPPDATA%`, where Windows' installer puts it. */
  localAppData?: string | undefined;
  exists(path: string): boolean;
  /** Opens an app or file as the OS does for a double-click. Rejects if it can't. */
  openPath(path: string): Promise<void>;
}

/** Where the app is on this computer, or null. */
export function ollamaAppPath({
  platform,
  home,
  localAppData,
  exists,
}: OllamaAppEnvironment): string | null {
  if (platform === "darwin") {
    const places = ["/Applications/Ollama.app", posix.join(home, "Applications", "Ollama.app")];
    return places.find(exists) ?? null;
  }
  if (platform === "win32" && localAppData) {
    const place = win32.join(localAppData, "Programs", "Ollama", "ollama app.exe");
    return exists(place) ? place : null;
  }
  return null;
}

export function createOllamaApp(environment: OllamaAppEnvironment): OllamaApp {
  return {
    installed: async () => ollamaAppPath(environment) !== null,
    async open() {
      const path = ollamaAppPath(environment);
      if (!path) throw new Error("Ollama isn't installed on this computer.");
      await environment.openPath(path);
    },
  };
}

/**
 * The smoke tests' stand-in: the app is on this computer ("installed") or isn't
 * ("missing"), and opening it launches nothing, only counts, in
 * `globalThis.incarnamindOllamaApp.opened`.
 */
export function createFakeOllamaApp(state: "installed" | "missing"): OllamaApp {
  const counter = { opened: 0 };
  (globalThis as { incarnamindOllamaApp?: typeof counter }).incarnamindOllamaApp = counter;
  return {
    installed: async () => state === "installed",
    async open() {
      if (state === "missing") throw new Error("Ollama isn't installed on this computer.");
      counter.opened += 1;
    },
  };
}
