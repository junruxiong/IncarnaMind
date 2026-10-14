import { posix, win32 } from "node:path";
import type { OllamaApp } from "../core";

export interface OllamaAppEnvironment {
  platform: NodeJS.Platform;
  home: string;
  localAppData?: string | undefined;
  exists(path: string): boolean;
  /** Rejects when the OS can't open it. */
  openPath(path: string): Promise<void>;
}

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
  // Linux runs Ollama as a service, with no app to open: the User is offered to get it.
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

/** The smoke tests' stand-in: launches nothing, and counts opens in `globalThis.incarnamindOllamaApp`. */
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
