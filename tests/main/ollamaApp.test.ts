import { describe, expect, test } from "vitest";
import {
  createFakeOllamaApp,
  createOllamaApp,
  type OllamaAppEnvironment,
  ollamaAppPath,
} from "../../src/main/ollamaApp";

function environment(
  platform: NodeJS.Platform,
  present: string[],
  opened: string[] = [],
): OllamaAppEnvironment {
  return {
    platform,
    home: platform === "win32" ? "C:\\Users\\Priya" : "/Users/priya",
    localAppData: "C:\\Users\\Priya\\AppData\\Local",
    exists: (path) => present.includes(path),
    openPath: async (path) => {
      opened.push(path);
    },
  };
}

describe("The Ollama app", () => {
  test("on a Mac is found in Applications, or in the User's own", () => {
    expect(ollamaAppPath(environment("darwin", ["/Applications/Ollama.app"]))).toBe(
      "/Applications/Ollama.app",
    );
    expect(ollamaAppPath(environment("darwin", ["/Users/priya/Applications/Ollama.app"]))).toBe(
      "/Users/priya/Applications/Ollama.app",
    );
    expect(ollamaAppPath(environment("darwin", []))).toBeNull();
  });

  test("on Windows is found where its installer puts it", () => {
    const app = "C:\\Users\\Priya\\AppData\\Local\\Programs\\Ollama\\ollama app.exe";

    expect(ollamaAppPath(environment("win32", [app]))).toBe(app);
    expect(ollamaAppPath(environment("win32", []))).toBeNull();
  });

  test("on Linux, where Ollama is a service, is not told apart from not installed", () => {
    expect(ollamaAppPath(environment("linux", ["/usr/local/bin/ollama"]))).toBeNull();
  });

  test("is opened where it is, and refused where it isn't", async () => {
    const opened: string[] = [];
    const there = createOllamaApp(environment("darwin", ["/Applications/Ollama.app"], opened));
    const missing = createOllamaApp(environment("darwin", [], opened));

    expect(await there.installed()).toBe(true);
    await there.open();
    expect(opened).toEqual(["/Applications/Ollama.app"]);
    expect(await missing.installed()).toBe(false);
    await expect(missing.open()).rejects.toThrow(/isn't installed/);
    expect(opened).toHaveLength(1);
  });

  test("the smoke tests' stand-in launches nothing and counts how often it was opened", async () => {
    const installed = createFakeOllamaApp("installed");
    const counter = (globalThis as { incarnamindOllamaApp?: { opened: number } })
      .incarnamindOllamaApp;

    expect(await installed.installed()).toBe(true);
    await installed.open();
    await installed.open();
    expect(counter?.opened).toBe(2);

    const missing = createFakeOllamaApp("missing");
    expect(await missing.installed()).toBe(false);
    await expect(missing.open()).rejects.toThrow();
  });
});
