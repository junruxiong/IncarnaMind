import { describe, expect, test } from "vitest";
import type { OllamaStatus } from "../../src/core/api";
import { ollamaLineAction } from "../../src/renderer/src/ollamaProblem";

const running: OllamaStatus = {
  running: true,
  baseUrl: "http://127.0.0.1:11434",
  models: [],
  cloudModels: [],
  recommendedModel: "qwen3.5:4b",
};
const notRunning = (installed: boolean): OllamaStatus => ({
  running: false,
  baseUrl: "http://127.0.0.1:11434",
  installed,
  recommendedModel: "qwen3.5:4b",
});

describe("The action under 'Ollama isn't running on this computer'", () => {
  test("is Open Ollama where its app is here, and Get Ollama where it isn't or can't be told", () => {
    expect(ollamaLineAction(notRunning(true), false)).toBe("open");
    expect(ollamaLineAction(notRunning(false), false)).toBe("get");
  });

  test("is Try again when Ollama runs by now, and the Question is kept to ask again", () => {
    expect(ollamaLineAction(running, false)).toBe("try-again");
  });

  test("waits while Ollama is being looked for or opened", () => {
    expect(ollamaLineAction(null, false)).toBeNull();
    expect(ollamaLineAction(notRunning(true), true)).toBeNull();
  });
});
