import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));

// Node imports the engine module itself, as the transcription process does once built (#1233): no
// bundler's interop gives a CommonJS package's exports their names, so they must be found on `default`.
const LOAD = `
  const { loadSherpa } = await import("./src/core/transcription/sherpa.ts");
  const sherpa = await loadSherpa();
  console.log(JSON.stringify({
    createAsync: typeof sherpa.OfflineRecognizer?.createAsync,
    vad: typeof sherpa.Vad,
  }));
`;

test("the engine module finds sherpa-onnx-node's recognizer and voice detector when Node imports it", {
  timeout: 60_000,
}, () => {
  const output = execFileSync(
    process.execPath,
    [
      "--disable-warning=MODULE_TYPELESS_PACKAGE_JSON",
      "--import",
      "./scripts/tsResolve.mjs",
      "--input-type=module",
      "-e",
      LOAD,
    ],
    { cwd: ROOT, encoding: "utf8" },
  );
  expect(JSON.parse(output.trim().split("\n").at(-1) ?? "{}")).toEqual({
    createAsync: "function",
    vad: "function",
  });
});
