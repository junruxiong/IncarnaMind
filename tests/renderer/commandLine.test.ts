import { describe, expect, test } from "vitest";
import {
  commandLine,
  splitCommandLine,
} from "../../src/renderer/src/components/connectors/commandLine";

describe("a pasted command line", () => {
  test("splits at spaces, keeping quoted parts and escaped spaces whole", () => {
    expect(splitCommandLine("node /abs/path/mcp-server.mjs")).toEqual([
      "node",
      "/abs/path/mcp-server.mjs",
    ]);
    expect(splitCommandLine('npx -y "@scope/server name"  --flag')).toEqual([
      "npx",
      "-y",
      "@scope/server name",
      "--flag",
    ]);
    expect(splitCommandLine("uvx my\\ tool 'a b' \"\"")).toEqual(["uvx", "my tool", "a b", ""]);
    expect(splitCommandLine("  npx  ")).toEqual(["npx"]);
    expect(splitCommandLine("")).toEqual([]);
  });

  test("is the inverse of how a command line is shown", () => {
    const parts = ["/Applications/My Tool/run", "--name", "a b", ""];
    expect(splitCommandLine(commandLine(parts[0] as string, parts.slice(1)))).toEqual(parts);
  });
});
