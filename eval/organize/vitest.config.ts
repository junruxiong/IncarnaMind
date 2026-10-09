import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { nodeWorkers } from "../../vitest.config";

export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  plugins: [nodeWorkers()],
  test: {
    include: ["eval/organize/*.check.ts"],
    environment: "node",
    disableConsoleIntercept: true,
    testTimeout: 6 * 60 * 60_000,
  },
});
