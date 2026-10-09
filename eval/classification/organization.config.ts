import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { nodeWorkers } from "../../vitest.config";
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  plugins: [nodeWorkers()],
  test: {
    include: ["eval/classification/organization.validation.ts"],
    environment: "node",
    disableConsoleIntercept: true,
    testTimeout: 300_000,
  },
});
