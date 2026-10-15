import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { HOME_GUARD } from "../../vitest.config";
import { evalCache } from "../lib/cache";

/** `npm run eval:ocr` (eval/ocr/README.md): never part of `npm test`. */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  test: {
    include: ["eval/ocr/*.check.ts"],
    environment: "node",
    setupFiles: [HOME_GUARD],
    env: { INCARNAMIND_EVAL_CACHE: evalCache() },
    disableConsoleIntercept: true,
    testTimeout: 2 * 60 * 60_000,
  },
});
