import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { HOME_GUARD } from "../../vitest.config";
import { evalCache } from "../lib/cache";

/** `npm run eval:pptx` (eval/pptx/README.md): fetches the corpus, so neither `npm test` nor `npm run eval` runs it. */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  test: {
    include: ["eval/pptx/**/*.pptx.ts"],
    environment: "node",
    setupFiles: [HOME_GUARD],
    env: { INCARNAMIND_EVAL_CACHE: evalCache() },
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    testTimeout: 60 * 60_000,
  },
});
