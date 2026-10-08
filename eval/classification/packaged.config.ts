import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "packaged-smoke.ts",
  timeout: 90_000,
  workers: 1,
  reporter: "list",
  outputDir: "../results/classification-auto/packaged-smoke",
});
