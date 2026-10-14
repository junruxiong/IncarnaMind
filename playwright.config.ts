import { defineConfig } from "@playwright/test";

/** Smoke tests drive the built Electron app (`npm run test:smoke` builds it first). */
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  // Two on CI, where each shard has four vCPUs; one elsewhere, so a run by hand loads the computer lightly.
  // Every test launches its own app on its own data folder and home folder (e2e/app.ts).
  workers: process.env.CI ? 2 : 1,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { trace: "retain-on-failure" },
});
