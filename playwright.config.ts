import { defineConfig } from "@playwright/test";

/** Smoke tests drive the built Electron app (`npm run test:smoke` builds it first). */
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  workers: 1,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { trace: "retain-on-failure" },
});
