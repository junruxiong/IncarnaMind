import { defineConfig } from "@playwright/test";

/**
 * The UI audit's own config: its files end in `.audit.ts`, so the smoke suite
 * (`playwright.config.ts`, `*.spec.ts`) never picks them up. Needs a test build:
 *
 *   npx electron-vite build --mode test
 *   npx playwright test -c e2e/audit/playwright.audit.config.ts [file]
 *
 * AUDIT_OUT is where screenshots and measurements go (default: test-results/ui-audit).
 */
export default defineConfig({
  testDir: ".",
  testMatch: /.*\.audit\.ts$/,
  timeout: 30 * 60_000,
  workers: 1,
  retries: 0,
  reporter: "list",
  use: { trace: "off" },
});
