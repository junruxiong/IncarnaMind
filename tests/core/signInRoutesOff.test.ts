import { expect, test, vi } from "vitest";
import { createTempDataFolder, startCore } from "../helpers/core";

vi.mock("../../src/core/buildFlags", () => ({ EXPERIMENTAL_SIGN_IN_ROUTES: false }));

// First run's "Use my Claude plan" and Settings' Claude row show only while the route is available.
test("a build without the experimental sign-in routes offers no Claude sign-in", async () => {
  const core = startCore(await createTempDataFolder());

  expect(await core.getClaudeAccount()).toMatchObject({ available: false });
  expect(await core.signInToClaude()).toMatchObject({
    ok: false,
    error: { kind: "unavailable" },
  });
});
