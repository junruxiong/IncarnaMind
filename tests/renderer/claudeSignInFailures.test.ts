import { expect, test } from "vitest";
import { CLAUDE_SIGN_IN_FAILURES } from "../../src/renderer/src/claudeSignInFailures";
import { translate } from "../../src/shared/i18n";

// A computer that can't keep the sign-in can't be made on a test machine; its line is checked here.
test.each(["en", "zh-CN"] as const)(
  "each way a Claude sign-in fails names Claude in %s",
  (language) => {
    const lines = Object.entries(CLAUDE_SIGN_IN_FAILURES).map(([kind, key]) => {
      const line = translate(language, key);
      expect(line, kind).not.toBe(key);
      expect(line, kind).toContain("Claude");
      return line;
    });
    const keys = new Set(Object.values(CLAUDE_SIGN_IN_FAILURES));
    expect(new Set(lines).size).toBe(keys.size);
  },
);
