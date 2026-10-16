import { expect, test } from "vitest";
import { untestedHintKey } from "../../src/renderer/src/untestedModels";
import { translate } from "../../src/shared/i18n";

test("a preset not yet through the citing evaluation says so on its models, in both languages; other providers don't", () => {
  const key = untestedHintKey({ kind: "openai-compatible", catalogId: "groq" });
  expect(key).toBe("providers.untested");
  if (!key) return;
  expect(translate("en", key)).toBe("Not yet tested for checked Citations");
  expect(translate("zh-CN", key)).toBe("尚未测试引用核对");
  expect(untestedHintKey({ kind: "anthropic", catalogId: "anthropic" })).toBeNull();
  expect(untestedHintKey({ kind: "siliconflow", catalogId: "siliconflow" })).toBeNull();
  expect(untestedHintKey({ kind: "openai-compatible", catalogId: "openai-compatible" })).toBeNull();
});
