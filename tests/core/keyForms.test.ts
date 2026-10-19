import { describe, expect, test } from "vitest";
import {
  KEY_FORM_PROVIDERS,
  keyFits,
  serviceOfKey,
} from "../../src/core/providers/catalog/keyForms";
import { catalogProvider } from "../../src/core/providers/catalog/providers";

const alnum = (length: number, seed = "aZ3") =>
  seed.repeat(Math.ceil(length / seed.length)).slice(0, length);

/** Keys as each provider's own form has them, made up: no real key is in a test. */
const EXAMPLES: readonly { provider: string; why: string; key: string }[] = [
  { provider: "anthropic", why: "sk-ant-", key: `sk-ant-api03-${alnum(40)}` },
  { provider: "anthropic", why: "an admin key", key: `sk-ant-admin01-${alnum(40)}` },
  { provider: "openai", why: "sk-proj-", key: `sk-proj-${alnum(60)}` },
  { provider: "openai", why: "sk-svcacct-", key: `sk-svcacct-${alnum(60)}` },
  { provider: "openai", why: "sk-admin-", key: `sk-admin-${alnum(60)}` },
  {
    provider: "openai",
    why: "the older form, with its marker",
    key: `sk-${alnum(20)}T3BlbkFJ${alnum(20)}`,
  },
  { provider: "google", why: "AIza", key: `AIzaSy${alnum(33, "a-Z_9")}` },
  { provider: "openrouter", why: "sk-or-", key: `sk-or-v1-${alnum(64, "0f9a")}` },
  { provider: "xai", why: "xai-", key: `xai-${alnum(80)}` },
  { provider: "kimi", why: "sk- and 48 letters and digits", key: `sk-${alnum(48)}` },
];

describe("A provider's key form (#1384)", () => {
  test.each(EXAMPLES)("$provider: $why is its own and no other's", ({ provider, key }) => {
    expect(serviceOfKey(key)).toBe(provider);
    expect(keyFits(provider, key)).toBe(true);
    for (const other of KEY_FORM_PROVIDERS.filter((each) => each !== provider)) {
      expect(keyFits(other, key), `${other} for ${provider}'s key`).toBe(false);
    }
  });

  test("every provider with a form has an example above, and is in the catalog", () => {
    for (const provider of KEY_FORM_PROVIDERS) {
      expect(
        EXAMPLES.some((each) => each.provider === provider),
        provider,
      ).toBe(true);
      expect(catalogProvider(provider), provider).toBeDefined();
    }
  });

  test.each([
    ["a key with no form of its own", "d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6"],
    ["a DeepSeek or Qwen key's shape", `sk-${"0123456789abcdef".repeat(2)}`],
    ["a Chinese key with a dot", `${"0123456789abcdef".repeat(2)}.${alnum(16)}`],
    ["too short for Kimi", `sk-${alnum(20)}`],
    ["not a key", "hello"],
    ["empty", ""],
  ])("%s is recognised as nobody's", (_what, key) => {
    expect(serviceOfKey(key)).toBeNull();
  });

  test("a provider whose keys have no form of its own has none to fail", () => {
    for (const provider of ["deepseek", "qwen", "mistral", "glm", "siliconflow"]) {
      expect(keyFits(provider, `sk-ant-api03-${alnum(40)}`), provider).toBeNull();
    }
  });

  test("spaces around a pasted key don't matter", () => {
    expect(serviceOfKey(`  sk-ant-api03-${alnum(40)}\n`)).toBe("anthropic");
    expect(keyFits("anthropic", ` sk-ant-api03-${alnum(40)} `)).toBe(true);
  });
});
