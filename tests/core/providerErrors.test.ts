import { APICallError, RetryError } from "ai";
import { describe, expect, test } from "vitest";
import { classifyProviderError } from "../../src/core/providers/providerErrors";
import { translate } from "../../src/shared/i18n";
import { en } from "../../src/shared/i18n/en";
import { zhCN } from "../../src/shared/i18n/zh-CN";

const refused = (message: string, statusCode = 400) =>
  new APICallError({
    message,
    url: "https://api.anthropic.com/v1/messages",
    requestBodyValues: {},
    statusCode,
    responseBody: JSON.stringify({
      type: "error",
      error: { type: "invalid_request_error", message },
    }),
    isRetryable: false,
  });

describe("A provider's refusal", () => {
  test("of a model that needs 30-day data retention, asked by an organization without it, says so", () => {
    const error = classifyProviderError(
      refused(
        "In order to access this model, your organization or workspace must have data retention enabled.",
      ),
    );
    expect(error.kind).toBe("data-retention");
    // Each line names the model it was asked for, and neither tells the User what to change in the Console.
    for (const key of ["answer.error.data-retention", "providers.test.data-retention"] as const) {
      expect(en[key], key).toContain("{model}");
      expect(zhCN[key], key).toContain("{model}");
      expect(en[key], key).toContain("30-day retention");
      expect(zhCN[key], key).toContain("30 天数据保留");
      expect(`${en[key]}${zhCN[key]}`, key).not.toMatch(/Console/);
    }
  });

  test("with another 400 stays the provider's own error", () => {
    expect(classifyProviderError(refused("messages: field required")).kind).toBe("provider");
  });

  test("of a key, said only in the body of a 400 (Google's model list), is a refused key", () => {
    const listRefused = new APICallError({
      message: "HTTP 400",
      url: "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
      requestBodyValues: undefined,
      statusCode: 400,
      responseBody: JSON.stringify({
        error: { code: 400, message: "API key not valid. Please pass a valid API key." },
      }),
    });
    expect(classifyProviderError(listRefused).kind).toBe("auth");
  });
});

describe("A key with a character that keys don't use", () => {
  const engine = (code: number) =>
    new TypeError(
      `Cannot convert argument to a ByteString because the character at index 19 has a value of ${code} which is greater than 255.`,
    );

  test("is said as a bad key, naming only that character, in both languages", () => {
    const error = classifyProviderError(engine(20316));

    expect(error.kind).toBe("auth");
    expect(error.wording).toEqual({
      key: "error.provider.keyCharacter",
      values: { character: "作" },
    });
    expect(error.message).toBe(
      "That doesn’t look like a key: it has a character that keys don’t use (作). Paste the key again.",
    );
    expect(
      translate(
        "zh-CN",
        error.wording?.key ?? "error.provider.keyCharacter",
        error.wording?.values,
      ),
    ).toBe("这看起来不像密钥：其中有密钥不会用到的字符（作）。重新粘贴密钥。");
  });

  test("names an invisible character by its code, not by a blank", () => {
    expect(classifyProviderError(engine(0x200b)).wording?.values).toEqual({ character: "U+200B" });
    expect(classifyProviderError(engine(0x2019)).wording?.values).toEqual({ character: "\u2019" });
  });

  test("is found inside the AI SDK's retry error too", () => {
    const retried = new RetryError({
      message: "Failed after 3 attempts.",
      reason: "maxRetriesExceeded",
      errors: [engine(20316)],
    });
    expect(classifyProviderError(retried).wording?.values).toEqual({ character: "作" });
  });

  test("leaves another TypeError as it was", () => {
    expect(classifyProviderError(new TypeError("fetch failed")).kind).toBe("network");
    expect(classifyProviderError(new TypeError("x is not a function")).kind).toBe("unknown");
  });
});
