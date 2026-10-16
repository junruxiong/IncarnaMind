import { APICallError } from "ai";
import { describe, expect, test } from "vitest";
import { classifyProviderError } from "../../src/core/providers/providerErrors";
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
      expect(en[key], key).toContain("30-day data retention");
      expect(zhCN[key], key).toContain("30 天数据保留");
      expect(`${en[key]}${zhCN[key]}`, key).not.toMatch(/Console/);
    }
  });

  test("with another 400 stays the provider's own error", () => {
    expect(classifyProviderError(refused("messages: field required")).kind).toBe("provider");
  });
});
