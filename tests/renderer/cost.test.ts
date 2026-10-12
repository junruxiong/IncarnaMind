import { describe, expect, test } from "vitest";
import {
  estimateText,
  formatMoney,
  formatSpent,
  notCheckedKey,
  runStopText,
} from "../../src/renderer/src/cost";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

const t =
  (language: "en" | "zh-CN") =>
  (key: MessageKey, params?: MessageParams): string =>
    translate(language, key, params);

describe("Cost in words", () => {
  test("amounts show cents, and a small one its first figures rather than nothing", () => {
    expect(formatMoney({ amount: 3.2, currency: "USD" }, "en")).toBe("$3.20");
    expect(formatMoney({ amount: 0.0013, currency: "USD" }, "en")).toBe("$0.0013");
    expect(formatMoney({ amount: 0, currency: "USD" }, "en")).toBe("$0");
    expect(formatMoney({ amount: 1.4, currency: "CNY" }, "zh-CN")).toBe("¥1.40");
  });

  test("spending in two currencies names both", () => {
    const spent = [
      { amount: 3.2, currency: "USD" as const },
      { amount: 1.4, currency: "CNY" as const },
    ];
    expect(formatSpent(spent, t("en"), "en")).toBe("$3.20 and CN¥1.40");
    expect(formatSpent(spent, t("zh-CN"), "zh-CN")).toBe("US$3.20 和 ¥1.40");
    expect(formatSpent([], t("en"), "en")).toBeNull();
  });

  test("a stopped run says why, with when to try again when the provider said", () => {
    const now = new Date(2026, 9, 12, 9, 0);
    const later = new Date(2026, 9, 12, 14, 30).toISOString();
    expect(runStopText({ reason: "cost-cap" }, t("en"), "en", now)).toBe("Monthly cap reached");
    expect(runStopText({ reason: "plan-limit", resetsAt: null }, t("en"), "en", now)).toBe(
      "Plan limit reached",
    );
    expect(runStopText({ reason: "plan-limit", resetsAt: later }, t("en"), "en", now)).toBe(
      "Plan limit reached · Try again after 2:30 PM",
    );
    expect(runStopText({ reason: "plan-limit", resetsAt: later }, t("zh-CN"), "zh-CN", now)).toBe(
      "已达到订阅用量上限 · 14:30 后可重试",
    );
  });

  test("the items a stop left say they weren't checked, and why", () => {
    expect(translate("en", notCheckedKey({ reason: "cost-cap" }))).toBe("not checked: cost cap");
    expect(translate("en", notCheckedKey({ reason: "plan-limit", resetsAt: null }))).toBe(
      "not checked: plan limit",
    );
    expect(translate("zh-CN", notCheckedKey({ reason: "cost-cap" }))).toBe("未核对：已达花费上限");
  });

  test("the estimate names calls, tokens and price, with the month so far under a cap; a plan or a local model shows none", () => {
    const estimate = {
      billing: "per-token" as const,
      calls: 24,
      tokens: 310_000,
      cost: { amount: 0.4, currency: "USD" as const },
      spentTowardCap: { amount: 12.4, currency: "USD" as const },
      cap: { amount: 20, currency: "USD" as const },
    };
    expect(estimateText(estimate, t("en"), "en")).toBe(
      "About 24 calls, 310K tokens, $0.40 · $12.40 of $20.00 used this month",
    );
    expect(estimateText({ ...estimate, cap: null, spentTowardCap: null }, t("en"), "en")).toBe(
      "About 24 calls, 310K tokens, $0.40",
    );
    expect(estimateText({ ...estimate, cost: null, cap: null }, t("zh-CN"), "zh-CN")).toBe(
      "约 24 次调用、31万 token",
    );
    expect(estimateText({ billing: "plan" }, t("en"), "en")).toBeNull();
    expect(estimateText({ billing: "free" }, t("en"), "en")).toBeNull();
  });
});
