import { describe, expect, test } from "vitest";
import { zhCN } from "../../src/shared/i18n/zh-CN";

const NOT_ALLOWED: readonly [string, string][] = [
  ["搜索网络", "联网搜索"],
  ["在网页上搜索", "联网搜索"],
  ["网页搜索", "联网搜索"],
  ["查找来源", "查找资料"],
];

describe("Chinese strings keep to the glossary", () => {
  test.each(NOT_ALLOWED)("no string says %s; it is %s", (wrong) => {
    const found = Object.entries(zhCN).filter(([, message]) => message.includes(wrong));
    expect(found.map(([key]) => key)).toEqual([]);
  });
});
