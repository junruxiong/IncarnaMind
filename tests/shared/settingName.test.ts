import { describe, expect, test } from "vitest";
import { en } from "../../src/shared/i18n/en";
import { zhCN } from "../../src/shared/i18n/zh-CN";

describe("The setting's name", () => {
  test("no interface string calls “Keep everything on this computer” local mode, in English or Chinese", () => {
    const offenders = [...Object.entries(en), ...Object.entries(zhCN)]
      .filter(([, text]) => /\blocal mode\b/i.test(text) || text.includes("本地模式"))
      .map(([key]) => key);
    expect(offenders).toEqual([]);
  });
});
