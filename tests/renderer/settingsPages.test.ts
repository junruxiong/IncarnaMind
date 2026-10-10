import { describe, expect, test } from "vitest";
import { isSettingsPage, settingsPages } from "../../src/renderer/src/settingsPages";
import { en } from "../../src/shared/i18n/en";
import { zhCN } from "../../src/shared/i18n/zh-CN";

describe("Settings pages", () => {
  test("are General, Models, Search, Tools, Connectors, Skills and Privacy, in that order", () => {
    expect(settingsPages).toEqual([
      "general",
      "models",
      "search",
      "tools",
      "connectors",
      "skills",
      "privacy",
    ]);
  });

  test("a page name is recognised, and anything else, such as a click event, is not", () => {
    for (const page of settingsPages) expect(isSettingsPage(page)).toBe(true);
    for (const old of ["chat-model", "approvals", "organization"]) {
      expect(isSettingsPage(old)).toBe(false);
    }
    expect(isSettingsPage({ type: "click" })).toBe(false);
    expect(isSettingsPage(undefined)).toBe(false);
  });

  test("have a name in both languages", () => {
    expect(en["settings.pages.models"]).toBe("Models");
    expect(zhCN["settings.pages.models"]).toBe("模型");
    expect(en["settings.pages.search"]).toBe("Search");
    expect(zhCN["settings.pages.search"]).toBe("搜索");
    expect(en["settings.pages.tools"]).toBe("Tools");
    expect(zhCN["settings.pages.tools"]).toBe("工具");
  });
});
