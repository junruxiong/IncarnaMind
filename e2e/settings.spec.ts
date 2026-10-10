import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { createDataFolder, dismissChatSetup, launchApp, removeDataFolder } from "./app";

/*
 * Settings has seven pages (General, Models, Search, Tools, Connectors, Skills,
 * Privacy), and every setting there was before is on one of them. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save a screenshot of each page.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

/** Each page, its name in the list, and the sections on it. */
const PAGES = [
  {
    page: "general",
    name: "General",
    sections: ["language-settings", "data-folder-settings", "about-settings"],
  },
  {
    page: "models",
    name: "Models",
    sections: [
      "chat-model-settings",
      "jev-settings",
      "experimental-settings",
      "organization-settings",
    ],
  },
  { page: "search", name: "Search", sections: ["embedding-settings", "rerank-settings"] },
  { page: "tools", name: "Tools", sections: ["approvals-settings"] },
  { page: "connectors", name: "Connectors", sections: ["connectors-settings"] },
  { page: "skills", name: "Skills", sections: ["skills-settings", "skill-scripts-settings"] },
  {
    page: "privacy",
    name: "Privacy",
    sections: ["consent-settings", "network-traffic"],
  },
] as const;

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function shot(window: Page, name: string) {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

test("Settings lists seven pages, and each holds its settings", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await window.getByRole("button", { name: "Settings" }).click();
    const settings = window.getByTestId("settings");
    const nav = settings.getByTestId("settings-nav");
    await expect(nav.getByRole("button")).toHaveText(PAGES.map((each) => each.name));

    for (const { page, name, sections } of PAGES) {
      await nav.getByRole("button", { name, exact: true }).click();
      await expect(settings).toHaveAttribute("data-page", page);
      await expect(settings.getByTestId("settings-page-title")).toHaveText(name);
      const content = settings.getByTestId(`settings-page-${page}`);
      for (const section of sections) {
        await expect(content.getByTestId(section), `${name}: ${section}`).toBeAttached();
      }
      await shot(window, `settings-${page}`);
      if (page === "models") {
        await content.getByTestId("organization-settings").scrollIntoViewIfNeeded();
        await shot(window, "settings-models-organization");
      }
    }
  } finally {
    await app.close();
  }
});

test("the list of pages works by keyboard, with a focus ring", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await window.getByRole("button", { name: "Settings" }).click();
    const settings = window.getByTestId("settings");
    const models = settings.getByTestId("settings-nav-models");
    // Tab reaches the list the way a keyboard user would (a focus() call shows no ring).
    for (let step = 0; step < 12; step += 1) {
      if (await models.evaluate((each) => each === document.activeElement)) break;
      await window.keyboard.press("Tab");
    }
    await expect(models).toBeFocused();
    expect(await models.evaluate((each) => getComputedStyle(each).outlineStyle)).not.toBe("none");
    await window.keyboard.press("Enter");
    await expect(settings).toHaveAttribute("data-page", "models");
    await window.keyboard.press("Tab");
    await expect(settings.getByTestId("settings-nav-search")).toBeFocused();
    await window.keyboard.press("Space");
    await expect(settings).toHaveAttribute("data-page", "search");
  } finally {
    await app.close();
  }
});

test("the Library's Organization button opens Models, where Organization is", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await window.getByTestId("open-library").click();
    await window.getByTestId("library").getByRole("button", { name: "Organization" }).click();
    const settings = window.getByTestId("settings");
    await expect(settings).toHaveAttribute("data-page", "models");
    await expect(settings.getByTestId("organization-settings")).toBeVisible();
  } finally {
    await app.close();
  }
});

test("the pages are named in Chinese too", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await window.evaluate(async () =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await window.getByRole("button", { name: "设置" }).click();
    const nav = window.getByTestId("settings-nav");
    await expect(nav.getByRole("button")).toHaveText([
      "通用",
      "模型",
      "搜索",
      "工具",
      "连接器",
      "技能",
      "隐私",
    ]);
    const names = ["general", "models", "search", "tools", "connectors", "skills", "privacy"];
    for (const page of names) {
      await window.getByTestId(`settings-nav-${page}`).click();
      await expect(window.getByTestId("settings")).toHaveAttribute("data-page", page);
      if (page === "models")
        await expect(window.getByTestId("settings-page-title")).toHaveText("模型");
      await shot(window, `settings-${page}-zh`);
      if (page === "models") {
        await window.getByTestId("organization-settings").scrollIntoViewIfNeeded();
        await shot(window, "settings-models-organization-zh");
      }
    }
  } finally {
    await app.close();
  }
});
