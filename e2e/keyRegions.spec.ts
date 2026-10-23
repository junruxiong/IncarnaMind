import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  addProviderForm,
  createDataFolder,
  launchApp,
  newMind,
  openSettings,
  removeDataFolder,
  setWindowSize,
} from "./app";

/*
 * A key from either region (#1572): Kimi, Qwen, GLM and SiliconFlow make keys per region, and the
 * key form asks no region. The scripted model refuses a key ending "-for-<host>" at any other
 * host, as a region refuses another's key, and one with "refused" in it everywhere.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const language = (window: Page, value: "en" | "zh-CN") =>
  window.evaluate(
    (chosen) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: chosen },
      }),
    value,
  );

/** The region a provider was saved on. */
const savedRegion = (window: Page, catalogId: string) =>
  window.evaluate(async (id) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const providers = await core.listChatProviders();
    return providers.find((each) => each.catalogId === id)?.endpoint ?? null;
  }, catalogId);

const keyMadeAt = (host: string) => `sk-e2e-for-${host}`;

test("in Settings, a key from either region saves with no region asked, and the provider's page says which, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    const settings = await openSettings(window, "models");
    // Each key is from the region the form doesn't try first: Kimi's by the language, the others' China.
    for (const { catalogId, value, host, region, line, sends } of [
      {
        catalogId: "kimi",
        value: "en",
        host: "api.moonshot.cn",
        region: "cn",
        line: "Using the China endpoint",
        sends: "Kimi (Moonshot AI)",
      },
      {
        catalogId: "kimi",
        value: "zh-CN",
        host: "api.moonshot.ai",
        region: "intl",
        line: "使用国际的 API 地址",
        sends: "Kimi（月之暗面）",
      },
      {
        catalogId: "qwen",
        value: "en",
        host: "dashscope-intl.aliyuncs.com",
        region: "intl",
        line: "Using the International (Singapore) endpoint",
        sends: "Qwen (Alibaba Cloud Model Studio)",
      },
      {
        catalogId: "glm",
        value: "zh-CN",
        host: "api.z.ai",
        region: "intl",
        line: "使用国际（Z.ai）的 API 地址",
        sends: "智谱 GLM",
      },
      {
        catalogId: "siliconflow",
        value: "en",
        host: "api.siliconflow.com",
        region: "intl",
        line: "Using the International endpoint",
        sends: "SiliconFlow",
      },
    ] as const) {
      await language(window, value);
      const form = await addProviderForm(settings, catalogId);
      await expect(form.getByRole("combobox", { name: /^(Region|地域)/ })).toHaveCount(0);
      // The line names the company, whose regions both are.
      await expect(form.getByTestId("provider-sends")).toContainText(sends);
      await form.locator('input[type="password"]').fill(keyMadeAt(host));
      await form.getByTestId("provider-use-key").click();

      const page = settings.getByTestId("provider-page");
      await expect(page.getByTestId("provider-page-region-line")).toContainText(line);
      await expect(page.getByTestId("provider-page-region-change")).toHaveText(
        value === "en" ? "Change" : "更改",
      );
      expect(await savedRegion(window, catalogId)).toBe(region);
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await settings.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-region-${catalogId}-${value}.png`,
        });
      }
      await settings.getByTestId("models-back").click();
      // Kimi is set up again from its other region's tile, below.
      if (catalogId === "kimi") {
        await window.evaluate(async () => {
          const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
          for (const each of await core.listChatProviders()) {
            if (each.catalogId === "kimi") await core.deleteChatProvider(each.id);
          }
        });
      }
    }
  } finally {
    await app.close();
  }
});

test("a key no region takes shows the region choice and where each region makes its keys; the region chosen is the only one tried", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    await language(window, "en");
    const settings = await openSettings(window, "models");
    const form = await addProviderForm(settings, "kimi");
    const key = form.locator('input[type="password"]');
    const useKey = form.getByTestId("provider-use-key");
    const problem = form.getByTestId("connection-test");
    const region = form.getByRole("combobox", { name: /^(Region|地域)/ });
    const note = form.getByTestId("provider-region-note");

    await key.fill("sk-refused-by-every-region");
    await useKey.click();

    await expect(problem).toContainText("This key isn’t valid for Kimi");
    await expect(region).toHaveValue("intl");
    await expect(note).toContainText(
      "A Kimi key works only in the region it was made in. Each region’s key page:",
    );
    await expect(note.getByRole("link")).toHaveText(["China ↗", "International ↗"]);
    await expect(note.getByRole("link").first()).toHaveAttribute(
      "href",
      "https://platform.kimi.com/console/api-keys",
    );
    await expect(note.getByRole("link").last()).toHaveAttribute(
      "href",
      "https://platform.kimi.ai/console/api-keys",
    );
    if (process.env.INCARNAMIND_SCREENSHOTS) {
      await settings.screenshot({
        path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-region-asked-en.png`,
      });
    }

    await language(window, "zh-CN");
    await expect(problem).toContainText("这个密钥对 Kimi");
    await expect(note).toContainText("Kimi 的密钥只能在创建它的地域使用。各地域的密钥页面：");
    await expect(note.getByRole("link")).toHaveText(["中国 ↗", "国际 ↗"]);
    if (process.env.INCARNAMIND_SCREENSHOTS) {
      await settings.screenshot({
        path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-region-asked-zh-CN.png`,
      });
    }
    await language(window, "en");

    // China chosen, an International key is tried there alone, and refused.
    await region.selectOption("cn");
    await expect(key).toHaveValue("");
    await expect(problem).toHaveCount(0);
    await key.fill(keyMadeAt("api.moonshot.ai"));
    await useKey.click();
    await expect(problem).toContainText("This key isn’t valid for Kimi");
    await expect(region).toHaveValue("cn");
    expect(await savedRegion(window, "kimi")).toBeNull();

    await region.selectOption("intl");
    await key.fill(keyMadeAt("api.moonshot.ai"));
    await useKey.click();
    await expect(settings.getByTestId("provider-page-region-line")).toContainText(
      "Using the International endpoint",
    );
    expect(await savedRegion(window, "kimi")).toBe("intl");
  } finally {
    await app.close();
  }
});

test("at first run, the key form asks no region either: Where do I find it? has each region's key page, and a key from the other region gets the card to ready, in English and Chinese", async () => {
  for (const { value, host, region, ready, pages, links } of [
    {
      value: "en",
      host: "api.moonshot.cn",
      region: "cn",
      ready: "Kimi (Moonshot AI) works",
      pages: "Kimi’s key pages, by region:",
      links: ["China ↗", "International ↗"],
    },
    {
      value: "zh-CN",
      host: "api.moonshot.ai",
      region: "intl",
      ready: "“Kimi（月之暗面）”可以使用了",
      pages: "Kimi 各地域的密钥页面：",
      links: ["中国 ↗", "国际 ↗"],
    },
  ] as const) {
    const folder = await createDataFolder();
    const { app, window } = await launchApp(folder, { fakeChat: true });
    try {
      await setWindowSize(app, window, 1100, 900);
      await acceptPlainTextKeys(window);
      await language(window, value);
      await newMind(window);
      await window.getByTestId("mind-editor").click();
      await window.keyboard.press("ControlOrMeta+j");
      await window.keyboard.type("What is in my Documents?");
      await window.keyboard.press("Enter");
      const card = window.getByTestId("model-setup");
      await card.getByTestId("setup-key").click();
      const form = card.getByTestId("provider-form");
      await form.getByRole("radio", { name: "Kimi", exact: true }).check({ force: true });
      await expect(form.getByRole("combobox", { name: /^(Region|地域)/ })).toHaveCount(0);
      await form.getByTestId("provider-key-help-toggle").click();
      const keyPages = form.getByTestId("provider-key-pages");
      await expect(keyPages).toContainText(pages);
      await expect(keyPages.getByRole("link")).toHaveText([...links]);
      if (process.env.INCARNAMIND_SCREENSHOTS) {
        await card.screenshot({
          path: `${process.env.INCARNAMIND_SCREENSHOTS}/key-region-first-run-${value}.png`,
        });
      }
      await form.locator('input[type="password"]').fill(keyMadeAt(host));
      await form.getByTestId("provider-use-key").click();

      await expect(card.getByTestId("setup-ready-title")).toHaveText(ready);
      expect(await savedRegion(window, "kimi")).toBe(region);
    } finally {
      await app.close();
      await removeDataFolder(folder);
    }
  }
});

test("a provider's page says its region in a quiet line; Change shows the regions in place, Esc goes back, and another opens its own form", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 900);
    await acceptPlainTextKeys(window);
    await window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await core.saveChatProvider({
        kind: "kimi",
        endpoint: "intl",
        apiKey: "sk-not-real",
        modelId: "kimi-k2.6",
      });
    });
    const settings = await openSettings(window, "models");
    await settings.getByTestId("provider-row").first().click();
    const page = settings.getByTestId("provider-page");
    const line = page.getByTestId("provider-page-region-line");
    const change = page.getByTestId("provider-page-region-change");
    const regions = page.getByTestId("provider-page-region");

    await expect(line).toContainText("Using the International endpoint");
    await expect(regions).toHaveCount(0);
    await language(window, "zh-CN");
    await expect(line).toContainText("使用国际的 API 地址");
    await expect(change).toHaveText("更改");
    await language(window, "en");

    await change.focus();
    await window.keyboard.press("Enter");
    await expect(regions).toBeFocused();
    await expect(regions).toHaveValue("intl");
    await window.keyboard.press("Escape");
    await expect(regions).toHaveCount(0);
    await expect(change).toBeFocused();
    await expect(settings).toBeVisible();

    await change.click();
    await regions.selectOption("cn");
    const form = settings.getByTestId("provider-add").getByTestId("provider-form");
    await expect(form.locator('input[type="password"]')).toHaveValue("");
    await expect(form.getByRole("combobox", { name: /^(Region|地域)/ })).toHaveCount(0);
  } finally {
    await app.close();
  }
});
