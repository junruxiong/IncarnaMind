import { describe, expect, test } from "vitest";
import { InvalidInputError } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";

describe("Settings", () => {
  test("a new data folder starts with the defaults", async () => {
    const core = startCore(await createTempDataFolder());

    expect(await core.getSettings()).toEqual({
      user: { language: "system", chatModel: null },
      device: { sidebarWidth: 270, viewerWidth: 420, chatSetupDismissed: false },
      language: "en",
    });
  });

  test.each([
    { system: ["zh-Hans-CN", "en-US"], expected: "zh-CN" },
    { system: ["zh-TW"], expected: "zh-CN" },
    { system: ["en-GB", "zh-CN"], expected: "en" },
    { system: ["fr-FR", "zh-CN"], expected: "zh-CN" },
    { system: ["fr-FR", "de-DE"], expected: "en" },
    { system: [], expected: "en" },
  ])(
    "by default the language follows the OS: $system → $expected",
    async ({ system, expected }) => {
      const core = startCore(await createTempDataFolder(), { systemLanguages: () => system });

      expect((await core.getSettings()).language).toBe(expected);
    },
  );

  test("a chosen language overrides the OS language", async () => {
    const core = startCore(await createTempDataFolder(), { systemLanguages: () => ["en-US"] });

    const settings = await core.updateSettings({ user: { language: "zh-CN" } });

    expect(settings.user.language).toBe("zh-CN");
    expect(settings.language).toBe("zh-CN");
    expect(await core.getSettings()).toEqual(settings);
  });

  test("choosing 'system' again goes back to the OS language", async () => {
    const core = startCore(await createTempDataFolder(), { systemLanguages: () => ["zh-CN"] });
    await core.updateSettings({ user: { language: "en" } });

    const settings = await core.updateSettings({ user: { language: "system" } });

    expect(settings.language).toBe("zh-CN");
  });

  test("per-User and per-device settings change independently", async () => {
    const core = startCore(await createTempDataFolder());

    await core.updateSettings({ device: { sidebarWidth: 320 } });
    const settings = await core.updateSettings({ user: { language: "zh-CN" } });

    expect(settings.user).toEqual({ language: "zh-CN", chatModel: null });
    expect(settings.device).toEqual({
      sidebarWidth: 320,
      viewerWidth: 420,
      chatSetupDismissed: false,
    });
  });

  test("settings survive a restart", async () => {
    const dataDir = await createTempDataFolder();
    const before = startCore(dataDir);
    await before.updateSettings({
      user: { language: "zh-CN" },
      device: { sidebarWidth: 300, viewerWidth: 500, chatSetupDismissed: true },
    });
    before.close();

    const after = startCore(dataDir);

    expect(await after.getSettings()).toEqual({
      user: { language: "zh-CN", chatModel: null },
      device: { sidebarWidth: 300, viewerWidth: 500, chatSetupDismissed: true },
      language: "zh-CN",
    });
  });

  test.each([
    { name: "an unknown language", patch: { user: { language: "fr" } } },
    { name: "an unknown setting", patch: { user: { theme: "dark" } } },
    { name: "a setting in the wrong group", patch: { device: { language: "en" } } },
    { name: "an unknown group", patch: { secrets: { apiKey: "sk-…" } } },
    { name: "a negative width", patch: { device: { sidebarWidth: -1 } } },
    { name: "a width that isn't a number", patch: { device: { viewerWidth: "wide" } } },
    {
      name: "a default model without a model",
      patch: { user: { chatModel: { providerId: "x" } } },
    },
    {
      name: "a default model on a provider that doesn't exist",
      patch: { user: { chatModel: { providerId: "no-such-provider", modelId: "gpt-5.4-mini" } } },
    },
  ])("rejects $name and changes nothing", async ({ patch }) => {
    const core = startCore(await createTempDataFolder());
    const before = await core.getSettings();

    await expect(
      core.updateSettings({ device: { sidebarWidth: 999 }, ...(patch as object) }),
    ).rejects.toThrow(InvalidInputError);
    expect(await core.getSettings()).toEqual(before);
  });
});
