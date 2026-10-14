import { describe, expect, test } from "vitest";
import { type Executor, InvalidInputError } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";

/**
 * Export uses Word until the User saves other options, and the deliverables leave Questions
 * out. A Word file tracks the User's edits and marks AI text and failed checks (#107).
 */
const DEFAULT_WORD = {
  includeQuestions: false,
  trackChanges: true,
  markAiText: true,
  commentFailedChecks: true,
} as const;

const DEFAULT_EXPORT = {
  format: "docx",
  docx: DEFAULT_WORD,
  pdf: { includeQuestions: false },
  html: { includeQuestions: false },
} as const;

const PDF_EXPORT = {
  format: "pdf",
  docx: { ...DEFAULT_WORD, markAiText: false },
  pdf: { includeQuestions: true },
  html: { includeQuestions: false },
  theirChanges: "accepted",
} as const;

describe("Settings", () => {
  test("a new data folder starts with the defaults", async () => {
    const core = startCore(await createTempDataFolder());

    expect(await core.getSettings()).toEqual({
      user: {
        language: "system",
        chatModel: null,
        suggestWhileWriting: false,
        leaveOutOwnFiles: true,
        exportOptions: DEFAULT_EXPORT,
        authorName: null,
        monthlyCap: null,
        webSearch: false,
      },
      device: {
        sidebarWidth: 248,
        sidebarHidden: false,
        chatSetupDismissed: false,
        gettingStarted: {
          started: false,
          citationChecked: false,
          indexed: false,
          askedOwn: false,
          hidden: false,
        },
        skillScriptsEnabled: true,
        skillScriptTimeoutSeconds: 60,
        suggestionOffers: 0,
        meetingsEnabled: false,
        meetingSystemAudio: true,
        meetingNoticeAccepted: false,
        motion: "system",
      },
      language: "en",
      scriptSandbox: "none",
      // No capture in these tests: Meetings can't record.
      meetingAudio: "none",
      author: "Priya Raman",
    });
  });

  test("say how Skill scripts run here: at the level of the Executor the app gave the core", async () => {
    const executor: Executor = {
      level: "os",
      run: async () => {
        throw new Error("Nothing runs in this test.");
      },
    };
    const core = startCore(await createTempDataFolder(), { executor });

    expect((await core.getSettings()).scriptSandbox).toBe("os");
    // Not a setting: it can't be changed.
    await expect(
      core.updateSettings({ device: { scriptSandbox: "none" } } as never),
    ).rejects.toThrow(InvalidInputError);
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

    expect(settings.user).toEqual({
      language: "zh-CN",
      chatModel: null,
      suggestWhileWriting: false,
      leaveOutOwnFiles: true,
      exportOptions: DEFAULT_EXPORT,
      authorName: null,
      monthlyCap: null,
      webSearch: false,
    });
    expect(settings.device).toEqual({
      sidebarWidth: 320,
      sidebarHidden: false,
      chatSetupDismissed: false,
      gettingStarted: {
        started: false,
        citationChecked: false,
        indexed: false,
        askedOwn: false,
        hidden: false,
      },
      skillScriptsEnabled: true,
      skillScriptTimeoutSeconds: 60,
      suggestionOffers: 0,
      meetingsEnabled: false,
      meetingSystemAudio: true,
      meetingNoticeAccepted: false,
      motion: "system",
    });
  });

  test("a width must be a number", async () => {
    const core = startCore(await createTempDataFolder());

    await expect(
      core.updateSettings({ device: { sidebarWidth: null as unknown as number } }),
    ).rejects.toThrow(InvalidInputError);
  });

  test("settings survive a restart", async () => {
    const dataDir = await createTempDataFolder();
    const before = startCore(dataDir);
    const device = {
      sidebarWidth: 300,
      sidebarHidden: true,
      chatSetupDismissed: true,
      gettingStarted: {
        started: true,
        citationChecked: true,
        indexed: false,
        askedOwn: false,
        hidden: true,
      },
      skillScriptsEnabled: false,
      skillScriptTimeoutSeconds: 15,
      suggestionOffers: 1,
      meetingsEnabled: true,
      meetingSystemAudio: false,
      meetingNoticeAccepted: true,
      motion: "reduced" as const,
    };
    await before.updateSettings({
      user: {
        language: "zh-CN",
        suggestWhileWriting: true,
        leaveOutOwnFiles: false,
        exportOptions: PDF_EXPORT,
        authorName: "P. Raman",
        monthlyCap: { amount: 20, currency: "USD" },
        webSearch: true,
      },
      device,
    });
    before.close();

    const after = startCore(dataDir);

    expect(await after.getSettings()).toEqual({
      user: {
        language: "zh-CN",
        chatModel: null,
        suggestWhileWriting: true,
        leaveOutOwnFiles: false,
        exportOptions: PDF_EXPORT,
        authorName: "P. Raman",
        monthlyCap: { amount: 20, currency: "USD" },
        webSearch: true,
      },
      device,
      language: "zh-CN",
      scriptSandbox: "none",
      meetingAudio: "none",
      author: "P. Raman",
    });
  });

  test.each([
    { name: "an unknown language", patch: { user: { language: "fr" } } },
    { name: "an unknown setting", patch: { user: { theme: "dark" } } },
    { name: "a setting in the wrong group", patch: { device: { language: "en" } } },
    { name: "an unknown group", patch: { secrets: { apiKey: "sk-…" } } },
    { name: "a negative width", patch: { device: { sidebarWidth: -1 } } },
    { name: "a width that isn't a number", patch: { device: { sidebarWidth: "wide" } } },
    { name: "the tabs the app kept before groups", patch: { device: { openMinds: ["mind-a"] } } },
    {
      name: "a checklist with a step missing",
      patch: { device: { gettingStarted: { started: true, hidden: false } } },
    },
    {
      name: "a script switch that isn't true or false",
      patch: { device: { skillScriptsEnabled: 1 } },
    },
    { name: "a script timeout of 0", patch: { device: { skillScriptTimeoutSeconds: 0 } } },
    {
      name: "a script timeout over an hour",
      patch: { device: { skillScriptTimeoutSeconds: 3601 } },
    },
    {
      name: "a script timeout in part seconds",
      patch: { device: { skillScriptTimeoutSeconds: 1.5 } },
    },
    { name: "an unknown motion choice", patch: { device: { motion: "slow" } } },
    {
      name: "an export format the app can't write",
      patch: { user: { exportOptions: { ...DEFAULT_EXPORT, format: "odt" } } },
    },
    {
      name: "export options with a format's options missing",
      patch: { user: { exportOptions: { format: "pdf", docx: { includeQuestions: false } } } },
    },
    {
      name: "export options whose Questions switch isn't true or false",
      patch: { user: { exportOptions: { ...DEFAULT_EXPORT, pdf: { includeQuestions: "yes" } } } },
    },
    {
      name: "a way to show someone's changes that isn't one",
      patch: { user: { exportOptions: { ...DEFAULT_EXPORT, theirChanges: "hidden" } } },
    },
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
