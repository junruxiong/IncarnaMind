import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  createDataFolder,
  dismissChatSetup,
  expectReadyDocuments,
  launchApp,
  openSettings,
  removeDataFolder,
  setWindowSize,
  showSettingsPage,
} from "./app";

/*
 * Settings → Agents (#351): turned on, an agent is connected after one plain
 * confirmation that names what it may read and where that goes; Codex's
 * settings file is written as shown, with Undo; an agent set up by pasting
 * its settings searches, and its activity lists each call; Disconnect stops
 * it. The home folder is a temporary one, so no agent of the User's is
 * touched, and copying is caught in the page, so the machine's clipboard isn't.
 */

let dataDir: string;
let sources: string;
let home: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await realpath(await createDataFolder());
  home = await realpath(await createDataFolder());
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
  await removeDataFolder(home);
});

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

/** Saves a screenshot when INCARNAMIND_SCREENSHOTS names a folder for them. */
async function shot(window: Page, name: string) {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const CODEX_SETTINGS = 'model = "gpt-6-luna"\n';

/** A Folder "Coast" with one Document in it. */
async function withCoast(window: Page) {
  const tides = join(sources, "Tides.md");
  await writeFile(
    tides,
    "# Tides\n\nSpring tides happen twice a month, at new moon and full moon, when the sun and the moon pull in line.\n",
  );
  await window.evaluate(async (path) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const coast = await bridge.createLibraryGroup({ name: "Coast", description: "The sea." });
    await bridge.addDocuments([path], { groupId: coast.id });
  }, tides);
  await expectReadyDocuments(window, 1);
}

/** What the page copies goes here instead of the machine's clipboard. */
async function catchCopies(window: Page) {
  await window.evaluate(() => {
    const caught = globalThis as unknown as { copied?: string };
    navigator.clipboard.writeText = async (text: string) => {
      caught.copied = text;
    };
  });
}

const copied = (window: Page) =>
  window.evaluate(() => (globalThis as unknown as { copied?: string }).copied ?? "");

test("off by default; Connect asks once, writes Codex's settings as shown, and Undo takes them back", async () => {
  await mkdir(join(home, ".codex"));
  const codexSettings = join(home, ".codex", "config.toml");
  await writeFile(codexSettings, CODEX_SETTINGS);
  const { app, window } = await launchApp(dataDir, { home });
  try {
    await dismissChatSetup(window);
    await withCoast(window);
    const page = await openSettings(window, "agents");
    const enabled = page.getByTestId("agents-enabled");
    await expect(enabled).not.toBeChecked();
    await expect(page).toContainText("Off: no agent can reach your Documents.");
    await expect(page.getByTestId("agents-clients")).toBeHidden();
    await shot(window, "agents-off");

    await enabled.click();
    await expect(enabled).toBeChecked();
    const list = page.getByTestId("agents-list");
    // Only the first-class agents and "Other agent".
    await expect(list.getByTestId("agent-client")).toHaveCount(7);
    await expect(page.getByTestId("agents-clients")).toContainText("can’t reach IncarnaMind");
    const codex = list.locator('[data-client="codex"]');
    await expect(codex).toContainText("ChatGPT app and Codex");
    await expect(codex).toContainText("ordinary Chat mode doesn’t use it");
    await expect(codex).toContainText(
      "IncarnaMind adds itself to ChatGPT app and Codex’s settings",
    );
    await shot(window, "agents-on");
    await codex.getByTestId("agent-connect").click();

    const dialog = window.getByTestId("connect-agent-dialog");
    await expect(dialog).toContainText("Let ChatGPT app and Codex read your Documents?");
    await expect(dialog).toContainText("Some plans may use it to train their models.");
    await expect(dialog.getByTestId("connect-agent-change")).toContainText(codexSettings);
    const skillFile = join(home, ".agents", "skills", "incarnamind", "SKILL.md");
    await expect(dialog.getByTestId("connect-agent-skill")).toContainText(skillFile);
    const allow = dialog.getByTestId("connect-agent-allow");
    await expect(allow).toBeDisabled();
    // Linux with no keyring running (CI): the dialog asks to keep the key as plain text first.
    await acceptPlainTextKeys(window, dialog);
    await dialog.getByTestId("connect-agent-folder").check();
    await dialog.getByTestId("connect-agent-change").locator("summary").click();
    await shot(window, "agents-connect-dialog");
    await allow.click();
    await expect(dialog).toBeHidden();

    const connected = page
      .getByTestId("agent-pairing")
      .filter({ hasText: "ChatGPT app and Codex" });
    await expect(connected.getByTestId("agent-setup-done")).toContainText(
      "Added to ChatGPT app and Codex’s settings.",
    );
    await expect(connected.getByTestId("agent-pairing-reads")).toHaveText("Reads Coast");
    await expect(connected.getByTestId("agent-pairing-skill")).toContainText("Skill installed");
    expect(await readFile(skillFile, "utf8")).toContain("name: incarnamind");
    await shot(window, "agents-connected");
    const written = await readFile(codexSettings, "utf8");
    expect(written).toContain("[mcp_servers.incarnamind]");
    expect(written.startsWith(CODEX_SETTINGS.trimEnd())).toBe(true);

    await connected.getByTestId("agent-setup-undo").click();
    await expect(page.getByTestId("agent-pairing")).toHaveCount(0);
    expect(await readFile(codexSettings, "utf8")).toBe(CODEX_SETTINGS);
    await expect(readFile(skillFile, "utf8")).rejects.toThrow(/ENOENT/);
  } finally {
    await app.close();
  }
});

test("an agent set up by pasting its settings searches, its activity shows each call, and Disconnect stops it", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  try {
    await dismissChatSetup(window);
    await withCoast(window);
    await catchCopies(window);
    await acceptPlainTextKeys(window);
    const page = await openSettings(window, "agents");
    await page.getByTestId("agents-enabled").click();
    await page
      .getByTestId("agents-list")
      .locator('[data-client="other"]')
      .getByTestId("agent-connect")
      .click();
    const dialog = window.getByTestId("connect-agent-dialog");
    await expect(dialog).toContainText("Let Other agent read your Documents?");
    await dialog.getByTestId("connect-agent-folder").check();
    await dialog.getByTestId("connect-agent-allow").click();

    const row = page.getByTestId("agent-pairing").filter({ hasText: "Other agent" });
    await expect(row.getByTestId("agent-setup-done")).toContainText("Copied.");
    const settings = JSON.parse(await copied(window)) as {
      mcpServers: { incarnamind: { command: string; args: string[]; env: Record<string, string> } };
    };
    const server = settings.mcpServers.incarnamind;

    const agent = new Client({ name: "test-agent", version: "1.0.0" });
    await agent.connect(
      new StdioClientTransport({
        command: server.command,
        args: server.args,
        env: server.env,
        stderr: "pipe",
      }),
    );
    try {
      await agent.callTool({ name: "search", arguments: { query: "spring tides" } });
      await row.getByTestId("agent-activity-toggle").click();
      await expect(row.getByTestId("agent-activity-call").first()).toContainText(
        "Searched “spring tides” · 1 result",
      );
      await shot(window, "agents-activity");

      // Its Folders change in place.
      await row.getByTestId("agent-folders").click();
      await row.getByTestId("agent-pairing-folders").getByRole("checkbox").uncheck();
      await expect(row.getByTestId("agent-pairing-reads")).toHaveText("Reads no folders yet");

      // Its name renames in place: a double-click, or F2; Esc cancels and leaves Settings open.
      await row.getByTestId("agent-pairing-name").dblclick();
      await row.getByTestId("agent-pairing-rename").fill("Research agent");
      await window.keyboard.press("Enter");
      const renamed = page.getByTestId("agent-pairing").filter({ hasText: "Research agent" });
      await expect(renamed.getByTestId("agent-pairing-name")).toHaveText("Research agent");
      await renamed.getByTestId("agent-pairing-name").focus();
      await window.keyboard.press("F2");
      await renamed.getByTestId("agent-pairing-rename").fill("Something else");
      await window.keyboard.press("Escape");
      await expect(renamed.getByTestId("agent-pairing-name")).toHaveText("Research agent");
      await expect(page).toBeVisible();

      await renamed.getByTestId("agent-disconnect").click();
      await expect(page.getByTestId("agent-pairing")).toHaveCount(0);
    } finally {
      await agent.close();
    }
  } finally {
    await app.close();
  }
});

test("an Other agent's Details show the command, arguments and environment, each copied whole, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  try {
    await dismissChatSetup(window);
    await withCoast(window);
    await catchCopies(window);
    await acceptPlainTextKeys(window);
    const page = await openSettings(window, "agents");
    await page.getByTestId("agents-enabled").click();
    await page
      .getByTestId("agents-list")
      .locator('[data-client="other"]')
      .getByTestId("agent-connect")
      .click();
    const dialog = window.getByTestId("connect-agent-dialog");
    await dialog.getByTestId("connect-agent-folder").check();
    await dialog.getByTestId("connect-agent-allow").click();

    const row = page.getByTestId("agent-pairing").filter({ hasText: "Other agent" });
    await expect(row.getByTestId("agent-setup-done")).toContainText("Details for technical users");
    const all = JSON.parse(await copied(window)) as {
      mcpServers: { incarnamind: { command: string; args: string[]; env: Record<string, string> } };
    };
    const server = all.mcpServers.incarnamind;
    const key = server.env.INCARNAMIND_AGENT_TOKEN ?? "";
    expect(key).not.toBe("");

    await row.locator("summary", { hasText: "Details for technical users" }).click();
    const launch = row.getByTestId("agent-launch");
    await expect(launch).toContainText("Command");
    await expect(launch).toContainText("Arguments");
    await expect(launch).toContainText("Environment");
    await expect(launch).toContainText("INCARNAMIND_AGENT_TOKEN=•••••••");
    expect(await launch.innerText()).not.toContain(key);
    await shot(window, "agents-other-details");

    const copyOf = async (id: string) => {
      await launch.getByTestId(`agent-launch-${id}`).getByTestId("agent-launch-copy").click();
      return copied(window);
    };
    const command = await copyOf("command");
    const args = (await copyOf("args")).split("\n");
    const env: Record<string, string> = {};
    for (const name of Object.keys(server.env)) {
      const line = await copyOf(`env-${name}`);
      env[name] = line.slice(name.length + 1);
    }
    expect(command).toBe(server.command);
    expect(args).toEqual(server.args);
    expect(env).toEqual(server.env);
    expect(env.INCARNAMIND_AGENT_TOKEN).toBe(key);

    const agent = new Client({ name: "test-agent", version: "1.0.0" });
    await agent.connect(new StdioClientTransport({ command, args, env, stderr: "pipe" }));
    try {
      const found = await agent.callTool({ name: "search", arguments: { query: "spring tides" } });
      expect(found.isError ?? false).toBe(false);
    } finally {
      await agent.close();
    }

    await window.evaluate(async () => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await bridge.updateSettings({ user: { language: "zh-CN" } });
    });
    await expect(launch).toContainText("命令");
    await expect(launch).toContainText("参数");
    await expect(launch).toContainText("环境变量");
    await expect(launch).toContainText("INCARNAMIND_AGENT_TOKEN=•••••••");
    await shot(window, "agents-other-details-zh");
  } finally {
    await app.close();
  }
});

test("in Chinese the Codex agent is named in Chinese in the list, the dialog and the connected row", async () => {
  await mkdir(join(home, ".codex"));
  const { app, window } = await launchApp(dataDir, { home });
  try {
    await dismissChatSetup(window);
    await withCoast(window);
    await acceptPlainTextKeys(window);
    await setWindowSize(app, window, 900, 660);
    await window.evaluate(async () => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await bridge.setAgentsEnabled(true);
      await bridge.updateSettings({ user: { language: "zh-CN" } });
    });
    await window.getByRole("button", { name: "设置" }).click();
    const page = await showSettingsPage(window, "agents");
    const codex = page.getByTestId("agents-list").locator('[data-client="codex"]');
    await shot(window, "agents-codex-name-zh-list");
    await expect(codex).toContainText("IncarnaMind 会把自己添加到 ChatGPT 应用和 Codex 的设置中");
    await codex.getByTestId("agent-connect").click();

    const dialog = window.getByTestId("connect-agent-dialog");
    await shot(window, "agents-codex-name-zh-dialog");
    await expect(dialog).toContainText("允许 ChatGPT 应用和 Codex 读取你的文档？");
    await acceptPlainTextKeys(window, dialog);
    await dialog.getByTestId("connect-agent-folder").check();
    await dialog.getByTestId("connect-agent-allow").click();
    await expect(dialog).toBeHidden();

    const connected = page.getByTestId("agent-pairing").filter({ hasText: "ChatGPT 应用和 Codex" });
    await shot(window, "agents-codex-name-zh-connected-900");
    await expect(connected.getByTestId("agent-setup-done")).toContainText(
      "已添加到 ChatGPT 应用和 Codex 的设置中。开始一个新的 ChatGPT 应用和 Codex 会话即可使用。",
    );
    await setWindowSize(app, window, 1440, 900);
    await shot(window, "agents-codex-name-zh-connected-wide");
    await expect(connected.getByTestId("agent-setup-done")).toBeVisible();
    await expect(page).not.toContainText("app and");
  } finally {
    await app.close();
  }
});

test("Local mode pauses agents, and the page reads in Chinese at a narrow window", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  try {
    await dismissChatSetup(window);
    await setWindowSize(app, window, 900, 660);
    await window.evaluate(async () => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await bridge.setAgentsEnabled(true);
      await bridge.setLocalOnly(true);
      await bridge.updateSettings({ user: { language: "zh-CN" } });
    });
    await window.getByRole("button", { name: "设置" }).click();
    const page = await showSettingsPage(window, "agents");
    await expect(window.getByTestId("settings-page-title")).toHaveText("智能体");
    await expect(page.getByTestId("agents-paused")).toContainText(
      "开启“所有内容都留在这台电脑上”时暂停使用",
    );
    await shot(window, "agents-paused-zh-narrow");
    // The first-class agents and "Other agent" (其他智能体), and nothing spills out at either width.
    const list = page.getByTestId("agents-list");
    await expect(list.getByTestId("agent-client")).toHaveCount(7);
    await expect(list.locator('[data-client="other"]')).toContainText("其他智能体");
    await expect(list.locator('[data-client="codex"]')).toContainText("普通对话模式不会用到");
    const fitsPage = () =>
      page
        .getByTestId("settings-page-agents")
        .evaluate((element) => element.scrollWidth <= element.clientWidth + 1);
    expect(await fitsPage()).toBe(true);
    await setWindowSize(app, window, 1440, 900);
    expect(await fitsPage()).toBe(true);
  } finally {
    await app.close();
  }
});
