/**
 * Setting IncarnaMind up in the User's agent from Settings → Agents (#351):
 * one click where the agent allows it, the key handed over behind the scenes,
 * the agent's settings file written only as shown, with Undo, and tidied when
 * the agent is disconnected. Every path is in a temporary home folder.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { beforeAll, describe, expect, it, vi } from "vitest";
import type { AgentLaunch, Core } from "../../src/core";
import { openPackage } from "../../src/core/documents/formats/zip";
import { agentsAdapter, buildAgentShim, connectLegacyAgent } from "../helpers/agents";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";

const TOKEN_ENV = "INCARNAMIND_AGENT_TOKEN";

async function started() {
  const dataDir = await createTempDataFolder();
  const home = await createTempDataFolder();
  const applications = await createTempDataFolder();
  const tempDir = await createTempDataFolder();
  const links: string[] = [];
  const opened: string[] = [];
  const shim = await buildAgentShim();
  const core = startCore(dataDir, {
    paths: { dataDir, documents: join(dataDir, "Documents"), tempDir },
    agents: {
      ...(await agentsAdapter(dataDir)),
      shimScript: shim,
      home,
      applications: [applications],
      openLink: async (url) => {
        links.push(url);
      },
    },
    shell: {
      openPath: async (path) => {
        opened.push(path);
      },
      showItemInFolder: () => undefined,
    },
  });
  const sources = await createSourceFolder();
  const [tides] = await addAndProcess(core, [
    await writeSourceFile(
      sources,
      "Tides.md",
      "# Tides\n\nSpring tides come at new and full moon, when the sun and the moon pull in line.\n",
    ),
  ]);
  if (!tides) throw new Error("Nothing was added.");
  const coast = await core.createLibraryGroup({ name: "Coast", description: "The sea." });
  await core.assignDocumentGroup(tides.id, coast.id);
  await core.setAgentsEnabled(true);
  return { core, home, applications, links, opened, shim, coast };
}

/** The settings an agent was given, as it would start the server. */
const launchIn = (entry: { command: string; args: string[]; env: Record<string, string> }) =>
  entry satisfies AgentLaunch;

async function searchesThrough(core: Core, launch: AgentLaunch) {
  const agent = await connectLegacyAgent(launch);
  const found = await agent.callTool({ name: "search", arguments: { query: "spring tides" } });
  expect(found.isError ?? false).toBe(false);
  void core;
}

describe("setting an agent up", () => {
  beforeAll(async () => {
    await buildAgentShim();
  }, 60_000);

  it("lists the agents on this computer first", async () => {
    const { core, home, applications } = await started();
    await mkdir(join(home, ".codex"));
    await mkdir(join(applications, "Cursor.app"));
    const clients = await core.listAgentClients();
    expect(clients.slice(0, 2).map((each) => [each.id, each.installed])).toEqual([
      ["codex", true],
      ["cursor", true],
    ]);
    expect(clients.at(-1)).toMatchObject({ id: "other", installed: false, setup: "manual" });
  });

  it("writes Codex's settings as shown, works from them, and takes them back with Undo", async () => {
    const { core, home, coast } = await started();
    const path = join(home, ".codex", "config.toml");
    await mkdir(join(home, ".codex"));
    const theirs = 'model = "gpt-6-luna"\n\n[mcp_servers.github]\ncommand = "gh-mcp"\n';
    await writeFile(path, theirs);

    const preview = await core.previewAgentSetup("codex");
    expect(preview?.path).toBe(path);
    expect(preview?.change).toContain("[mcp_servers.incarnamind]");
    expect(preview?.change).toContain("•••••••");

    const { pairing, result } = await core.setUpAgent({ client: "codex", folderIds: [coast.id] });
    expect(result).toEqual({ kind: "wrote", path });
    expect(pairing).toMatchObject({ name: "Codex", client: "codex", folderIds: [coast.id] });
    const written = await readFile(path, "utf8");
    expect(written.startsWith(theirs.trimEnd())).toBe(true);
    const token = (await core.getAgentLaunch(pairing.id)).env[TOKEN_ENV] ?? "";
    expect(written).toContain(`INCARNAMIND_AGENT_TOKEN = "${token}"`);
    await searchesThrough(core, await core.getAgentLaunch(pairing.id));

    await core.undoAgentSetup(pairing.id);
    expect(await readFile(path, "utf8")).toBe(theirs);
    expect((await core.getAgents()).pairings).toEqual([]);
  });

  it("adds itself to an mcpServers file, keeps the User's servers, and leaves when disconnected", async () => {
    const { core, home, coast } = await started();
    const path = join(home, ".gemini", "settings.json");
    await mkdir(join(home, ".gemini"));
    await writeFile(
      path,
      JSON.stringify({ theme: "dark", mcpServers: { notes: { command: "n" } } }),
    );

    const { pairing, result } = await core.setUpAgent({
      client: "gemini-cli",
      folderIds: [coast.id],
    });
    expect(result.kind).toBe("wrote");
    const settings = JSON.parse(await readFile(path, "utf8"));
    expect(settings.theme).toBe("dark");
    expect(Object.keys(settings.mcpServers)).toEqual(["notes", "incarnamind"]);
    await searchesThrough(core, launchIn(settings.mcpServers.incarnamind));

    await core.disconnectAgent(pairing.id);
    expect(JSON.parse(await readFile(path, "utf8"))).toEqual({
      theme: "dark",
      mcpServers: { notes: { command: "n" } },
    });
  });

  it("leaves a settings file it can't read alone, and gives the settings to paste instead", async () => {
    const { core, home, coast } = await started();
    const path = join(home, ".qwen", "settings.json");
    await mkdir(join(home, ".qwen"));
    await writeFile(path, "{ // comments aren't JSON\n}");
    const { pairing, result } = await core.setUpAgent({
      client: "qwen-code",
      folderIds: [coast.id],
    });
    expect(result).toMatchObject({ kind: "copy", how: "settings", path });
    expect(await readFile(path, "utf8")).toBe("{ // comments aren't JSON\n}");
    if (result.kind !== "copy") throw new Error("Expected settings to copy.");
    const token = (await core.getAgentLaunch(pairing.id)).env[TOKEN_ENV] ?? "";
    expect(result.text).toContain(token);
    expect(result.shown).not.toContain(token);
  });

  it("opens Cursor and VS Code on their own install links", async () => {
    const { core, links, coast } = await started();
    const cursor = await core.setUpAgent({ client: "cursor", folderIds: [coast.id] });
    const vscode = await core.setUpAgent({ client: "vscode", folderIds: [coast.id] });
    expect(cursor.result).toEqual({ kind: "opened", how: "link" });
    expect(vscode.result).toEqual({ kind: "opened", how: "link" });

    const cursorUrl = new URL(links[0] ?? "");
    expect(cursorUrl.protocol).toBe("cursor:");
    const config = JSON.parse(
      Buffer.from(cursorUrl.searchParams.get("config") ?? "", "base64").toString(),
    );
    await searchesThrough(core, launchIn(config));
    expect(links[1]?.startsWith("vscode:mcp/install?")).toBe(true);
    const vscodeConfig = JSON.parse(decodeURIComponent(links[1]?.split("?")[1] ?? ""));
    expect(vscodeConfig).toMatchObject({ name: "incarnamind", type: "stdio" });
  });

  it("opens Claude on an install bundle carrying the shim, in English and Chinese", async () => {
    const { core, opened, shim, coast } = await started();
    const { pairing, result } = await core.setUpAgent({
      client: "claude-desktop",
      folderIds: [coast.id],
    });
    expect(result).toEqual({ kind: "opened", how: "bundle" });
    expect(opened[0]).toMatch(/IncarnaMind\.mcpb$/);
    const bundle = openPackage(await readFile(opened[0] ?? ""));
    const manifest = JSON.parse(new TextDecoder().decode(await bundle.read("manifest.json")));
    expect(manifest).toMatchObject({
      manifest_version: "0.3",
      name: "incarnamind",
      server: { type: "node", entry_point: "server/index.js" },
    });
    const token = (await core.getAgentLaunch(pairing.id)).env[TOKEN_ENV];
    expect(manifest.server.mcp_config.env[TOKEN_ENV]).toBe(token);
    expect(manifest.server.mcp_config.env).not.toHaveProperty("ELECTRON_RUN_AS_NODE");
    expect(Buffer.from((await bundle.read("server/index.js")) ?? [])).toEqual(await readFile(shim));
    const zh = JSON.parse(
      new TextDecoder().decode(await bundle.read("mcpb-resources/zh-Hans.json")),
    );
    expect(zh.description).toMatch(/引文/);
  });

  it("gives Claude Code one line, its key hidden where it shows", async () => {
    const { core, coast } = await started();
    const { pairing, result } = await core.setUpAgent({
      client: "claude-code",
      folderIds: [coast.id],
    });
    if (result.kind !== "copy") throw new Error("Expected a line to copy.");
    const token = (await core.getAgentLaunch(pairing.id)).env[TOKEN_ENV] ?? "";
    expect(result.how).toBe("command");
    expect(result.text).toMatch(/^claude mcp add --scope user /);
    expect(result.text).toContain(token);
    expect(result.shown).not.toContain(token);
    expect(result.shown).toContain("•••••••");
  });

  it("leaves no pairing when the agent's app can't be opened", async () => {
    const { core, links, coast } = await started();
    links.push = () => {
      throw new Error("No app can open this link.");
    };
    await expect(core.setUpAgent({ client: "cursor", folderIds: [coast.id] })).rejects.toThrow(
      /No app/,
    );
    expect((await core.getAgents()).pairings).toEqual([]);
  });

  it("names a second copy of the same agent apart, and needs a Folder", async () => {
    const { core, coast } = await started();
    await core.setUpAgent({ client: "other", folderIds: [coast.id] });
    const second = await core.setUpAgent({ client: "other", folderIds: [coast.id] });
    expect(second.pairing.name).toBe("Another agent 2");
    await expect(core.setUpAgent({ client: "codex", folderIds: [] })).rejects.toThrow(/Folder/);
  });

  it("shows a Citation an agent asks to show, in front", async () => {
    const { core, coast } = await started();
    const { pairing } = await core.setUpAgent({ client: "other", folderIds: [coast.id] });
    const agent = await connectLegacyAgent(await core.getAgentLaunch(pairing.id));
    const found = await agent.callTool({ name: "search", arguments: { query: "spring tides" } });
    const cite = (found.structuredContent as { passages: { cite: string }[] }).passages[0]?.cite;
    const shown = vi.fn();
    core.on("agents.citationShown", shown);

    const result = await agent.callTool({ name: "show_citation", arguments: { cite } });
    expect(result.isError ?? false).toBe(false);
    expect(shown).toHaveBeenCalledWith(
      expect.objectContaining({
        quote: expect.stringContaining("Spring tides come at new and full moon"),
        citation: expect.objectContaining({ check: "found" }),
      }),
    );
    const unknown = await agent.callTool({
      name: "show_citation",
      arguments: { cite: "[cite:zzzzzzzz]" },
    });
    expect(unknown.isError).toBe(true);
  });
});
