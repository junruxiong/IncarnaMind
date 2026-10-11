/**
 * Setting IncarnaMind up in the User's agent from Settings → Agents (#351),
 * in one click wherever the agent allows it (see ./clients for how each one
 * takes it). Every way pairs the agent first (./index) and gives it its
 * token behind the scenes: inside a bundle or a link the agent's app
 * installs, in its settings file, or in what the User pastes, where what is
 * shown hides it.
 *
 * Writing an agent's settings file keeps everything else in it, and is shown
 * before it is done (`preview`) and can be taken back right after (`undo`,
 * while the file is as IncarnaMind left it). A file IncarnaMind can't read is
 * left alone: the User gets the settings to paste instead. Disconnecting a
 * pairing takes IncarnaMind out of the file it was written to, if it is
 * still its own entry there.
 */
import { existsSync } from "node:fs";
import { mkdir, readFile, rm, unlink, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { deflateRawSync } from "node:zlib";
import { AGENT_TOKEN_ENV } from "../../shared/agentSocket";
import { translate } from "../../shared/i18n";
import type { MessageKey, MessageParams } from "../../shared/i18n/types";
import type {
  AgentClient,
  AgentLaunch,
  AgentPairing,
  AgentSetupPreview,
  AgentSetupResult,
  SetUpAgent,
} from "../api";
import { writeZip } from "../documents/formats/zip";
import { InvalidInputError, isRecord, NotFoundError } from "../errors";
import {
  AGENT_CLIENT_DEFINITIONS,
  type ClientDefinition,
  claudeCodeCommand,
  clientDefinition,
  codexTable,
  cursorLink,
  HIDDEN_KEY,
  hideToken,
  isInstalled,
  mcpServersSnippet,
  SERVER_KEY,
  serverEntry,
  vscodeLink,
  withCodexTable,
  withMcpServersEntry,
  withoutCodexTable,
} from "./clients";

/** How long a bundle waits in the temporary folder for the agent's app to install it. */
const BUNDLE_KEPT_MS = 15 * 60_000;

export interface AgentSetupOptions {
  /** Pairing (see ./index). */
  agents: {
    settings(): { pairings: AgentPairing[] };
    connect(input: unknown): Promise<{ pairing: AgentPairing; launch: AgentLaunch }>;
    disconnect(pairingId: unknown): Promise<void>;
    launch(pairingId: unknown): Promise<AgentLaunch>;
    /** How the server starts with this token, for showing settings before there is one. */
    launchFor(token: string): AgentLaunch;
  };
  /** The User's home folder, where agents keep their settings. */
  home: string;
  /** Where apps are installed (macOS's Applications folders), to tell which agents are here. */
  applications: readonly string[];
  platform: NodeJS.Platform;
  /** Where a bundle waits for the agent's app to install it. */
  tempDir: string;
  /** The shim's script (src/main/agentShim.ts, built), which a bundle carries. */
  shimScript: string | null;
  version: string;
  /** Opens an agent's install link in its app. */
  openLink: ((url: string) => Promise<void>) | null;
  /** Opens a file in the app it belongs to, e.g. a bundle in Claude. */
  openPath: ((path: string) => Promise<void>) | null;
  say(key: MessageKey, params?: MessageParams): string;
  reportError(error: unknown): void;
}

/** A settings file IncarnaMind just wrote, to take back. */
interface Written {
  path: string;
  before: string | null;
  after: string;
}

async function readText(path: string): Promise<string | null> {
  try {
    return await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}

/** The bundle format's own placeholders, which Claude fills in when it installs the bundle. */
const BUNDLE_FOLDER = "$" + "{__dirname}";
const BUNDLE_LOCALE = "$" + "{locale}";

/** The manifest of Claude's bundle: the shim, run on the app's own Node. */
function bundleManifest(launch: AgentLaunch, version: string, description: string) {
  return {
    manifest_version: "0.3",
    name: SERVER_KEY,
    display_name: "IncarnaMind",
    version,
    description,
    author: { name: "IncarnaMind", url: "https://github.com/junruxiong/IncarnaMind" },
    homepage: "https://github.com/junruxiong/IncarnaMind",
    license: "Apache-2.0",
    server: {
      type: "node",
      entry_point: "server/index.js",
      mcp_config: {
        command: "node",
        args: [`${BUNDLE_FOLDER}/server/index.js`],
        // The shim needs only where the app is and its key: not Electron's own variable.
        env: Object.fromEntries(
          Object.entries(launch.env).filter(([name]) => name !== "ELECTRON_RUN_AS_NODE"),
        ),
      },
    },
    localization: { resources: `mcpb-resources/${BUNDLE_LOCALE}.json`, default_locale: "en-US" },
  };
}

export function createAgentSetup(options: AgentSetupOptions) {
  const t = options.say;
  /** Settings files written, by pairing, while their Undo stands. */
  const written = new Map<string, Written>();
  const pathOf = (client: ClientDefinition) =>
    client.file ? join(options.home, ...client.file.path) : null;

  /** The file with IncarnaMind's entry for `launch` set, or taken out (null). */
  const changed = (client: ClientDefinition, before: string | null, launch: AgentLaunch | null) => {
    if (client.file?.format === "codex-toml") {
      return launch ? withCodexTable(before, codexTable(launch)) : withoutCodexTable(before ?? "");
    }
    return withMcpServersEntry(before, launch ? serverEntry(launch) : null);
  };

  /** What a setup adds to the file: Codex's table, or the entry under `mcpServers`. */
  const changeShown = (client: ClientDefinition, launch: AgentLaunch) =>
    client.file?.format === "codex-toml"
      ? codexTable(launch)
      : `${JSON.stringify({ mcpServers: { [SERVER_KEY]: serverEntry(launch) } }, null, 2)}\n`;

  const nameFor = (client: ClientDefinition) => {
    const base = client.name || t("agents.client.other");
    const taken = new Set(
      options.agents.settings().pairings.map((pairing) => pairing.name.toLocaleLowerCase()),
    );
    for (let n = 1; ; n++) {
      const name = n === 1 ? base : `${base} ${n}`;
      if (!taken.has(name.toLocaleLowerCase())) return name;
    }
  };

  const copy = (
    how: "command" | "settings",
    text: string,
    token: string,
    path?: string,
  ): Extract<AgentSetupResult, { kind: "copy" }> => ({
    kind: "copy",
    how,
    text,
    shown: hideToken(text, token),
    ...(path && { path }),
  });

  /** Claude's bundle, in the temporary folder for its app to install, then gone. */
  const openBundle = async (pairingId: string, launch: AgentLaunch) => {
    if (!options.shimScript || !options.openPath) {
      throw new Error("This copy of IncarnaMind can't make an install file for Claude.");
    }
    const shim = await readFile(options.shimScript);
    const bundle = writeZip(
      [
        {
          name: "manifest.json",
          data: JSON.stringify(
            bundleManifest(launch, options.version, translate("en", "agents.bundle.description")),
            null,
            2,
          ),
        },
        { name: "server/index.js", data: shim },
        {
          name: "mcpb-resources/zh-Hans.json",
          data: JSON.stringify({ description: translate("zh-CN", "agents.bundle.description") }),
        },
      ],
      deflateRawSync,
    );
    const folder = join(options.tempDir, `incarnamind-agent-${pairingId}`);
    await mkdir(folder, { recursive: true, mode: 0o700 });
    const path = join(folder, "IncarnaMind.mcpb");
    await writeFile(path, bundle, { mode: 0o600 });
    // It holds the pairing's key: it goes once the app has had time to install it.
    setTimeout(() => {
      rm(folder, { recursive: true, force: true }).catch(options.reportError);
    }, BUNDLE_KEPT_MS).unref?.();
    await options.openPath(path);
  };

  /** Adds IncarnaMind to the agent's settings file; the settings to paste if it can't be read. */
  const writeSettings = async (
    client: ClientDefinition,
    pairingId: string,
    launch: AgentLaunch,
  ): Promise<AgentSetupResult> => {
    const path = pathOf(client) as string;
    const token = launch.env[AGENT_TOKEN_ENV] ?? "";
    const before = await readText(path);
    let after: string;
    try {
      after = changed(client, before, launch);
    } catch {
      return copy("settings", changeShown(client, launch), token, path);
    }
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, after, before === null ? { mode: 0o600 } : {});
    written.set(pairingId, { path, before, after });
    return { kind: "wrote", path };
  };

  const install = async (
    client: ClientDefinition,
    pairingId: string,
    launch: AgentLaunch,
  ): Promise<AgentSetupResult> => {
    const token = launch.env[AGENT_TOKEN_ENV] ?? "";
    switch (client.setup) {
      case "bundle":
        await openBundle(pairingId, launch);
        return { kind: "opened", how: "bundle" };
      case "link": {
        if (!options.openLink) throw new Error("This copy of IncarnaMind can't open other apps.");
        await options.openLink(client.id === "cursor" ? cursorLink(launch) : vscodeLink(launch));
        return { kind: "opened", how: "link" };
      }
      case "file":
        return writeSettings(client, pairingId, launch);
      case "command":
        return copy("command", claudeCodeCommand(launch, options.platform), token);
      case "manual":
        return copy("settings", mcpServersSnippet(launch), token);
    }
  };

  return {
    /** Every agent, those on this computer first, in the catalog's order otherwise. */
    clients(): AgentClient[] {
      const where = { home: options.home, applications: options.applications };
      const all = AGENT_CLIENT_DEFINITIONS.map((client) => ({
        id: client.id,
        name: client.name,
        installed: client.id !== "other" && isInstalled(client, where, existsSync),
        setup: client.setup,
      }));
      return [...all.filter((each) => each.installed), ...all.filter((each) => !each.installed)];
    },

    preview(clientId: unknown): AgentSetupPreview | null {
      const client = clientDefinition(clientId);
      if (!client) throw new InvalidInputError("Unknown agent.");
      const path = pathOf(client);
      if (!path) return null;
      // The key is made when the User confirms; what is shown hides it anyway.
      return { path, change: changeShown(client, options.agents.launchFor(HIDDEN_KEY)) };
    },

    async setUp(input: unknown): Promise<SetUpAgent> {
      if (!isRecord(input)) throw new InvalidInputError("setUpAgent expects an object.");
      const client = clientDefinition(input.client);
      if (!client) throw new InvalidInputError("Unknown agent.");
      if (!Array.isArray(input.folderIds) || input.folderIds.length === 0) {
        throw new InvalidInputError("Choose at least one Folder for the agent to read.");
      }
      const { pairing, launch } = await options.agents.connect({
        name: nameFor(client),
        folderIds: input.folderIds,
        client: client.id,
      });
      try {
        return { pairing, result: await install(client, pairing.id, launch) };
      } catch (error) {
        // Nothing half made: the pairing goes with the setup that failed.
        await options.agents.disconnect(pairing.id).catch(options.reportError);
        throw error;
      }
    },

    async undo(pairingId: unknown): Promise<void> {
      if (typeof pairingId !== "string") throw new InvalidInputError("A pairing id must be text.");
      const file = written.get(pairingId);
      if (file) {
        if ((await readText(file.path)) !== file.after) {
          throw new InvalidInputError(t("agents.setup.undoChanged", { path: file.path }));
        }
        if (file.before === null) await unlink(file.path);
        else await writeFile(file.path, file.before);
        written.delete(pairingId);
      }
      await options.agents.disconnect(pairingId);
    },

    async settingsFor(pairingId: unknown) {
      const pairing = options.agents.settings().pairings.find((each) => each.id === pairingId);
      if (!pairing) throw new NotFoundError("That agent isn't connected.");
      const launch = await options.agents.launch(pairing.id);
      const token = launch.env[AGENT_TOKEN_ENV] ?? "";
      if (pairing.client === "claude-code") {
        return copy("command", claudeCodeCommand(launch, options.platform), token);
      }
      return copy("settings", mcpServersSnippet(launch), token);
    },

    /**
     * Before a pairing is disconnected: takes IncarnaMind out of the agent's
     * settings file it was written to, if the entry there is still its own.
     */
    async removeFromSettings(pairingId: unknown): Promise<void> {
      const pairing = options.agents.settings().pairings.find((each) => each.id === pairingId);
      const client = clientDefinition(pairing?.client);
      const path = client && pathOf(client);
      if (!pairing || !client || !path) return;
      written.delete(pairing.id);
      const token = (await options.agents.launch(pairing.id)).env[AGENT_TOKEN_ENV] ?? "";
      const before = await readText(path);
      if (before === null || token === "" || !before.includes(token)) return;
      try {
        await writeFile(path, changed(client, before, null));
      } catch (error) {
        // The User's file is theirs to tidy: the key in it no longer works anyway.
        options.reportError(error);
      }
    },
  };
}

export type AgentSetup = ReturnType<typeof createAgentSetup>;
