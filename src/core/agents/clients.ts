/**
 * The agents IncarnaMind can be set up in (#351), and how each takes it, so
 * that connecting one is one click wherever the agent allows it:
 *
 * - "bundle": Claude's desktop app installs an `.mcpb` bundle opened with it
 *   (the shim inside, run on the app's own Node);
 * - "link": Cursor and VS Code install a server from a link of their own;
 * - "file": IncarnaMind adds itself to the agent's settings file (Codex's
 *   TOML, or the `mcpServers` JSON most agents keep), shown first, with Undo;
 * - "command": Claude Code takes one line pasted into a terminal;
 * - "manual": the rest take the same settings pasted into their own screen.
 *
 * Whether an agent is on this computer is a guess from its app or the folder
 * it keeps its settings in, used only to list it first. The settings files
 * and folders are the agents' own documented ones.
 */
import { join } from "node:path";
import type { AgentClientId, AgentLaunch, AgentSetupKind } from "../api";

/** The name IncarnaMind has in every agent's settings. */
export const SERVER_KEY = "incarnamind";

export interface ClientDefinition {
  id: AgentClientId;
  /** Its own name, which no language translates; "other" is named by the interface. */
  name: string;
  setup: AgentSetupKind;
  /** Files or folders under the home folder whose presence says it is installed. */
  home?: readonly (readonly string[])[];
  /** Its app, on macOS. */
  apps?: readonly string[];
  /** Where its settings file is, under the home folder, and how it is written. */
  file?: { path: readonly string[]; format: "mcp-servers-json" | "codex-toml" };
}

export const AGENT_CLIENT_DEFINITIONS: readonly ClientDefinition[] = [
  {
    id: "claude-desktop",
    name: "Claude",
    setup: "bundle",
    apps: ["Claude.app"],
    home: [
      ["Library", "Application Support", "Claude"],
      ["AppData", "Roaming", "Claude"],
      [".config", "Claude"],
    ],
  },
  {
    id: "claude-code",
    name: "Claude Code",
    setup: "command",
    home: [[".claude.json"], [".claude"]],
  },
  {
    id: "codex",
    name: "Codex",
    setup: "file",
    apps: ["Codex.app", "ChatGPT.app"],
    home: [[".codex"]],
    file: { path: [".codex", "config.toml"], format: "codex-toml" },
  },
  { id: "cursor", name: "Cursor", setup: "link", apps: ["Cursor.app"], home: [[".cursor"]] },
  {
    id: "vscode",
    name: "VS Code",
    setup: "link",
    apps: ["Visual Studio Code.app"],
    home: [[".vscode"]],
  },
  {
    id: "gemini-cli",
    name: "Gemini CLI",
    setup: "file",
    home: [[".gemini"]],
    file: { path: [".gemini", "settings.json"], format: "mcp-servers-json" },
  },
  {
    id: "kimi-code",
    name: "Kimi Code CLI",
    setup: "file",
    home: [[".kimi-code"]],
    file: { path: [".kimi-code", "mcp.json"], format: "mcp-servers-json" },
  },
  {
    id: "qwen-code",
    name: "Qwen Code",
    setup: "file",
    home: [[".qwen"]],
    file: { path: [".qwen", "settings.json"], format: "mcp-servers-json" },
  },
  {
    id: "codebuddy",
    name: "CodeBuddy",
    setup: "file",
    apps: ["CodeBuddy.app", "CodeBuddy CN.app"],
    home: [[".codebuddy"]],
    file: { path: [".codebuddy", ".mcp.json"], format: "mcp-servers-json" },
  },
  {
    id: "minimax-code",
    name: "MiniMax Code",
    setup: "file",
    home: [[".minimax"]],
    file: { path: [".minimax", "mcp.json"], format: "mcp-servers-json" },
  },
  { id: "trae", name: "Trae", setup: "manual", apps: ["Trae.app", "Trae CN.app"] },
  { id: "qoder", name: "Qoder", setup: "manual", apps: ["Qoder.app"], home: [[".qoder"]] },
  { id: "comate", name: "Comate", setup: "manual", home: [[".comate"]] },
  { id: "cherry-studio", name: "Cherry Studio", setup: "manual", apps: ["Cherry Studio.app"] },
  { id: "chatbox", name: "Chatbox", setup: "manual", apps: ["Chatbox.app"] },
  { id: "deepchat", name: "DeepChat", setup: "manual", apps: ["DeepChat.app"] },
  {
    id: "goose",
    name: "Goose",
    setup: "manual",
    apps: ["Goose.app"],
    home: [[".config", "goose"]],
  },
  { id: "opencode", name: "OpenCode", setup: "manual", home: [[".config", "opencode"]] },
  { id: "cline", name: "Cline", setup: "manual", home: [[".cline"]] },
  { id: "kilo", name: "Kilo Code", setup: "manual", home: [[".config", "kilo"]] },
  { id: "openclaw", name: "OpenClaw", setup: "manual", home: [[".openclaw"]] },
  { id: "pi", name: "pi", setup: "manual", home: [[".pi"]] },
  { id: "other", name: "", setup: "manual" },
];

export function clientDefinition(id: unknown): ClientDefinition | null {
  return AGENT_CLIENT_DEFINITIONS.find((each) => each.id === id) ?? null;
}

/** Whether the agent looks installed: its app, or the folder it keeps its settings in. */
export function isInstalled(
  client: ClientDefinition,
  where: { home: string; applications: readonly string[] },
  exists: (path: string) => boolean,
): boolean {
  if (client.home?.some((segments) => exists(join(where.home, ...segments)))) return true;
  return (
    client.apps?.some((app) => where.applications.some((dir) => exists(join(dir, app)))) ?? false
  );
}

/** The server's entry, as most agents' `mcpServers` settings take it. */
export const serverEntry = (launch: AgentLaunch) => ({
  command: launch.command,
  args: launch.args,
  env: launch.env,
});

/** Settings to paste: `{ "mcpServers": { "incarnamind": … } }`. */
export const mcpServersSnippet = (launch: AgentLaunch) =>
  `${JSON.stringify({ mcpServers: { [SERVER_KEY]: serverEntry(launch) } }, null, 2)}\n`;

/** A word quoted for the User's shell: Windows' (PowerShell) or a POSIX one. */
function quote(word: string, platform: NodeJS.Platform): string {
  if (platform === "win32") return `"${word.replaceAll('"', '`"')}"`;
  return /^[\w@%+=:,./-]+$/.test(word) ? word : `'${word.replaceAll("'", "'\\''")}'`;
}

/** The line that adds IncarnaMind to Claude Code, for every project. */
export function claudeCodeCommand(launch: AgentLaunch, platform: NodeJS.Platform): string {
  const env = Object.entries(launch.env).flatMap(([name, value]) => [
    "--env",
    quote(`${name}=${value}`, platform),
  ]);
  return [
    "claude mcp add --scope user",
    ...env,
    SERVER_KEY,
    "--",
    quote(launch.command, platform),
    ...launch.args.map((arg) => quote(arg, platform)),
  ].join(" ");
}

/** Cursor's install link: it shows the server and asks the User to install it. */
export function cursorLink(launch: AgentLaunch): string {
  const config = Buffer.from(JSON.stringify(serverEntry(launch))).toString("base64");
  return `cursor://anysphere.cursor-deeplink/mcp/install?name=${SERVER_KEY}&config=${encodeURIComponent(config)}`;
}

/** VS Code's install link: it shows the server and asks the User to install it. */
export function vscodeLink(launch: AgentLaunch): string {
  const config = { name: SERVER_KEY, type: "stdio", ...serverEntry(launch) };
  return `vscode:mcp/install?${encodeURIComponent(JSON.stringify(config))}`;
}

/** A TOML basic string: JSON's escapes are TOML's. */
const tomlString = (value: string) => JSON.stringify(value);

/** Codex's table for IncarnaMind. `ask` writes a whole Answer, so its Tools get 5 minutes. */
export function codexTable(launch: AgentLaunch): string {
  const env = Object.entries(launch.env)
    .map(([name, value]) => `${name} = ${tomlString(value)}`)
    .join(", ");
  return [
    `[mcp_servers.${SERVER_KEY}]`,
    `command = ${tomlString(launch.command)}`,
    `args = [${launch.args.map(tomlString).join(", ")}]`,
    `env = { ${env} }`,
    "tool_timeout_sec = 300",
    "",
  ].join("\n");
}

/** Codex's settings with IncarnaMind's table in place of any it had, or added at the end. */
export function withCodexTable(before: string | null, table: string): string {
  const text = before ?? "";
  const without = withoutCodexTable(text);
  const base = without.trimEnd();
  return base === "" ? table : `${base}\n\n${table}`;
}

/** Codex's settings without IncarnaMind's table; unchanged if it has none. */
export function withoutCodexTable(text: string): string {
  const lines = text.split("\n");
  const header = `[mcp_servers.${SERVER_KEY}]`;
  const start = lines.findIndex((line) => line.trim() === header);
  if (start === -1) return text;
  let end = lines.findIndex((line, index) => index > start && /^\s*\[/.test(line));
  if (end === -1) end = lines.length;
  return [...lines.slice(0, start), ...lines.slice(end)].join("\n").replace(/\n{3,}/g, "\n\n");
}

/**
 * An `mcpServers` settings file with IncarnaMind's entry set (`entry`) or
 * taken out (null). Throws when the file isn't a JSON object, so nothing of
 * the User's is lost: they are given the settings to paste instead.
 */
export function withMcpServersEntry(before: string | null, entry: object | null): string {
  let settings: Record<string, unknown> = {};
  if (before !== null && before.trim() !== "") {
    const parsed: unknown = JSON.parse(before);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("The settings file isn't a JSON object.");
    }
    settings = parsed as Record<string, unknown>;
  }
  const servers = settings.mcpServers;
  if (servers !== undefined && (typeof servers !== "object" || servers === null)) {
    throw new Error("The settings file's mcpServers isn't an object.");
  }
  const next: Record<string, unknown> = { ...(servers as Record<string, unknown> | undefined) };
  if (entry) next[SERVER_KEY] = entry;
  else delete next[SERVER_KEY];
  return `${JSON.stringify({ ...settings, mcpServers: next }, null, 2)}\n`;
}

/** How a pairing's key shows wherever settings are shown rather than copied. */
export const HIDDEN_KEY = "•••••••";

/** A settings text with the pairing's key hidden. */
export const hideToken = (text: string, token: string) =>
  token === "" ? text : text.replaceAll(token, HIDDEN_KEY);
