/**
 * A local Connector's configuration: checking what the User typed, and
 * reading the `mcpServers` JSON that Claude Desktop and Cursor keep theirs in:
 *
 *   { "mcpServers": { "github": { "command": "npx", "args": ["-y", "…"], "env": { "GITHUB_TOKEN": "…" } } } }
 */
import type { ConnectorImportEntry } from "../api";
import { InvalidInputError, isRecord } from "../errors";

const MAX_NAME_LENGTH = 100;
const MAX_COMMAND_LENGTH = 4096;
const MAX_ARGS = 200;

/** What starts a local Connector. `env` holds secrets: it goes to the keychain, never the database. */
export interface LocalConnectorConfig {
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
}

export function parseName(value: unknown): string {
  if (typeof value !== "string") throw new InvalidInputError("A Connector's name must be text.");
  const name = value.trim();
  if (!name) throw new InvalidInputError("A Connector's name can't be empty.");
  if (name.length > MAX_NAME_LENGTH) {
    throw new InvalidInputError(
      `A Connector's name can't be longer than ${MAX_NAME_LENGTH} characters.`,
    );
  }
  return name;
}

function parseCommand(value: unknown): string {
  if (typeof value !== "string") throw new InvalidInputError("A Connector's command must be text.");
  const command = value.trim();
  if (!command) throw new InvalidInputError("Enter the command that starts the Connector.");
  if (command.length > MAX_COMMAND_LENGTH || command.includes("\0")) {
    throw new InvalidInputError("That command isn't valid.");
  }
  return command;
}

function parseArgs(value: unknown): string[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || value.length > MAX_ARGS) {
    throw new InvalidInputError("A Connector's arguments must be a list of text.");
  }
  return value.map((arg) => {
    // Configurations sometimes give numbers bare, e.g. a port.
    if (typeof arg === "number" || typeof arg === "boolean") return String(arg);
    if (typeof arg !== "string" || arg.includes("\0")) {
      throw new InvalidInputError("A Connector's arguments must be a list of text.");
    }
    return arg;
  });
}

/** A name a variable can have in every shell: no "=", spaces or control characters. */
const ENV_NAME = /^[^=\s\p{Cc}]+$/u;

function parseEnv(value: unknown): Record<string, string> {
  if (value === undefined || value === null) return {};
  if (!isRecord(value)) {
    throw new InvalidInputError("A Connector's environment must map names to values.");
  }
  const env: Record<string, string> = {};
  for (const [rawName, raw] of Object.entries(value)) {
    const name = rawName.trim();
    if (!ENV_NAME.test(name)) {
      throw new InvalidInputError(`"${rawName}" isn't a valid environment variable name.`);
    }
    if (typeof raw !== "string" && typeof raw !== "number" && typeof raw !== "boolean") {
      throw new InvalidInputError(`The value of ${name} must be text.`);
    }
    const text = String(raw);
    if (text.includes("\0")) throw new InvalidInputError(`The value of ${name} isn't valid.`);
    env[name] = text;
  }
  return env;
}

export function parseAddInput(input: unknown): LocalConnectorConfig {
  if (!isRecord(input)) throw new InvalidInputError("addConnector expects an object.");
  return {
    name: parseName(input.name),
    command: parseCommand(input.command),
    args: parseArgs(input.args),
    env: parseEnv(input.env),
  };
}

/** A server of an `mcpServers` configuration, and its configuration when it can be added. */
export interface ImportCandidate {
  entry: ConnectorImportEntry;
  config: LocalConnectorConfig | null;
}

/** Transports of remote servers, as Cursor and VS Code name them. */
const REMOTE_TYPES = new Set(["sse", "http", "streamable-http", "streamablehttp"]);

const looksLikeServer = (value: unknown) =>
  isRecord(value) && (typeof value.command === "string" || typeof value.url === "string");

/** The object that maps server names to servers, wherever the configuration keeps it. */
function serversOf(parsed: unknown): Record<string, unknown> | null {
  if (!isRecord(parsed)) return null;
  if (isRecord(parsed.mcpServers)) return parsed.mcpServers;
  // VS Code's mcp.json.
  if (isRecord(parsed.servers)) return parsed.servers;
  // Just the servers, pasted without the "mcpServers" around them.
  const values = Object.values(parsed);
  return values.length > 0 && values.every(looksLikeServer) ? parsed : null;
}

function candidate(key: string, value: unknown): ImportCandidate {
  const name = key.trim();
  const base = { name, command: null, args: [], env: [] };
  if (!name || name.length > MAX_NAME_LENGTH || !isRecord(value)) {
    return { entry: { ...base, action: "invalid" }, config: null };
  }
  const type = typeof value.type === "string" ? value.type.toLowerCase() : "";
  if (typeof value.url === "string" || REMOTE_TYPES.has(type)) {
    return { entry: { ...base, action: "remote" }, config: null };
  }
  try {
    const config = {
      name,
      command: parseCommand(value.command),
      args: parseArgs(value.args),
      env: parseEnv(value.env),
    };
    const entry: ConnectorImportEntry = {
      name,
      command: config.command,
      args: config.args,
      env: Object.keys(config.env),
      action: "add",
    };
    return { entry, config };
  } catch {
    const command = typeof value.command === "string" ? value.command : null;
    return { entry: { ...base, command, action: "invalid" }, config: null };
  }
}

/**
 * Every server in an `mcpServers` configuration, in its order, marked "add"
 * unless it is remote or invalid. Whether a name is taken is for the caller.
 */
export function readMcpServers(json: unknown): ImportCandidate[] {
  if (typeof json !== "string") throw new InvalidInputError("The configuration must be text.");
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new InvalidInputError(`That isn't valid JSON: ${reason}`);
  }
  const servers = serversOf(parsed);
  if (!servers || Object.keys(servers).length === 0) {
    throw new InvalidInputError(
      'No MCP servers found: expected an "mcpServers" object, as Claude Desktop and Cursor write it.',
    );
  }
  return Object.entries(servers).map(([key, value]) => candidate(key, value));
}
