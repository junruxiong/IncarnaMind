/**
 * Connectors (CONTEXT.md): external services the User has connected, through
 * which Answers can look things up. This ticket brings local ones: MCP
 * servers that run as programs on this computer, spoken to over their
 * standard input and output with the official MCP SDK.
 *
 * - Storage: one row per Connector, sync-ready and without secrets. Its
 *   environment's values (usually API keys) are one keychain secret.
 * - Lifecycle: enabled Connectors start when the core starts, or on first use
 *   if they haven't yet. A process that stops is started again, waiting
 *   longer after each failure. Turning one off, deleting it, or closing the
 *   core stops it.
 * - Answers: the Tools of each ready Connector that it marks read-only,
 *   namespaced by Connector. Tools that may change something wait for
 *   approvals (#38). Every call checks consent for the "connectors" flow
 *   first: even a server on this computer can reach the internet.
 */
import { randomUUID } from "node:crypto";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import type { Tool } from "@modelcontextprotocol/sdk/types.js";
import type { ProcessLauncher } from "../adapters";
import type { ExternalTool } from "../answers/engine";
import type {
  Connector,
  ConnectorError,
  ConnectorImportEntry,
  ConnectorImportResult,
  ConnectorState,
  ConnectorTool,
  ExternalService,
} from "../api";
import type { Consent } from "../consent";
import {
  ConsentDeclinedError,
  InvalidInputError,
  isRecord,
  NotFoundError,
  SecretStorageError,
} from "../errors";
import type { Secrets } from "../secrets";
import type { Database } from "../storage";
import { type LocalConnectorConfig, parseAddInput, readMcpServers } from "./config";
import { startError, stoppedError } from "./errors";
import { ChildProcessTransport } from "./transport";

/** What the "connectors" flow sends to each Connector. */
export const CONNECTORS_FLOW_SENDS = ["tool-arguments"] as const;

export interface ConnectorTiming {
  /** How long a Connector may take to start and answer MCP's handshake: `npx -y` may download its package first. */
  connectTimeoutMs: number;
  /** How long one Tool call may take, without progress. */
  callTimeoutMs: number;
  /** How long an Answer waits for Connectors that are still connecting before going on without them. */
  readyWaitMs: number;
  /** The waits before each restart after a failure; the last one repeats. */
  restartDelaysMs: readonly number[];
  /** Failures in a row before IncarnaMind stops restarting a Connector by itself. */
  maxRestarts: number;
  /** A Connector that stayed up this long has its failure count reset. */
  stableMs: number;
}

export const CONNECTOR_TIMING: ConnectorTiming = {
  connectTimeoutMs: 60_000,
  callTimeoutMs: 120_000,
  readyWaitMs: 10_000,
  restartDelaysMs: [1_000, 2_000, 4_000, 8_000, 16_000],
  maxRestarts: 5,
  stableMs: 60_000,
};

/** The most of a Tool's result an Answer passes to the model. */
const MAX_RESULT_CHARS = 20_000;
/** The most of a Tool's description the model is given. */
const MAX_DESCRIPTION_CHARS = 1_000;
/** What providers accept as a Tool's name. */
const MAX_TOOL_NAME_LENGTH = 64;
const CLIENT_INFO = { name: "IncarnaMind", version: "1.0.0" };

interface ConnectorRow {
  id: string;
  name: string;
  transport: string;
  config: string;
  enabled: number;
  created_at: string;
  updated_at: string;
}

/** The `config` column for "stdio": no secrets, only the names of the environment variables. */
interface StoredConfig {
  command: string;
  args: string[];
  env: string[];
}

interface ToolInfo extends ConnectorTool {
  inputSchema: Record<string, unknown>;
}

/** A Connector's process and connection, kept in memory. */
interface Runtime {
  /** Counts starts and stops: work from an earlier one sees it is stale and gives up. */
  generation: number;
  state: Exclude<ConnectorState, "off">;
  error: ConnectorError | null;
  tools: ToolInfo[] | null;
  transport: ChildProcessTransport | null;
  /** Set while "ready". */
  client: Client | null;
  connectedAt: number;
  /** Settles when the start under way has connected or failed. */
  starting: Promise<void> | null;
  /** Failures in a row, for the wait before the next restart. */
  failures: number;
  retryTimer: ReturnType<typeof setTimeout> | null;
}

class MissingSecretsError extends Error {
  override name = "MissingSecretsError";
}

export interface ConnectorsOptions {
  db: Database;
  now: () => string;
  secrets: Secrets;
  processes: ProcessLauncher;
  consent: Consent;
  /** Something changed: the list, as `list()` returns it. */
  onChange(connectors: Connector[]): void;
  /** A Connector was added, deleted, or turned on or off. */
  onEnabledChange(): void;
  reportError(error: unknown): void;
  timing?: Partial<ConnectorTiming>;
}

export type Connectors = ReturnType<typeof createConnectors>;

const COLUMNS = "id, name, transport, config, enabled, created_at, updated_at";
const envSecret = (connectorId: string) => `connector:${connectorId}:env`;
const serviceOf = (row: { id: string; name: string }): ExternalService => ({
  id: `connector:${row.id}`,
  name: row.name,
});
const sameName = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();

function storedConfig(row: ConnectorRow): StoredConfig {
  try {
    const parsed: unknown = JSON.parse(row.config);
    if (isRecord(parsed) && typeof parsed.command === "string") {
      const texts = (value: unknown) =>
        Array.isArray(value)
          ? value.filter((item): item is string => typeof item === "string")
          : [];
      return { command: parsed.command, args: texts(parsed.args), env: texts(parsed.env) };
    }
  } catch {
    // Unreadable: treated as a Connector with no command, which fails to start.
  }
  return { command: "", args: [], env: [] };
}

function toToolInfo(tool: Tool): ToolInfo {
  return {
    name: tool.name,
    title: tool.title ?? tool.annotations?.title ?? null,
    description: tool.description ?? "",
    readOnly: tool.annotations?.readOnlyHint === true,
    inputSchema: tool.inputSchema,
  };
}

/** Lowercase letters, digits and "_", for a Tool name the model calls: "Linear (work)" → "linear_work". */
function slug(text: string, fallback: string): string {
  const cleaned = text
    .normalize("NFKD")
    .replace(/[^A-Za-z0-9_-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "")
    .toLowerCase();
  return cleaned || fallback;
}

/** What a Tool call returned, as text for the model. */
function resultText(result: Record<string, unknown>): string {
  const parts: string[] = [];
  const content = Array.isArray(result.content) ? result.content : [];
  for (const item of content) {
    if (!isRecord(item)) continue;
    switch (item.type) {
      case "text":
        if (typeof item.text === "string") parts.push(item.text);
        break;
      case "resource": {
        const resource = isRecord(item.resource) ? item.resource : {};
        if (typeof resource.text === "string") parts.push(resource.text);
        else parts.push(`[A file, ${String(resource.uri ?? "")}, which IncarnaMind can't read.]`);
        break;
      }
      case "resource_link":
        parts.push(`[A link: ${String(item.name ?? "")} ${String(item.uri ?? "")}]`);
        break;
      case "image":
      case "audio":
        parts.push(
          `[An ${item.type === "image" ? "image" : "audio clip"}, which IncarnaMind can't pass on.]`,
        );
        break;
    }
  }
  if (parts.length === 0 && result.structuredContent !== undefined) {
    parts.push(JSON.stringify(result.structuredContent));
  }
  // Older servers give `toolResult` instead of content.
  if (parts.length === 0 && result.toolResult !== undefined) {
    parts.push(JSON.stringify(result.toolResult));
  }
  const text = parts.join("\n\n").trim();
  if (!text) return "(The Tool returned nothing.)";
  return text.length > MAX_RESULT_CHARS
    ? `${text.slice(0, MAX_RESULT_CHARS)}\n[The rest was cut: the result was too long.]`
    : text;
}

/** Waits for `promise`, unless `signal` aborts first. */
function untilAborted<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
}

export function createConnectors(options: ConnectorsOptions) {
  const { db, now, secrets, processes, consent } = options;
  const timing: ConnectorTiming = { ...CONNECTOR_TIMING, ...options.timing };
  const runtimes = new Map<string, Runtime>();
  let closed = false;

  const liveRows = () =>
    db.all<ConnectorRow>(
      `SELECT ${COLUMNS} FROM connectors WHERE deleted_at IS NULL ORDER BY name COLLATE NOCASE, created_at`,
    );
  const rowById = (id: string) =>
    db.get<ConnectorRow>(`SELECT ${COLUMNS} FROM connectors WHERE id = ? AND deleted_at IS NULL`, [
      id,
    ]);
  const requireRow = (id: unknown): ConnectorRow => {
    if (typeof id !== "string" || id === "") {
      throw new InvalidInputError("A Connector id must be a non-empty string.");
    }
    const row = rowById(id);
    if (!row) throw new NotFoundError("That Connector doesn't exist.");
    return row;
  };

  const toConnector = (row: ConnectorRow): Connector => {
    const config = storedConfig(row);
    const enabled = row.enabled === 1;
    const runtime = runtimes.get(row.id);
    return {
      id: row.id,
      name: row.name,
      transport: "stdio",
      command: config.command,
      args: config.args,
      env: config.env,
      enabled,
      state: enabled ? (runtime?.state ?? "connecting") : "off",
      error: enabled ? (runtime?.error ?? null) : null,
      tools:
        enabled && runtime?.tools
          ? runtime.tools.map(({ name, title, description, readOnly }) => ({
              name,
              title,
              description,
              readOnly,
            }))
          : null,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  };

  const list = () => liveRows().map(toConnector);
  const changed = () => {
    if (!closed) options.onChange(list());
  };

  consent.registry.register({
    id: "connectors",
    sends: CONNECTORS_FLOW_SENDS,
    async services() {
      return liveRows()
        .filter((row) => row.enabled === 1)
        .map(serviceOf);
    },
  });

  const runtimeOf = (id: string): Runtime => {
    let runtime = runtimes.get(id);
    if (!runtime) {
      runtime = {
        generation: 0,
        state: "connecting",
        error: null,
        tools: null,
        transport: null,
        client: null,
        connectedAt: 0,
        starting: null,
        failures: 0,
        retryTimer: null,
      };
      runtimes.set(id, runtime);
    }
    return runtime;
  };

  const clearRetry = (runtime: Runtime) => {
    if (runtime.retryTimer) clearTimeout(runtime.retryTimer);
    runtime.retryTimer = null;
  };

  /** Ends the current process, gracefully unless `now`. Its callbacks see a new generation and stand down. */
  const stopProcess = (runtime: Runtime, { now = false } = {}) => {
    runtime.generation++;
    clearRetry(runtime);
    runtime.starting = null;
    const transport = runtime.transport;
    runtime.transport = null;
    runtime.client = null;
    if (!transport) return;
    if (now) transport.kill();
    else transport.close().catch(options.reportError);
  };

  const readEnv = async (id: string, names: readonly string[]) => {
    if (names.length === 0) return {};
    const raw = await secrets.tryGet(envSecret(id));
    let env: unknown;
    try {
      env = raw === null ? null : JSON.parse(raw);
    } catch {
      env = null;
    }
    if (!isRecord(env) || names.some((name) => typeof env[name] !== "string")) {
      throw new MissingSecretsError("The Connector's environment can't be read on this device.");
    }
    return env as Record<string, string>;
  };

  /** Records a failure; failures the process may get over are retried, waiting longer each time. */
  const fail = (id: string, runtime: Runtime, error: ConnectorError) => {
    runtime.failures++;
    const retryable = error.kind !== "missing-command" && error.kind !== "missing-secrets";
    const retrying = retryable && runtime.failures <= timing.maxRestarts;
    runtime.state = "error";
    runtime.error = { ...error, retrying };
    runtime.tools = null;
    if (retrying) {
      const delays = timing.restartDelaysMs;
      const wait = delays[Math.min(runtime.failures - 1, delays.length - 1)] ?? 1_000;
      runtime.retryTimer = setTimeout(() => {
        runtime.retryTimer = null;
        if (!closed && rowById(id)?.enabled === 1) start(id);
      }, wait);
      runtime.retryTimer.unref?.();
    }
    changed();
  };

  /** The process of a ready Connector ended without being asked to. */
  const lost = (id: string, runtime: Runtime, generation: number, command: string) => {
    if (closed || runtime.generation !== generation) return;
    const transport = runtime.transport;
    runtime.transport = null;
    runtime.client = null;
    if (Date.now() - runtime.connectedAt >= timing.stableMs) runtime.failures = 0;
    fail(
      id,
      runtime,
      stoppedError(
        command,
        transport?.exit ?? null,
        transport?.stderrTail ?? "",
        transport?.error ?? null,
      ),
    );
  };

  async function connect(id: string, runtime: Runtime, generation: number): Promise<void> {
    const current = () => !closed && runtime.generation === generation;
    const row = rowById(id);
    if (row?.enabled !== 1) return;
    const { command, args, env: envNames } = storedConfig(row);
    let transport: ChildProcessTransport | null = null;
    try {
      const env = await readEnv(id, envNames);
      if (!current()) return;
      const child = await processes.spawn(command, args, { env });
      transport = new ChildProcessTransport(child);
      if (!current()) {
        transport.kill();
        return;
      }
      runtime.transport = transport;
      let ready = false;
      const client = new Client(CLIENT_INFO, {
        capabilities: {},
        listChanged: {
          tools: {
            onChanged: (error, tools) => {
              if (error || !tools || !ready || !current()) return;
              runtime.tools = tools.map(toToolInfo);
              changed();
            },
          },
        },
      });
      // Lines that aren't MCP, e.g. a server logging to its standard output, are skipped.
      client.onerror = () => undefined;
      client.onclose = () => {
        if (ready) lost(id, runtime, generation, command);
      };
      await client.connect(transport, { timeout: timing.connectTimeoutMs });
      const tools: ToolInfo[] = [];
      if (client.getServerCapabilities()?.tools) {
        let cursor: string | undefined;
        do {
          const page = await client.listTools(cursor ? { cursor } : undefined, {
            timeout: timing.connectTimeoutMs,
          });
          tools.push(...page.tools.map(toToolInfo));
          cursor = page.nextCursor;
        } while (cursor && tools.length < 1_000);
      }
      if (!current()) {
        transport.close().catch(options.reportError);
        return;
      }
      ready = true;
      runtime.client = client;
      runtime.connectedAt = Date.now();
      runtime.state = "ready";
      runtime.error = null;
      runtime.tools = tools;
      changed();
    } catch (error) {
      let ended: Parameters<typeof startError>[2] = null;
      if (transport) {
        // A moment for the process to say how it ended, e.g. its exit code.
        await Promise.race([
          transport.ended,
          new Promise((resolve) => setTimeout(resolve, 250).unref()),
        ]);
        ended = {
          exit: transport.exit,
          stderr: transport.stderrTail,
          processError: transport.error,
        };
        transport.kill();
      }
      if (!current()) return;
      runtime.transport = null;
      runtime.client = null;
      fail(
        id,
        runtime,
        error instanceof MissingSecretsError
          ? {
              kind: "missing-secrets",
              command: null,
              install: null,
              message: error.message,
              retrying: false,
            }
          : startError(command, error, ended),
      );
    }
  }

  /** Starts a Connector, stopping its process first if one runs. */
  function start(id: string): void {
    const runtime = runtimeOf(id);
    stopProcess(runtime);
    const generation = runtime.generation;
    runtime.state = "connecting";
    runtime.error = null;
    runtime.tools = null;
    changed();
    runtime.starting = connect(id, runtime, generation)
      .catch(options.reportError)
      .finally(() => {
        if (runtime.generation === generation) runtime.starting = null;
      });
  }

  /** Turned off or deleted: the process stops, and the failure count is forgotten. */
  const stop = (id: string) => {
    const runtime = runtimes.get(id);
    if (!runtime) return;
    stopProcess(runtime);
    runtime.state = "connecting";
    runtime.error = null;
    runtime.tools = null;
    runtime.failures = 0;
  };

  const assertNameFree = (name: string) => {
    if (liveRows().some((row) => sameName(row.name, name))) {
      throw new InvalidInputError(`A Connector named "${name}" already exists.`);
    }
  };

  /** Stores a new Connector, its environment in the keychain first: if that is refused, nothing is added. */
  const insert = async (config: LocalConnectorConfig): Promise<string> => {
    assertNameFree(config.name);
    const id = randomUUID();
    const envNames = Object.keys(config.env);
    if (envNames.length > 0) await secrets.set(envSecret(id), JSON.stringify(config.env));
    const stored: StoredConfig = { command: config.command, args: config.args, env: envNames };
    const at = now();
    try {
      assertNameFree(config.name);
      db.run(
        `INSERT INTO connectors (id, name, transport, config, enabled, created_at, updated_at)
         VALUES (?, ?, 'stdio', ?, 1, ?, ?)`,
        [id, config.name, JSON.stringify(stored), at, at],
      );
    } catch (error) {
      if (envNames.length > 0) await secrets.delete(envSecret(id)).catch(options.reportError);
      throw error;
    }
    return id;
  };

  const previewImport = (json: unknown) => {
    const candidates = readMcpServers(json);
    const taken = new Set(liveRows().map((row) => row.name.toLowerCase()));
    for (const { entry } of candidates) {
      if (entry.action !== "add") continue;
      const key = entry.name.toLowerCase();
      if (taken.has(key)) entry.action = "exists";
      else taken.add(key);
    }
    return candidates;
  };

  /** Each enabled Connector that hasn't been started (or whose start is long over) starts. */
  const startIdle = (rows: readonly ConnectorRow[]) => {
    for (const row of rows) {
      const runtime = runtimes.get(row.id);
      const idle =
        !runtime || (runtime.state === "connecting" && !runtime.starting && !runtime.retryTimer);
      if (idle) start(row.id);
    }
  };

  /** A Connector's Tool, as an Answer offers it to the model. */
  const answerTool = (row: ConnectorRow, tool: ToolInfo, name: string): ExternalTool => {
    const service = serviceOf(row);
    const description = tool.description || tool.title || tool.name;
    return {
      name,
      description: `From the User's Connector "${row.name}". ${description}`.slice(
        0,
        MAX_DESCRIPTION_CHARS,
      ),
      inputSchema: tool.inputSchema,
      source: { connectorId: row.id, connectorName: row.name, tool: tool.name },
      async call(input, signal) {
        try {
          await untilAborted(consent.ensure("connectors", service), signal);
        } catch (error) {
          if (error instanceof ConsentDeclinedError) {
            throw new Error(
              `The User didn't allow sending data to the Connector "${row.name}", so ${tool.name} wasn't called. Go on without it.`,
            );
          }
          throw error;
        }
        const client = runtimes.get(row.id)?.client;
        if (!client) throw new Error(`The Connector "${row.name}" isn't connected right now.`);
        const result = await client.callTool({ name: tool.name, arguments: input }, undefined, {
          signal,
          timeout: timing.callTimeoutMs,
          resetTimeoutOnProgress: true,
        });
        const text = resultText(result);
        if (result.isError === true) throw new Error(text);
        return text;
      },
    };
  };

  return {
    list,

    async add(input: unknown): Promise<Connector> {
      const id = await insert(parseAddInput(input));
      start(id);
      options.onEnabledChange();
      return toConnector(requireRow(id));
    },

    setEnabled(id: unknown, enabled: unknown): Connector {
      const row = requireRow(id);
      if (typeof enabled !== "boolean")
        throw new InvalidInputError("enabled must be true or false.");
      if ((row.enabled === 1) !== enabled) {
        db.run("UPDATE connectors SET enabled = ?, updated_at = ? WHERE id = ?", [
          enabled ? 1 : 0,
          now(),
          row.id,
        ]);
      }
      if (enabled) {
        const runtime = runtimes.get(row.id);
        if (!runtime?.client && !runtime?.starting) start(row.id);
      } else {
        stop(row.id);
      }
      changed();
      options.onEnabledChange();
      return toConnector(requireRow(row.id));
    },

    restart(id: unknown): Connector {
      const row = requireRow(id);
      if (row.enabled !== 1) throw new InvalidInputError("Turn the Connector on first.");
      runtimeOf(row.id).failures = 0;
      start(row.id);
      return toConnector(requireRow(row.id));
    },

    async delete(id: unknown): Promise<void> {
      const row = requireRow(id);
      stop(row.id);
      runtimes.delete(row.id);
      const at = now();
      db.run("UPDATE connectors SET deleted_at = ?, updated_at = ? WHERE id = ?", [at, at, row.id]);
      // Added again later, it is a new Connector: it asks again.
      consent.revoke("connectors", serviceOf(row).id);
      await secrets.delete(envSecret(row.id)).catch(options.reportError);
      changed();
      options.onEnabledChange();
    },

    previewImport: (json: unknown): ConnectorImportEntry[] =>
      previewImport(json).map((candidate) => candidate.entry),

    async import(json: unknown): Promise<ConnectorImportResult> {
      const candidates = previewImport(json);
      const adding = candidates.filter((each) => each.entry.action === "add");
      // Every environment goes to the keychain: check it can take them before adding any.
      if (adding.some((each) => Object.keys(each.config?.env ?? {}).length > 0)) {
        const { canSave, protection } = secrets.status();
        if (!canSave) throw new SecretStorageError(protection);
      }
      const ids: string[] = [];
      for (const { config } of adding) if (config) ids.push(await insert(config));
      for (const id of ids) start(id);
      if (ids.length > 0) options.onEnabledChange();
      return {
        added: ids.map((id) => toConnector(requireRow(id))),
        skipped: candidates.filter((each) => each.entry.action !== "add").map((each) => each.entry),
      };
    },

    /** Whether any Connector is on: Answers may then send Tool results to the chat model. */
    anyEnabled: () => liveRows().some((row) => row.enabled === 1),

    /** Starts every enabled Connector, e.g. when the app starts. */
    startAll(): void {
      startIdle(liveRows().filter((row) => row.enabled === 1));
    },

    /**
     * The read-only Tools of every enabled Connector that is ready, for one
     * Answer. Connectors not yet started start now; those still connecting
     * are waited for, a little.
     */
    async toolsForAnswer(signal: AbortSignal): Promise<ExternalTool[]> {
      const rows = liveRows().filter((row) => row.enabled === 1);
      if (rows.length === 0) return [];
      startIdle(rows);
      const starting = rows.flatMap((row) => runtimes.get(row.id)?.starting ?? []);
      if (starting.length > 0) {
        let timer: ReturnType<typeof setTimeout> | undefined;
        await untilAborted(
          Promise.race([
            Promise.allSettled(starting),
            new Promise((resolve) => {
              timer = setTimeout(resolve, timing.readyWaitMs);
            }),
          ]),
          signal,
        ).finally(() => clearTimeout(timer));
      }
      const tools: ExternalTool[] = [];
      const names = new Set<string>();
      const unique = (name: string) => {
        let candidate = name.slice(0, MAX_TOOL_NAME_LENGTH);
        for (let n = 2; names.has(candidate); n++) {
          const suffix = `_${n}`;
          candidate = `${name.slice(0, MAX_TOOL_NAME_LENGTH - suffix.length)}${suffix}`;
        }
        names.add(candidate);
        return candidate;
      };
      const prefixes = new Set<string>();
      for (const row of rows) {
        const runtime = runtimes.get(row.id);
        if (runtime?.state !== "ready" || !runtime.tools) continue;
        // Two Connectors whose names give the same prefix: the second gets a number.
        let prefix = slug(row.name, "connector").slice(0, 24);
        for (let n = 2; prefixes.has(prefix); n++)
          prefix = `${slug(row.name, "connector").slice(0, 21)}_${n}`;
        prefixes.add(prefix);
        for (const tool of runtime.tools) {
          if (!tool.readOnly) continue;
          tools.push(answerTool(row, tool, unique(`${prefix}__${slug(tool.name, "tool")}`)));
        }
      }
      return tools;
    },

    /** Stops every Connector's process at once, e.g. when the app quits. */
    close(): void {
      if (closed) return;
      closed = true;
      for (const runtime of runtimes.values()) stopProcess(runtime, { now: true });
      runtimes.clear();
    },
  };
}
