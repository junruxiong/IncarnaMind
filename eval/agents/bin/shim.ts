/**
 * The bake-off's stdio shim: what an outside agent starts as IncarnaMind's
 * MCP server. It connects to the app's socket (#351), says which pairing it
 * is with one line, then relays MCP's newline-delimited JSON-RPC both ways.
 *
 * Two things only the bake-off needs, so the server stays as it ships:
 * - INCARNAMIND_BAKEOFF_TOOLS limits the Tools the agent sees (arm Q sees
 *   only `ask`, arm M only the fine-grained Tools); a call to any other is
 *   answered here as an error and never reaches the app.
 * - INCARNAMIND_BAKEOFF_LOG names a file each request's round trip through
 *   the socket is appended to, to report the time per call at the shim.
 * - INCARNAMIND_BAKEOFF_INSTRUCTIONS replaces the server's instructions, so
 *   an arm isn't told to use a Tool it was given none of.
 *
 * Plain Node with no imports outside it, so an agent can start it with
 * `node shim.ts` from any folder.
 */
import { appendFileSync } from "node:fs";
import { createConnection } from "node:net";

const socketPath = process.env.INCARNAMIND_AGENT_SOCKET;
const token = process.env.INCARNAMIND_AGENT_TOKEN;
const allowed = (process.env.INCARNAMIND_BAKEOFF_TOOLS ?? "")
  .split(",")
  .map((each) => each.trim())
  .filter(Boolean);
const logFile = process.env.INCARNAMIND_BAKEOFF_LOG;
const instructions = process.env.INCARNAMIND_BAKEOFF_INSTRUCTIONS;

if (!socketPath || !token) {
  process.stderr.write(
    "IncarnaMind's shim needs INCARNAMIND_AGENT_SOCKET and INCARNAMIND_AGENT_TOKEN.\n",
  );
  process.exit(2);
}

type Id = string | number;
interface Message {
  jsonrpc?: string;
  id?: Id;
  method?: string;
  params?: { name?: string };
  result?: { tools?: { name: string }[]; instructions?: string };
  error?: unknown;
}

const pending = new Map<
  Id,
  { method: string; tool: string | null; started: number; bytes: number }
>();
const toolLists = new Set<Id>();
/** Requests whose result carries the server's instructions, in either protocol era. */
const openings = new Set<Id>();

function log(record: Record<string, unknown>): void {
  if (!logFile) return;
  try {
    appendFileSync(logFile, `${JSON.stringify(record)}\n`);
  } catch {
    // Timing is a report, never a reason to break the agent's connection.
  }
}

const isAllowed = (tool: string | undefined) =>
  allowed.length === 0 || (tool !== undefined && allowed.includes(tool));

const socket = createConnection(socketPath);
socket.setNoDelay(true);
socket.on("error", (error) => {
  process.stderr.write(`IncarnaMind's shim couldn't reach the app: ${error.message}\n`);
  process.exit(1);
});
socket.on("close", () => process.exit(0));
socket.write(`${JSON.stringify({ incarnamind: 1, token })}\n`);

/** Calls `onLine` for each complete line of a stream. */
function lines(stream: NodeJS.ReadableStream, onLine: (line: string) => void): void {
  let buffer = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk: string) => {
    buffer += chunk;
    for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
      const line = buffer.slice(0, end);
      buffer = buffer.slice(end + 1);
      if (line.trim()) onLine(line);
    }
  });
}

function parse(line: string): Message | null {
  try {
    const value: unknown = JSON.parse(line);
    return typeof value === "object" && value !== null ? (value as Message) : null;
  } catch {
    return null;
  }
}

// The agent's requests, to the app.
lines(process.stdin, (line) => {
  const message = parse(line);
  if (
    message?.method === "tools/call" &&
    message.id !== undefined &&
    !isAllowed(message.params?.name)
  ) {
    const refusal = {
      jsonrpc: "2.0",
      id: message.id,
      result: {
        content: [{ type: "text", text: `Unknown tool: ${message.params?.name ?? ""}` }],
        isError: true,
      },
    };
    process.stdout.write(`${JSON.stringify(refusal)}\n`);
    log({
      method: "tools/call",
      tool: message.params?.name ?? null,
      refused: true,
      at: Date.now(),
    });
    return;
  }
  if (message?.id !== undefined && message.method) {
    pending.set(message.id, {
      method: message.method,
      tool: message.method === "tools/call" ? (message.params?.name ?? null) : null,
      started: performance.now(),
      bytes: Buffer.byteLength(line),
    });
    if (message.method === "tools/list") toolLists.add(message.id);
    if (message.method === "initialize" || message.method === "server/discover") {
      openings.add(message.id);
    }
  }
  socket.write(`${line}\n`);
});
process.stdin.on("end", () => socket.end());

// The app's replies, to the agent.
lines(socket, (line) => {
  let out = line;
  const message = parse(line);
  if (message?.id !== undefined && message.method === undefined) {
    const sent = pending.get(message.id);
    if (sent) {
      pending.delete(message.id);
      log({
        method: sent.method,
        tool: sent.tool,
        ms: Number((performance.now() - sent.started).toFixed(3)),
        requestBytes: sent.bytes,
        responseBytes: Buffer.byteLength(line),
        error: message.error !== undefined,
        at: Date.now(),
      });
    }
    if (toolLists.has(message.id) && allowed.length > 0 && message.result?.tools) {
      toolLists.delete(message.id);
      const tools = message.result.tools.filter((tool) => allowed.includes(tool.name));
      out = JSON.stringify({ ...message, result: { ...message.result, tools } });
    }
    if (
      openings.has(message.id) &&
      instructions &&
      typeof message.result?.instructions === "string"
    ) {
      openings.delete(message.id);
      out = JSON.stringify({ ...message, result: { ...message.result, instructions } });
    }
  }
  process.stdout.write(`${out}\n`);
});
