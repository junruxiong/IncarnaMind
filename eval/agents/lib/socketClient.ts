/**
 * Calls IncarnaMind's MCP Tools from the harness itself, over the same socket
 * and handshake as the shim (#351): arm H's host re-checking the agent's text
 * inside IncarnaMind, and the retrieval probe. One connection per session of
 * calls; the protocol before 2026-07-28 (`initialize`), as most agents speak.
 */
import { createConnection, type Socket } from "node:net";
import { AGENT_SOCKET_ENV, AGENT_TOKEN_ENV, handshakeLine } from "../../../src/shared/agentSocket";

export interface ToolResult {
  /** The Tool's typed result, or its text parsed as JSON. */
  data: unknown;
  text: string;
  isError: boolean;
  ms: number;
}

interface Reply {
  id?: number;
  result?: {
    structuredContent?: unknown;
    content?: { type?: string; text?: string }[];
    isError?: boolean;
  };
  error?: { message?: string };
}

export interface SocketClient {
  call(tool: string, input: Record<string, unknown>): Promise<ToolResult>;
  close(): void;
}

/** Opens a session as the pairing whose launch environment is `env`. */
export async function openSocketClient(env: Record<string, string>): Promise<SocketClient> {
  const path = env[AGENT_SOCKET_ENV];
  const token = env[AGENT_TOKEN_ENV];
  if (!path || !token) throw new Error("The pairing's launch has no socket or token.");
  const socket: Socket = createConnection(path);
  await new Promise<void>((resolve, reject) => {
    socket.once("connect", resolve);
    socket.once("error", reject);
  });
  const waiting = new Map<number, (reply: Reply) => void>();
  let buffer = "";
  socket.setEncoding("utf8");
  socket.on("data", (chunk: string) => {
    buffer += chunk;
    for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
      const line = buffer.slice(0, end);
      buffer = buffer.slice(end + 1);
      try {
        const reply = JSON.parse(line) as Reply;
        if (typeof reply.id === "number") {
          waiting.get(reply.id)?.(reply);
          waiting.delete(reply.id);
        }
      } catch {
        // A notification, or not ours.
      }
    }
  });
  let next = 1;
  const request = (method: string, params: unknown) =>
    new Promise<Reply>((resolve) => {
      const id = next++;
      waiting.set(id, resolve);
      socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    });
  socket.write(handshakeLine(token));
  const init = await request("initialize", {
    protocolVersion: "2025-11-25",
    capabilities: {},
    clientInfo: { name: "incarnamind-bakeoff", version: "0.0.0" },
  });
  if (init.error) throw new Error(`initialize: ${init.error.message}`);
  socket.write(`${JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })}\n`);
  return {
    async call(tool, input) {
      const started = performance.now();
      const reply = await request("tools/call", { name: tool, arguments: input });
      const ms = performance.now() - started;
      if (reply.error) {
        return { data: null, text: reply.error.message ?? "", isError: true, ms };
      }
      const text = (reply.result?.content ?? []).map((part) => part.text ?? "").join("");
      let data = reply.result?.structuredContent ?? null;
      if (data === null) {
        try {
          data = JSON.parse(text);
        } catch {
          data = null;
        }
      }
      return { data, text, isError: reply.result?.isError === true, ms };
    },
    close: () => socket.destroy(),
  };
}
