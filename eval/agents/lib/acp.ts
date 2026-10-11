/**
 * Arm H: IncarnaMind hosting the User's agent over the Agent Client Protocol
 * (ACP v1), headless (#353's spike). The harness is the ACP client: it starts
 * `codex-acp`, opens a session with IncarnaMind's MCP server (the same shim
 * and pairing as arm M), sends the prompt and collects what streams back.
 * Permission requests for IncarnaMind's own Tools are allowed once; any other
 * (a shell command, a file change) is refused, as #353 starts an agent in its
 * most restrictive mode. A minimal client over newline-delimited JSON-RPC, so
 * the evaluation adds no dependency.
 */
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import type { AgentRun, ToolCall } from "./codex";
import { NO_USAGE } from "./usage";

export interface AcpOptions {
  /** `node` and the adapter's entry script (dist/index.js). */
  command: string;
  args: string[];
  env: Record<string, string>;
  cwd: string;
  prompt: string;
  /** IncarnaMind's MCP server, as session/new takes it. */
  mcpServer: { name: string; command: string; args: string[]; env: Record<string, string> };
  /** Tool names (as the agent reports them) whose permission requests are allowed. */
  allowTools: RegExp;
  timeoutMs: number;
  eventsFile: string;
}

export interface AcpRun extends AgentRun {
  /** Permission requests: what was asked and what the harness answered. */
  permissions: { title: string; allowed: boolean }[];
  /** The last usage the agent reported (ACP's `usage_update`), as it gave it. */
  usageReported: unknown;
}

interface JsonRpc {
  jsonrpc?: string;
  id?: number | string;
  method?: string;
  params?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: { code?: number; message?: string };
}

interface ToolCallUpdate {
  toolCallId?: string;
  title?: string;
  kind?: string;
  status?: string;
  rawInput?: unknown;
  rawOutput?: unknown;
  content?: { type?: string; content?: { type?: string; text?: string } }[];
  _meta?: { is_mcp_tool_call?: boolean };
}

const asText = (value: unknown) =>
  value === undefined || value === null
    ? ""
    : typeof value === "string"
      ? value
      : JSON.stringify(value);

export function runAcp(options: AcpOptions): Promise<AcpRun> {
  return new Promise((resolve) => {
    const started = performance.now();
    const at = () => Math.round(performance.now() - started);
    const child = spawn(options.command, options.args, {
      env: options.env,
      cwd: options.cwd,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const events = createWriteStream(options.eventsFile);
    const pending = new Map<number, (message: JsonRpc) => void>();
    const calls = new Map<string, ToolCall>();
    const order: string[] = [];
    const permissions: { title: string; allowed: boolean }[] = [];
    let next = 1;
    /** The agent's messages in order, each streamed in chunks under its own id. */
    const messages = new Map<string, string>();
    let usageReported: unknown = null;
    /** Tokens in the session's context, as the agent last reported them (ACP v1's only usage). */
    let contextTokens = 0;
    let ready = 0;
    let promptSent = 0;
    let promptDone = 0;
    let error: string | null = null;
    let finished = false;
    let stderr = "";

    const send = (payload: JsonRpc) => {
      const line = JSON.stringify({ jsonrpc: "2.0", ...payload });
      events.write(`> ${line}\n`);
      child.stdin.write(`${line}\n`);
    };
    const request = (method: string, params: Record<string, unknown>) =>
      new Promise<JsonRpc>((done) => {
        const id = next++;
        pending.set(id, done);
        send({ id, method, params });
      });

    const finish = (status: AgentRun["status"]) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      const totalMs = at();
      const list = order.map((id) => calls.get(id) as ToolCall);
      child.stdin.end();
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 3_000).unref();
      events.end();
      resolve({
        // Everything the agent streamed is what IncarnaMind would show, message by message.
        finalText: [...messages.values()]
          .map((text) => text.trim())
          .filter(Boolean)
          .join("\n\n"),
        status,
        error: status === "done" ? null : (error ?? stderr.trim().split("\n").slice(-3).join(" ")),
        calls: list,
        modelCalls: list.length + 1,
        modelCallsEstimated: true,
        usage: { ...NO_USAGE, calls: list.length + 1, inputTokens: contextTokens },
        startupMs: ready,
        taskMs: Math.max(0, (promptDone || totalMs) - promptSent),
        totalMs,
        costUsd: null,
        permissions,
        usageReported,
      });
    };

    const onUpdate = (update: Record<string, unknown>) => {
      const kind = update.sessionUpdate;
      if (kind === "agent_message_chunk") {
        const content = update.content as { type?: string; text?: string } | undefined;
        const id = typeof update.messageId === "string" ? update.messageId : "";
        if (content?.type === "text")
          messages.set(id, (messages.get(id) ?? "") + (content.text ?? ""));
      } else if (kind === "tool_call" || kind === "tool_call_update") {
        const call = update as ToolCallUpdate;
        const id = call.toolCallId ?? `call-${order.length}`;
        let entry = calls.get(id);
        if (!entry) {
          // codex-acp reports an MCP Tool call as kind "execute", marked in `_meta`.
          const mcp =
            call._meta?.is_mcp_tool_call === true || (call.title ?? "").startsWith("mcp.");
          entry = {
            kind: mcp
              ? "mcp"
              : call.kind === "execute"
                ? "command"
                : call.kind === "edit"
                  ? "file"
                  : "mcp",
            name: call.title ?? "",
            input: asText(call.rawInput),
            output: "",
            ok: true,
            startedMs: at(),
            endedMs: null,
          };
          calls.set(id, entry);
          order.push(id);
        }
        if (call.title) entry.name = call.title;
        if (call.rawInput !== undefined && !entry.input) entry.input = asText(call.rawInput);
        const output =
          asText(call.rawOutput) ||
          (call.content ?? []).map((part) => part.content?.text ?? "").join("");
        if (output) entry.output = output;
        if (call.status === "completed" || call.status === "failed") {
          entry.endedMs = at();
          entry.ok = call.status === "completed";
        }
      } else if (kind === "usage_update") {
        usageReported = update;
        if (typeof update.used === "number") contextTokens = update.used;
      }
    };

    let buffer = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      buffer += chunk;
      for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
        const line = buffer.slice(0, end);
        buffer = buffer.slice(end + 1);
        if (!line.trim()) continue;
        events.write(`< ${line}\n`);
        let incoming: JsonRpc;
        try {
          incoming = JSON.parse(line) as JsonRpc;
        } catch {
          continue;
        }
        if (incoming.method === "session/update") {
          onUpdate((incoming.params?.update ?? {}) as Record<string, unknown>);
        } else if (incoming.method === "session/request_permission" && incoming.id !== undefined) {
          const params = incoming.params ?? {};
          const toolCall = (params.toolCall ?? {}) as ToolCallUpdate;
          const title = `${toolCall.title ?? ""} ${asText(toolCall.rawInput)}`.trim();
          const choices = (params.options ?? []) as { optionId: string; kind: string }[];
          const allowed = options.allowTools.test(title);
          const choice = choices.find((option) =>
            allowed ? option.kind === "allow_once" : option.kind.startsWith("reject"),
          );
          permissions.push({ title, allowed });
          send({
            id: incoming.id,
            result: choice
              ? { outcome: { outcome: "selected", optionId: choice.optionId } }
              : { outcome: { outcome: "cancelled" } },
          });
        } else if (incoming.method !== undefined && incoming.id !== undefined) {
          // The client offers no file system or terminal; say so to anything else asked of it.
          send({
            id: incoming.id,
            error: { code: -32601, message: "Not offered by this client." },
          });
        } else if (typeof incoming.id === "number") {
          pending.get(incoming.id)?.(incoming);
          pending.delete(incoming.id);
        }
      }
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      if (stderr.length > 20_000) stderr = stderr.slice(-20_000);
    });
    child.on("close", () => {
      if (!finished) {
        error ??= "codex-acp exited before the prompt finished.";
        finish("failed");
      }
    });
    const timer = setTimeout(() => {
      error = `Stopped after ${options.timeoutMs / 1000} s.`;
      finish("timed-out");
    }, options.timeoutMs);

    (async () => {
      const init = await request("initialize", {
        protocolVersion: 1,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: "incarnamind-bakeoff", version: "0.0.0" },
      });
      if (init.error) throw new Error(`initialize: ${init.error.message}`);
      const session = await request("session/new", {
        cwd: options.cwd,
        mcpServers: [
          {
            name: options.mcpServer.name,
            command: options.mcpServer.command,
            args: options.mcpServer.args,
            env: Object.entries(options.mcpServer.env).map(([name, value]) => ({ name, value })),
          },
        ],
      });
      if (session.error) throw new Error(`session/new: ${session.error.message}`);
      const sessionId = session.result?.sessionId;
      if (typeof sessionId !== "string") throw new Error("session/new gave no session id.");
      ready = at();
      promptSent = at();
      const prompted = await request("session/prompt", {
        sessionId,
        prompt: [{ type: "text", text: options.prompt }],
      });
      promptDone = at();
      if (prompted.error) throw new Error(`session/prompt: ${prompted.error.message}`);
      const stop = prompted.result?.stopReason;
      if (stop !== "end_turn") error = `The turn ended with "${asText(stop)}".`;
      finish(stop === "end_turn" ? "done" : "failed");
    })().catch((caught: unknown) => {
      error = caught instanceof Error ? caught.message : String(caught);
      finish("failed");
    });
  });
}
