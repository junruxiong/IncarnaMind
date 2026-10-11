/**
 * Arm CC: the same MCP server as arm M, in Claude Code, headless
 * (`claude --bare -p`). `--bare` reads Anthropic credentials only from
 * ANTHROPIC_API_KEY (never a sign-in or the keychain), so the run spends the
 * evaluation's API key, never anyone's Claude plan; the temporary HOME and
 * `--strict-mcp-config` keep every other setting and MCP server out.
 */
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import type { AgentRun, ToolCall } from "./codex";

export interface ClaudeOptions {
  home: string;
  workDir: string;
  path: string;
  model: string;
  apiKey: string;
  prompt: string;
  /** A JSON file in Claude Code's `--mcp-config` format. */
  mcpConfigFile: string;
  /** Tools it may use without asking (anything else is refused in print mode). */
  allowedTools: string[];
  timeoutMs: number;
  eventsFile: string;
}

interface Content {
  type?: string;
  id?: string;
  name?: string;
  input?: unknown;
  text?: string;
  tool_use_id?: string;
  content?: unknown;
  is_error?: boolean;
}

interface StreamEvent {
  type?: string;
  subtype?: string;
  message?: { content?: Content[] };
  result?: string;
  num_turns?: number;
  total_cost_usd?: number;
  usage?: {
    input_tokens?: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
    output_tokens?: number;
  };
  is_error?: boolean;
}

const asText = (value: unknown): string =>
  typeof value === "string"
    ? value
    : Array.isArray(value)
      ? value.map((part: { text?: string }) => part.text ?? "").join("")
      : JSON.stringify(value ?? "");

export function runClaudeCode(options: ClaudeOptions): Promise<AgentRun> {
  const args = [
    "--bare",
    "-p",
    "--output-format",
    "stream-json",
    "--verbose",
    "--no-session-persistence",
    "--model",
    options.model,
    "--mcp-config",
    options.mcpConfigFile,
    "--strict-mcp-config",
    // Takes several values, so the prompt goes on stdin rather than after it.
    "--allowedTools",
    options.allowedTools.join(","),
  ];
  const env: Record<string, string> = {
    PATH: options.path,
    HOME: options.home,
    TMPDIR: process.env.TMPDIR ?? "/tmp",
    LANG: "en_US.UTF-8",
    ANTHROPIC_API_KEY: options.apiKey,
    DISABLE_AUTOUPDATER: "1",
    DISABLE_TELEMETRY: "1",
  };
  return new Promise((resolve) => {
    const started = performance.now();
    const at = () => Math.round(performance.now() - started);
    const child = spawn("claude", args, {
      env,
      cwd: options.workDir,
      stdio: ["pipe", "pipe", "pipe"],
    });
    child.stdin.end(options.prompt);
    const events = createWriteStream(options.eventsFile);
    const calls = new Map<string, ToolCall>();
    const order: string[] = [];
    let finalText = "";
    let initAt: number | null = null;
    let result: StreamEvent | null = null;
    let stderr = "";
    let buffer = "";
    let timedOut = false;
    let lastText = "";

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      events.write(chunk);
      buffer += chunk;
      for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
        const line = buffer.slice(0, end);
        buffer = buffer.slice(end + 1);
        let event: StreamEvent;
        try {
          event = JSON.parse(line) as StreamEvent;
        } catch {
          continue;
        }
        if (event.type === "system" && event.subtype === "init" && initAt === null) initAt = at();
        if (event.type === "assistant") {
          for (const part of event.message?.content ?? []) {
            if (part.type === "text" && part.text) lastText = part.text;
            if (part.type === "tool_use" && part.id) {
              const name = part.name ?? "";
              calls.set(part.id, {
                kind: name.startsWith("mcp__") ? "mcp" : name === "Bash" ? "command" : "file",
                name,
                input: asText(part.input),
                output: "",
                ok: true,
                startedMs: at(),
                endedMs: null,
              });
              order.push(part.id);
            }
          }
        } else if (event.type === "user") {
          for (const part of event.message?.content ?? []) {
            if (part.type !== "tool_result" || !part.tool_use_id) continue;
            const call = calls.get(part.tool_use_id);
            if (!call) continue;
            call.output = asText(part.content);
            call.ok = !part.is_error;
            call.endedMs = at();
          }
        } else if (event.type === "result") {
          result = event;
          finalText = event.result ?? lastText;
        }
      }
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      if (stderr.length > 20_000) stderr = stderr.slice(-20_000);
    });
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5_000).unref();
    }, options.timeoutMs);
    child.on("close", (code) => {
      clearTimeout(timer);
      events.end();
      const totalMs = at();
      const ended = result as StreamEvent | null;
      const usage = ended?.usage ?? {};
      const cached = usage.cache_read_input_tokens ?? 0;
      const written = usage.cache_creation_input_tokens ?? 0;
      const failed = !timedOut && (code !== 0 || !ended || ended.is_error === true);
      const ready = initAt ?? 0;
      resolve({
        finalText: finalText || lastText,
        status: timedOut ? "timed-out" : failed ? "failed" : "done",
        error: timedOut
          ? `Stopped after ${options.timeoutMs / 1000} s.`
          : failed
            ? stderr.trim().split("\n").slice(-3).join(" ") ||
              `Claude Code exited with ${code}${ended?.subtype ? ` (${ended.subtype})` : ""}.`
            : null,
        calls: order.map((id) => calls.get(id) as ToolCall),
        modelCalls: ended?.num_turns ?? order.length + 1,
        modelCallsEstimated: ended?.num_turns === undefined,
        usage: {
          calls: ended?.num_turns ?? order.length + 1,
          inputTokens: (usage.input_tokens ?? 0) + cached + written,
          cachedInputTokens: cached,
          cacheWriteTokens: written,
          outputTokens: usage.output_tokens ?? 0,
          reasoningTokens: 0,
        },
        startupMs: ready,
        taskMs: Math.max(0, totalMs - ready),
        totalMs,
        costUsd: ended?.total_cost_usd ?? null,
      });
    });
  });
}
