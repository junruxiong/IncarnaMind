/**
 * Runs one task through Codex (`codex exec --json`) and reads what happened
 * from its JSONL events: the final message, each MCP Tool call and shell
 * command, and the turn's token usage.
 *
 * A clean Codex for every run: a new session kept nowhere (`--ephemeral`), no
 * user config or rules, a temporary HOME (so no `~/.agents/skills` of the
 * owner's), the read-only sandbox unless the arm needs more, and no
 * environment inherited from this process. Codex finds its sign-in through
 * CODEX_HOME; the harness never reads it.
 */
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import type { ModelUsage } from "./usage";

export interface CodexOptions {
  /** The Codex binary. */
  bin: string;
  codexHome: string;
  home: string;
  workDir: string;
  model: string;
  effort: string;
  prompt: string;
  /** Extra `-c key=value` settings, e.g. the MCP server or the sandbox's. */
  config: string[];
  /** `--disable` and `--enable` these features. */
  disable: string[];
  enable: string[];
  /** The sandbox mode, or "profile" when `config` selects a permissions profile instead. */
  sandbox: "read-only" | "profile";
  /** Variables for Codex itself. */
  env: Record<string, string>;
  /** PATH for Codex and the commands it runs. */
  path: string;
  timeoutMs: number;
  /** Every event, as Codex printed it, for looking into a run. */
  eventsFile: string;
}

export interface ToolCall {
  /** "mcp" for an MCP Tool, "command" for a shell command, "file" for a file change. */
  kind: "mcp" | "command" | "file" | "web";
  name: string;
  input: string;
  output: string;
  ok: boolean;
  /** Milliseconds since the run started, when it started and ended. */
  startedMs: number | null;
  endedMs: number | null;
}

export interface AgentRun {
  finalText: string;
  /** "done", or why not. */
  status: "done" | "failed" | "timed-out";
  error: string | null;
  calls: ToolCall[];
  /** Model calls: reported by the agent, or estimated as its Tool calls plus one. */
  modelCalls: number;
  modelCallsEstimated: boolean;
  usage: ModelUsage;
  /** Milliseconds from start to the agent being ready (its turn started, its MCP servers up). */
  startupMs: number;
  /** Milliseconds from the agent being ready to its final text. */
  taskMs: number;
  /** Milliseconds from spawning it to its exit. */
  totalMs: number;
  /** Cost in dollars, when the agent reports it (Claude Code). */
  costUsd: number | null;
}

interface CodexItem {
  id?: string;
  type?: string;
  text?: string;
  command?: string;
  aggregated_output?: string;
  exit_code?: number | null;
  status?: string;
  server?: string;
  tool?: string;
  arguments?: unknown;
  result?: { content?: { type?: string; text?: string }[]; structured_content?: unknown } | null;
  error?: { message?: string } | null;
  changes?: { path?: string; kind?: string }[];
  query?: string;
  message?: string;
}

interface CodexEvent {
  type?: string;
  item?: CodexItem;
  usage?: {
    input_tokens?: number;
    cached_input_tokens?: number;
    output_tokens?: number;
    reasoning_output_tokens?: number;
  };
  error?: { message?: string };
  message?: string;
}

const text = (value: unknown) => (typeof value === "string" ? value : JSON.stringify(value ?? ""));

function resultText(item: CodexItem): string {
  if (item.error?.message) return item.error.message;
  const parts = item.result?.content ?? [];
  const joined = parts.map((part) => part.text ?? "").join("");
  return joined || text(item.result?.structured_content ?? "");
}

export function runCodex(options: CodexOptions): Promise<AgentRun> {
  const args = [
    "exec",
    "--json",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "--color",
    "never",
    "-C",
    options.workDir,
    "-m",
    options.model,
    // A permissions profile (arm C) replaces the sandbox mode; `--sandbox` would override it.
    ...(options.sandbox === "profile" ? [] : ["-s", options.sandbox]),
    "-c",
    `model_reasoning_effort="${options.effort}"`,
    "-c",
    'approval_policy="never"',
    ...options.disable.flatMap((feature) => ["--disable", feature]),
    ...options.enable.flatMap((feature) => ["--enable", feature]),
    ...options.config.flatMap((setting) => ["-c", setting]),
    options.prompt,
  ];
  const env: Record<string, string> = {
    PATH: options.path,
    HOME: options.home,
    CODEX_HOME: options.codexHome,
    TMPDIR: process.env.TMPDIR ?? "/tmp",
    LANG: "en_US.UTF-8",
    ...options.env,
  };
  return new Promise((resolve) => {
    const started = performance.now();
    const at = () => Math.round(performance.now() - started);
    const child = spawn(options.bin, args, {
      env,
      cwd: options.workDir,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const events = createWriteStream(options.eventsFile);
    const calls = new Map<string, ToolCall>();
    const order: string[] = [];
    let finalText = "";
    let turnStarted: number | null = null;
    let turnEnded: number | null = null;
    let error: string | null = null;
    let usage: ModelUsage = {
      calls: 0,
      inputTokens: 0,
      cachedInputTokens: 0,
      cacheWriteTokens: 0,
      outputTokens: 0,
      reasoningTokens: 0,
    };
    let stderr = "";
    let buffer = "";
    let timedOut = false;

    const onItem = (item: CodexItem, done: boolean) => {
      const id = item.id ?? `item-${order.length}`;
      const kind =
        item.type === "mcp_tool_call"
          ? "mcp"
          : item.type === "command_execution"
            ? "command"
            : item.type === "file_change"
              ? "file"
              : item.type === "web_search"
                ? "web"
                : null;
      if (item.type === "agent_message" && done && typeof item.text === "string") {
        finalText = item.text;
      }
      if (item.type === "error" && item.message) error = item.message;
      if (!kind) return;
      let call = calls.get(id);
      if (!call) {
        call = {
          kind,
          name:
            kind === "mcp"
              ? `${item.server ?? ""}.${item.tool ?? ""}`
              : kind === "command"
                ? "shell"
                : kind === "file"
                  ? "file_change"
                  : "web_search",
          input:
            kind === "mcp"
              ? text(item.arguments)
              : kind === "command"
                ? (item.command ?? "")
                : kind === "file"
                  ? text(item.changes)
                  : (item.query ?? ""),
          output: "",
          ok: true,
          startedMs: at(),
          endedMs: null,
        };
        calls.set(id, call);
        order.push(id);
      }
      if (done) {
        call.endedMs = at();
        if (kind === "mcp") {
          call.output = resultText(item);
          call.ok = item.status !== "failed" && !item.error;
        } else if (kind === "command") {
          call.output = item.aggregated_output ?? "";
          call.ok = item.exit_code === 0;
        } else if (kind === "file") {
          call.ok = item.status === "completed";
        }
      }
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      events.write(chunk);
      buffer += chunk;
      for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
        const line = buffer.slice(0, end);
        buffer = buffer.slice(end + 1);
        let event: CodexEvent;
        try {
          event = JSON.parse(line) as CodexEvent;
        } catch {
          continue;
        }
        if (event.type === "turn.started" && turnStarted === null) turnStarted = at();
        else if (event.type === "turn.completed") {
          turnEnded = at();
          const used = event.usage ?? {};
          usage = {
            calls: 0,
            inputTokens: usage.inputTokens + (used.input_tokens ?? 0),
            cachedInputTokens: usage.cachedInputTokens + (used.cached_input_tokens ?? 0),
            cacheWriteTokens: 0,
            outputTokens: usage.outputTokens + (used.output_tokens ?? 0),
            reasoningTokens: usage.reasoningTokens + (used.reasoning_output_tokens ?? 0),
          };
        } else if (event.type === "turn.failed") {
          turnEnded = at();
          error = event.error?.message ?? "The turn failed.";
        } else if (event.type === "error") {
          error = event.message ?? "Codex reported an error.";
        } else if (event.item && (event.type === "item.started" || event.type === "item.updated")) {
          onItem(event.item, false);
        } else if (event.item && event.type === "item.completed") {
          onItem(event.item, true);
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
      const list = order.map((id) => calls.get(id) as ToolCall);
      const ready = turnStarted ?? 0;
      const ended = turnEnded ?? totalMs;
      const failed = !timedOut && (code !== 0 || (error !== null && !finalText));
      resolve({
        finalText,
        status: timedOut ? "timed-out" : failed ? "failed" : "done",
        error: timedOut
          ? `Stopped after ${options.timeoutMs / 1000} s.`
          : failed
            ? (error ??
              (stderr.trim().split("\n").slice(-3).join(" ") || `Codex exited with ${code}.`))
            : null,
        calls: list,
        modelCalls: list.filter((call) => call.kind !== "file").length + 1,
        modelCallsEstimated: true,
        usage: { ...usage, calls: list.filter((call) => call.kind !== "file").length + 1 },
        startupMs: ready,
        taskMs: Math.max(0, ended - ready),
        totalMs,
        costUsd: null,
      });
    });
  });
}
