/**
 * The bake-off's settings, from INCARNAMIND_EVAL_* variables (eval/agents/README.md).
 * Keys come only from INCARNAMIND_EVAL_* variables, read from the process or
 * from an env file named by INCARNAMIND_EVAL_ENV_FILE: never OPENAI_API_KEY or
 * the like, and no child process inherits this process's environment, so a
 * key set for other tools can't change which account a run uses.
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { type ChatSettings, readConfig } from "../../lib/config";

export const ARMS = ["A0", "A0-L", "M", "Q", "C", "H", "CC", "MQ"] as const;
export type Arm = (typeof ARMS)[number];

/** The arms an outside agent drives, through IncarnaMind's socket. */
export const OUTSIDE_ARMS: readonly Arm[] = ["M", "Q", "C", "H", "CC", "MQ"];

export const ARM_LABELS: Record<Arm, string> = {
  A0: "Built-in assistant, Haiku 5.5",
  "A0-L": "Built-in assistant, local qwen3.5:4b",
  M: "MCP, fine-grained Tools (Codex)",
  Q: "MCP, ask only (Codex)",
  C: "CLI + Agent Skill (Codex)",
  H: "ACP host, headless (codex-acp)",
  CC: "MCP, fine-grained Tools (Claude Code, Haiku 5.5)",
  MQ: "MCP, fine-grained Tools and ask (Codex)",
};

export interface AgentsConfig {
  root: string;
  /** Where the built-in models are kept between runs (the evaluation's cache). */
  cacheDir: string;
  /** Each session writes into a new folder here, or continues `resume`. */
  resultsDir: string;
  /** A session folder to continue: runs already recorded there are skipped. */
  resume: string | null;
  /** A session folder to score again from its records (sentences, judge), and report again; nothing is run. */
  rescore: string | null;
  /** The model behind A0, `ask` (Q) and the judge: the evaluation's chat model. */
  chat: ChatSettings;
  arms: Arm[];
  /** Task ids, e.g. T1; all ten when not set. */
  tasks: string[] | null;
  /** Runs of each (task, arm) pair; A0-L and CC run once whatever this says. */
  repeats: number;
  /** Codex's model and reasoning effort for every Codex arm. */
  codexModel: string;
  codexEffort: string;
  /** Codex's own folder: its sign-in. Never read by the harness; Codex reads it. */
  codexHome: string;
  /** The Codex binary every Codex arm runs, H's app server included, so they are the same agent. */
  codexBin: string;
  /** More `-c` settings for the Codex MCP arms, e.g. how Tools are approved. */
  codexMcpExtra: string[];
  /** The folder `@agentclientprotocol/codex-acp` was installed into (arm H). */
  codexAcpDir: string | null;
  /** Local model for A0-L, through Ollama. */
  localModel: string;
  localNumCtx: number;
  ollamaUrl: string;
  /** Stops a run that takes longer, counted as a failure. */
  runTimeoutMs: number;
  /** Before each run, wait until the one-minute load average is at most this (0: never wait). */
  maxLoad: number;
  /** The longest that wait lasts; the run then goes ahead, its load recorded. */
  calmWaitMs: number;
  /** Judge uncited claims and correctness with the chat model (on by default). */
  judge: boolean;
  /** Keep each run's temporary folders. */
  keep: boolean;
  /** Where each run's temporary home and working folders go. */
  scratch: string;
}

type Env = Readonly<Record<string, string | undefined>>;

const PREFIX = "INCARNAMIND_EVAL_";

/** INCARNAMIND_EVAL_* lines of an env file; every other line is ignored, and nothing is logged. */
export function readEnvFile(path: string): Record<string, string> {
  const values: Record<string, string> = {};
  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    const match = /^\s*(INCARNAMIND_EVAL_[A-Z0-9_]+)\s*=\s*(.*?)\s*$/.exec(line);
    if (!match?.[1]) continue;
    values[match[1]] = (match[2] ?? "").replace(/^(["'])(.*)\1$/, "$2");
  }
  return values;
}

/** `-c` settings separated by ";", e.g. `a=1;b="x"`. */
const settings = (raw: string | undefined) =>
  (raw ?? "")
    .split(";")
    .map((each) => each.trim())
    .filter(Boolean);

/** A command's path on PATH, or the name itself when it isn't found. */
function onPath(name: string, path: string | undefined): string {
  for (const dir of (path ?? "").split(":")) {
    const candidate = join(dir, name);
    if (dir && existsSync(candidate)) return candidate;
  }
  return name;
}

function list<T extends string>(raw: string, allowed: readonly T[], name: string): T[] {
  return raw
    .split(",")
    .map((each) => each.trim())
    .filter(Boolean)
    .map((each) => {
      const found = allowed.find((value) => value === each);
      if (!found)
        throw new Error(`${PREFIX}${name} takes some of ${allowed.join(", ")}, not "${each}".`);
      return found;
    });
}

export function readAgentsConfig(root: string, processEnv: Env = process.env): AgentsConfig {
  const file = processEnv[`${PREFIX}ENV_FILE`];
  const env: Env = { ...(file && existsSync(file) ? readEnvFile(file) : {}), ...processEnv };
  const value = (name: string) => {
    const raw = env[`${PREFIX}${name}`]?.trim();
    return raw ? raw : undefined;
  };
  const base = readConfig(root, env);
  if (!base.chat) {
    throw new Error(
      `Set ${PREFIX}CHAT_KIND, ${PREFIX}CHAT_MODEL and ${PREFIX}CHAT_KEY (or ${PREFIX}ENV_FILE): A0, ask and the judge need the chat model.`,
    );
  }
  const repeats = Number(value("AGENTS_REPEATS") ?? 3);
  if (!Number.isInteger(repeats) || repeats < 1) {
    throw new Error(`${PREFIX}AGENTS_REPEATS must be a whole number of at least 1.`);
  }
  const arms = value("AGENTS_ARMS");
  const tasks = value("AGENTS_TASKS");
  return {
    root,
    cacheDir: base.cacheDir,
    resultsDir: join(root, "eval", "results"),
    resume: value("AGENTS_RESUME") ? resolve(value("AGENTS_RESUME") as string) : null,
    rescore: value("AGENTS_RESCORE") ? resolve(value("AGENTS_RESCORE") as string) : null,
    chat: base.chat,
    arms: arms ? list(arms, ARMS, "AGENTS_ARMS") : ARMS.filter((arm) => arm !== "MQ"),
    tasks: tasks
      ? tasks
          .split(",")
          .map((each) => each.trim())
          .filter(Boolean)
      : null,
    repeats,
    codexModel: value("AGENTS_CODEX_MODEL") ?? "gpt-6-luna",
    codexEffort: value("AGENTS_CODEX_EFFORT") ?? "low",
    codexHome: value("AGENTS_CODEX_HOME") ?? join(homedir(), ".codex"),
    codexBin: value("AGENTS_CODEX_BIN") ?? onPath("codex", env.PATH),
    codexMcpExtra: settings(value("AGENTS_CODEX_MCP_EXTRA")),
    codexAcpDir: value("AGENTS_CODEX_ACP_DIR") ?? null,
    localModel: value("AGENTS_LOCAL_MODEL") ?? "qwen3.5:4b",
    localNumCtx: Number(value("AGENTS_LOCAL_NUM_CTX") ?? 8192),
    ollamaUrl: value("AGENTS_OLLAMA_URL") ?? "http://127.0.0.1:11434",
    runTimeoutMs: Number(value("AGENTS_RUN_TIMEOUT_S") ?? 300) * 1000,
    maxLoad: Number(value("AGENTS_MAX_LOAD") ?? 0),
    calmWaitMs: Number(value("AGENTS_CALM_WAIT_S") ?? 600) * 1000,
    judge: value("AGENTS_JUDGE") !== "off",
    keep: value("KEEP_DATA") === "1",
    scratch: resolve(value("AGENTS_SCRATCH") ?? join(tmpdir(), "incarnamind-bakeoff")),
  };
}
