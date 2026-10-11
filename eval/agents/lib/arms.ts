/**
 * One run of one contestant on one task: set it up, run it, score it.
 *
 * - A0, A0-L: the built-in assistant, a Question in a Mind of its own scoped
 *   to "In scope", on Haiku 5.5 or a local model (../../lib/citations `askOne`).
 * - M, Q, MQ: Codex with IncarnaMind's MCP server (#351) through the bake-off's
 *   shim, which shows M the fine-grained Tools, Q only `ask`, MQ both.
 * - C: Codex with the CLI on its PATH and an Agent Skill in its working folder.
 * - H: codex-acp, with M's server given at session start; afterwards the host
 *   checks the agent's text inside IncarnaMind (`check_citations`), as #353 would.
 * - CC: Claude Code (`--bare`, the evaluation's API key) with M's server.
 *
 * Every outside arm gets a new pairing, scoped to "In scope", so its tokens,
 * activity and refusals are its own.
 */
import { execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { type Dirent, existsSync, readFileSync } from "node:fs";
import { chmod, mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { AgentActivity, ChatProvider, Core } from "../../../src/core";
import type { Database } from "../../../src/core/storage";
import { type AnswerRecord, askOne } from "../../lib/citations";
import type { EvalQuestion } from "../../lib/evaluationSet";
import type { Log } from "../../lib/log";
import { runAcp } from "./acp";
import type { Bakeoff, Task } from "./bakeoff";
import { promptFor } from "./bakeoff";
import type { CanaryListener } from "./canary";
import { runClaudeCode } from "./claude";
import { type AgentRun, runCodex, type ToolCall } from "./codex";
import type { AgentsConfig, Arm } from "./config";
import type { FixtureLibrary } from "./fixtureLibrary";
import { judge } from "./judge";
import {
  type RunMetrics,
  type ScoredCitation,
  type Sentence,
  scoreAnswerCitations,
  scoreRun,
  scoreTokens,
  sentencesOf,
} from "./score";
import { openSocketClient } from "./socketClient";
import { addUsage, type ModelUsage, NO_USAGE, type UsageMeter } from "./usage";
import { ASK_TOOLS, FINE_TOOLS, instructionsFor, skillMarkdown, type Wording } from "./wording";

export interface Session {
  config: AgentsConfig;
  bakeoff: Bakeoff;
  fixture: FixtureLibrary;
  core: Core;
  db: Database;
  meter: UsageMeter;
  canary: CanaryListener;
  cloud: ChatProvider;
  local: ChatProvider | null;
  /** The harness's own files an agent starts. */
  node: string;
  shim: string;
  cli: string;
  /** Where each run's folders go, and where its events are kept. */
  runsDir: string;
  eventsDir: string;
  /** The owner's own Codex skills and MCP servers, switched off in every Codex run (paths and names only). */
  ownerSkills: string[];
  ownerMcpServers: string[];
  /** Paths whose direct reading counts as going round IncarnaMind. */
  privatePaths: string[];
  log: Log;
}

export interface ShimCall {
  method: string;
  tool: string | null;
  ms?: number;
  refused?: boolean;
  via?: string;
}

export interface RunRecord {
  arm: Arm;
  task: string;
  repeat: number;
  runId: string;
  startedAt: string;
  prompt: string;
  status: "done" | "failed" | "timed-out";
  error: string | null;
  finalText: string;
  startupMs: number;
  taskMs: number;
  totalMs: number;
  /** H only: the host's check of the agent's text inside IncarnaMind, added to its task time. */
  recheckMs: number | null;
  modelCalls: number;
  modelCallsEstimated: boolean;
  /** The agent's own model. */
  agentUsage: ModelUsage;
  /** IncarnaMind's model: the built-in Answer, or the Answers behind `ask`. */
  innerUsage: ModelUsage;
  /** Dollars on the evaluation's API key (Haiku 5.5): IncarnaMind's model, and Claude Code's. */
  costUsd: number;
  /** Codex plan credits, at the plan's rates (B.4). */
  credits: number;
  calls: ToolCall[];
  shimCalls: ShimCall[];
  activity: Pick<
    AgentActivity,
    "tool" | "summary" | "outcome" | "problem" | "count" | "durationMs"
  >[];
  citations: ScoredCitation[];
  sentences: Sentence[];
  metrics: RunMetrics;
  permissions?: { title: string; allowed: boolean }[];
  usageReported?: unknown;
  judgeUsage: ModelUsage;
}

// ---------------------------------------------------------------------------
// Prices (B.4): Haiku 5.5 at list prices; Codex in plan credits.

const HAIKU = { input: 0.1, cached: 0.01, cacheWrite: 0.125, output: 0.5 };
const LUNA_CREDITS = { input: 2.5, cached: 0.25, output: 12.5 };

export function haikuUsd(usage: ModelUsage): number {
  const fresh = Math.max(0, usage.inputTokens - usage.cachedInputTokens - usage.cacheWriteTokens);
  return (
    (fresh * HAIKU.input +
      usage.cachedInputTokens * HAIKU.cached +
      usage.cacheWriteTokens * HAIKU.cacheWrite +
      usage.outputTokens * HAIKU.output) /
    1e6
  );
}

export function lunaCredits(usage: ModelUsage): number {
  const fresh = Math.max(0, usage.inputTokens - usage.cachedInputTokens);
  return (
    (fresh * LUNA_CREDITS.input +
      usage.cachedInputTokens * LUNA_CREDITS.cached +
      usage.outputTokens * LUNA_CREDITS.output) /
    1e6
  );
}

// ---------------------------------------------------------------------------

const TOML = (value: string) => JSON.stringify(value);
const tomlTable = (entries: Record<string, string>) =>
  `{${Object.entries(entries)
    .map(([key, value]) => `${key}=${TOML(value)}`)
    .join(",")}}`;

/** Codex features no run needs, off so the owner's plugins, apps and hooks stay out. */
const CODEX_DISABLED = [
  "plugins",
  "apps",
  "hooks",
  "memories",
  "multi_agent",
  "multi_agent_v2",
  "browser_use",
  "computer_use",
  "image_generation",
  "in_app_browser",
  "remote_plugin",
  "tool_suggest",
  "goals",
];

/**
 * Settings that keep the owner's own Codex setup out of a run. `codex exec` skips the owner's
 * config.toml, so naming its MCP servers there would declare broken ones; the app server H runs
 * reads it, so there each is switched off by name.
 */
function ownerOverrides(session: Session, readsUserConfig: boolean): string[] {
  const skills = session.ownerSkills.map((path) => `{path=${TOML(path)},enabled=false}`);
  return [
    ...(skills.length > 0 ? [`skills.config=[${skills.join(",")}]`] : []),
    ...(readsUserConfig
      ? session.ownerMcpServers.map((name) => `mcp_servers.${name}.enabled=false`)
      : []),
    'web_search="disabled"',
  ];
}

const searchTools = /search|ask/;

function readShimLog(path: string): ShimCall[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as ShimCall];
      } catch {
        return [];
      }
    });
}

async function plantedFile(dirs: readonly string[], name: string): Promise<boolean> {
  for (const dir of dirs) {
    const stack = [dir];
    while (stack.length > 0) {
      const current = stack.pop() as string;
      let entries: Dirent[];
      try {
        entries = await readdir(current, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const entry of entries) {
        if (entry.name === name) return true;
        if (entry.isDirectory() && !entry.isSymbolicLink() && entry.name !== "node_modules") {
          stack.push(join(current, entry.name));
        }
      }
    }
  }
  return false;
}

/** A task as the evaluation asks a Question (its expected passage isn't used here: the scorer checks). */
function asQuestion(task: Task, prompt: string): EvalQuestion {
  const expected = task.expected[0];
  return {
    id: task.id,
    language: task.language,
    crossLingual: false,
    question: prompt,
    expected: {
      document: expected?.document ?? "",
      pages: expected?.pages ?? [1, 1],
      quote: expected?.quote ?? "",
    },
  };
}

const trim = (text: string, max = 6000) => (text.length > max ? `${text.slice(0, max)}…` : text);

/** Runs one (arm, task, repeat) and scores it. */
export async function runOne(
  session: Session,
  arm: Arm,
  task: Task,
  repeat: number,
): Promise<RunRecord> {
  const runId = `${arm}-${task.id}-r${repeat}-${randomBytes(3).toString("hex")}`;
  const prompt = promptFor(session.bakeoff, task);
  const startedAt = new Date().toISOString();
  const since = Date.now();
  session.meter.take();
  const record =
    arm === "A0" || arm === "A0-L"
      ? await runBuiltIn(session, arm, task, prompt, runId)
      : await runOutside(session, arm, task, prompt, runId);
  // No text: nothing to judge, and nothing correct.
  const judged = !session.config.judge
    ? null
    : record.sentences.length === 0
      ? { correct: "no" as const, claims: [], usage: NO_USAGE }
      : await judge(session.config.chat, task, prompt, record.sentences).catch((error: unknown) => {
          session.log(`  judge failed: ${error instanceof Error ? error.message : String(error)}`);
          return null;
        });
  const sentences = record.sentences.map((sentence, index) => ({
    ...sentence,
    claim: judged ? judged.claims.includes(index) : null,
  }));
  const uncitedClaims = judged ? judged.claims.length : null;
  return {
    ...record,
    arm,
    task: task.id,
    repeat,
    runId,
    startedAt,
    prompt,
    sentences,
    metrics: {
      ...record.metrics,
      uncitedClaims,
      unmarkedUncitedClaims: uncitedClaims === null ? null : record.marks ? 0 : uncitedClaims,
      correct: judged?.correct ?? null,
      injection: {
        ...record.metrics.injection,
        listenerHits: session.canary.hitsSince(since).length,
      },
    },
    judgeUsage: judged?.usage ?? NO_USAGE,
  };
}

type ArmResult = Omit<
  RunRecord,
  "arm" | "task" | "repeat" | "runId" | "startedAt" | "prompt" | "judgeUsage"
> & { marks: boolean };

async function runBuiltIn(
  session: Session,
  arm: "A0" | "A0-L",
  task: Task,
  prompt: string,
  runId: string,
): Promise<ArmResult> {
  const { fixture, config } = session;
  const provider = arm === "A0" ? session.cloud : session.local;
  if (!provider) throw new Error("A0-L needs the local model, which wasn't set up.");
  const modelId = arm === "A0" ? config.chat.modelId : config.localModel;
  const answer: AnswerRecord = await askOne(
    fixture.library,
    asQuestion(task, prompt),
    1,
    config.runTimeoutMs,
    { scopeFolderIds: fixture.scope.folderIds, providerId: provider.id, modelId },
  );
  const inner = session.meter.take();
  const citations = scoreAnswerCitations(answer, fixture);
  const finalText = answer.sentences.map((sentence) => sentence.text).join("\n");
  const retrieved = new Set(
    citations.flatMap((citation) => (citation.documentKey ? [citation.documentKey] : [])),
  );
  const metrics = scoreRun({
    task,
    bakeoff: session.bakeoff,
    finalText,
    citations,
    calls: [],
    innerSearches: answer.searches,
    innerText: finalText,
    retrievedKeys: retrieved,
    marksUncited: true,
    scopeRefusals: 0,
    listenerHits: 0,
    fileWritten: false,
    privatePaths: session.privatePaths,
  });
  // The Answer's own sentence split, which is how the editor anchors its Citations.
  const sentences: Sentence[] = answer.sentences.map((sentence) => ({ ...sentence, claim: null }));
  metrics.sentences = sentences.length;
  metrics.uncitedSentences = sentences.filter((sentence) => !sentence.cited).length;
  session.log(
    `  ${runId}: ${answer.status}, ${answer.searches.length} searches (${answer.searches.join(" | ")})`,
  );
  return {
    status:
      answer.status === "done" ? "done" : answer.status === "timed-out" ? "timed-out" : "failed",
    error: answer.error ? `${answer.error.kind}: ${answer.error.message}` : null,
    finalText,
    startupMs: 0,
    taskMs: Math.round(answer.seconds * 1000),
    totalMs: Math.round(answer.seconds * 1000),
    recheckMs: null,
    modelCalls: inner.calls,
    modelCallsEstimated: false,
    agentUsage: NO_USAGE,
    innerUsage: inner,
    costUsd: arm === "A0" ? haikuUsd(inner) : 0,
    credits: 0,
    calls: answer.searches.map((query) => ({
      kind: "mcp" as const,
      name: "built-in.search",
      input: query,
      output: "",
      ok: true,
      startedMs: null,
      endedMs: null,
    })),
    shimCalls: [],
    activity: [],
    citations,
    sentences,
    metrics,
    marks: true,
  };
}

/** The Tools, instructions and CLI verbs each outside arm gets. */
function armSetup(arm: Arm): { tools: readonly string[]; wording: Wording } {
  if (arm === "Q") return { tools: ASK_TOOLS, wording: "ask" };
  if (arm === "MQ") return { tools: [...FINE_TOOLS, ...ASK_TOOLS], wording: "both" };
  return { tools: FINE_TOOLS, wording: "fine" };
}

async function runOutside(
  session: Session,
  arm: Arm,
  task: Task,
  prompt: string,
  runId: string,
): Promise<ArmResult> {
  const { core, config, fixture } = session;
  const runDir = join(session.runsDir, runId);
  const home = join(runDir, "home");
  const work = join(runDir, "work");
  await mkdir(home, { recursive: true });
  await mkdir(work, { recursive: true });
  const shimLog = join(session.eventsDir, `${runId}.shim.jsonl`);
  const eventsFile = join(session.eventsDir, `${runId}.events.jsonl`);
  const { tools, wording } = armSetup(arm);
  const connected = await core.connectAgent({ name: runId, folderIds: fixture.scope.folderIds });
  const serverEnv: Record<string, string> = {
    ...connected.launch.env,
    INCARNAMIND_BAKEOFF_TOOLS: tools.join(","),
    INCARNAMIND_BAKEOFF_LOG: shimLog,
    INCARNAMIND_BAKEOFF_INSTRUCTIONS: instructionsFor(wording),
  };
  const server = { command: connected.launch.command, args: connected.launch.args };
  const systemPath = process.env.PATH ?? "/usr/bin:/bin";
  let run: AgentRun;
  let permissions: { title: string; allowed: boolean }[] | undefined;
  let usageReported: unknown;
  let recheckMs: number | null = null;
  try {
    if (arm === "M" || arm === "Q" || arm === "MQ" || arm === "C") {
      const config_: string[] = [...ownerOverrides(session, false)];
      let path = systemPath;
      let sandbox: "read-only" | "profile" = "read-only";
      const enable: string[] = [];
      if (arm === "C") {
        // The CLI on the PATH, its pairing in a wrapper only this run has; the Agent Skill in the
        // working folder, where Codex looks for a project's skills.
        const bin = join(runDir, "bin");
        await mkdir(bin, { recursive: true });
        const wrapper = join(bin, "incarnamind");
        const exports = Object.entries(serverEnv)
          .filter(([key]) => key !== "INCARNAMIND_BAKEOFF_INSTRUCTIONS")
          .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
          .join(" ");
        await writeFile(
          wrapper,
          `#!/bin/sh\n${exports} exec ${JSON.stringify(session.node)} --disable-warning=MODULE_TYPELESS_PACKAGE_JSON ${JSON.stringify(session.cli)} "$@"\n`,
        );
        await chmod(wrapper, 0o700);
        const skillDir = join(work, ".agents", "skills", "incarnamind");
        await mkdir(skillDir, { recursive: true });
        await writeFile(join(skillDir, "SKILL.md"), skillMarkdown(cliHelp(session, serverEnv)));
        path = `${bin}:${systemPath}`;
        // Codex's sandbox refuses Unix sockets by default (EPERM). The least privilege that lets
        // the CLI reach IncarnaMind (found with `codex sandbox`): read-only, network only through
        // Codex's proxy with no domain allowed, and this one socket allowed.
        const socket = connected.launch.env.INCARNAMIND_AGENT_SOCKET ?? "";
        config_.push(
          'default_permissions="bakeoffcli"',
          'permissions.bakeoffcli.extends=":read-only"',
          "permissions.bakeoffcli.network.enabled=true",
          "permissions.bakeoffcli.network.domains={}",
          `permissions.bakeoffcli.network.unix_sockets={${TOML(socket)}="allow"}`,
        );
        enable.push("network_proxy");
        sandbox = "profile";
      } else {
        config_.push(
          `mcp_servers.incarnamind.command=${TOML(server.command)}`,
          `mcp_servers.incarnamind.args=${JSON.stringify(server.args)}`,
          `mcp_servers.incarnamind.env=${tomlTable(serverEnv)}`,
          "mcp_servers.incarnamind.startup_timeout_sec=20",
          "mcp_servers.incarnamind.tool_timeout_sec=120",
          ...config.codexMcpExtra,
        );
      }
      run = await runCodex({
        bin: config.codexBin,
        codexHome: config.codexHome,
        home,
        workDir: work,
        model: config.codexModel,
        effort: config.codexEffort,
        prompt,
        config: config_,
        disable: CODEX_DISABLED,
        enable,
        sandbox,
        env: {},
        path,
        timeoutMs: config.runTimeoutMs,
        eventsFile,
      });
    } else if (arm === "H") {
      if (!config.codexAcpDir)
        throw new Error("Arm H needs INCARNAMIND_EVAL_AGENTS_CODEX_ACP_DIR.");
      // codex-acp runs `codex app-server`, which reads Codex's own config: a wrapper gives it the
      // same overrides as the other Codex arms, so the owner's servers, plugins and hooks stay off.
      const wrapper = join(runDir, "codex");
      const overrides = [
        ...CODEX_DISABLED.flatMap((feature) => ["--disable", feature]),
        ...ownerOverrides(session, true).flatMap((setting) => ["-c", setting]),
        "-c",
        `model=${TOML(config.codexModel)}`,
        "-c",
        `model_reasoning_effort=${TOML(config.codexEffort)}`,
      ]
        .map((part) => `'${part.replaceAll("'", "'\\''")}'`)
        .join(" ");
      await writeFile(
        wrapper,
        `#!/bin/sh\nif [ "$1" = "app-server" ]; then shift; exec ${JSON.stringify(config.codexBin)} app-server ${overrides} "$@"; fi\nexec ${JSON.stringify(config.codexBin)} "$@"\n`,
      );
      await chmod(wrapper, 0o700);
      const acp = await runAcp({
        command: session.node,
        args: [
          join(config.codexAcpDir, "node_modules/@agentclientprotocol/codex-acp/dist/index.js"),
        ],
        env: {
          PATH: systemPath,
          HOME: home,
          CODEX_HOME: config.codexHome,
          TMPDIR: process.env.TMPDIR ?? "/tmp",
          LANG: "en_US.UTF-8",
          CODEX_PATH: wrapper,
          INITIAL_AGENT_MODE: "read-only",
          NO_BROWSER: "1",
          CODEX_CONFIG: JSON.stringify({
            model: config.codexModel,
            model_reasoning_effort: config.codexEffort,
          }),
        },
        cwd: work,
        prompt,
        mcpServer: { name: "incarnamind", ...server, env: serverEnv },
        allowTools: /incarnamind|search|read_passages|cite|check_citations/i,
        timeoutMs: config.runTimeoutMs,
        eventsFile,
      });
      run = acp;
      permissions = acp.permissions;
      usageReported = acp.usageReported;
      // The host checks the agent's text inside IncarnaMind, as #353 would before showing it.
      if (run.finalText) {
        const client = await openSocketClient(connected.launch.env);
        try {
          const checked = await client.call("check_citations", { text: run.finalText });
          recheckMs = Math.round(checked.ms);
        } finally {
          client.close();
        }
      }
    } else if (arm === "CC") {
      const mcpConfigFile = join(runDir, "mcp.json");
      await writeFile(
        mcpConfigFile,
        JSON.stringify({ mcpServers: { incarnamind: { ...server, env: serverEnv } } }),
        { mode: 0o600 },
      );
      if (!config.chat.apiKey) throw new Error("Arm CC needs the evaluation's API key.");
      run = await runClaudeCode({
        home,
        workDir: work,
        path: systemPath,
        model: config.chat.modelId,
        apiKey: config.chat.apiKey,
        prompt,
        mcpConfigFile,
        allowedTools: FINE_TOOLS.map((tool) => `mcp__incarnamind__${tool}`),
        timeoutMs: config.runTimeoutMs,
        eventsFile,
      });
    } else {
      throw new Error(`Unknown arm ${arm}.`);
    }

    const inner = session.meter.take();
    const activity = await core.listAgentActivity(connected.pairing.id);
    const citations = await scoreTokens(run.finalText, {
      db: session.db,
      core,
      pairingId: connected.pairing.id,
      fixture,
    });
    const retrieved = new Set(
      activity.flatMap((call) =>
        call.documentIds.flatMap((id) => {
          const key = fixture.keyOf(id);
          return key ? [key] : [];
        }),
      ),
    );
    const askAnswers = run.calls
      .filter((call) => /\bask\b/.test(call.name) || /incarnamind\s+ask/.test(call.input))
      .map((call) => call.output)
      .join("\n");
    const fileWritten = await plantedFile([work, home], session.bakeoff.canaries.injection.file);
    const metrics = scoreRun({
      task,
      bakeoff: session.bakeoff,
      finalText: run.finalText,
      citations,
      calls: run.calls,
      innerSearches: [],
      innerText: askAnswers,
      retrievedKeys: retrieved,
      marksUncited: arm === "H",
      scopeRefusals: activity.filter((call) => call.problem === "outside-scope").length,
      listenerHits: 0,
      fileWritten,
      privatePaths: session.privatePaths,
    });
    const sentences: Sentence[] = sentencesOf(run.finalText, task.language).map((sentence) => ({
      ...sentence,
      claim: null,
    }));
    const searches = run.calls.filter(
      (call) =>
        (call.kind === "mcp" && searchTools.test(call.name)) ||
        /incarnamind\s+(search|ask)/.test(call.input),
    ).length;
    session.log(
      `  ${runId}: ${run.status}${run.error ? ` (${run.error})` : ""}, ${run.calls.length} calls (${searches} searches/asks), ${metrics.citations} Citations`,
    );
    const agentUsage = run.usage;
    const innerUsage = inner;
    return {
      status: run.status,
      error: run.error,
      finalText: run.finalText,
      startupMs: run.startupMs,
      taskMs: run.taskMs + (recheckMs ?? 0),
      totalMs: run.totalMs + (recheckMs ?? 0),
      recheckMs,
      modelCalls: run.modelCalls,
      modelCallsEstimated: run.modelCallsEstimated,
      agentUsage,
      innerUsage,
      costUsd: haikuUsd(innerUsage) + (arm === "CC" ? haikuUsd(agentUsage) : 0),
      credits: arm === "CC" ? 0 : lunaCredits(agentUsage),
      calls: run.calls.map((call) => ({ ...call, output: trim(call.output) })),
      shimCalls: readShimLog(shimLog),
      activity: activity.map(({ tool, summary, outcome, problem, count, durationMs }) => ({
        tool,
        summary,
        outcome,
        problem,
        count,
        durationMs,
      })),
      citations,
      sentences,
      metrics,
      ...(permissions && { permissions }),
      ...(usageReported !== undefined && { usageReported }),
      marks: arm === "H",
    };
  } finally {
    await core.disconnectAgent(connected.pairing.id).catch(() => undefined);
    if (!config.keep) await rm(runDir, { recursive: true, force: true });
  }
}

/** The CLI's help as arm C's pairing sees it (the verbs it offers), for its Agent Skill. */
function cliHelp(session: Session, env: Record<string, string>): string {
  return execFileSync(
    session.node,
    ["--disable-warning=MODULE_TYPELESS_PACKAGE_JSON", session.cli, "--help"],
    {
      env: {
        PATH: process.env.PATH ?? "",
        INCARNAMIND_BAKEOFF_TOOLS: env.INCARNAMIND_BAKEOFF_TOOLS ?? "",
      },
      encoding: "utf8",
    },
  );
}

/** Sums what a list of runs used, for the report. */
export function totalUsage(records: readonly RunRecord[], pick: (record: RunRecord) => ModelUsage) {
  return records.reduce((sum, record) => addUsage(sum, pick(record)), NO_USAGE);
}
