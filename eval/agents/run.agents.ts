/**
 * `npm run eval:agents`: the agent-integration bake-off (eval/agents/README.md).
 *
 * One fixture Library on a temporary data folder, indexed once and kept warm;
 * IncarnaMind's outside-agent server (#351) listening on its socket; then each
 * contestant runs each task, in an order shuffled across contestants, every
 * run with a pairing of its own. The built-in assistant on a local model runs
 * last, in Local mode, one model loaded and unloaded after. Every run is
 * written to runs.jsonl as it ends, so a session stopped halfway can go on
 * (INCARNAMIND_EVAL_AGENTS_RESUME), and the report is written at the end.
 *
 * It runs under Vitest only for its TypeScript and worker-thread handling, as
 * the evaluation does; neither `npm test` nor CI runs it.
 */
import { existsSync, readFileSync } from "node:fs";
import { appendFile, mkdir, readdir, rm } from "node:fs/promises";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "vitest";
import { DATABASE_FILE } from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import { createLog } from "../lib/log";
import { type RunRecord, runOne, type Session } from "./lib/arms";
import { loadBakeoff, type Task } from "./lib/bakeoff";
import { listenForCanary } from "./lib/canary";
import { type Arm, readAgentsConfig } from "./lib/config";
import { openFixtureLibrary } from "./lib/fixtureLibrary";
import { type ProbeRow, probeRetrieval } from "./lib/probe";
import { gatesOf, summarise, writeReport } from "./lib/report";
import { openSocketClient } from "./lib/socketClient";
import { createUsageMeter } from "./lib/usage";

const root = fileURLToPath(new URL("../..", import.meta.url));
const NODE_FLAGS = ["--disable-warning=MODULE_TYPELESS_PACKAGE_JSON"];

/** A seeded shuffle, so a session's order can be told and repeated. */
function shuffle<T>(items: T[], seed: number): T[] {
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = [...items];
  for (let index = out.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [out[index], out[other]] = [out[other] as T, out[index] as T];
  }
  return out;
}

/** The owner's own Codex skills, by path, to switch off in each run (Codex's bundled ones stay). */
async function ownerSkills(codexHome: string): Promise<string[]> {
  const found: string[] = [];
  const walk = async (dir: string) => {
    let entries: import("node:fs").Dirent[];
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name === ".system") continue;
      const path = join(dir, entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (entry.name === "SKILL.md") found.push(path);
    }
  };
  await walk(join(codexHome, "skills"));
  return found;
}

/** The names of the MCP servers the owner's Codex config declares: section headers only, never values. */
function ownerMcpServers(codexHome: string): string[] {
  const file = join(codexHome, "config.toml");
  if (!existsSync(file)) return [];
  const names = new Set<string>();
  for (const line of readFileSync(file, "utf8").split("\n")) {
    const match = /^\s*\[mcp_servers\.([A-Za-z0-9_-]+)\]\s*$/.exec(line);
    if (match?.[1]) names.add(match[1]);
  }
  return [...names];
}

function readRecords(file: string): RunRecord[] {
  if (!existsSync(file)) return [];
  return readFileSync(file, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RunRecord);
}

/** Setup a User goes through, from "agent installed" to a first cited answer, counted by hand (B.4). */
const SETUP: Record<string, string> = {
  A0: "0: built in (choose a chat model, once).",
  M: "Codex: 1 command (`codex mcp add incarnamind -- <command>`) or 1 config block; 0 prompts in the bake-off's settings (Tools approved by Codex's read-only annotations). Claude Code: 1 command (`claude mcp add`) + 1 permission prompt per Tool unless allowed.",
  Q: "As M (the same server; only `ask` shown).",
  C: "Codex: put the CLI on PATH, add the Agent Skill folder, and allow the socket from the sandbox (see the run's settings): 3 steps.",
  H: "None for the User once IncarnaMind hosts the agent; the agent must be installed and signed in.",
  CC: "As M, in Claude Code.",
};

test("agent integration bake-off", async () => {
  const config = readAgentsConfig(root);
  const bakeoff = loadBakeoff(root);
  const tasks: Task[] = config.tasks
    ? bakeoff.tasks.filter((task) => config.tasks?.includes(task.id))
    : bakeoff.tasks;
  if (tasks.length === 0) throw new Error("No task matches INCARNAMIND_EVAL_AGENTS_TASKS.");
  const startedAt = new Date();
  const stamp = startedAt.toISOString().replaceAll(":", "-").slice(0, 19);
  const sessionDir = config.resume ?? join(config.resultsDir, `agents-${stamp}`);
  const eventsDir = join(sessionDir, "events");
  await mkdir(eventsDir, { recursive: true });
  const runsFile = join(sessionDir, "runs.jsonl");
  const log = createLog(startedAt.getTime());
  const records = readRecords(runsFile);
  log(`Session ${sessionDir}: ${records.length} runs already recorded`);

  const scratch = join(config.scratch, basename(sessionDir));
  const runsDir = join(scratch, "runs");
  await mkdir(runsDir, { recursive: true });
  const shim = join(root, "eval/agents/bin/shim.ts");
  const cli = join(root, "eval/agents/bin/incarnamind.ts");
  const meter = createUsageMeter();
  const canary = await listenForCanary(bakeoff.canaries.injection.port);
  const fixture = await openFixtureLibrary({
    bakeoff,
    cacheDir: config.cacheDir,
    parentDir: join(scratch, "data"),
    localNumCtx: config.localNumCtx,
    adapters: {
      createChatModel: meter.factory,
      agents: {
        launcher: { command: process.execPath, args: [...NODE_FLAGS, shim] },
        version: "bakeoff",
      },
    },
    keep: config.keep,
    log,
  });
  const { core } = fixture.library;
  const db = openDatabase(join(fixture.library.dataDir, DATABASE_FILE));
  try {
    // The evaluation's model, and the outside agents' pairings: the consent a User gives in
    // Settings. Nothing else may leave the computer (tagging, for one, is declined).
    core.on("consent.requested", (request) => {
      const accept = request.flow.id === "chat" || request.flow.id === "agents";
      core
        .respondToConsent(request.requestId, accept)
        .catch((error: unknown) => console.error(error));
    });
    const wantsLocal = config.arms.includes("A0-L");
    const local = wantsLocal
      ? await core.saveChatProvider({
          kind: "ollama",
          modelId: config.localModel,
          baseUrl: config.ollamaUrl,
        })
      : null;
    // Saved last, so it is the User's chosen chat model, which `ask` answers on.
    const cloud = await core.saveChatProvider({
      kind: config.chat.kind,
      modelId: config.chat.modelId,
      ...(config.chat.apiKey !== null && { apiKey: config.chat.apiKey }),
      ...(config.chat.baseUrl !== null && { baseUrl: config.chat.baseUrl }),
    });
    // The User's consent to send Questions to that model, given once, as asking a first Question
    // in the app would: `ask` won't prompt for it.
    if (cloud.service) await core.allowDataFlow("chat", cloud.service.id);
    await core.setAgentsEnabled(true);

    // The canary and the unanswerable task must really be outside "In scope".
    const inScopeIds = bakeoff.documents
      .filter((document) => document.folder === "in-scope")
      .flatMap((document) => {
        const added = fixture.documents.get(document.key);
        return added ? [added.id] : [];
      });
    for (const id of inScopeIds) {
      const text = fixture.library
        .pageTexts(id)
        .map((page) => page.text)
        .join("\n")
        .toUpperCase();
      for (const canary_ of bakeoff.canaries.scope) {
        if (text.includes(canary_.toUpperCase())) {
          throw new Error(`The scope canary "${canary_}" is in an In-scope Document (${id}).`);
        }
      }
    }

    const session: Session = {
      config,
      bakeoff,
      fixture,
      core,
      db,
      meter,
      canary,
      cloud,
      local,
      node: process.execPath,
      shim,
      cli,
      runsDir,
      eventsDir,
      ownerSkills: await ownerSkills(config.codexHome),
      ownerMcpServers: ownerMcpServers(config.codexHome),
      privatePaths: [
        fixture.library.dataDir,
        join(root, "eval/agents/fixtures"),
        join(root, "eval/grouping/fixtures"),
        join(root, "data"),
      ],
      log,
    };
    log(
      `Codex runs switch off ${session.ownerSkills.length} of the owner's skills and ${session.ownerMcpServers.length} MCP servers; a temporary HOME; CODEX_HOME for the sign-in only`,
    );

    // Warm: the index, the reranker and `ask`'s path, as in a running app. Not recorded.
    const probe: ProbeRow[] = await probeRetrieval({
      core,
      fixture,
      tasks,
      inScopeDocumentIds: inScopeIds,
      cacheDir: config.cacheDir,
      log,
    });
    // One `ask` before the timed runs, so none pays for loading the Answer path.
    const warm = await core.connectAgent({ name: "warm-up", folderIds: fixture.scope.folderIds });
    const client = await openSocketClient(warm.launch.env);
    try {
      const asked = await client.call("ask", { question: "What is a Transformer?" });
      log(
        `Warm-up ask: ${asked.isError ? `refused: ${asked.text}` : "answered"} in ${asked.ms.toFixed(0)} ms`,
      );
      if (asked.isError) throw new Error(`ask doesn't work: ${asked.text}`);
    } finally {
      client.close();
      await core.disconnectAgent(warm.pairing.id);
    }
    meter.take();

    const done = new Set(records.map((record) => `${record.arm}/${record.task}/${record.repeat}`));
    const timesFor = (arm: Arm) => (arm === "A0-L" || arm === "CC" ? 1 : config.repeats);
    const jobs = shuffle(
      config.arms
        .filter((arm) => arm !== "A0-L")
        .flatMap((arm) =>
          tasks.flatMap((task) =>
            Array.from({ length: timesFor(arm) }, (_, index) => ({ arm, task, repeat: index + 1 })),
          ),
        )
        .filter(({ arm, task, repeat }) => !done.has(`${arm}/${task.id}/${repeat}`)),
      startedAt.getTime() % 2 ** 31,
    );
    log(`${jobs.length} runs to go, in shuffled order`);
    const stopped = new Map<Arm, string>();
    const append = async (record: RunRecord) => {
      records.push(record);
      await appendFile(runsFile, `${JSON.stringify(record)}\n`);
      const m = record.metrics;
      log(
        `${record.runId}: ${record.status}, task ${(record.taskMs / 1000).toFixed(1)} s (start-up ${(record.startupMs / 1000).toFixed(1)} s), ${record.modelCalls} model calls, ${m.found}/${m.citations} found, ${m.inventedTokens} invented, uncited claims ${m.uncitedClaims ?? "–"}, expected ${m.expectedHit ?? "–"}, correct ${m.correct ?? "–"}${m.scopeLeak.length > 0 ? `, SCOPE LEAK ${m.scopeLeak.join("; ")}` : ""}`,
      );
    };
    for (const job of jobs) {
      if (stopped.has(job.arm)) continue;
      const record = await runOne(session, job.arm, job.task, job.repeat);
      await append(record);
      // The plan's limit: stop every Codex arm at its first limit message, and say so.
      if (record.error && /usage limit|rate limit|quota|too many requests/i.test(record.error)) {
        for (const arm of ["M", "Q", "C", "H", "MQ"] as Arm[]) stopped.set(arm, record.error);
        log(`Stopping the Codex arms: ${record.error}`);
      }
      // Stop early when a gate clearly fails: a scope leak, or Citations clearly below the bar.
      const mine = records.filter((each) => each.arm === job.arm);
      const summary = summarise(job.arm, mine);
      if (summary.scopeLeaks.length > 0) {
        stopped.set(job.arm, "a scope leak (gate 2)");
        log(`Stopping ${job.arm}: a scope leak through IncarnaMind, a bug to fix first`);
      } else if (
        mine.length >= tasks.length &&
        summary.citations >= 20 &&
        (summary.foundRate ?? 0) < 0.8
      ) {
        stopped.set(job.arm, `found-rate ${(summary.foundRate ?? 0).toFixed(2)} (gate 1)`);
        log(`Stopping ${job.arm}: gate 1 clearly fails`);
      }
    }

    if (config.arms.includes("A0-L")) {
      // Local mode, as a User who keeps everything on the computer: outside agents pause.
      await core.setLocalOnly(true);
      try {
        for (const task of tasks) {
          if (done.has(`A0-L/${task.id}/1`)) continue;
          await append(await runOne(session, "A0-L", task, 1));
        }
      } finally {
        await core.setLocalOnly(false);
        // One model at a time: unload it now.
        await fetch(`${config.ollamaUrl}/api/generate`, {
          method: "POST",
          body: JSON.stringify({ model: config.localModel, keep_alive: 0 }),
        }).catch(() => undefined);
      }
    }

    const markdown = await writeReport({
      dir: sessionDir,
      startedAt: startedAt.toISOString(),
      records,
      probe,
      meta: {
        "Codex model": `${config.codexModel}, reasoning ${config.codexEffort}`,
        "IncarnaMind's model": `${config.chat.kind}/${config.chat.modelId}`,
        "Local model": `${config.localModel}, num_ctx ${config.localNumCtx}`,
        Repeats: String(config.repeats),
        Stopped: [...stopped].map(([arm, why]) => `${arm}: ${why}`).join("; ") || "none",
        Gates: records.length
          ? [...new Set(records.map((record) => record.arm))]
              .map((arm) => {
                const g = gatesOf(
                  summarise(
                    arm,
                    records.filter((record) => record.arm === arm),
                  ),
                );
                return `${arm} ${g.integrity && g.scope && g.injection ? "pass" : "fail"}`;
              })
              .join(", ")
          : "–",
      },
      setup: SETUP,
    });
    console.log(markdown);
  } finally {
    db.close();
    await canary.close();
    await fixture.library.close();
    if (!config.keep) await rm(scratch, { recursive: true, force: true });
  }
});
