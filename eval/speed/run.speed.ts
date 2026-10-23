/**
 * `npm run eval:speed` (eval/speed/README.md): the speed baseline of #1624. Each flow in ./flows,
 * on each model, in each language, a few times, over the sample reports, with the built-in
 * reranker on this computer as Users have it. Prints the tables of ./report and writes them to
 * eval/results/speed-<time>/.
 *
 * INCARNAMIND_EVAL_SPEED_MODELS: the models, "<kind>/<model id>" comma-separated, for example
 *   "anthropic/claude-haiku-5-5,kimi/kimi-k2.6". Each kind's key is INCARNAMIND_EVAL_SPEED_KEY_<KIND>
 *   (INCARNAMIND_EVAL_SPEED_KEY_ANTHROPIC…), and its endpoint INCARNAMIND_EVAL_SPEED_ENDPOINT_<KIND>.
 * INCARNAMIND_EVAL_SPEED_FLOWS: flow ids to run, comma-separated (default: all).
 * INCARNAMIND_EVAL_SPEED_LANGUAGES: "en", "zh-CN" or both (default: both).
 * INCARNAMIND_EVAL_SPEED_RUNS: how many times each flow runs per model and language (default: 3).
 */
import { mkdir, writeFile } from "node:fs/promises";
import { loadavg } from "node:os";
import { join, resolve } from "node:path";
import { expect, test } from "vitest";
import { chatProviderKinds, type SaveChatProviderInput } from "../../src/core";
import { queryDatabase } from "../../tests/helpers/core";
import { reports } from "../everyday/asking";
import { readConfig } from "../lib/config";
import { createWorkerEmbedder } from "../lib/embedder";
import { openLibrary } from "../lib/library";
import { createLog } from "../lib/log";
import { createWorkerCrossEncoder } from "../lib/rerank";
import { FLOWS, type Language } from "./flows";
import { type FlowRun, flowTable, requestTable, type TimingRecord } from "./report";

const ROOT = resolve(__dirname, "../..");
const env = (name: string) => process.env[`INCARNAMIND_EVAL_SPEED_${name}`]?.trim() || null;
const list = (name: string) =>
  env(name)
    ?.split(",")
    .map((each) => each.trim())
    .filter(Boolean) ?? null;

function modelsAsked(): SaveChatProviderInput[] {
  return (list("MODELS") ?? []).map((each) => {
    const [kind, ...rest] = each.split("/");
    const found = chatProviderKinds.find((known) => known === kind);
    if (!found || found === "chatgpt" || rest.length === 0) {
      throw new Error(`INCARNAMIND_EVAL_SPEED_MODELS: "${each}" isn't "<kind>/<model id>".`);
    }
    const apiKey = env(`KEY_${found.toUpperCase().replaceAll("-", "_")}`);
    const endpoint = env(`ENDPOINT_${found.toUpperCase().replaceAll("-", "_")}`);
    return {
      kind: found,
      modelId: rest.join("/"),
      ...(apiKey && { apiKey }),
      ...(endpoint && { endpoint }),
    };
  });
}

test("the speed of every AI flow, on each model and language", async () => {
  const log = createLog();
  const config = readConfig(ROOT);
  const models = modelsAsked();
  if (models.length === 0) throw new Error("Set INCARNAMIND_EVAL_SPEED_MODELS.");
  const wanted = list("FLOWS");
  const flows = wanted ? FLOWS.filter((flow) => wanted.includes(flow.id)) : FLOWS;
  const languages = (list("LANGUAGES") ?? ["en", "zh-CN"]) as Language[];
  const runs = Number(env("RUNS") ?? "3");
  const results: FlowRun[] = [];
  const library = await openLibrary({
    name: "speed",
    embedder: createWorkerEmbedder(),
    modelCache: join(config.cacheDir, "models"),
    reranker: createWorkerCrossEncoder(config.rerankOnCpu),
    documents: await reports(),
    embeddings: false,
    keep: config.keepData,
    log,
  });
  try {
    const { core, dataDir } = library;
    // Setting the variables is the consent to send the requests and Passages to each model.
    core.on("consent.requested", (request) => void core.respondToConsent(request.requestId, true));
    for (const model of models) {
      const name = `${model.kind}/${model.modelId}`;
      await core.saveChatProvider(model);
      for (const flow of flows) {
        for (const language of languages) {
          await core.updateSettings({ user: { language } });
          for (let run = 1; run <= runs; run++) {
            const since = new Date().toISOString();
            log(
              `${name} · ${flow.id} · ${language} · ${run}/${runs} · load ${loadavg()[0]?.toFixed(1)}`,
            );
            let signs: FlowRun["signs"] = null;
            let error: string | null = null;
            try {
              signs = await flow.run({ core, language });
            } catch (thrown) {
              error = thrown instanceof Error ? thrown.message : String(thrown);
              log(`  failed: ${error}`);
            }
            const records = queryDatabase<TimingRecord>(
              dataDir,
              "SELECT * FROM ai_timings WHERE started_at >= ? ORDER BY started_at, rowid",
              [since],
            );
            results.push({ flow: flow.id, model: name, language, signs, error, records });
          }
        }
      }
    }
  } finally {
    await library.close();
  }
  const report = [
    "# Speed baseline",
    "",
    `${new Date().toISOString()} · ${runs} runs a flow, model and language · times are median / 90th percentile.`,
    "",
    "## Flows",
    "",
    flowTable(results),
    "",
    "## Requests and local steps",
    "",
    requestTable(results),
    "",
  ].join("\n");
  console.log(report);
  const out = join(config.resultsDir, `speed-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  await mkdir(out, { recursive: true });
  await writeFile(join(out, "report.md"), report);
  await writeFile(join(out, "report.json"), JSON.stringify(results, null, 2));
  log(`Report in ${out}`);
  expect(results.length).toBe(models.length * flows.length * languages.length * runs);
});
