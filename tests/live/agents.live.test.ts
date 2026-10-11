/**
 * An outside agent's `ask` (#351) against real chat models: an MCP client
 * starts the shim, as Claude Code or Codex would, and asks in English and in
 * Chinese; IncarnaMind answers on the User's model and returns its
 * Citations as tokens, which `check_citations` then finds. Each model runs
 * only when its key (or, for Ollama, its name) is in the environment, and is
 * skipped otherwise, so `npx vitest run` makes no request without one.
 *
 *   INCARNAMIND_LIVE_ANTHROPIC_KEY   Claude (INCARNAMIND_LIVE_ANTHROPIC_MODEL, Haiku by default)
 *   INCARNAMIND_LIVE_OPENAI_KEY      OpenAI (INCARNAMIND_LIVE_OPENAI_MODEL, its quick-tasks model by default)
 *   INCARNAMIND_LIVE_OLLAMA_MODEL    a model in Ollama on this computer, e.g. qwen3.5:4b, unloaded after
 *
 * Set INCARNAMIND_LIVE_REPORT to a file to keep what each model gave back.
 */
import { appendFile } from "node:fs/promises";
import { CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";
import { describe, expect, test } from "vitest";
import {
  type ChatProviderKind,
  createAiSdkChatModel,
  createOllamaModels,
  type OllamaModels,
} from "../../src/core";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import { outputTokensFor } from "../../src/core/providers/ollamaModels";
import { agentsAdapter, connectLegacyAgent } from "../helpers/agents";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";

const OLLAMA_URL = "http://127.0.0.1:11434";
const MAX_NUM_CTX = 8_192;

function smallWindow(models: OllamaModels): OllamaModels {
  return {
    ...models,
    describe: async (baseUrl, model) => {
      const profile = await models.describe(baseUrl, model);
      if (!profile) return null;
      const numCtx = Math.min(profile.settings.numCtx, MAX_NUM_CTX);
      return {
        ...profile,
        settings: {
          ...profile.settings,
          numCtx,
          outputTokens: outputTokensFor(numCtx, profile.settings.think),
        },
      };
    },
  };
}

const LIVE: { name: string; kind: ChatProviderKind; apiKey: string | undefined; model: string }[] =
  [
    {
      name: "Anthropic",
      kind: "anthropic",
      apiKey: process.env.INCARNAMIND_LIVE_ANTHROPIC_KEY,
      model:
        process.env.INCARNAMIND_LIVE_ANTHROPIC_MODEL ??
        (catalogProvider("anthropic")?.roles.quickTasks as string),
    },
    {
      name: "OpenAI",
      kind: "openai",
      apiKey: process.env.INCARNAMIND_LIVE_OPENAI_KEY,
      model:
        process.env.INCARNAMIND_LIVE_OPENAI_MODEL ??
        (catalogProvider("openai")?.roles.quickTasks as string),
    },
    {
      name: "Ollama",
      kind: "ollama",
      apiKey: process.env.INCARNAMIND_LIVE_OLLAMA_MODEL ? "" : undefined,
      model: process.env.INCARNAMIND_LIVE_OLLAMA_MODEL ?? "",
    },
  ];

const TIDES = [
  "# Tides",
  "",
  "Spring tides happen at new moon and at full moon, when the Sun and the Moon pull together. Neap tides happen at the quarter moons, when they pull at right angles.",
].join("\n");
const CHAOXI =
  "# 潮汐\n\n大潮出现在新月和满月前后，这时太阳和月亮的引力叠加在一起。小潮出现在上弦月和下弦月前后。";

const QUESTIONS = ["When do spring tides happen?", "大潮在什么时候出现？"];

interface Answered {
  status: string;
  answer: string;
  citations: { cite: string; quote: string }[];
  unchecked: number;
}

describe.each(LIVE)("An outside agent's ask, live: $name", (live) => {
  test.skipIf(live.apiKey === undefined)(
    `${live.name}: answers in English and Chinese, every Citation a token that checks as found`,
    { timeout: 900_000 },
    async () => {
      const dataDir = await createTempDataFolder();
      const sources = await createTempDataFolder();
      const core = startCore(dataDir, {
        createChatModel: createAiSdkChatModel,
        agents: await agentsAdapter(dataDir),
        // What the model can do (Tools, its window), as the app looks it up, but never a window
        // over 8,192 tokens, so the test doesn't load a large one.
        ...(live.kind === "ollama" && { ollamaModels: smallWindow(createOllamaModels()) }),
      });
      await core.saveChatProvider({
        kind: live.kind,
        ...(live.apiKey && { apiKey: live.apiKey }),
        modelId: live.model,
      });
      // The User allowed the chat model before: `ask` never waits for them.
      for (const flow of await core.listRegisteredDataFlows()) {
        if (flow.id !== "chat") continue;
        for (const { flow: each } of flow.services)
          await core.allowDataFlow("chat", each.service.id);
      }
      const documents = await addAndProcess(core, [
        await writeSourceFile(sources, "Tides.md", TIDES),
        await writeSourceFile(sources, "潮汐.md", CHAOXI),
      ]);
      const coast = await core.createLibraryGroup({ name: "Coast", description: "The sea." });
      for (const document of documents) await core.assignDocumentGroup(document.id, coast.id);
      await core.setAgentsEnabled(true);
      const { launch } = await core.connectAgent({ name: "Live agent", folderIds: [coast.id] });
      const agent = await connectLegacyAgent(launch);

      try {
        for (const question of QUESTIONS) {
          const started = Date.now();
          const result = await agent.callTool(
            { name: "ask", arguments: { question } },
            CallToolResultSchema,
            { timeout: 600_000 },
          );
          const seconds = (Date.now() - started) / 1000;
          expect(result.isError ?? false).toBe(false);
          const answered = result.structuredContent as unknown as Answered;
          const checked = (
            await agent.callTool({ name: "check_citations", arguments: { text: answered.answer } })
          ).structuredContent as { citations: { check: string }[]; uncited: string[] };
          const report = {
            model: `${live.name} ${live.model}`,
            question,
            seconds,
            answered,
            checked,
          };
          if (process.env.INCARNAMIND_LIVE_REPORT) {
            await appendFile(process.env.INCARNAMIND_LIVE_REPORT, `${JSON.stringify(report)}\n`);
          }
          expect(answered.status).toBe("done");
          expect(answered.citations.length).toBeGreaterThan(0);
          for (const citation of answered.citations) {
            expect(answered.answer).toContain(citation.cite);
          }
          expect(checked.citations.every((each) => each.check === "found")).toBe(true);
        }
      } finally {
        if (live.kind === "ollama") {
          // One model at a time on this computer: unload it now.
          await fetch(`${OLLAMA_URL}/api/generate`, {
            method: "POST",
            body: JSON.stringify({ model: live.model, keep_alive: 0 }),
          }).catch(() => undefined);
        }
      }
    },
  );
});
