/**
 * A Question asked beside a Mind (#544, #564) against real models: a short
 * Mind, read whole, and one of over 50,000 tokens, read in part, the part a
 * Question is about far from where the User is, in English and in Chinese;
 * then a second Question in the same conversation, which a provider's prompt
 * cache can answer from the Mind it already read. Each model runs only when
 * its key (or, for Ollama, its name) is in the environment, and is skipped
 * otherwise; the variables are those of ./assistantAnswers.live.test.ts.
 *
 * Set INCARNAMIND_LIVE_REPORT to a file to keep, for each Question, what the
 * Answer said, how much of the Mind it read, and the tokens and cost of its
 * requests, the cached ones too.
 */
import { appendFile } from "node:fs/promises";
import { describe, expect, onTestFinished, test } from "vitest";
import type { ChatProviderKind, ConversationAnswer, Core } from "../../src/core";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import {
  createOllamaModels,
  type OllamaModels,
  outputTokensFor,
} from "../../src/core/providers/ollamaModels";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { heading, note, writeMind } from "../helpers/minds";

interface Live {
  name: string;
  kind: ChatProviderKind;
  apiKey: string | undefined;
  model: string;
  local: boolean;
}

const LIVE: Live[] = [
  {
    name: "Anthropic",
    kind: "anthropic",
    apiKey: process.env.INCARNAMIND_LIVE_ANTHROPIC_KEY,
    model:
      process.env.INCARNAMIND_LIVE_ANTHROPIC_MODEL ??
      (catalogProvider("anthropic")?.roles.quickTasks as string),
    local: false,
  },
  {
    name: "OpenAI",
    kind: "openai",
    apiKey: process.env.INCARNAMIND_LIVE_OPENAI_KEY,
    model:
      process.env.INCARNAMIND_LIVE_OPENAI_MODEL ??
      (catalogProvider("openai")?.roles.quickTasks as string),
    local: false,
  },
  {
    name: "Ollama",
    kind: "ollama",
    apiKey: process.env.INCARNAMIND_LIVE_OLLAMA_MODEL ? "" : undefined,
    model: process.env.INCARNAMIND_LIVE_OLLAMA_MODEL ?? "",
    local: true,
  },
  {
    name: "Ollama cloud",
    kind: "ollama",
    apiKey: process.env.INCARNAMIND_LIVE_OLLAMA_CLOUD_MODEL ? "" : undefined,
    model: process.env.INCARNAMIND_LIVE_OLLAMA_CLOUD_MODEL ?? "",
    local: false,
  },
];

/** The app's own look at a model in Ollama, a local one's context kept to 8,192 tokens. */
function ollamaAsTheAppSees(): OllamaModels {
  const app = createOllamaModels();
  return {
    async describe(baseUrl, model) {
      const profile = await app.describe(baseUrl, model);
      if (!profile) return null;
      const numCtx = Math.min(profile.settings.numCtx, 8_192);
      const outputTokens = outputTokensFor(numCtx, profile.settings.think);
      return { ...profile, settings: { ...profile.settings, numCtx, outputTokens } };
    },
    loaded: (baseUrl, model, numCtx) => app.loaded(baseUrl, model, numCtx),
  };
}

const STATIONS = ["Avonmouth", "Portishead", "Newport", "Cardiff", "Barry", "Minehead"];

/** About 500 tokens of field notes from one station, in English. */
const englishNote = (index: number) => {
  const station = STATIONS[index % STATIONS.length];
  const day = (index % 28) + 1;
  return [
    `Visit ${index}: ${station}, ${day} May.`,
    "The gauge was read every quarter of an hour from the slipway, and the readings were copied into the logbook before the next tide.",
    "Wind came from the south-west, light to moderate, with a long swell that made the lower readings hard to take by eye.",
    "Sediment samples were bagged at the high-water line and at mid-tide, each labelled with the time and the depth of the sample.",
  ]
    .join(" ")
    .repeat(5);
};

/** About 500 tokens of field notes from one station, in Chinese. */
const chineseNote = (index: number) => {
  const station = ["盐官", "老盐仓", "大缺口", "九溪", "萧山", "海宁"][index % 6];
  return [
    `第${index}次观测：${station}，五月${(index % 28) + 1}日。`,
    "每一刻钟在码头读一次潮位尺，下一次涨潮前把读数抄进记录本。",
    "风从西南来，风力轻到中等，涌浪很长，低潮时的读数很难用肉眼看清。",
    "在高潮线和半潮处各取一份泥沙样本，每份都写上时间和取样深度。",
  ]
    .join("")
    .repeat(4);
};

const LANGUAGES = [
  {
    language: "en",
    short: ["Spring tides come at new and full moon.", "Neap tides come at the quarter moons."],
    far: "The Severn bore reached 2.4 metres at Minsterworth on 3 March, the highest we recorded.",
    noteOf: englishNote,
    questions: ["What height did the bore reach, and where?", "On which day was that?"],
    summarise: "Summarise my notes",
  },
  {
    language: "zh-CN",
    short: ["大潮出现在新月和满月。", "小潮出现在上弦月和下弦月。"],
    far: "钱塘江大潮在九月十八日于盐官观潮点达到三点二米，是我们记录到的最高潮。",
    noteOf: chineseNote,
    questions: ["钱塘江大潮在哪里达到多高？", "那是哪一天？"],
    summarise: "总结我的笔记",
  },
] as const;

/** Asks in the conversation (a new one when null) beside `mindId`, and waits until its Answer has ended. */
async function ask(core: Core, conversationId: string | null, text: string, mindId: string) {
  const started = Date.now();
  const result = await core.askInConversation({ conversationId, text, mindId });
  if (!result.asked) throw new Error(`Not asked: ${JSON.stringify(result)}`);
  await new Promise<void>((resolve) => {
    const stops = [
      core.on("answer.finished", ({ answerId }) => answerId === result.answerId && done()),
      core.on("answer.failed", ({ answerId }) => answerId === result.answerId && done()),
    ];
    function done() {
      for (const stop of stops) stop();
      resolve();
    }
  });
  const { messages } = await core.getConversation(result.conversation.id);
  const answer = messages.find((message) => message.id === result.answerId);
  if (answer?.kind !== "answer") throw new Error("No Answer.");
  return { answer, conversationId: result.conversation.id, seconds: (Date.now() - started) / 1000 };
}

interface Requests {
  input: number;
  cached: number;
  written: number;
  output: number;
  cost: number;
}

/** What the chat requests recorded since `after` (rowid) counted and cost. */
function requestsSince(dataDir: string, after: number): Requests & { last: number } {
  const rows = queryDatabase<{
    rowid: number;
    input_tokens: number | null;
    cached_input_tokens: number | null;
    cache_write_tokens: number | null;
    output_tokens: number | null;
    cost: number | null;
  }>(
    dataDir,
    "SELECT rowid, input_tokens, cached_input_tokens, cache_write_tokens, output_tokens, cost FROM model_requests WHERE rowid > ? AND flow = 'chat' ORDER BY rowid",
    [after],
  );
  return rows.reduce(
    (sum, row) => ({
      input: sum.input + (row.input_tokens ?? 0),
      cached: sum.cached + (row.cached_input_tokens ?? 0),
      written: sum.written + (row.cache_write_tokens ?? 0),
      output: sum.output + (row.output_tokens ?? 0),
      cost: sum.cost + (row.cost ?? 0),
      last: Math.max(sum.last, row.rowid),
    }),
    { input: 0, cached: 0, written: 0, output: 0, cost: 0, last: after },
  );
}

const readOf = (answer: ConversationAnswer) =>
  answer.read
    ? answer.read.partial
      ? `read part (${Math.round((answer.steps?.readShare ?? 0) * 100)}%)`
      : "read whole"
    : "read none";

describe.each(LIVE)("A Question beside a Mind, live: $name", (live) => {
  test.skipIf(live.apiKey === undefined)(
    `${live.name}: a short Mind is read whole, a long one in part with the section asked about, and a second Question reads it again`,
    { timeout: 1_200_000 },
    async () => {
      const dataDir = await createTempDataFolder();
      const core = startCore(dataDir, {
        createChatModel: createAiSdkChatModel,
        ...(live.kind === "ollama" && {
          ollamaModels: live.local ? ollamaAsTheAppSees() : createOllamaModels(),
        }),
      });
      core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
      await core.saveChatProvider({
        kind: live.kind,
        ...(live.apiKey && { apiKey: live.apiKey }),
        modelId: live.model,
      });
      if (live.local) await core.setLocalOnly(true);

      const report: string[] = [`## ${live.name} · ${live.model}`];
      onTestFinished(async () => {
        if (process.env.INCARNAMIND_LIVE_REPORT)
          await appendFile(process.env.INCARNAMIND_LIVE_REPORT, `${report.join("\n")}\n\n`);
      });
      let seen = 0;
      const line = (label: string, answer: ConversationAnswer, seconds: number) => {
        const spent = requestsSince(dataDir, seen);
        seen = spent.last;
        report.push(
          `- ${label}: ${answer.status}${answer.errorKind ? ` (${answer.errorKind})` : ""}, ${readOf(answer)}, ${seconds.toFixed(1)} s; input ${spent.input} (cached ${spent.cached}, written ${spent.written}), output ${spent.output}, $${spent.cost.toFixed(5)}`,
          `  > ${answer.text.replace(/\s+/g, " ").slice(0, 300)}`,
        );
      };

      for (const each of LANGUAGES) {
        const short = await core.createMind({ title: each.language === "en" ? "Tides" : "潮汐" });
        const shortClient = await connectToMind(core, short.id);
        writeMind(
          shortClient,
          each.short.map((text) => note(text)),
        );
        await shortClient.settled();
        const summary = await ask(core, null, each.summarise, short.id);
        line(`${each.language} short Mind, "${each.summarise}"`, summary.answer, summary.seconds);
        expect(summary.answer.status).toBe("done");

        const long = await core.createMind({
          title: each.language === "en" ? "Field notes" : "观测记录",
        });
        const longClient = await connectToMind(core, long.id);
        const notes = Array.from({ length: 120 }, (_, index) => note(each.noteOf(index + 1)));
        writeMind(longClient, [
          heading(1, each.language === "en" ? "Field notes" : "观测记录"),
          ...notes.slice(0, 9),
          note(each.far),
          ...notes.slice(9),
        ]);
        await longClient.settled();
        const [first, second] = each.questions;
        const asked = await ask(core, null, first, long.id);
        line(`${each.language} long Mind, "${first}"`, asked.answer, asked.seconds);
        const again = await ask(core, asked.conversationId, second, long.id);
        line(`${each.language} long Mind, then "${second}"`, again.answer, again.seconds);
        expect(asked.answer.status).toBe("done");
        expect(asked.answer.read?.partial).toBe(true);
      }
    },
  );
});
