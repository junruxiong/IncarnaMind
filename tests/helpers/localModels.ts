/**
 * A core whose chat model is a local model on the Ollama server stub (see
 * ./ollama), with the app's own model factory and model lookups, and the
 * models the stub plays.
 */
import type { Core } from "../../src/core";
import { createOllamaModels } from "../../src/core";
import { answerEnded } from "./citations";
import { createTempDataFolder, startCore } from "./core";
import { addAndProcess, writeSourceFile } from "./documents";
import { connectToMind, type MindClient } from "./mindClient";
import type { OllamaChatReply, OllamaModelStub, OllamaServer } from "./ollama";
import { waitForTagging } from "./tags";

export const GIB = 1024 ** 3;

/** Qwen3.5-4B as Ollama lists it: Tools and thinking, a long context, little model_info (an MLX build). */
export const QWEN35: OllamaModelStub = {
  name: "qwen3.5:4b",
  capabilities: ["completion", "vision", "thinking", "tools"],
  size: 3_973_305_013,
  modelInfo: {
    "general.architecture": "qwen3_5",
    "qwen3_5.block_count": 32,
    "qwen3_5.context_length": 262_144,
  },
  thinking: { values: [false, true], default: true },
};

/** The 2023 Mistral build: no Tools. */
export const MISTRAL: OllamaModelStub = {
  name: "mistral:latest",
  capabilities: ["completion"],
  size: 4_108_917_344,
  modelInfo: {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.embedding_length": 4096,
    "llama.context_length": 32_768,
  },
};

export const BGE_M3: OllamaModelStub = { name: "bge-m3:latest", capabilities: ["embedding"] };

export const TIDES = [
  "# Tides",
  "",
  "Spring tides happen at new moon and at full moon, when the Sun and the Moon pull in line.",
  "",
  "Neap tides happen at the quarter moons, when the Sun and the Moon pull at right angles.",
].join("\n");

/**
 * A core whose chat model is `model` on the Ollama stub, with `memory` bytes
 * of memory (all of it free) for choosing its window: 32 GB by default.
 * Documents are added, and tagged (the chat model tags them too), before it returns.
 */
export async function setUpLocalModel(
  ollama: { baseUrl: string },
  model: string,
  { memory = 32 * GIB, documents = [] as { name: string; contents: string }[] } = {},
) {
  const core = startCore(await createTempDataFolder(), {
    // The real model factory and lookups, against the stub.
    createChatModel: undefined,
    ollamaModels: createOllamaModels({ memoryBytes: () => memory }),
  });
  await core.saveChatProvider({ kind: "ollama", baseUrl: ollama.baseUrl, modelId: model });
  if (documents.length > 0) {
    const sources = await createTempDataFolder();
    const paths = await Promise.all(
      documents.map((each) => writeSourceFile(sources, each.name, each.contents)),
    );
    const added = await addAndProcess(core, paths);
    await waitForTagging(
      core,
      added.map((document) => document.id),
      (document) => document.tagging !== "pending" && document.tagging !== "tagging",
    );
  }
  const mind = await core.createMind({ title: "Tides" });
  const client = await connectToMind(core, mind.id);
  return { core, mind, client };
}

/** A chat request's messages. */
export const messagesOf = (body: Record<string, unknown>) =>
  (body.messages as { role: string; content: string }[]) ?? [];

/** A request that writes an Answer, rather than tags a Document. */
export const isAnswer = (body: Record<string, unknown>) =>
  messagesOf(body).some(
    (message) => message.role === "system" && message.content.startsWith("You answer Questions"),
  );

/** Replies to Answer requests with `reply` (counting only those), and to tagging with no Tags. */
export function answering(
  reply: (body: Record<string, unknown>, index: number) => OllamaChatReply,
) {
  let answered = 0;
  return (body: Record<string, unknown>): OllamaChatReply =>
    isAnswer(body) ? reply(body, answered++) : { content: JSON.stringify({ tags: [] }) };
}

/** The Answer requests the stub got. */
export const answerChats = (ollama: OllamaServer) =>
  ollama.chats.filter((chat) => isAnswer(chat.body));

/** Asks a Question already written and waits for its Answer to end. */
export async function askAndEnd(
  core: Core,
  client: MindClient,
  mindId: string,
  questionId: string,
) {
  await client.settled();
  const result = await core.askQuestion({ mindId, questionId });
  if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
  return { answerId: result.answerId, ended: await answerEnded(core, result.answerId) };
}
