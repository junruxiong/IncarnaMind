import { Tokenizer } from "@huggingface/tokenizers";
import { describe, expect, test } from "vitest";
import { BUILT_IN_RERANKING_MODEL, type CrossEncoder, DEFAULT_RERANK_MODELS } from "../../src/core";
import {
  type CrossEncoderChannel,
  type CrossEncoderRequest,
  type CrossEncoderResponse,
  createChannelCrossEncoder,
  serveCrossEncoder,
} from "../../src/core/reranking/channel";
import { createFakeCrossEncoder, fakeRelevance } from "../../src/core/reranking/fake";
import { createRerankingModel, rerankText } from "../../src/core/reranking/index";
import { pairIds } from "../../src/core/reranking/onnx";
import { createTempDataFolder, NO_MODEL_FILES } from "../helpers/core";

const FILES = { model: "m.onnx", tokenizer: "t.json", tokenizerConfig: "c.json", maxTokens: 512 };

/** A tiny tokenizer with XLM-RoBERTa's pair template: "<s> A </s></s> B </s>". */
function tinyTokenizer(): Tokenizer {
  const vocab: Record<string, number> = { "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3 };
  for (const word of "what is the capital of france paris lies on seine a b c d e f g h".split(
    " ",
  )) {
    vocab[word] = Object.keys(vocab).length;
  }
  const special = (id: number, content: string) => ({
    id,
    content,
    special: true,
    single_word: false,
    lstrip: false,
    rstrip: false,
    normalized: false,
  });
  return new Tokenizer(
    {
      version: "1.0",
      added_tokens: [special(0, "<s>"), special(2, "</s>")],
      normalizer: null,
      pre_tokenizer: { type: "Whitespace" },
      post_processor: {
        type: "RobertaProcessing",
        sep: ["</s>", 2],
        cls: ["<s>", 0],
        trim_offsets: true,
        add_prefix_space: false,
      },
      decoder: null,
      model: { type: "WordLevel", vocab, unk_token: "<unk>" },
    },
    { model_max_length: 512 },
  );
}

describe("The reranking models", () => {
  test("the built-in model is pinned, has a permissive licence and isn't a service", () => {
    const model = BUILT_IN_RERANKING_MODEL;
    expect(model.licence).toMatch(/^(Apache-2\.0|MIT)$/);
    expect(model.source.baseUrl).toMatch(/\/resolve\/[0-9a-f]{40}\/$/);
    expect(model.source.files.map((file) => file.path)).toEqual(
      expect.arrayContaining([
        model.files.model,
        model.files.tokenizer,
        model.files.tokenizerConfig,
      ]),
    );
    for (const file of model.source.files) expect(file.sha256).toMatch(/^[0-9a-f]{64}$/);
    // The built-in model isn't a service: it has no default model to name.
    expect(Object.keys(DEFAULT_RERANK_MODELS)).toEqual(["cohere", "voyage"]);
  });
});

describe("A query and a Passage, as one pair of tokens", () => {
  test("follow the model's pair template", () => {
    const tokenizer = tinyTokenizer();
    expect(pairIds(tokenizer, "what is", "paris lies", 512)).toEqual(
      tokenizer.encode("what is", { text_pair: "paris lies" }).ids,
    );
  });

  test("a long pair loses the end of the Passage, keeping the end-of-text token", () => {
    const tokenizer = tinyTokenizer();
    const ids = pairIds(tokenizer, "what is", "a b c d e f g h", 10);
    expect(ids).toHaveLength(10);
    const vocab = tokenizer.get_vocab();
    // <s> what is </s></s> a b c d </s>
    expect(ids).toEqual([
      0,
      vocab.get("what"),
      vocab.get("is"),
      2,
      2,
      vocab.get("a"),
      vocab.get("b"),
      vocab.get("c"),
      vocab.get("d"),
      2,
    ]);
  });

  test("a very long query is cut too, so the Passage keeps most of the room", () => {
    const tokenizer = tinyTokenizer();
    const ids = pairIds(tokenizer, "a b c d e f g h a b c d e f g h", "paris lies on seine", 16);
    expect(ids).toHaveLength(12);
    const vocab = tokenizer.get_vocab();
    // The query keeps a quarter of 16 tokens, the Passage all of its own.
    expect(ids.slice(0, 6)).toEqual([0, ...["a", "b", "c", "d"].map((w) => vocab.get(w)), 2]);
    expect(ids.slice(-5)).toEqual([
      ...["paris", "lies", "on", "seine"].map((w) => vocab.get(w)),
      2,
    ]);
  });
});

describe("The built-in reranking model", () => {
  const hits = [
    { id: "a", documentName: "Ships", text: "Boats float on water." },
    { id: "b", documentName: "Lighthouse", text: "A lighthouse guides ships at night." },
    { id: "c", documentName: "Bread", text: "Bread rises in the oven." },
    { id: "d", documentName: "Cakes", text: "Cakes rise in the oven too." },
  ];

  test("orders hits by the model's score, from 0 to 1, keeping ties in their order", async () => {
    const dataDir = await createTempDataFolder();
    const statuses: string[] = [];
    const model = createRerankingModel({
      definition: BUILT_IN_RERANKING_MODEL,
      source: NO_MODEL_FILES,
      dataDir,
      crossEncoder: createFakeCrossEncoder(),
      emitStatus: (status) => statuses.push(status.state),
    });

    const ranked = await model.rerank("lighthouse ships", hits);

    expect(ranked?.map((hit) => hit.id)).toEqual(["b", "a", "c", "d"]);
    for (const hit of ranked ?? []) {
      expect(hit.score).toBeGreaterThan(0);
      expect(hit.score).toBeLessThan(1);
    }
    // c and d score the same: they keep search's order.
    expect(ranked?.[2]?.score).toBe(ranked?.[3]?.score);
    model.close();
  });

  test("reads each hit as its Document's name, then its text, normalised", async () => {
    const dataDir = await createTempDataFolder();
    const read: string[][] = [];
    const crossEncoder: CrossEncoder = {
      load: async () => {},
      score: async (query, texts) => {
        read.push([query, ...texts]);
        return texts.map(() => 0);
      },
      close: () => {},
    };
    const model = createRerankingModel({
      definition: BUILT_IN_RERANKING_MODEL,
      source: NO_MODEL_FILES,
      dataDir,
      crossEncoder,
      emitStatus: () => {},
    });

    await model.rerank("Ｑｕｅｒｙ", [{ documentName: "⼤ Report", text: "ﬁne text" }]);

    // NFKC, the Kangxi radical folded, and no space next to a CJK character, as for embedding.
    expect(read).toEqual([["Query", "大Report\nfine text"]]);
    expect(rerankText({ documentName: "Notes", text: "Body" })).toBe("Notes\nBody");
    model.close();
  });

  test("reads nothing while its files aren't downloaded", async () => {
    const dataDir = await createTempDataFolder();
    let scored = false;
    const model = createRerankingModel({
      definition: BUILT_IN_RERANKING_MODEL,
      source: {
        baseUrl: "http://127.0.0.1:9/",
        files: [{ path: "m.onnx", size: 10, sha256: "0".repeat(64) }],
      },
      dataDir,
      crossEncoder: {
        load: async () => {},
        score: async (_query, texts) => {
          scored = true;
          return texts.map(() => 0);
        },
        close: () => {},
      },
      emitStatus: () => {},
    });

    expect(model.status().state).toBe("not-downloaded");
    expect(await model.rerank("q", hits)).toBeNull();
    expect(scored).toBe(false);
    model.close();
  });

  test("the fake scores the share of the query's words a text has", () => {
    expect(fakeRelevance("lighthouse ships", "A lighthouse guides ships")).toBe(1);
    expect(fakeRelevance("lighthouse ships", "Boats and ships")).toBe(0.5);
    expect(fakeRelevance("灯塔", "海边的灯塔")).toBe(1);
    expect(fakeRelevance("", "anything")).toBe(0);
  });
});

/** Both ends of the channel in one process, as a utility process would serve them. */
function linkedChannel(served: CrossEncoder) {
  let onRequest: (request: CrossEncoderRequest) => void = () => {};
  let onResponse: (response: CrossEncoderResponse) => void = () => {};
  let onExit: (code: number | null) => void = () => {};
  const started: string[] = [];
  serveCrossEncoder(served, {
    onRequest: (listener) => {
      onRequest = listener;
    },
    respond: (response) => queueMicrotask(() => onResponse(response)),
  });
  const start = (): CrossEncoderChannel => {
    started.push("started");
    return {
      send: (request) => queueMicrotask(() => onRequest(request)),
      onResponse: (listener) => {
        onResponse = listener;
      },
      onExit: (listener) => {
        onExit = listener;
      },
      stop: () => onExit(0),
    };
  };
  return { start, started, crash: (code: number) => onExit(code) };
}

describe("The reranking channel", () => {
  test("loads once, then scores texts in order, over messages", async () => {
    const link = linkedChannel(createFakeCrossEncoder());
    const crossEncoder = createChannelCrossEncoder(link.start);

    await crossEncoder.load(FILES);
    await crossEncoder.load(FILES);
    expect(await crossEncoder.score("ships", ["ships at sea", "bread"])).toEqual([1, 0]);
    expect(link.started).toHaveLength(1);
  });

  test("scores fail before a load, and errors come back as errors", async () => {
    const failing: CrossEncoder = {
      load: async () => {},
      score: async () => {
        throw new Error("The model failed on this text.");
      },
      close: () => {},
    };
    const crossEncoder = createChannelCrossEncoder(linkedChannel(failing).start);

    await expect(crossEncoder.score("q", ["t"])).rejects.toThrow(/isn't loaded/);
    await crossEncoder.load(FILES);
    await expect(crossEncoder.score("q", ["t"])).rejects.toThrow("The model failed on this text.");
  });

  test("when the process stops, pending requests fail and the next load starts a new one", async () => {
    let release = () => {};
    const slow: CrossEncoder = {
      load: async () => {},
      score: (_query, texts) =>
        new Promise((resolve) => {
          release = () => resolve(texts.map(() => 0));
        }),
      close: () => {},
    };
    const link = linkedChannel(slow);
    const crossEncoder = createChannelCrossEncoder(link.start);
    await crossEncoder.load(FILES);

    const scoring = crossEncoder.score("q", ["t"]);
    await new Promise((resolve) => setTimeout(resolve, 10));
    link.crash(1);
    await expect(scoring).rejects.toThrow(/stopped \(exit code 1\)/);
    release();

    await crossEncoder.load(FILES);
    expect(link.started).toHaveLength(2);
  });
});
