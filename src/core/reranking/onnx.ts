/**
 * Runs a built-in reranking model, a cross-encoder, with onnxruntime-node and
 * the Hugging Face tokenizer, on the CPU. Like the embedding model (see
 * ../embedding/onnx), it blocks whatever thread it runs on, so the desktop
 * app runs it in an Electron utility process (src/main/rerankerProcess.ts).
 *
 * A cross-encoder reads the query and a text together, as one pair
 * ("<s> query </s></s> text </s>" for the XLM-RoBERTa models), and gives one
 * logit: how well the text answers the query. A pair longer than the model
 * reads loses the end of the text, keeping the end-of-text token, as
 * sentence-transformers' CrossEncoder cuts it. One pair at a time, like the
 * embedding model: with int8 models, batching makes a score depend on the
 * other texts in the batch.
 */
import { readFile } from "node:fs/promises";
import type { CrossEncoder, RerankingModelFiles } from "../adapters";

type OnnxRuntime = typeof import("onnxruntime-node");
type InferenceSession = import("onnxruntime-node").InferenceSession;
type Tokenizer = import("@huggingface/tokenizers").Tokenizer;

interface Loaded {
  key: string;
  session: InferenceSession;
  tokenizer: Tokenizer;
  maxTokens: number;
  ort: OnnxRuntime;
}

/** A query keeps at most this share of the tokens a pair may have; the text gets the rest. */
const QUERY_SHARE = 0.25;

/** The token ids of the pair (query, text), cut to `maxTokens`. */
export function pairIds(tokenizer: Tokenizer, query: string, text: string, maxTokens: number) {
  let question = query;
  const queryTokens = tokenizer.encode(query, { add_special_tokens: false }).ids;
  const queryLimit = Math.floor(maxTokens * QUERY_SHARE);
  // A very long query is cut too, so the text still has room.
  if (queryTokens.length > queryLimit) {
    question = tokenizer.decode(queryTokens.slice(0, queryLimit), { skip_special_tokens: true });
  }
  const { ids } = tokenizer.encode(question, { text_pair: text });
  // The text comes last, followed by one end-of-text token: cut the text's end, keep that token.
  return ids.length > maxTokens ? [...ids.slice(0, maxTokens - 1), ids.at(-1) as number] : ids;
}

export function createOnnxCrossEncoder(): CrossEncoder {
  let loaded: Loaded | undefined;
  let loading: Promise<Loaded> | undefined;

  async function load(files: RerankingModelFiles): Promise<Loaded> {
    // Loaded only when needed: the native runtime is large.
    const ort = await import("onnxruntime-node");
    const { Tokenizer } = await import("@huggingface/tokenizers");
    const [tokenizerJson, tokenizerConfig] = await Promise.all([
      readFile(files.tokenizer, "utf8"),
      readFile(files.tokenizerConfig, "utf8"),
    ]);
    const tokenizer = new Tokenizer(JSON.parse(tokenizerJson), JSON.parse(tokenizerConfig));
    const session = await ort.InferenceSession.create(files.model, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });
    return { key: JSON.stringify(files), session, tokenizer, maxTokens: files.maxTokens, ort };
  }

  async function scoreOne(model: Loaded, query: string, text: string): Promise<number> {
    const { session, tokenizer, maxTokens, ort } = model;
    const ids = pairIds(tokenizer, query, text, maxTokens);
    const length = ids.length;
    const tensor = (values: BigInt64Array) => new ort.Tensor("int64", values, [1, length]);
    const feeds: Record<string, import("onnxruntime-node").Tensor> = {
      input_ids: tensor(BigInt64Array.from(ids, (id) => BigInt(id))),
      attention_mask: tensor(new BigInt64Array(length).fill(1n)),
    };
    // XLM-RoBERTa has one token type; a model that asks for them gets zeros.
    if (session.inputNames.includes("token_type_ids")) {
      feeds.token_type_ids = tensor(new BigInt64Array(length));
    }
    const output = await session.run(feeds);
    const logits = output.logits ?? Object.values(output)[0];
    const score = (logits?.data as Float32Array | undefined)?.[0];
    for (const tensorOut of Object.values(output)) tensorOut.dispose();
    if (score === undefined || !Number.isFinite(score)) {
      throw new Error("The reranking model returned no score.");
    }
    return score;
  }

  return {
    async load(files) {
      const key = JSON.stringify(files);
      if (loaded?.key === key) return;
      loading ??= load(files).finally(() => {
        loading = undefined;
      });
      const next = await loading;
      if (loaded && loaded !== next) void loaded.session.release();
      loaded = next;
    },

    async score(query, texts) {
      const model = loaded;
      if (!model) throw new Error("The reranking model isn't loaded.");
      const scores: number[] = [];
      for (const text of texts) scores.push(await scoreOne(model, query, text));
      return scores;
    },

    close() {
      void loaded?.session.release();
      loaded = undefined;
    },
  };
}
