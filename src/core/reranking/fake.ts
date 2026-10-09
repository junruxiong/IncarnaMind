/**
 * A deterministic stand-in for the reranking model, for tests and the smoke
 * tests' test-only launch flag (INCARNAMIND_TEST_EMBEDDER=fake). It needs no
 * files and no native code. A text scores the share of the query's words it
 * contains (single characters for CJK), so a text with more of them comes first.
 */
import { setTimeout as sleep } from "node:timers/promises";
import type { CrossEncoder } from "../adapters";

export interface FakeCrossEncoderOptions {
  /** Waits this long for each text. */
  delayMs?: number;
}

const HAN = /\p{Script=Han}/u;

function words(text: string): Set<string> {
  const found = new Set<string>();
  for (const word of text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
    if (HAN.test(word)) for (const character of word) found.add(character);
    else found.add(word);
  }
  return found;
}

/** The fake's score for a text: the share of the query's words in it, from 0 to 1. */
export function fakeRelevance(query: string, text: string): number {
  const wanted = words(query);
  if (wanted.size === 0) return 0;
  const present = words(text);
  let shared = 0;
  for (const word of wanted) if (present.has(word)) shared++;
  return shared / wanted.size;
}

export function createFakeCrossEncoder(options: FakeCrossEncoderOptions = {}): CrossEncoder {
  const { delayMs = 0 } = options;
  let loaded = false;
  return {
    async load() {
      loaded = true;
    },
    async score(query, texts) {
      if (!loaded) throw new Error("The fake reranking model isn't loaded.");
      if (delayMs > 0) await sleep(delayMs * texts.length);
      return texts.map((text) => fakeRelevance(query, text));
    },
    close() {
      loaded = false;
    },
  };
}
