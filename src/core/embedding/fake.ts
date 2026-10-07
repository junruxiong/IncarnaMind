/**
 * A deterministic stand-in for the embedding model: for tests, and for the
 * smoke tests' test-only launch flag (INCARNAMIND_TEST_EMBEDDER=fake). It needs
 * no files and no native code. A text's vector counts its words and their
 * character trigrams (pairs of characters in CJK text), hashed into the
 * dimensions, so texts sharing words or parts of words come out similar.
 */
import { setTimeout as sleep } from "node:timers/promises";
import type { Embedder } from "../adapters";

export interface FakeEmbedderOptions {
  /** Defaults to the built-in model's 384. */
  dimensions?: number;
  /** Waits this long for each text, so a smoke test can watch "embedding". */
  delayMs?: number;
}

/** FNV-1a: a small, stable string hash. */
function hash(text: string): number {
  let value = 0x811c9dc5;
  for (let index = 0; index < text.length; index++) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 0x01000193);
  }
  return value >>> 0;
}

const HAN_RUN = /\p{Script=Han}+/u;

/** The text's features: words, and their character trigrams (CJK: character pairs). */
function features(text: string): string[] {
  const found: string[] = [];
  for (const word of text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
    const characters = Array.from(word);
    if (HAN_RUN.test(word)) {
      for (let index = 0; index + 1 < characters.length; index++) {
        found.push(characters.slice(index, index + 2).join(""));
      }
      if (characters.length === 1) found.push(word);
      continue;
    }
    found.push(word);
    const padded = Array.from(` ${word} `);
    for (let index = 0; index + 3 <= padded.length; index++) {
      found.push(`#${padded.slice(index, index + 3).join("")}`);
    }
  }
  return found;
}

/** The fake's vector for a text, prefix ("passage: ", "query: ") ignored. Not normalised. */
export function fakeVector(text: string, dimensions = 384): Float32Array {
  const vector = new Float32Array(dimensions);
  for (const feature of features(text.replace(/^(passage|query): /, ""))) {
    const value = hash(feature);
    const slot = value % dimensions;
    vector[slot] = (vector[slot] ?? 0) + ((value & 0x80000000) === 0 ? 1 : -1);
  }
  // A constant component, so no text gets an all-zero vector.
  vector[0] = (vector[0] ?? 0) + 0.01;
  return vector;
}

export function createFakeEmbedder(options: FakeEmbedderOptions = {}): Embedder {
  const { dimensions = 384, delayMs = 0 } = options;
  let loaded = false;
  return {
    async load() {
      loaded = true;
    },
    async embed(text) {
      if (!loaded) throw new Error("The fake embedding model isn't loaded.");
      if (delayMs > 0) await sleep(delayMs);
      return fakeVector(text, dimensions);
    },
    close() {
      loaded = false;
    },
  };
}
