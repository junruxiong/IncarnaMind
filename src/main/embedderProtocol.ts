/** Messages between the main process and the embedding utility process. */
import type { EmbeddingModelFiles } from "../core";

export type EmbedderCommand =
  | { type: "load"; files: EmbeddingModelFiles }
  | { type: "embed"; text: string };

/** `id` pairs a response with its request. */
export type EmbedderRequest = EmbedderCommand & { id: number };

/** A vector for "embed", nothing more for "load", or what went wrong. */
export interface EmbedderResponse {
  id: number;
  vector?: Float32Array;
  error?: string;
}
