/**
 * The citation-style engine's WebAssembly module (wasm/citation-styles, ADR-0016), loaded
 * without wasm-bindgen: the module exports a few plain functions over its memory and speaks
 * JSON both ways (see its src/lib.rs). Pure Node: the thread (./styleWorker) and tests load it.
 */
import type { CslItem } from "./csl";
import type { EngineRun } from "./types";

interface Exports {
  memory: WebAssembly.Memory;
  alloc(len: number): number;
  dealloc(ptr: number, len: number): void;
  engine_new(ptr: number, len: number): number;
  engine_render(handle: number, ptr: number, len: number): number;
  engine_drop(handle: number): void;
  output_ptr(): number;
  output_len(): number;
}

/** What a style says about itself, read when it is parsed. */
export interface StyleDescription {
  /** Whether its citations are in the text or in notes. */
  class: "in-text" | "note";
  title: string;
  defaultLocale: string | null;
  hasBibliography: boolean;
}

/** One reference in a citation, as the engine takes it. */
export interface EngineCite {
  /** The `id` of an item in the request. */
  id: string;
  /** "12", "12-13"; or with the label "custom", any text, printed as it is. */
  locator?: string;
  /** A CSL locator label ("page", "line") or "custom". */
  label?: string;
  /** The locale this reference's terms come from, instead of the style's. */
  locale?: string;
}

export interface EngineCitation {
  items: EngineCite[];
  /** The note's number, for note styles. */
  noteIndex?: number;
}

export interface EngineRequest {
  items: CslItem[];
  citations: EngineCitation[];
  /** The locale for the style's terms; none: the style's own. */
  locale?: string;
}

export interface EngineEntry {
  id: string;
  firstField: EngineRun[] | null;
  runs: EngineRun[];
}

export interface EngineOutput {
  citations: EngineRun[][];
  bibliography: EngineEntry[];
  hangingIndent: boolean;
  secondFieldAlign: "flush" | "margin" | null;
}

/** A style the engine can't use, with the parent a dependent style points to. */
export class StyleError extends Error {
  readonly parent: string | null;

  constructor(message: string, parent: string | null) {
    super(message);
    this.name = "StyleError";
    this.parent = parent;
  }
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/**
 * One instance of the engine. A trap (a panic in Hayagriva) leaves it unusable: the caller
 * drops it and makes another from the same module.
 */
export class CslEngine {
  private readonly exports: Exports;

  constructor(module: WebAssembly.Module) {
    const instance = new WebAssembly.Instance(module, {});
    this.exports = instance.exports as unknown as Exports;
  }

  /** Calls `fn` with `json` written into the module's memory; returns its result and output. */
  private call(json: unknown, fn: (ptr: number, len: number) => number) {
    const bytes = encoder.encode(JSON.stringify(json));
    const ptr = this.exports.alloc(bytes.length);
    try {
      new Uint8Array(this.exports.memory.buffer, ptr, bytes.length).set(bytes);
      const result = fn(ptr, bytes.length);
      const output = new Uint8Array(
        this.exports.memory.buffer,
        this.exports.output_ptr(),
        this.exports.output_len(),
      );
      return { result, output: JSON.parse(decoder.decode(output)) as unknown };
    } finally {
      this.exports.dealloc(ptr, bytes.length);
    }
  }

  /** Parses a style and its locales; returns its handle and what it says about itself. */
  parse(style: string, locales: string[]): { handle: number; description: StyleDescription } {
    const { result, output } = this.call({ style, locales }, (ptr, len) =>
      this.exports.engine_new(ptr, len),
    );
    if (result === 0) {
      const failure = output as { error?: unknown; parent?: unknown };
      throw new StyleError(
        typeof failure.error === "string" ? failure.error : "The style can't be read.",
        typeof failure.parent === "string" ? failure.parent : null,
      );
    }
    return { handle: result, description: output as StyleDescription };
  }

  render(handle: number, request: EngineRequest): EngineOutput {
    const { result, output } = this.call(request, (ptr, len) =>
      this.exports.engine_render(handle, ptr, len),
    );
    if (result === 0) {
      const failure = output as { error?: unknown };
      throw new Error(typeof failure.error === "string" ? failure.error : "Formatting failed.");
    }
    return output as EngineOutput;
  }

  drop(handle: number): void {
    this.exports.engine_drop(handle);
  }
}
