/**
 * Spike for #755, not for merging. The spike is driven from its e2e spec through
 * `window.__spike755`, set before a Mind opens: which approach paginates ("plus":
 * tiptap-pagination-plus; "custom": the layout pass in ./customPages), and the Word file's
 * layout properties (spikes/755-page-view/wordLayout.ts).
 */
import type { SpikeLook } from "../../../../../spikes/755-page-view/wordLayout";

export type SpikeMode = "off" | "look" | "plus" | "custom";

export interface SpikeStats {
  /** One entry per layout run: what started it, how long each part took, in ms. */
  runs: {
    reason: string;
    pages: number;
    pagesLaidOut: number;
    measureMs: number;
    layoutMs: number;
    totalMs: number;
    converged: boolean;
    background: boolean;
    at: number;
  }[];
  keydownAt: number;
  updateAt: number;
  settledAt: number;
  backgroundDoneAt: number;
  composingSkips: number;
}

export interface SpikeConfig {
  mode: SpikeMode;
  look: SpikeLook | null;
  /** Face names the file asks for, to the face LibreOffice drew them with. */
  fontMap: Record<string, string>;
  /** How many pages after the edited one are laid out before the frame; the rest in the background. */
  syncPages: number;
  /** The gap between sheets: 24 px on screen, none when printing. */
  gap?: number;
  stats: SpikeStats;
}

declare global {
  interface Window {
    __spike755?: SpikeConfig;
  }
}

export const spikeConfig = (): SpikeConfig | null => window.__spike755 ?? null;

export const spikeMode = (): SpikeMode => window.__spike755?.mode ?? "off";

export function emptyStats(): SpikeStats {
  return {
    runs: [],
    keydownAt: 0,
    updateAt: 0,
    settledAt: 0,
    backgroundDoneAt: 0,
    composingSkips: 0,
  };
}
