import type { WordLook } from "../../../core/api";

/** Below this, prose reflows and tables scroll in their frames (DESIGN.md, the Word look). */
const LEAST_WORD_ZOOM = 0.8;

/** The content's `--word-pt` follows the frame's width (`cqw`); the frame's stays whole, for its max-width. */
export function wordLookZoomCss(scope: string, look: WordLook): string[] {
  if (!look.width) return [];
  return [
    `${scope} { container-type: inline-size; --word-unit: var(--word-pt); }`,
    `${scope} .mind-editor { --word-pt: clamp(calc(var(--word-unit) * ${LEAST_WORD_ZOOM}), calc(100cqw / ${look.width}), var(--word-unit)); }`,
  ];
}
