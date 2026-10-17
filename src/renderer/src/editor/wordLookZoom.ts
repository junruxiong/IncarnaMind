import type { WordLook } from "../../../core/api";
import { widestTable } from "./wordTableCss";

/** The size Word gives text when a file says none. */
export const DEFAULT_BODY = 11;

/** The body is never drawn smaller: below it, prose reflows and tables wrap (#1035). */
const LEAST_BODY_PX = 11;

/** The page's width in points: the text's, or the widest table's where it reaches further. */
export const pageWidth = (look: WordLook) => Math.max(look.width ?? 0, widestTable(look));

/** The content's `--word-pt` follows the frame's width (`cqw`); the frame's stays whole, for its max-width. */
export function wordLookZoomCss(scope: string, look: WordLook): string[] {
  const page = pageWidth(look);
  if (page === 0) return [];
  return [
    `${scope} { container-type: inline-size; --word-unit: var(--word-pt); }`,
    `${scope} .mind-editor { --word-pt: clamp(calc(${LEAST_BODY_PX}px / ${look.body ?? DEFAULT_BODY}), calc(100cqw / ${page}), var(--word-unit)); }`,
  ];
}
