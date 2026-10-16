import type { WordBorder } from "../../../core/api";

const BORDER_STYLES = new Map([
  ["double", "double"],
  ["dotted", "dotted"],
  ["dashed", "dashed"],
  ["dashSmallGap", "dashed"],
  ["dotDash", "dashed"],
  ["dotDotDash", "dotted"],
]);

export const pt = (value: number) => `calc(var(--word-pt) * ${value})`;

export const attribute = (name: string, key: string) => `[${name}="${key.replace(/["\\]/g, "")}"]`;

export const rule = (selector: string, declarations: string[]) =>
  declarations.length > 0 ? `${selector} { ${declarations.join("; ")}; }` : "";

export function border(value: WordBorder): string {
  const width = Math.max(value.width, 0.25);
  return `${pt(width)} ${BORDER_STYLES.get(value.style) ?? "solid"} ${value.color ? `#${value.color}` : "currentColor"}`;
}
