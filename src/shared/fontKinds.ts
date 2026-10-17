const SERIF =
  /^(?:times|georgia|cambria|garamond|book antiqua|palatino|constantia|baskerville|century|bookman|didot|hoefler|songti|simsun|source serif|noto serif|minion|caslon|charter|perpetua|rockwell|stsong|fangsong|nsimsun|mincho|source han serif|宋体|仿宋|明朝)/i;
const MONO =
  /^(?:consolas|courier|lucida console|menlo|monaco|andale mono|source code|jetbrains|sf mono)/i;

export type FontKind = "serif" | "mono" | "sans";

/** What a font's name says it is; a name it doesn't know counts as sans. */
export function fontKind(name: string): FontKind {
  return SERIF.test(name) ? "serif" : MONO.test(name) ? "mono" : "sans";
}
