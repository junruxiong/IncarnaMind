/**
 * Whether a plain-text file reads as code, so the viewer sets it in a
 * monospaced font as an editor would (its indentation and columns line up);
 * anything else is prose, set in the Mind's serif. Pure.
 */

/** The lines looked at: enough to tell, without reading a long file through. */
const SAMPLE_LINES = 400;

const CODE_LINE =
  /(?:[;{}]\s*$|^\s*(?:def|class|function|import|from|return|if|for|while|const|let|var|public|private|#include|package|fn|func)\b|[=!<>]=|=>|->|::|\)\s*\{|^\s*\/\/|^\s*#\s*(?:include|define)\b)/;
const TABLE_RULE = /^\s*[+|][-=+|:\s]{3,}[+|]\s*$/;
const BOX_DRAWING = /[─-╿]/;
/**
 * Two or more spaces between words, as columns lined up with spaces have;
 * not after a sentence's end, where some writers put two.
 */
const COLUMNS = /[^\s.!?:;,"')] {2,}\S/;

export function looksLikeCode(text: string): boolean {
  if (text.startsWith("#!")) return true;
  const lines = text.split(/\r?\n/, SAMPLE_LINES).filter((line) => line.trim() !== "");
  if (lines.length === 0) return false;
  const share = (test: (line: string) => boolean) => lines.filter(test).length / lines.length;
  if (share((line) => TABLE_RULE.test(line) || BOX_DRAWING.test(line)) >= 0.2) return true;
  const indented = share((line) => /^(?: {2,}|\t)\S/.test(line));
  const code = share((line) => CODE_LINE.test(line));
  if (code >= 0.4 || (code >= 0.25 && indented >= 0.25)) return true;
  // Columns lined up with spaces, on most lines: a table or a log.
  return lines.length >= 3 && share((line) => COLUMNS.test(line.trim())) >= 0.6;
}
