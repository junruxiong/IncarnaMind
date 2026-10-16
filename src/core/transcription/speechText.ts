/**
 * Text as an engine gives it back (#981). sherpa-onnx writes a result as JSON
 * without escaping the control characters its text holds (a newline, a tab),
 * which its own reader then refuses ("Bad control character in string literal
 * in JSON"), so the words of a stretch of speech were lost. And a control
 * character in a line would break the Transcript's file: a blank line ends a cue.
 */

/** A control character: C0, and DEL. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: they are what it looks for
const CONTROL = /[\u0000-\u001f\u007f]/;
// biome-ignore lint/suspicious/noControlCharactersInRegex: they are what it looks for
const CONTROLS = /[\u0000-\u001f\u007f]+/g;
const WHITESPACE = /^[\t\n\v\f\r]+$/;

/** A result's JSON with the control characters inside its strings escaped, so it reads. */
function escapedInStrings(json: string): string {
  let out = "";
  let inString = false;
  for (let at = 0; at < json.length; at++) {
    const char = json[at] as string;
    if (!inString) {
      inString = char === '"';
      out += char;
    } else if (char === "\\") {
      out += char + (json[++at] ?? "");
    } else {
      if (char === '"') inString = false;
      out += CONTROL.test(char) ? `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}` : char;
    }
  }
  return out;
}

/** The text of an engine's result JSON, however it escaped what the text holds. Throws if it isn't one. */
export function textOfResult(json: string): string {
  const result: unknown = JSON.parse(escapedInStrings(json));
  const text = (result as { text?: unknown } | null)?.text;
  return typeof text === "string" ? text : "";
}

/** Where Qwen3-ASR's words begin: it writes "language English<asr_text>" before them. */
const ASR_TEXT = "<asr_text>";

/**
 * The words after the model's own header. sherpa-onnx drops the header, unless
 * the model began with other tokens first ("提uggestion\nlanguage English<asr_text>…"),
 * and then it is in the text; nothing before the marker is speech.
 */
export function wordsOnly(text: string): string {
  const at = text.lastIndexOf(ASR_TEXT);
  return at < 0 ? text : text.slice(at + ASR_TEXT.length);
}

/**
 * What a stretch of speech said, kept: a control character that is a break
 * (a newline, a tab) is a space, any other is left out, and the words stay
 * in one line.
 */
export function cleanSpeech(text: string): string {
  return text
    .replace(CONTROLS, (run) => (WHITESPACE.test(run) ? " " : ""))
    .replace(/\s+/g, " ")
    .trim();
}
