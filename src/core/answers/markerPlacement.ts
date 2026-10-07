/**
 * Placing the Citation markers a model left out. Small local models often
 * give the records of their Citations (through `cite`, or in the
 * structured-output JSON) but write few or none of their markers ("[^1]")
 * into the Answer, and a record without its marker is dropped
 * (docs/research/ollama-integration.md). Once the Answer's text is complete,
 * the engine calls `withMissingMarkers` with the records the core took, and
 * each missing marker is put in:
 *
 * 1. after the sentence the record supports: its `sentence`, when the record
 *    names one, else the Answer sentence that shares the most words with its
 *    quote, if it shares enough (see `matches`);
 * 2. failing that, at the end of the paragraph the record is closest to: the
 *    one that shares the most words with its quote; or, when none shares a
 *    word (an Answer in another language than the Passage), the one holding
 *    the nearest marker by number that the model wrote or step 1 placed; or,
 *    with no such marker, the paragraph at the record's place among the
 *    missing records (the first ones near the top).
 *
 * A marker goes before the sentence's closing punctuation ("a claim[^2].")
 * and after any marker already there. Code, headings, math and footnote
 * definitions are left alone.
 *
 * It reads only a record's marker number and text (`sentence`, `quote`), so a
 * change to how a record says where its quote is (pages, or a Location)
 * doesn't touch it.
 */
import type { AnswerEngineEvent, AnswerTools } from "./engine";

/** What placing a marker needs of a Citation record. */
export interface MarkerRecord {
  /** The marker's number: 1 for "[^1]". */
  marker: number;
  /** The quote from the Passage. */
  quote: string;
  /** The Answer's sentence the record supports, when the model gives it. */
  sentence?: string | null;
}

/** How a missing marker was placed. */
export interface PlacedMarker {
  marker: number;
  /** "sentence": after the sentence it supports. "paragraph": at the end of the paragraph it is closest to. */
  at: "sentence" | "paragraph";
}

const MARKER = /\[\^(\d{1,4})\]/g;
/**
 * A sentence ends at one of these, with any markers right after it, followed
 * by a space or the end of its line. CJK ones need no space.
 */
const SENTENCE_END = /[.!?](?:\[\^\d{1,4}\])*(?=["'”’)\]]*(?:\s|$))|[。！？](?:\[\^\d{1,4}\])*/g;
/** Closing punctuation a marker goes before, with the quotes and brackets that close the sentence. */
const CLOSING = /[.!?。！？…:;：；,，]+["'”’」』)）\]]*$/u;
/** Lines no marker goes into. */
const SKIPPED_LINE = /^\s*(?:#{1,6}\s|\||\[\^\d{1,4}\]:|(?:-{3,}|\*{3,}|_{3,})\s*$)/;
const FENCE = /^\s*(?:```|~~~)/;
const MATH_FENCE = /^\s*\$\$\s*$/;

/** The least an Answer sentence must share with a quote to be the one it supports. */
const MIN_SHARED = 2;
const MIN_SIMILARITY = 0.2;

/** Common English words, which say nothing about which sentence a quote supports. */
const STOP_WORDS = new Set(
  "a an and are as at be been but by can do does for from had has have he her his i if in into is it its may more most not of on or our she so such than that the their them then there these they this those to was we were what when which while who will with would you your".split(
    " ",
  ),
);

/** The words of a text for comparing: lower-case words and numbers, and each two neighbouring CJK characters. */
export function wordsOf(text: string): Set<string> {
  const words = new Set<string>();
  const plain = text.replace(MARKER, " ").toLowerCase();
  for (const [run] of plain.matchAll(/[\p{L}\p{N}]+/gu)) {
    // A run may mix scripts, e.g. "transformer模型".
    for (const [part] of run.matchAll(
      /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+|[^\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+/gu,
    )) {
      if (/^[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(part)) {
        const chars = [...part];
        if (chars.length === 1) words.add(part);
        for (let at = 0; at + 1 < chars.length; at++) words.add(`${chars[at]}${chars[at + 1]}`);
      } else if (!STOP_WORDS.has(part)) {
        words.add(part);
      }
    }
  }
  return words;
}

/** How alike two sets of words are: the shared words, and their Dice similarity. */
function likeness(
  a: ReadonlySet<string>,
  b: ReadonlySet<string>,
): { shared: number; score: number } {
  let shared = 0;
  for (const word of a) if (b.has(word)) shared++;
  const total = a.size + b.size;
  return { shared, score: total === 0 ? 0 : (2 * shared) / total };
}

/** Whether a sentence is close enough to a record's text to be the one it supports. */
const matches = (likeness: { shared: number; score: number }) =>
  likeness.shared >= MIN_SHARED && likeness.score >= MIN_SIMILARITY;

interface Span {
  start: number;
  end: number;
}

interface Paragraph extends Span {
  sentences: Span[];
}

/** The paragraphs a marker may go into, with their sentences, by offset in `text`. */
function paragraphsOf(text: string): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  let current: Paragraph | null = null;
  let fenced = false;
  let math = false;
  let offset = 0;
  for (const line of text.split("\n")) {
    const start = offset;
    const end = start + line.length;
    offset = end + 1;
    const fence = FENCE.test(line);
    const mathFence = !fenced && MATH_FENCE.test(line);
    if (fence) fenced = !fenced;
    if (mathFence) math = !math;
    if (fence || mathFence || fenced || math || line.trim() === "" || SKIPPED_LINE.test(line)) {
      current = null;
      continue;
    }
    if (!current) {
      current = { start, end, sentences: [] };
      paragraphs.push(current);
    }
    current.end = end;
    // Each line ends a sentence (a list item, a line of its own); so does closing punctuation.
    let from = start;
    for (const found of line.matchAll(SENTENCE_END)) {
      const to = start + found.index + found[0].length;
      if (text.slice(from, to).trim()) current.sentences.push({ start: from, end: to });
      from = to;
    }
    if (text.slice(from, end).trim()) current.sentences.push({ start: from, end });
  }
  return paragraphs.filter((paragraph) => paragraph.sentences.length > 0);
}

/**
 * Where a marker goes in a sentence or paragraph ending at `end`: after any
 * markers there, before its closing punctuation, without trailing space.
 */
function insertionPoint(text: string, span: Span): number {
  let end = span.end;
  while (end > span.start && /\s/.test(text[end - 1] as string)) end--;
  const body = text.slice(span.start, end);
  // Markers the model wrote after the punctuation ("a claim.[^1]"): after them.
  if (/\[\^\d{1,4}\]$/.test(body)) return end;
  const closing = CLOSING.exec(body);
  return closing ? span.start + closing.index : end;
}

/**
 * The Answer with each record's marker placed, for records whose marker
 * isn't in it, and how each was placed.
 */
export function placeMissingMarkers(
  answer: string,
  records: readonly MarkerRecord[],
): { text: string; placed: PlacedMarker[] } {
  const present = new Set([...answer.matchAll(MARKER)].map((found) => Number(found[1])));
  const missing = [...new Map(records.map((record) => [record.marker, record])).values()]
    .filter((record) => !present.has(record.marker))
    .sort((a, b) => a.marker - b.marker);
  const paragraphs = paragraphsOf(answer);
  if (missing.length === 0 || paragraphs.length === 0) return { text: answer, placed: [] };

  const sentences = paragraphs.flatMap((paragraph, index) =>
    paragraph.sentences.map((span) => ({
      span,
      paragraph: index,
      words: wordsOf(answer.slice(span.start, span.end)),
    })),
  );
  const paragraphWords = paragraphs.map((paragraph) =>
    wordsOf(answer.slice(paragraph.start, paragraph.end)),
  );
  /**
   * Markers by paragraph that say what a paragraph is about: those the model
   * wrote, and those placed after the sentence they support.
   */
  const markersIn = paragraphs.map(
    (paragraph) =>
      new Set(
        [...answer.slice(paragraph.start, paragraph.end).matchAll(MARKER)].map((found) =>
          Number(found[1]),
        ),
      ),
  );

  const insertions: { at: number; marker: number }[] = [];
  const placed: PlacedMarker[] = [];
  missing.forEach((record, rank) => {
    // 1. The sentence it supports: the record's own, else the one most like its quote.
    let sentence: (typeof sentences)[number] | null = null;
    for (const probe of [record.sentence, record.quote]) {
      if (!probe?.trim()) continue;
      const words = wordsOf(probe);
      let bestScore = 0;
      for (const each of sentences) {
        const alike = likeness(words, each.words);
        if (matches(alike) && alike.score > bestScore) {
          bestScore = alike.score;
          sentence = each;
        }
      }
      if (sentence) break;
    }
    if (sentence) {
      insertions.push({ at: insertionPoint(answer, sentence.span), marker: record.marker });
      markersIn[sentence.paragraph]?.add(record.marker);
      placed.push({ marker: record.marker, at: "sentence" });
      return;
    }
    // 2. The paragraph it is closest to.
    const words = wordsOf(`${record.sentence ?? ""} ${record.quote}`);
    let paragraph = -1;
    let most = 0;
    paragraphWords.forEach((each, index) => {
      const { shared } = likeness(words, each);
      if (shared > most) {
        most = shared;
        paragraph = index;
      }
    });
    if (paragraph === -1) {
      // The paragraph with the nearest marker by number, before it if tied.
      let nearest = Number.POSITIVE_INFINITY;
      markersIn.forEach((markers, index) => {
        for (const marker of markers) {
          const distance = Math.abs(marker - record.marker) + (marker > record.marker ? 0.5 : 0);
          if (distance < nearest) {
            nearest = distance;
            paragraph = index;
          }
        }
      });
    }
    if (paragraph === -1) {
      paragraph = Math.min(
        paragraphs.length - 1,
        Math.floor((rank * paragraphs.length) / missing.length),
      );
    }
    const target = paragraphs[paragraph] as Paragraph;
    insertions.push({ at: insertionPoint(answer, target), marker: record.marker });
    placed.push({ marker: record.marker, at: "paragraph" });
  });

  // From the end, so earlier offsets hold; markers at one place in order.
  insertions.sort((a, b) => b.at - a.at || b.marker - a.marker);
  let text = answer;
  for (const { at, marker } of insertions) {
    text = `${text.slice(0, at)}[^${marker}]${text.slice(at)}`;
  }
  return { text, placed: placed.sort((a, b) => a.marker - b.marker) };
}

/**
 * The Answer with the markers of `records` that are missing from it placed,
 * for the records the core accepted (`accepted`): a marker without a valid
 * record would only be removed again. A record whose quote (and sentence)
 * has no word, such as an empty one or "…", is left to be dropped: nothing
 * says where it goes. The engine calls this once the text is complete.
 */
export function withMissingMarkers(
  answer: string,
  records: readonly MarkerRecord[],
  accepted: (marker: number) => boolean,
): { text: string; placed: PlacedMarker[] } {
  const usable = records.filter(
    (record) =>
      Number.isInteger(record.marker) &&
      record.marker >= 1 &&
      // With no word in its quote or its sentence (empty, or only "…" or quotation marks),
      // nothing says where it goes, or what the check would find.
      (wordsOf(record.quote).size > 0 || wordsOf(record.sentence ?? "").size > 0) &&
      accepted(record.marker),
  );
  return usable.length === 0 ? { text: answer, placed: [] } : placeMissingMarkers(answer, usable);
}

/** Engine events that change the Answer's text from `from` to `to`: what follows their common start is replaced. */
export function* textEdits(from: string, to: string): Generator<AnswerEngineEvent> {
  let common = 0;
  while (common < from.length && common < to.length && from[common] === to[common]) common++;
  if (from.length > common) yield { type: "text-retracted", length: from.length - common };
  if (to.length > common) yield { type: "text-delta", text: to.slice(common) };
}

/**
 * Where the engine handles a finished Answer's records: the edits that put in
 * the markers the model left out of `text` for records the core took (see
 * `AnswerTools.hasRecord`), and how many it put in.
 */
export function* missingMarkerEvents(
  text: string,
  records: readonly MarkerRecord[],
  tools: Pick<AnswerTools, "hasRecord">,
): Generator<AnswerEngineEvent> {
  const { text: placed, placed: markers } = withMissingMarkers(
    text,
    records,
    (marker) => tools.hasRecord?.(marker) ?? false,
  );
  if (markers.length === 0) return;
  yield* textEdits(text, placed);
  yield { type: "markers-placed", count: markers.length };
}
