/**
 * The language a Document is written in, told from a sample of its text, so
 * the search Tool can tell an Answer which languages the Documents in scope
 * are in (ADR-0007). Search matches a query best in a Document's own
 * language: the built-in embedding model and keyword search don't bridge
 * languages well (ADR-0009), so a Question in another language is also
 * searched in theirs.
 *
 * Small on purpose: CJK languages by their scripts, and a few Latin-script
 * languages by their most common words. Anything else is no language, and
 * isn't named.
 */

/** A language as the model is told it, in English. */
export type TextLanguage = string;

/** Common words that tell Latin-script languages apart: few appear in more than one list. */
const LATIN_WORDS: Readonly<Record<TextLanguage, ReadonlySet<string>>> = {
  English: new Set(
    "the and of to is that with for this are was from which have has not be by it as".split(" "),
  ),
  French: new Set("les des est une dans pour qui sur avec pas sont du ce cette aux".split(" ")),
  German: new Set("der die und das ist nicht mit sich auf für eine dem werden auch".split(" ")),
  Spanish: new Set("el los las y por para del se es como más pero sus".split(" ")),
  Italian: new Set("il di che per non della sono gli nel anche delle alla".split(" ")),
  Portuguese: new Set("os do da em não uma são mais ao das dos pelo".split(" ")),
  Dutch: new Set("het een van en dat op voor niet zijn worden ook maar".split(" ")),
};

/** At least this many common words, so a few names or figures aren't a language. */
const MIN_COMMON_WORDS = 3;

const count = (text: string, pattern: RegExp) => text.match(pattern)?.length ?? 0;

/** The language `text` is mostly written in, or null when it can't tell. */
export function detectLanguage(text: string): TextLanguage | null {
  const han = count(text, /\p{Script=Han}/gu);
  const kana = count(text, /[\p{Script=Hiragana}\p{Script=Katakana}]/gu);
  const hangul = count(text, /\p{Script=Hangul}/gu);
  const words = text.toLowerCase().match(/\p{Script=Latin}+/gu) ?? [];
  // A CJK character stands for about a word.
  const cjk = han + kana + hangul;
  if (cjk > 0 && cjk >= words.length) {
    if (hangul > han + kana) return "Korean";
    if (kana >= (han + kana) * 0.2) return "Japanese";
    return "Chinese";
  }
  let best: TextLanguage | null = null;
  let bestCount = 0;
  for (const [language, common] of Object.entries(LATIN_WORDS)) {
    const found = words.filter((word) => common.has(word)).length;
    if (found > bestCount) {
      best = language;
      bestCount = found;
    }
  }
  return bestCount >= MIN_COMMON_WORDS ? best : null;
}

/** How many Documents are in each language: the most common first, then by name. */
export function languageList(
  languages: Iterable<TextLanguage | null>,
): { language: TextLanguage; documents: number }[] {
  const counts = new Map<TextLanguage, number>();
  for (const language of languages) {
    if (language) counts.set(language, (counts.get(language) ?? 0) + 1);
  }
  return [...counts]
    .map(([language, documents]) => ({ language, documents }))
    .sort((a, b) => b.documents - a.documents || a.language.localeCompare(b.language));
}
