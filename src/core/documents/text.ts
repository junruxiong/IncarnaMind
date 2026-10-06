/** Character classes shared by text extraction, Passage building and keyword search. */

/**
 * Scripts written without spaces between words (Chinese, Japanese, Korean),
 * as the body of a regular-expression character class. Needs the `u` flag.
 */
export const CJK = "\\p{Script=Han}\\p{Script=Hiragana}\\p{Script=Katakana}\\p{Script=Hangul}";

const cjkCharacter = new RegExp(`[${CJK}]`, "u");

export const isCjk = (character: string): boolean => cjkCharacter.test(character);

export const hasCjk = (text: string): boolean => cjkCharacter.test(text);
