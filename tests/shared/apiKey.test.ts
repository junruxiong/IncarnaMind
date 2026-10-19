import { describe, expect, test } from "vitest";
import {
  checkApiKey,
  cleanApiKey,
  nameCharacter,
  strayKeyCharacter,
} from "../../src/shared/apiKey";

const KEY = "sk-test-1234567890-not-a-real-key";

describe("A pasted key is cleaned at its ends", () => {
  test.each([
    ["spaces", `  ${KEY}  `],
    ["a line break and a tab", `\n\t${KEY}\r\n`],
    ["a zero-width space", `\u200B${KEY}\u200B`],
    ["a byte order mark", `\uFEFF${KEY}`],
    ["a no-break space", `\u00A0${KEY}\u00A0`],
    ["an ideographic space", `\u3000${KEY}\u3000`],
    ["a zero-width joiner and a word joiner", `\u200D\u2060${KEY}\u2060`],
    ["a mix of them", ` \u200B\uFEFF\n${KEY}\u00A0\u200B \n`],
  ])("without %s", (_name, pasted) => {
    expect(cleanApiKey(pasted)).toBe(KEY);
    expect(checkApiKey(pasted)).toEqual({ key: KEY, stray: null });
  });

  test("a key that is only spaces and invisible characters is blank", () => {
    expect(cleanApiKey(" \u200B\uFEFF\n\u00A0")).toBe("");
  });

  test("a key that has nothing around it is left as it is", () => {
    expect(cleanApiKey(KEY)).toBe(KEY);
    expect(strayKeyCharacter(KEY)).toBeNull();
    for (const key of ["AIzaSyA-1234_abcDEF", "sk-ant-api03-Ab_Cd-0123456789==", "a.b~c/d+e:f"]) {
      expect(checkApiKey(key)).toEqual({ key, stray: null });
    }
  });
});

describe("A character keys don't use", () => {
  test("a Chinese word is found with its place and the first character", () => {
    expect(strayKeyCharacter("sk-test-1234作品-not-a-real-key")).toEqual({
      index: 12,
      character: "作",
    });
  });

  test("a typographic quote is found", () => {
    expect(strayKeyCharacter("sk-test-\u2019abc")).toEqual({ index: 8, character: "\u2019" });
    expect(strayKeyCharacter("sk-\u201Cabc\u201D")).toEqual({ index: 3, character: "\u201C" });
  });

  test("an invisible character inside the key is found, though one at its end is cleaned", () => {
    expect(strayKeyCharacter("sk-12\u200B34")).toEqual({ index: 5, character: "\u200B" });
    expect(strayKeyCharacter("sk-12\uFEFF34")).toEqual({ index: 5, character: "\uFEFF" });
    expect(strayKeyCharacter("sk-12\u00A034")).toEqual({ index: 5, character: "\u00A0" });
    expect(checkApiKey("sk-12\u200B34\u200B").stray).toEqual({ index: 5, character: "\u200B" });
  });

  test("a letter with an accent, a full-width letter and an emoji are found whole", () => {
    expect(strayKeyCharacter("sk-é")).toEqual({ index: 3, character: "é" });
    expect(strayKeyCharacter("ｓk-1")).toEqual({ index: 0, character: "ｓ" });
    expect(strayKeyCharacter("sk-😀1")).toEqual({ index: 3, character: "😀" });
  });

  test("is named by itself when it can be seen, else by its code, never with the rest of the key", () => {
    expect(nameCharacter("作")).toBe("作");
    expect(nameCharacter("\u2019")).toBe("\u2019");
    expect(nameCharacter("\u200B")).toBe("U+200B");
    expect(nameCharacter("\u00A0")).toBe("U+00A0");
    expect(nameCharacter("\uFEFF")).toBe("U+FEFF");
    expect(nameCharacter(" ")).toBe("U+0020");
  });
});
