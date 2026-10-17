/**
 * Finding a citation style (#910): the picker's search over the common list and every style of
 * the pack, in either interface language.
 */
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { createCatalogue } from "../../src/core/citationStyles/catalogue";
import {
  commonEntries,
  searchStyles,
  styleEntries,
} from "../../src/renderer/src/citationStyleSearch";
import { CITATION_STYLES } from "../helpers/core";

const catalogue = createCatalogue(join(CITATION_STYLES, "styles.pack")).catalogue();

describe.each(["en", "zh-CN"] as const)("searching the styles, in the %s interface", (language) => {
  const entries = styleEntries(language, catalogue);
  const first = (query: string) => {
    const found = searchStyles(entries, query);
    return (found.journalsFirst ? found.journals : found.styles)[0]?.id;
  };

  test.each([
    ["Nature", "nature"],
    ["nature", "nature"],
    ["The Lancet", "the-lancet"],
    ["物理学报", "acta-physica-sinica"],
    ["国标", "gb-t-7714-2015-numeric"],
    ["APA", "apa"],
    ["Cite Them Right", "harvard"],
    ["Nature Genetics", "nature-genetics"],
  ])("“%s” puts %s first", (query, id) => {
    expect(first(query)).toBe(id);
  });

  test("a journal's style is found among the journals, a style among the styles", () => {
    const found = searchStyles(entries, "Nature");
    expect(found.styles[0]?.id).toBe("nature");
    expect(found.journals.map((entry) => entry.id)).toContain("nature-genetics");
    expect(found.journals.find((entry) => entry.id === "nature-genetics")?.follows).toBe("Nature");
    expect(found.styles.some((entry) => entry.follows !== undefined)).toBe(false);
  });

  test("words in any order find a style, and nothing finds nothing", () => {
    expect(searchStyles(entries, "lancet the").styles.map((entry) => entry.id)).toContain(
      "the-lancet",
    );
    const none = searchStyles(entries, "qqqzzzxxx");
    expect([...none.styles, ...none.journals]).toEqual([]);
  });

  test("the common list holds every common style, GB/T 7714 first only in Chinese", () => {
    const common = commonEntries(entries, language).map((entry) => entry.id);
    expect(common).toHaveLength(12);
    expect(common[0]).toBe(language === "zh-CN" ? "gb-t-7714-2015-numeric" : "numbered");
  });
});

test("every style the pack lists is offered once, with the common ones", () => {
  const entries = styleEntries("en", catalogue);
  expect(new Set(entries.map((entry) => entry.id)).size).toBe(entries.length);
  // Each common style stands for one of the pack's, so the count says how many there are.
  expect(entries.length).toBe(catalogue.count + 1);
});
