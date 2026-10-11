/**
 * The citation styles IncarnaMind ships (#289, ADR-0016): what the core formats with, and what
 * the style picker lists. Each CSL style is an unmodified file from the CSL project's repository
 * in resources/citation-styles/styles. "Numbered" is IncarnaMind's own: Citations numbered in
 * the text, each a footnote naming its Document and Location, as before styles existed.
 * Pure: the core and the renderer import it.
 */
import type { Language } from "../core/language";
import type { MessageKey } from "./i18n";

export const CITATION_STYLE_IDS = [
  "numbered",
  "apa",
  "mla",
  "harvard",
  "chicago-notes",
  "chicago-author-date",
  "ieee",
  "gb-t-7714-2015-numeric",
  "gb-t-7714-2015-author-date",
  "gb-t-7714-2025-numeric",
  "gb-t-7714-2025-author-date",
  "vancouver",
] as const;

export type CitationStyleId = (typeof CITATION_STYLE_IDS)[number];

/**
 * How a style's Citations read in the text:
 * - "numbered": IncarnaMind's numbered markers, each a footnote naming the Document and Location;
 * - "in-text": the style's own form in the text, "(Bowler et al., 2010, p. 4)" or "[1]";
 * - "note": a note (a footnote in Word) in the style's form, its number in the text.
 */
export type CitationForm = "numbered" | "in-text" | "note";

export interface CitationStyle {
  id: CitationStyleId;
  /** Its name in the picker and on its button: "APA 7". */
  label: MessageKey;
  /** The CSL file in resources/citation-styles/styles; null for Numbered. */
  file: string | null;
  form: CitationForm;
  /**
   * The language its own words are in, and its Location words ("slide 4", "第 4 张幻灯片"):
   * Chinese for GB/T 7714, English for the rest.
   */
  language: Language;
  /** What else the picker's search finds it by, beyond its label in either language. */
  keywords: readonly string[];
  /**
   * The Bibliography's heading in an export, as the style's guide names it, in its language;
   * null for Numbered, whose exports have none.
   */
  heading: string | null;
}

const GB_T = ["GB/T 7714", "GBT 7714", "国标", "国家标准", "guobiao"];

export const CITATION_STYLES: readonly CitationStyle[] = [
  {
    id: "numbered",
    label: "citationStyle.numbered",
    file: null,
    form: "numbered",
    language: "en",
    keywords: ["footnote", "document", "page", "脚注", "编号"],
    heading: null,
  },
  {
    id: "apa",
    label: "citationStyle.apa",
    file: "apa.csl",
    form: "in-text",
    language: "en",
    keywords: ["American Psychological Association", "author-date"],
    heading: "References",
  },
  {
    id: "mla",
    label: "citationStyle.mla",
    file: "modern-language-association.csl",
    form: "in-text",
    language: "en",
    keywords: ["Modern Language Association", "works cited"],
    heading: "Works Cited",
  },
  {
    id: "harvard",
    label: "citationStyle.harvard",
    file: "harvard-cite-them-right.csl",
    form: "in-text",
    language: "en",
    keywords: ["Cite Them Right", "author-date"],
    heading: "Reference list",
  },
  {
    id: "chicago-notes",
    label: "citationStyle.chicago-notes",
    file: "chicago-notes-bibliography.csl",
    form: "note",
    language: "en",
    keywords: ["Chicago Manual of Style", "footnote", "notes and bibliography", "脚注"],
    heading: "Bibliography",
  },
  {
    id: "chicago-author-date",
    label: "citationStyle.chicago-author-date",
    file: "chicago-author-date.csl",
    form: "in-text",
    language: "en",
    keywords: ["Chicago Manual of Style", "author-date"],
    heading: "References",
  },
  {
    id: "ieee",
    label: "citationStyle.ieee",
    file: "ieee.csl",
    form: "in-text",
    language: "en",
    keywords: ["Institute of Electrical and Electronics Engineers", "numeric"],
    heading: "References",
  },
  {
    id: "gb-t-7714-2015-numeric",
    label: "citationStyle.gb-t-7714-2015-numeric",
    file: "china-national-standard-gb-t-7714-2015-numeric.csl",
    form: "in-text",
    language: "zh-CN",
    keywords: [...GB_T, "numeric", "顺序编码"],
    heading: "参考文献",
  },
  {
    id: "gb-t-7714-2015-author-date",
    label: "citationStyle.gb-t-7714-2015-author-date",
    file: "china-national-standard-gb-t-7714-2015-author-date.csl",
    form: "in-text",
    language: "zh-CN",
    keywords: [...GB_T, "author-date", "著者-出版年"],
    heading: "参考文献",
  },
  {
    id: "gb-t-7714-2025-numeric",
    label: "citationStyle.gb-t-7714-2025-numeric",
    file: "china-national-standard-gb-t-7714-2025-numeric.csl",
    form: "in-text",
    language: "zh-CN",
    keywords: [...GB_T, "numeric", "顺序编码"],
    heading: "参考文献",
  },
  {
    id: "gb-t-7714-2025-author-date",
    label: "citationStyle.gb-t-7714-2025-author-date",
    file: "china-national-standard-gb-t-7714-2025-author-date.csl",
    form: "in-text",
    language: "zh-CN",
    keywords: [...GB_T, "author-date", "著者-出版年"],
    heading: "参考文献",
  },
  {
    id: "vancouver",
    label: "citationStyle.vancouver",
    file: "nlm-citation-sequence.csl",
    form: "in-text",
    language: "en",
    keywords: ["NLM", "Citing Medicine", "numeric", "medicine"],
    heading: "References",
  },
];

/**
 * A Mind that hasn't chosen is Numbered: many Minds cite contracts and reports that no
 * citation style could cite better, so none turns into "(n.d.)" by itself.
 */
export const DEFAULT_CITATION_STYLE: CitationStyleId = "numbered";

/**
 * The styles the prompt in a Mind that cites papers or books offers (`CitationStylePrompt`),
 * the interface language's own first.
 */
export const PROMPTED_CITATION_STYLES: Readonly<Record<Language, readonly CitationStyleId[]>> = {
  en: ["apa", "gb-t-7714-2015-numeric"],
  "zh-CN": ["gb-t-7714-2015-numeric", "apa"],
};

export const isCitationStyleId = (value: unknown): value is CitationStyleId =>
  typeof value === "string" && (CITATION_STYLE_IDS as readonly string[]).includes(value);

export function citationStyle(id: CitationStyleId): CitationStyle {
  const style = CITATION_STYLES.find((each) => each.id === id);
  if (!style) throw new Error(`There is no citation style ${id}.`);
  return style;
}
