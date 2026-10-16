/**
 * The pack's styles as the core knows them (ADR-0016): its index, read the first time the picker
 * or a Mind's style needs it and never at start, so first run and Numbered Minds don't touch the
 * pack. A journal's style resolves to the style it follows, in the journal's own locale; a style
 * with no locale of its own formats in the Mind's language.
 */
import {
  CITATION_STYLES,
  type CitationStyleId,
  commonCitationStyle,
  NUMBERED_STYLE,
  packIdOf,
  type ResolvedCitationStyle,
  type StyleCatalogue,
  type StyleSummary,
  styleIdOf,
} from "../../shared/citationStyles";
import type { Language } from "../language";
import { type PackIndex, PackFiles, type PackStyle, readPackIndex } from "./pack";

/** The standard whose styles raise a Location after the work (`CitationStyle.locators`). */
const RAISED_LOCATORS = /gb-t-7714/;

export function createCatalogue(packPath: string | undefined) {
  let read: { index: PackIndex & { dataOffset: number }; byId: Map<string, PackStyle> } | null =
    null;
  let files: PackFiles | null = null;
  const commonPackIds = new Set(CITATION_STYLES.flatMap((style) => style.packId ?? []));

  const pack = () => {
    if (packPath === undefined) throw new Error("The citation styles aren't installed.");
    if (!read) {
      const index = readPackIndex(packPath);
      read = { index, byId: new Map(index.styles.map((style) => [style.id, style])) };
    }
    return read;
  };

  return {
    /** Every style the pack has, for the picker; the common list's own are the common entries. */
    catalogue(): StyleCatalogue {
      const { index, byId } = pack();
      const styles: StyleSummary[] = [];
      for (const style of index.styles) {
        if (commonPackIds.has(style.id)) continue;
        const summary: StyleSummary = {
          id: styleIdOf(style.id),
          title: style.title,
          format: style.format,
          note: style.note === true,
          fields: style.fields ?? [],
        };
        if (style.short) summary.short = style.short;
        if (style.parent) summary.follows = byId.get(style.parent)?.title ?? style.parent;
        styles.push(summary);
      }
      return { count: index.styles.length, styles };
    },

    /** Whether a style is one this version can format in: common, or in the pack. */
    has(id: CitationStyleId): boolean {
      if (commonCitationStyle(id)) return true;
      const packId = packIdOf(id);
      return packId !== null && pack().byId.has(packId);
    },

    /** How a Mind in `id` is formatted, `language` being the Mind's: "zh-CN" for Chinese text. */
    resolve(id: CitationStyleId, language: Language): ResolvedCitationStyle {
      const common = commonCitationStyle(id);
      if (common) {
        if (common.packId === null) return { id, ...NUMBERED_STYLE };
        return {
          id,
          form: common.form,
          language: common.language,
          heading: common.heading,
          locators: common.locators,
          engine: { packId: common.packId },
        };
      }
      const packId = packIdOf(id);
      const own = packId === null || packPath === undefined ? undefined : pack().byId.get(packId);
      if (!own) return { id, ...NUMBERED_STYLE, unavailable: true };
      const parent = own.parent ? pack().byId.get(own.parent) : undefined;
      const style = parent ?? own;
      const locale = own.locale ?? parent?.locale ?? (language === "zh-CN" ? "zh-CN" : "en-US");
      const chinese = locale.startsWith("zh");
      const note = own.note === true || style.note === true;
      return {
        id,
        form: note ? "note" : "in-text",
        language: chinese ? "zh-CN" : "en",
        heading: chinese ? "参考文献" : note ? "Bibliography" : "References",
        locators: RAISED_LOCATORS.test(style.id) ? "raised-after" : "in-citation",
        engine: { packId: style.id, locale },
      };
    },

    /** The licence notice made from every style's and locale's `<info>`. */
    notice(): string {
      const { index } = pack();
      files ??= new PackFiles(packPath as string, index, 1);
      return files.text(index.notice);
    },
  };
}

export type Catalogue = ReturnType<typeof createCatalogue>;
