import type { LibrarySettings, TaggingClassifier } from "../core/api";

/** Only this verified local decision-model family receives PDF page images. */
export const supportsLibraryImages = (model: string) => /^clef-flash(?::|$)/i.test(model.trim());

/** The model Organize uses: the one picked by hand, else the one allowed in the Library; null for none yet. */
export const organizeModelOf = (settings: LibrarySettings): TaggingClassifier | null =>
  settings.classifier.kind === "automatic" ? settings.organizeWith : settings.classifier;

/**
 * Whether Organize tags new Documents rather than automatic tagging: once it has a model and there
 * are Folders, while it files them by itself.
 */
export const organizes = (
  library: { groups: readonly unknown[]; settings: LibrarySettings } | null,
): boolean =>
  !!library &&
  library.groups.length > 0 &&
  library.settings.automatic &&
  organizeModelOf(library.settings) !== null;
