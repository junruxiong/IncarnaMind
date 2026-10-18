import type { LibrarySettings, TaggingClassifier } from "../core/api";

/** Only this verified local decision-model family receives PDF page images. */
export const supportsLibraryImages = (model: string) => /^clef-flash(?::|$)/i.test(model.trim());

/** The model Organize uses: the one picked by hand, else the one allowed in the Library; null for none yet. */
export const organizeModelOf = (settings: LibrarySettings): TaggingClassifier | null =>
  settings.classifier.kind === "automatic" ? settings.organizeWith : settings.classifier;

/** Whether Organize can tag and file Documents: once it has a model and there are Folders. */
export const organizes = (
  library: { groups: readonly unknown[]; settings: LibrarySettings } | null,
): boolean => !!library && library.groups.length > 0 && organizeModelOf(library.settings) !== null;

/** Whether Organize tags new Documents rather than automatic tagging: it can, and files them by itself. */
export const organizesNewDocuments = (
  library: { groups: readonly unknown[]; settings: LibrarySettings } | null,
): boolean => organizes(library) && !!library?.settings.automatic;
