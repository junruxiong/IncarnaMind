/** Only this verified local decision-model family receives PDF page images. */
export const supportsLibraryImages = (model: string) => /^clef-flash(?::|$)/i.test(model.trim());
