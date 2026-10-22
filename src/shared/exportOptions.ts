import type { ExportFormat, ExportMindOptions, ExportPreferences } from "../core/api";

/** Whether a format has the Include Questions option: Markdown keeps them, and a source pack has none. */
export const hasQuestionsOption = (
  format: ExportFormat,
): format is Exclude<ExportFormat, "markdown" | "source-pack"> =>
  format !== "markdown" && format !== "source-pack";

/**
 * What an export of `format` asks the core for, with the User's choices for it; null for Word
 * until the User has said whether to mark the text IncarnaMind wrote, which nothing decides for them.
 */
export const optionsFor = (
  format: ExportFormat,
  preferences: ExportPreferences,
): ExportMindOptions | null => {
  if (format === "docx") {
    const { markAiText, ...choices } = preferences.docx;
    return markAiText === null ? null : { format, ...choices, markAiText: markAiText === "mark" };
  }
  if (format === "source-pack") return { format };
  if (format === "share-page") {
    return {
      format,
      ...preferences["share-page"],
      theirChanges: preferences.theirChanges ?? "shown",
    };
  }
  const theirChanges = preferences.theirChanges ?? "shown";
  return hasQuestionsOption(format)
    ? { format, includeQuestions: preferences[format].includeQuestions, theirChanges }
    : { format, theirChanges };
};

/** The Word copy offered when quitting: it marks the text IncarnaMind wrote only if the User chose to. */
export const quitCopyOptions = (preferences: ExportPreferences): ExportMindOptions => ({
  format: "docx",
  markAiText: preferences.docx.markAiText === "mark",
});
