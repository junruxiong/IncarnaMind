import type { ExportFormat, ExportMindOptions, ExportPreferences } from "../core/api";

/** Whether a format has the Include Questions option: Markdown keeps them, and a source pack has none. */
export const hasQuestionsOption = (
  format: ExportFormat,
): format is Exclude<ExportFormat, "markdown" | "source-pack"> =>
  format !== "markdown" && format !== "source-pack";

/** What an export of `format` asks the core for, with the User's choices for it. */
export const optionsFor = (
  format: ExportFormat,
  preferences: ExportPreferences,
): ExportMindOptions => {
  if (format === "docx") return { format, ...preferences.docx };
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
