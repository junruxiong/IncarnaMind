import { useEffect, useState } from "react";
import type { ExportFormat, Mind, MindExportPreview } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { core, files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { QuoteNotFoundIcon } from "./icons";
import { buttonClass, primaryButtonClass } from "./providers/shared";
import { useModal } from "./useModal";

const FORMATS: readonly { value: ExportFormat; label: MessageKey; hint: MessageKey }[] = [
  { value: "docx", label: "export.format.docx", hint: "export.format.docx.hint" },
  { value: "markdown", label: "export.format.markdown", hint: "export.format.markdown.hint" },
];

/**
 * Exports a Mind: the format, whether a .docx includes the Questions, and how
 * many Citations are unverified, before the system save dialog asks where.
 * Open while `mind` is set.
 */
export function ExportDialog({ mind, onClose }: { mind: Mind | null; onClose(): void }) {
  const t = useT();
  const dialog = useModal(mind !== null);
  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      data-testid="export-dialog"
      aria-labelledby="export-title"
      className="m-auto w-[28rem] max-w-[calc(100vw-2rem)] rounded-[9px] bg-white p-5 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      <h2 id="export-title" className="text-lg font-semibold">
        {t("export.title")}
      </h2>
      {/* Mounted only while open, so each export starts from the defaults and counts afresh. */}
      {mind && <ExportForm mind={mind} onDone={onClose} />}
    </dialog>
  );
}

function ExportForm({ mind, onDone }: { mind: Mind; onDone(): void }) {
  const t = useT();
  const [format, setFormat] = useState<ExportFormat>("docx");
  const [questionsWanted, setQuestionsWanted] = useState(false);
  const [preview, setPreview] = useState<MindExportPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // Markdown is the archive, so it always keeps the Questions.
  const includeQuestions = format === "markdown" || questionsWanted;

  useEffect(() => {
    let current = true;
    setPreview(null);
    setError(null);
    core.previewMindExport(mind.id, { format, includeQuestions }).then(
      (next) => {
        if (current) setPreview(next);
      },
      (failure: unknown) => {
        if (current) setError(errorMessage(failure));
      },
    );
    return () => {
      current = false;
    };
  }, [mind.id, format, includeQuestions]);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const path = await files.saveMindExport(mind.id, { format, includeQuestions });
      // Null: the User cancelled the save dialog, so this one stays open.
      if (path) onDone();
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setSaving(false);
    }
  };

  const leftOut = preview && !includeQuestions ? preview.questions : 0;

  return (
    <div className="mt-4 flex flex-col gap-4">
      <fieldset>
        <legend className="mb-1 text-sm font-medium">{t("export.format")}</legend>
        {FORMATS.map((option) => (
          <label key={option.value} className="flex items-start gap-2 py-1 text-sm">
            <input
              type="radio"
              name="export-format"
              value={option.value}
              data-testid={`export-format-${option.value}`}
              checked={format === option.value}
              onChange={() => setFormat(option.value)}
              className="mt-[3px]"
            />
            <span>
              {t(option.label)}
              <span className="block text-xs text-gray-500">{t(option.hint)}</span>
            </span>
          </label>
        ))}
      </fieldset>

      {format === "docx" && (
        <div className="text-sm">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              data-testid="export-include-questions"
              checked={questionsWanted}
              onChange={(event) => setQuestionsWanted(event.target.checked)}
            />
            {t("export.includeQuestions")}
          </label>
          {leftOut > 0 && (
            <p className="mt-1 ml-6 text-xs text-gray-500">
              {leftOut === 1
                ? t("export.questionsLeftOut.one")
                : t("export.questionsLeftOut.other", { count: leftOut })}
            </p>
          )}
        </div>
      )}

      <CitationSummary preview={preview} />

      {error && (
        <p role="alert" className="text-sm break-words text-red-700">
          {t("export.failed", { message: error })}
        </p>
      )}

      <div className="flex justify-end gap-2">
        <button type="button" onClick={onDone} className={buttonClass}>
          {t("export.cancel")}
        </button>
        <button
          type="button"
          data-testid="export-save"
          disabled={saving || preview === null}
          onClick={() => void save()}
          className={primaryButtonClass}
        >
          {saving ? t("export.saving") : t("export.save")}
        </button>
      </div>
    </div>
  );
}

/** How many of the exported Citations are unverified, with a warning when any are. */
function CitationSummary({ preview }: { preview: MindExportPreview | null }) {
  const t = useT();
  if (!preview) {
    return <p className="text-sm text-gray-500">{t("export.checking")}</p>;
  }
  const { citations: total, unverifiedCitations: count } = preview;
  if (count > 0) {
    return (
      <p
        data-testid="export-citations"
        data-unverified={count}
        className="flex items-start gap-2 rounded-[9px] bg-amber-50 px-3 py-2 text-sm text-amber-900"
      >
        <QuoteNotFoundIcon className="mt-[2px] size-4 shrink-0" />
        <span>
          {count === 1
            ? t("export.unverifiedCitations.one")
            : t("export.unverifiedCitations.other", { count, total })}
        </span>
      </p>
    );
  }
  return (
    <p data-testid="export-citations" data-unverified={0} className="text-sm text-gray-600">
      {total === 0
        ? t("export.noCitations")
        : total === 1
          ? t("export.allFound.one")
          : t("export.allFound.other", { count: total })}
    </p>
  );
}
