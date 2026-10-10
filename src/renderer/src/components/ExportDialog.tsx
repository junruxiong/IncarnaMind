import { useEffect, useRef, useState } from "react";
import type { ExportFormat, Mind, MindExportPreview } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { core, files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { saveFailureReason } from "../problems";
import { QuoteNotFoundIcon } from "./icons";
import { ProblemLine } from "./ProblemLine";
import {
  buttonClass,
  choiceListClass,
  choiceRadioClass,
  choiceRowClass,
  choiceTextClass,
  choiceTitleClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTitleClass,
  fieldLabelClass,
  hintClass,
  primaryButtonClass,
} from "./ui";
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
      className={`${dialogClass} w-[30rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id="export-title" className={dialogTitleClass}>
          {t("export.title")}
        </h2>
        {/* Mounted only while open, so each export starts from the defaults and counts afresh. */}
        {mind && <ExportForm mind={mind} onDone={onClose} />}
      </div>
    </dialog>
  );
}

function ExportForm({ mind, onDone }: { mind: Mind; onDone(): void }) {
  const t = useT();
  const [format, setFormat] = useState<ExportFormat>("docx");
  const [questionsWanted, setQuestionsWanted] = useState(false);
  const [preview, setPreview] = useState<MindExportPreview | null>(null);
  /** What went wrong, where: counting the Citations, or writing the file. */
  const [problem, setProblem] = useState<{ step: "preview" | "save"; message: string } | null>(
    null,
  );
  const [saving, setSaving] = useState(false);
  /** Where the export was saved, once it is. */
  const [saved, setSaved] = useState<string | null>(null);
  // Markdown is the archive, so it always keeps the Questions.
  const includeQuestions = format === "markdown" || questionsWanted;

  useEffect(() => {
    let current = true;
    setPreview(null);
    setProblem(null);
    core.previewMindExport(mind.id, { format, includeQuestions }).then(
      (next) => {
        if (current) setPreview(next);
      },
      (failure: unknown) => {
        if (current) setProblem({ step: "preview", message: errorMessage(failure) });
      },
    );
    return () => {
      current = false;
    };
  }, [mind.id, format, includeQuestions]);

  const save = async () => {
    setSaving(true);
    setProblem(null);
    try {
      const path = await files.saveMindExport(mind.id, { format, includeQuestions });
      // Null: the User cancelled the save dialog, so this one stays open.
      if (path) setSaved(path);
    } catch (failure) {
      setProblem({ step: "save", message: errorMessage(failure) });
    } finally {
      setSaving(false);
    }
  };

  const leftOut = preview && !includeQuestions ? preview.questions : 0;

  if (saved) return <Exported path={saved} onDone={onDone} />;

  return (
    <div className="flex flex-col gap-4">
      <fieldset>
        <legend className={`mb-1.5 ${fieldLabelClass}`}>{t("export.format")}</legend>
        <div className={choiceListClass}>
          {FORMATS.map((option) => (
            <label key={option.value} className={choiceRowClass}>
              <input
                type="radio"
                name="export-format"
                value={option.value}
                data-testid={`export-format-${option.value}`}
                checked={format === option.value}
                onChange={() => setFormat(option.value)}
                className={choiceRadioClass}
              />
              <span className="flex min-w-0 flex-col gap-0.5">
                <span className={choiceTitleClass}>{t(option.label)}</span>
                <span className={choiceTextClass}>{t(option.hint)}</span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {format === "docx" && (
        <div>
          <label className="flex items-center gap-2 text-ui text-ink">
            <input
              type="checkbox"
              data-testid="export-include-questions"
              checked={questionsWanted}
              onChange={(event) => setQuestionsWanted(event.target.checked)}
              className="size-4 shrink-0 accent-ink"
            />
            {t("export.includeQuestions")}
          </label>
          {leftOut > 0 && (
            <p className={`ml-6 ${hintClass}`}>
              {leftOut === 1
                ? t("export.questionsLeftOut.one")
                : t("export.questionsLeftOut.other", { count: leftOut })}
            </p>
          )}
        </div>
      )}

      <CitationSummary preview={preview} />

      {problem && (
        <ProblemLine
          testId="export-problem"
          // A file that can't be written goes somewhere else; a count that failed has no other way.
          action={
            problem.step === "save"
              ? {
                  label: t("export.chooseAnother"),
                  onClick: () => void save(),
                  testId: "export-choose-another",
                  disabled: saving,
                }
              : undefined
          }
        >
          {problem.step === "save"
            ? t("export.failed.save", { reason: reasonOf(problem.message, t) })
            : t("export.failed.preview", { reason: problem.message })}
        </ProblemLine>
      )}

      <div className={dialogActionsClass}>
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

/** Where the export went, with a way to see it there. */
function Exported({ path, onDone }: { path: string; onDone(): void }) {
  const t = useT();
  const [error, setError] = useState<string | null>(null);
  const done = useRef<HTMLButtonElement>(null);
  // The Export button that had the focus is gone: Done takes it.
  useEffect(() => done.current?.focus(), []);
  const name = path.split(/[\\/]/).at(-1) ?? path;
  const show = async () => {
    setError(null);
    try {
      await files.showExportInFolder(path);
      onDone();
    } catch (failure) {
      setError(errorMessage(failure));
    }
  };
  return (
    <div className="flex flex-col gap-4">
      <ProblemLine
        role="status"
        testId="export-done"
        title={path}
        action={{
          label: navigator.userAgent.includes("Mac")
            ? t("export.showInFinder")
            : t("export.showInFolder"),
          onClick: () => void show(),
          testId: "export-show",
        }}
      >
        {t("export.done", { name })}
      </ProblemLine>
      {error && (
        <ProblemLine testId="export-problem">
          {t("export.failed.show", { reason: error })}
        </ProblemLine>
      )}
      <div className={dialogActionsClass}>
        <button
          type="button"
          ref={done}
          data-testid="export-close"
          onClick={onDone}
          className={primaryButtonClass}
        >
          {t("export.close")}
        </button>
      </div>
    </div>
  );
}

/** Why a file couldn't be written, in a few words: ours when we know, else the system's. */
function reasonOf(message: string, t: ReturnType<typeof useT>): string {
  const key = saveFailureReason(message);
  return key ? t(key) : message;
}

/** How many of the exported Citations are unverified, with a warning when any are. */
function CitationSummary({ preview }: { preview: MindExportPreview | null }) {
  const t = useT();
  if (!preview) {
    return <p className="text-[13px] leading-5 text-ink-meta">{t("export.checking")}</p>;
  }
  const { citations: total, unverifiedCitations: count } = preview;
  if (count > 0) {
    // Not found is a Citation check, so it takes the not-found colours.
    return (
      <p
        data-testid="export-citations"
        data-unverified={count}
        className="flex items-start gap-2 rounded-lg bg-warning-wash px-3 py-2 text-[13px] leading-5 text-warning"
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
    <p
      data-testid="export-citations"
      data-unverified={0}
      className="text-[13px] leading-5 text-ink-secondary"
    >
      {total === 0
        ? t("export.noCitations")
        : total === 1
          ? t("export.allFound.one")
          : t("export.allFound.other", { count: total })}
    </p>
  );
}
