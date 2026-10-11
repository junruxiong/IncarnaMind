import { useEffect, useRef, useState } from "react";
import type { ExportFormat, Mind, MindExportPreview } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { core, files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { folderName } from "../linkedFolders";
import { chooseOutputLocation } from "../outputFiles";
import { saveFailureReason } from "../problems";
import { QuoteNotFoundIcon } from "./icons";
import { type ProblemAction, ProblemLine } from "./ProblemLine";
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
 * many Citations are unverified, before it is saved in the Output location
 * (the first save asks where that is). Open while `mind` is set.
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
  /**
   * What went wrong, where: counting the Citations, writing the file, or the
   * Output location being in a Linked folder the User said no to.
   */
  const [problem, setProblem] = useState<
    | { step: "preview" | "save"; message: string }
    | { step: "not-allowed"; linkedFolderId: string; folder: string }
    | null
  >(null);
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

  /** Saves, after `first` (a fix to what stopped the last try), unless it returns false. */
  const save = async (first?: () => Promise<boolean>) => {
    setSaving(true);
    try {
      if (first && !(await first())) return;
      setProblem(null);
      const result = await core.saveMindExport(mind.id, { format, includeQuestions });
      if (result.status === "saved") setSaved(result.file.path);
      if (result.status === "not-allowed") {
        setProblem({
          step: "not-allowed",
          linkedFolderId: result.linkedFolderId,
          folder: folderName(result.linkedFolderPath),
        });
      }
      // "cancelled": the User didn't choose where new files go, so this dialog stays as it was.
    } catch (failure) {
      setProblem({ step: "save", message: errorMessage(failure) });
    } finally {
      setSaving(false);
    }
  };

  /** The problem line's one action: fix what stopped the save, then save. */
  const fix = (): ProblemAction | undefined => {
    if (problem?.step === "save") {
      return {
        label: t("export.changeLocation"),
        onClick: () => void save(async () => (await chooseOutputLocation()) !== null),
        testId: "export-change-location",
        disabled: saving,
      };
    }
    if (problem?.step === "not-allowed") {
      const { linkedFolderId } = problem;
      return {
        label: t("export.allow"),
        onClick: () =>
          void save(async () => {
            await core.setLinkedFolderWriting(linkedFolderId, true);
            return true;
          }),
        testId: "export-allow",
        disabled: saving,
      };
    }
    // A count that failed has no other way.
    return undefined;
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
        <ProblemLine testId="export-problem" action={fix()}>
          {problem.step === "save"
            ? t("export.failed.save", { reason: reasonOf(problem.message, t) })
            : problem.step === "not-allowed"
              ? t("export.failed.save", {
                  reason: t("export.failed.notAllowed", { folder: problem.folder }),
                })
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
      await files.showSavedFile(path);
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
