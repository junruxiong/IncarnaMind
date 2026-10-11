import { type FormEvent, useEffect, useId, useState } from "react";
import {
  EXPORT_FORMATS,
  type ExportFormat,
  type ExportPreferences,
  type Mind,
  type MindCitationStyle,
  type OutputSettings,
} from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { styleLabel, useCitationStylePicker } from "../citationStylePicker";
import { shortcutLabel } from "../commands/keymap";
import { core } from "../core";
import { errorMessage } from "../errors";
import { exportPreferences, FORMAT_LABELS } from "../exports";
import { useT } from "../i18n";
import { folderName } from "../linkedFolders";
import { chooseOutputLocation } from "../outputFiles";
import { useAppStore } from "../store";
import { ChevronDownLineIcon, CloseLineIcon } from "./lineIcons";
import { ProblemLine } from "./ProblemLine";
import {
  buttonClass,
  dialogActionsClass,
  dialogFrameClass,
  dialogTitleClass,
  hintClass,
  primaryButtonClass,
  ruledListClass,
} from "./ui";
import { useModal } from "./useModal";

/** What each format does with Citations and Questions, under the format. */
const FORMAT_NOTES: Readonly<Record<ExportFormat, MessageKey>> = {
  docx: "export.options.docx",
  pdf: "export.options.pdf",
  markdown: "export.options.markdown",
};

/** A ruled row of the sheet: its words on the left, its control on the right. */
const rowClass = "flex min-h-11 flex-wrap items-center justify-between gap-x-4 gap-y-1 py-2";

/**
 * The Mind's citation style (#289), which its export is written in, with the button that
 * changes it: the same picker as the References block's. It changes at once, not on Save.
 */
function CitationStyleRow({ mindId }: { mindId: string }) {
  const t = useT();
  const [style, setStyle] = useState<MindCitationStyle | null>(null);
  const picking = useCitationStylePicker((state) => state.open?.mindId === mindId);
  useEffect(() => {
    // Read again once the picker closes: the User may have chosen another.
    if (picking) return;
    let current = true;
    core.getMindCitationStyle(mindId).then(
      (found) => {
        if (current) setStyle(found);
      },
      (error: unknown) => useAppStore.getState().reportError(error),
    );
    return () => {
      current = false;
    };
  }, [mindId, picking]);
  return (
    <div className={rowClass}>
      <span id="export-citation-style" className="text-ui text-ink">
        {t("export.citationStyle")}
      </span>
      <button
        type="button"
        data-testid="export-citation-style"
        aria-labelledby="export-citation-style export-citation-style-value"
        aria-haspopup="dialog"
        onClick={(event) => useCitationStylePicker.getState().show(mindId, event.currentTarget)}
        className={`${buttonClass} min-w-0 gap-1`}
      >
        <span id="export-citation-style-value" className="truncate">
          {style ? styleLabel(style.style, t) : "…"}
        </span>
        <ChevronDownLineIcon className="size-3.5 shrink-0 text-ink-meta" />
      </button>
    </div>
  );
}

/** Where exports go: the Output location, which the first export asks for, and how to change it. */
function SaveToRow() {
  const t = useT();
  const [output, setOutput] = useState<OutputSettings | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    let current = true;
    core.getOutputSettings().then(
      (found) => {
        if (current) setOutput(found);
      },
      (error: unknown) => useAppStore.getState().reportError(error),
    );
    return () => {
      current = false;
    };
  }, []);
  const change = async () => {
    setProblem(null);
    try {
      const next = await chooseOutputLocation();
      if (next) setOutput(next);
    } catch (failure) {
      setProblem(errorMessage(failure));
    }
  };
  const location = output?.location ?? null;
  return (
    <div className="flex flex-col gap-1 py-2">
      <div className={rowClass}>
        <div className="flex min-w-0 flex-col">
          <span className="text-ui text-ink">{t("export.saveTo")}</span>
          <span
            data-testid="export-save-to"
            title={location ?? undefined}
            className="truncate text-[13px] leading-5 text-ink-meta"
          >
            {output === null
              ? "…"
              : location
                ? folderName(location)
                : t("outputLocation.notChosen")}
          </span>
        </div>
        <button
          type="button"
          data-testid="export-save-to-change"
          onClick={() => void change()}
          className={buttonClass}
        >
          {t(location ? "outputLocation.change" : "outputLocation.choose")}
        </button>
      </div>
      {problem && (
        <ProblemLine testId="export-save-to-problem">
          {t("outputLocation.failed", { reason: problem })}
        </ProblemLine>
      )}
    </div>
  );
}

/**
 * The Export options sheet (#100): the format Export and ⇧⌘E use, each format's own options,
 * the Mind's citation style and where exports go. It remembers what was last saved; Cancel,
 * Esc and ✕ leave the options as they were. Open while `mind` is set.
 */
export function ExportDialog({ mind, onClose }: { mind: Mind | null; onClose(): void }) {
  const dialog = useModal(mind !== null);
  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      data-testid="export-dialog"
      aria-labelledby="export-title"
      // A sheet hanging from the tab strip, as the design has it, rather than a dialog in the middle.
      className={`${dialogFrameClass} mt-11 mb-auto max-h-[calc(100vh-68px)] w-[min(520px,calc(100vw-48px))] overflow-y-auto rounded-t-none`}
    >
      {/* Mounted only while open, so each opening starts from the options last saved. */}
      {mind && <ExportOptionsForm mind={mind} onDone={onClose} />}
    </dialog>
  );
}

function ExportOptionsForm({ mind, onDone }: { mind: Mind; onDone(): void }) {
  const t = useT();
  const formatId = useId();
  const questionsId = useId();
  const hintId = useId();
  const hint = t("export.options.saveHint", { shortcut: shortcutLabel("mind.export") });
  const [draft, setDraft] = useState<ExportPreferences>(exportPreferences);
  const [questions, setQuestions] = useState(0);
  const [saving, setSaving] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const { format } = draft;

  // How many Questions the Mind has, which a deliverable leaves out unless asked: said under its switch.
  useEffect(() => {
    let current = true;
    core.previewMindExport(mind.id, { format: "markdown" }).then(
      (preview) => {
        if (current) setQuestions(preview.questions);
      },
      (error: unknown) => useAppStore.getState().reportError(error),
    );
    return () => {
      current = false;
    };
  }, [mind.id]);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setProblem(null);
    try {
      const settings = await core.updateSettings({ user: { exportOptions: draft } });
      useAppStore.setState({ settings });
      onDone();
    } catch (failure) {
      setProblem(errorMessage(failure));
    } finally {
      setSaving(false);
    }
  };

  const includeQuestions = format === "markdown" || draft[format].includeQuestions;
  const leftOut = includeQuestions ? 0 : questions;

  return (
    <form onSubmit={(event) => void save(event)} className="flex flex-col">
      <header className="flex items-center justify-between gap-3 pt-4 pr-4 pb-2.5 pl-6">
        <h2 id="export-title" className={dialogTitleClass}>
          {t("export.options.title")}
        </h2>
        <button
          type="button"
          data-testid="export-options-close"
          aria-label={t("export.options.close")}
          title={t("export.options.close")}
          onClick={onDone}
          className="inline-flex size-8 shrink-0 items-center justify-center rounded-md text-ink-secondary hover:bg-hover hover:text-ink"
        >
          <CloseLineIcon className="size-4" />
        </button>
      </header>

      <div className={`mx-6 ${ruledListClass}`}>
        <div className={rowClass}>
          <label htmlFor={formatId} className="text-ui text-ink">
            {t("export.format")}
          </label>
          <span className="relative flex shrink-0">
            <select
              id={formatId}
              data-testid="export-format"
              value={format}
              onChange={(event) =>
                setDraft({ ...draft, format: event.target.value as ExportFormat })
              }
              className="h-8 w-52 max-w-full appearance-none rounded-sm border border-rule-strong bg-sheet pr-8 pl-2.5 text-ui text-ink focus:border-accent"
            >
              {EXPORT_FORMATS.map((each) => (
                <option key={each} value={each}>
                  {t(FORMAT_LABELS[each])}
                </option>
              ))}
            </select>
            <ChevronDownLineIcon className="pointer-events-none absolute top-2.5 right-2.5 size-3.5 text-ink-meta" />
          </span>
        </div>

        <p data-testid="export-format-note" className="py-3 text-[13px] leading-5 text-ink-meta">
          {t(FORMAT_NOTES[format])}
        </p>

        {format !== "markdown" && (
          <div>
            <div className={rowClass}>
              <label htmlFor={questionsId} className="text-ui text-ink">
                {t("export.includeQuestions")}
              </label>
              <input
                id={questionsId}
                type="checkbox"
                role="switch"
                data-testid="export-include-questions"
                checked={draft[format].includeQuestions}
                aria-checked={draft[format].includeQuestions}
                onChange={(event) =>
                  setDraft({ ...draft, [format]: { includeQuestions: event.target.checked } })
                }
                className="switch"
              />
            </div>
            {leftOut > 0 && (
              <p className={`-mt-2 pb-2 ${hintClass}`}>
                {leftOut === 1
                  ? t("export.questionsLeftOut.one")
                  : t("export.questionsLeftOut.other", { count: leftOut })}
              </p>
            )}
          </div>
        )}

        <CitationStyleRow mindId={mind.id} />
        <SaveToRow />
      </div>

      {problem && (
        <ProblemLine testId="export-problem" className="mx-6 mt-3">
          {t("export.failed.options", { reason: problem })}
        </ProblemLine>
      )}

      <div className={`${dialogActionsClass} px-6 pt-5 pb-6`}>
        <button type="button" onClick={onDone} className={buttonClass}>
          {t("export.cancel")}
        </button>
        <button
          type="submit"
          data-testid="export-save"
          disabled={saving}
          aria-describedby={hintId}
          title={hint}
          className={primaryButtonClass}
        >
          {t("export.options.save")}
        </button>
      </div>
      <span id={hintId} className="sr-only">
        {hint}
      </span>
    </form>
  );
}
