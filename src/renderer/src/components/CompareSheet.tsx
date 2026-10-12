import type { Editor } from "@tiptap/core";
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import type * as Y from "yjs";
import { MAX_COMPARISON_DOCUMENTS, MAX_COMPARISON_QUESTIONS } from "../../../shared/comparisons";
import {
  closeCompareSheet,
  makeComparisonTable,
  type SheetSource,
  sheetDocuments,
  updateCompareSheet,
  useComparisons,
} from "../comparisons";
import { core } from "../core";
import { estimateText } from "../cost";
import { RemoveIcon, TableIcon } from "../editor/icons";
import { ModelChip, useMindModel } from "../editor/ModelChip";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";
import { ChevronDownLineIcon, PlusLineIcon } from "./lineIcons";
import { ProblemLine } from "./ProblemLine";
import { ReadinessLine } from "./providers/ChatReadinessNotice";
import { buttonStyle, menuClass, menuItemClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * Compare Documents (#98; the canvas board Final-37-Compare-Start): at the
 * foot of the Mind, the Documents of its Folder, ticked, two to a row, with
 * the Folder to take them from; the Questions, one per column, "+ Question"
 * adding one; "Anything else the table should show?"; and, at its foot, the
 * "Compare Documents" chip (× closes it), the estimate for a cloud model
 * billed per token (none for a plan or this computer, #83), the model chip
 * and "Make the table". Esc closes it, ⌘↵ makes the table.
 */
export function CompareSheet({
  mindId,
  doc,
  editor,
}: {
  mindId: string;
  doc: Y.Doc;
  editor: Editor;
}) {
  const t = useT();
  const language = useLanguage();
  const sheet = useComparisons((state) => state.sheets[mindId]);
  const problem = useComparisons((state) => state.problems[mindId]);
  const allDocuments = useAppStore((state) => state.documents);
  const library = useAppStore((state) => state.library);
  const settings = useAppStore((state) => state.settings);
  const model = useMindModel(doc);
  const region = useRef<HTMLElement>(null);
  const questionFields = useRef<(HTMLInputElement | null)[]>([]);
  const focusQuestion = useRef<number | null>(null);
  const [estimate, setEstimate] = useState<string | null>(null);

  const source = sheet?.source ?? "all";
  // biome-ignore lint/correctness/useExhaustiveDependencies: listed again when the Library or its Documents change.
  const documents = useMemo(
    () =>
      sheetDocuments(source).sort((a, b) =>
        a.name.localeCompare(b.name, language, { numeric: true }),
      ),
    [source, allDocuments, library, settings, language],
  );
  const checked = useMemo(() => {
    const ticked = new Set(sheet?.checked ?? []);
    return documents.filter((document) => ticked.has(document.id));
  }, [documents, sheet?.checked]);
  const questions = sheet?.questions ?? [];
  const asked = questions.filter((question) => question.trim() !== "");
  const cells = checked.length * asked.length;

  // The first Question's field takes the focus as the sheet opens.
  const open = sheet !== undefined;
  // biome-ignore lint/correctness/useExhaustiveDependencies: only as it opens, not as Questions are typed.
  useEffect(() => {
    if (!open) return;
    if (questions.length === 0) updateCompareSheet(mindId, { questions: [""] });
    focusQuestion.current = 0;
    region.current?.scrollIntoView({ block: "nearest" });
  }, [open, mindId]);
  // Once its field is drawn: the first, as the sheet opens, has to wait for its Question.
  useEffect(() => {
    const index = focusQuestion.current;
    const field = index === null ? null : questionFields.current[index];
    if (!field) return;
    focusQuestion.current = null;
    field.focus();
  });

  // The estimate, again as the cells or the model change; none for a plan or this computer.
  useEffect(() => {
    if (!open || cells === 0) {
      setEstimate(null);
      return;
    }
    let current = true;
    core
      .estimateRun({ items: cells, ...(model && { model }) })
      .then((result) => {
        if (current) setEstimate(estimateText(result, t, language));
      })
      .catch(() => {
        if (current) setEstimate(null);
      });
    return () => {
      current = false;
    };
  }, [open, cells, model, t, language]);

  if (!sheet) return null;

  const close = () => {
    closeCompareSheet(mindId);
    editor.commands.focus();
  };
  const make = async () => {
    if (checked.length === 0 || asked.length === 0) return;
    const made = await makeComparisonTable(
      mindId,
      checked.map((document) => document.id),
      model,
    );
    if (made && !editor.isDestroyed) editor.commands.focus();
  };
  const setQuestions = (next: string[]) => updateCompareSheet(mindId, { questions: next });
  const addQuestion = (at = questions.length) => {
    if (questions.length >= MAX_COMPARISON_QUESTIONS) return;
    setQuestions([...questions.slice(0, at), "", ...questions.slice(at)]);
    focusQuestion.current = at;
  };
  const toggle = (documentId: string) => {
    const ticked = new Set(sheet.checked);
    if (ticked.has(documentId)) ticked.delete(documentId);
    else ticked.add(documentId);
    updateCompareSheet(mindId, { checked: [...ticked] });
  };
  const chooseSource = (next: SheetSource) =>
    updateCompareSheet(mindId, {
      source: next,
      checked: sheetDocuments(next)
        .slice(0, MAX_COMPARISON_DOCUMENTS)
        .map((document) => document.id),
    });

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "Escape" && !event.defaultPrevented) {
      event.preventDefault();
      close();
    } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      void make();
    }
  };
  const onQuestionKey = (index: number) => (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing || event.metaKey || event.ctrlKey) return;
    if (event.key === "Enter") {
      event.preventDefault();
      if (index === questions.length - 1) addQuestion();
      else questionFields.current[index + 1]?.focus();
    } else if (event.key === "Backspace" && questions[index] === "" && questions.length > 1) {
      event.preventDefault();
      setQuestions(questions.filter((_, at) => at !== index));
      focusQuestion.current = Math.max(0, index - 1);
    }
  };

  const folderName =
    source === "all"
      ? t("comparison.sheet.allDocuments")
      : (library?.groups.find((group) => group.id === source)?.name ??
        t("comparison.sheet.allDocuments"));
  const empty =
    documents.length === 0
      ? "comparison.sheet.noDocuments"
      : checked.length === 0
        ? "comparison.sheet.noneChecked"
        : asked.length === 0
          ? "comparison.sheet.noQuestions"
          : null;

  return (
    <section
      ref={region}
      data-testid="compare-sheet"
      aria-label={t("comparison.sheet.region")}
      className="mind-measure compare-sheet"
      onKeyDown={onKeyDown}
    >
      <div className="compare-sheet-body">
        <div className="compare-sheet-documents">
          <div className="compare-sheet-heading">
            <SourceMenu
              label={t("comparison.sheet.documentsIn", { folder: folderName })}
              onChoose={chooseSource}
            />
            <span data-testid="compare-sheet-counted" className="compare-sheet-count">
              {t("comparison.sheet.counted", { checked: checked.length, total: documents.length })}
            </span>
          </div>
          {documents.length === 0 ? (
            <ProblemLine
              role="status"
              testId="compare-sheet-no-documents"
              action={{
                label: t("sidebar.plus.addDocuments"),
                onClick: () => void useAppStore.getState().pickDocuments(),
              }}
            >
              {t("comparison.sheet.noDocuments")}
            </ProblemLine>
          ) : (
            <ul className="compare-sheet-list">
              {documents.map((document) => (
                <li key={document.id}>
                  <label className="compare-sheet-document" title={document.name}>
                    <input
                      type="checkbox"
                      data-testid="compare-sheet-document"
                      checked={sheet.checked.includes(document.id)}
                      onChange={() => toggle(document.id)}
                    />
                    <span className="truncate">{document.name}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="compare-sheet-questions">
          <div className="compare-sheet-heading">
            <span>{t("comparison.sheet.questions")}</span>
            <span className="compare-sheet-count">{asked.length}</span>
          </div>
          <ol className="compare-sheet-question-list">
            {questions.map((question, index) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: a Question is its place in the list while it is typed.
              <li key={index} className="compare-sheet-question">
                <span className="compare-sheet-number">{index + 1}</span>
                <input
                  ref={(field) => {
                    questionFields.current[index] = field;
                  }}
                  data-testid="compare-sheet-question"
                  aria-label={t("comparison.addQuestion.label")}
                  placeholder={t("comparison.addQuestion.placeholder")}
                  value={question}
                  onChange={(event) =>
                    setQuestions(
                      questions.map((each, at) => (at === index ? event.target.value : each)),
                    )
                  }
                  onKeyDown={onQuestionKey(index)}
                />
                {questions.length > 1 && (
                  <button
                    type="button"
                    className="compare-sheet-remove"
                    aria-label={t("comparison.sheet.removeQuestion", {
                      question: question || String(index + 1),
                    })}
                    onClick={() => setQuestions(questions.filter((_, at) => at !== index))}
                  >
                    <RemoveIcon className="size-3" />
                  </button>
                )}
              </li>
            ))}
          </ol>
          {questions.length < MAX_COMPARISON_QUESTIONS && (
            <button
              type="button"
              data-testid="compare-sheet-add-question"
              className="compare-sheet-add"
              onClick={() => addQuestion()}
            >
              <PlusLineIcon className="size-3.5" />
              {t("comparison.addQuestion")}
            </button>
          )}
        </div>
      </div>
      <input
        data-testid="compare-sheet-instruction"
        className="compare-sheet-instruction"
        aria-label={t("comparison.sheet.instruction")}
        placeholder={t("comparison.sheet.instruction")}
        value={sheet.instruction}
        onChange={(event) => updateCompareSheet(mindId, { instruction: event.target.value })}
      />
      {problem &&
        (problem.reason === "not-ready" ? (
          <ReadinessLine
            readiness={problem.readiness}
            testId="compare-sheet-not-ready"
            className="compare-sheet-problem"
            setUpLabel={t("comparison.notReady")}
            refused
          />
        ) : (
          <ProblemLine testId="compare-sheet-not-sent" className="compare-sheet-problem">
            {t("answer.error.consent-declined")}
          </ProblemLine>
        ))}
      <div className="compare-sheet-foot">
        <span className="compare-sheet-chip">
          <TableIcon className="size-3.5" />
          {t("comparison.sheet.region")}
          <button
            type="button"
            data-testid="compare-sheet-close"
            aria-label={t("comparison.sheet.close")}
            title={t("comparison.sheet.close")}
            onClick={close}
          >
            <RemoveIcon className="size-3" />
          </button>
        </span>
        <span
          data-testid="compare-sheet-note"
          className="compare-sheet-note"
          title={estimate ?? undefined}
        >
          {empty ? t(empty) : estimate}
        </span>
        <ModelChip doc={doc} onChosen={() => region.current?.focus()} />
        <button
          type="button"
          data-testid="compare-sheet-make"
          className={buttonStyle("primary", "sm")}
          disabled={empty !== null || sheet.making}
          onClick={() => void make()}
        >
          {t("comparison.sheet.make")}
        </button>
      </div>
    </section>
  );
}

/** "Documents in {Folder} ⌄": the Folder to take them from, or All Documents. */
function SourceMenu({ label, onChoose }: { label: string; onChoose(source: SheetSource): void }) {
  const t = useT();
  const groups = useAppStore((state) => state.library?.groups) ?? [];
  const menu = usePopoverMenu();
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="compare-sheet-source"
        className="compare-sheet-source"
        title={t("comparison.sheet.chooseFolder")}
      >
        <span className="truncate">{label}</span>
        <ChevronDownLineIcon className="size-3 shrink-0" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("comparison.sheet.chooseFolder")}
        className={menuClass}
      >
        {menu.open && (
          <>
            {groups.map((group) => (
              <button
                key={group.id}
                type="button"
                role="menuitem"
                className={menuItemClass}
                onClick={() => {
                  onChoose(group.id);
                  menu.close();
                }}
              >
                {group.name}
              </button>
            ))}
            <button
              type="button"
              role="menuitem"
              className={menuItemClass}
              onClick={() => {
                onChoose("all");
                menu.close();
              }}
            >
              {t("comparison.sheet.allDocuments")}
            </button>
          </>
        )}
      </div>
    </>
  );
}
