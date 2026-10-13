import type { Editor } from "@tiptap/core";
import { type FormEvent, useEffect, useState } from "react";
import type * as Y from "yjs";
import type { MessageKey } from "../../../shared/i18n";
import {
  countBasisOf,
  observeWordCountSettings,
  setCountBasis,
  setWordTarget,
  wordTargetOf,
} from "../../../shared/mindWordCount";
import {
  COUNT_BASES,
  type CountBasis,
  countFor,
  mostlyCjk,
  NO_COUNT,
  parseTarget,
  type TextCount,
} from "../../../shared/wordCount";
import { CheckLineIcon } from "../components/lineIcons";
import { ProblemMark } from "../components/ProblemLine";
import { usePopoverMenu } from "../components/usePopoverMenu";
import { useLanguage, useT } from "../i18n";
import { countMind } from "./wordCountText";

/**
 * How long the editor is left alone before the count is worked out. Counting
 * runs after this pause, never in a keystroke's own turn, so typing stays as
 * fast in a long Mind as in a short one.
 */
export const COUNT_IDLE_MS = 300;

const BASIS_NAMES: Record<CountBasis, MessageKey> = {
  words: "wordCount.basis.words",
  characters: "wordCount.basis.characters",
  charactersNoSpaces: "wordCount.basis.charactersNoSpaces",
};

interface Counts {
  total: TextCount;
  selection: TextCount | null;
}

/** The Mind's counts, worked out a moment after the text or the selection last changed. */
function useCounts(editor: Editor): Counts {
  const [counts, setCounts] = useState<Counts>({ total: NO_COUNT, selection: null });

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const count = () => {
      timer = undefined;
      if (editor.isDestroyed) return;
      const { from, to } = editor.state.selection;
      setCounts(countMind(editor.state.doc, { from, to }));
    };
    const later = () => {
      clearTimeout(timer);
      timer = setTimeout(count, COUNT_IDLE_MS);
    };
    count();
    editor.on("update", later);
    editor.on("selectionUpdate", later);
    return () => {
      clearTimeout(timer);
      editor.off("update", later);
      editor.off("selectionUpdate", later);
    };
  }, [editor]);

  return counts;
}

/** The Mind's target and what its count shows, kept up to date as they change here or elsewhere. */
function useCountSettings(doc: Y.Doc) {
  const read = () => ({ target: wordTargetOf(doc), basis: countBasisOf(doc) });
  const [settings, setSettings] = useState(read);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `read` only reads `doc`.
  useEffect(() => {
    setSettings(read());
    return observeWordCountSettings(doc, () => setSettings(read()));
  }, [doc]);
  return settings;
}

/**
 * The quiet count under the Mind's title: words for English and characters for
 * Chinese (what the hover says), the selection's count of the whole while text
 * is selected, and the progress to the Mind's target if it has one. Its menu
 * chooses what is counted (words, or characters with or without spaces) and
 * sets the target. Both are remembered for the Mind (`mindWordCount`).
 */
export function WordCount({ editor, doc }: { editor: Editor; doc: Y.Doc }) {
  const t = useT();
  const language = useLanguage();
  const counts = useCounts(editor);
  const { target, basis } = useCountSettings(doc);
  const menu = usePopoverMenu();
  const [draft, setDraft] = useState("");
  const [invalid, setInvalid] = useState(false);

  const format = (value: number) => new Intl.NumberFormat(language).format(value);
  // Words, unless the writing is Chinese or Japanese (or the User asked for characters).
  const counted = basis === "words" && !mostlyCjk(counts.total) ? "words" : "characters";
  const unit = (value: number) => {
    const name = counted;
    return t(value === 1 ? `wordCount.unit.${name}.one` : `wordCount.unit.${name}`);
  };

  const total = countFor(counts.total, basis);
  const selected = counts.selection ? countFor(counts.selection, basis) : null;
  const text =
    selected !== null
      ? t("wordCount.selected", {
          selected: format(selected),
          total: format(total),
          unit: unit(total),
        })
      : target !== null
        ? t("wordCount.target", {
            count: format(total),
            target: format(target),
            unit: unit(target),
          })
        : t("wordCount.total", { count: format(total), unit: unit(total) });
  const percent = target === null ? 0 : Math.min(100, Math.round((total / target) * 100));

  const setTarget = (event: FormEvent) => {
    event.preventDefault();
    const value = parseTarget(draft);
    if (value === null) {
      setInvalid(true);
      return;
    }
    setWordTarget(doc, value);
    menu.close();
  };

  return (
    <div className="mind-count" data-testid="mind-count">
      <button
        type="button"
        {...menu.buttonProps}
        data-testid="mind-count-button"
        data-total={total}
        data-selected={selected ?? undefined}
        title={t("wordCount.rule")}
        onClick={() => {
          setDraft(target === null ? "" : format(target));
          setInvalid(false);
        }}
        className="mind-count-button"
      >
        {text}
      </button>
      {target !== null && (
        <span
          role="progressbar"
          aria-label={t("wordCount.progress", { percent })}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          data-testid="mind-count-progress"
          className="mind-count-progress"
        >
          <span className="mind-count-fill" style={{ width: `${percent}%` }} />
        </span>
      )}
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("wordCount.menu")}
        data-testid="mind-count-menu"
        className="model-menu count-menu"
      >
        {menu.open && (
          <>
            <div className="model-menu-list">
              <p className="model-menu-intro">{t("wordCount.scope")}</p>
              {COUNT_BASES.map((each) => (
                <button
                  key={each}
                  type="button"
                  role="menuitemradio"
                  aria-checked={each === basis}
                  data-testid="mind-count-basis"
                  data-basis={each}
                  onClick={() => {
                    setCountBasis(doc, each);
                    menu.close();
                  }}
                  className="model-menu-item"
                >
                  <CheckLineIcon
                    className="model-menu-check"
                    style={{ visibility: each === basis ? "visible" : "hidden" }}
                  />
                  <span className="model-menu-name">{t(BASIS_NAMES[each])}</span>
                </button>
              ))}
            </div>
            <form onSubmit={setTarget} className="count-target">
              <label htmlFor={`${menu.menuProps.id}-target`}>{t("wordCount.target.label")}</label>
              <input
                id={`${menu.menuProps.id}-target`}
                type="text"
                className="count-target-field"
                inputMode="numeric"
                data-testid="mind-count-target"
                value={draft}
                placeholder={t("wordCount.target.placeholder")}
                aria-invalid={invalid}
                onChange={(event) => {
                  setDraft(event.target.value);
                  setInvalid(false);
                }}
                // Keys that move through the menu's items belong to the field while it has the focus.
                onKeyDown={(event) => {
                  if (event.key !== "Escape") event.stopPropagation();
                }}
              />
              <button
                type="submit"
                role="menuitem"
                data-testid="mind-count-set"
                className="count-target-set"
              >
                {t("wordCount.target.set")}
              </button>
              {invalid && (
                <p role="alert" className="count-target-problem">
                  <ProblemMark />
                  {t("wordCount.target.invalid")}
                </p>
              )}
            </form>
            {target !== null && (
              <div className="model-menu-foot">
                <button
                  type="button"
                  role="menuitem"
                  data-testid="mind-count-remove"
                  className="model-menu-link"
                  onClick={() => {
                    setWordTarget(doc, null);
                    menu.close();
                  }}
                >
                  {t("wordCount.target.remove")}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
