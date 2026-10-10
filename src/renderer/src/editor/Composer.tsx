import { type Editor, isMacOS } from "@tiptap/core";
import { Selection } from "@tiptap/pm/state";
import {
  type ChangeEvent,
  type ClipboardEvent,
  type KeyboardEvent,
  type Ref,
  type RefObject,
  useEffect,
  useId,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type * as Y from "yjs";
import type { SearchScope } from "../../../core/api";
import { mindModelOf } from "../../../shared/mindModel";
import { hasSearchScope, type ScopeKind, scopeIds } from "../../../shared/searchScope";
import { useAnswers } from "../answers";
import { SearchIcon } from "../components/icons";
import { ProblemLine } from "../components/ProblemLine";
import { ReadinessExplanation, settingsPageFor } from "../components/providers/ChatReadinessNotice";
import {
  composerKey,
  type Draft,
  isLongPaste,
  NO_SCOPE,
  type Paste,
  useComposer,
} from "../composer";
import { core } from "../core";
import { useT } from "../i18n";
import type { ScopeChoice } from "../scope";
import { skillDescription } from "../skills";
import { useAppStore } from "../store";
import { AnswerAnnouncer } from "./AnswerAnnouncer";
import {
  cursorBelowAnswer,
  placeQuestion,
  type QuestionToAsk,
  questionPlace,
  takeBackQuestion,
  whenAnswerShown,
} from "./composerAsk";
import { AskArrowIcon, CollapseIcon, ExpandIcon } from "./icons";
import { ModelChip } from "./ModelChip";
import { PasteChip, usePasteLabel } from "./PasteChip";
import { SkillChip } from "./QuestionView";
import { ScopeChips } from "./ScopeChips";
import { type ChoiceListHandle, ScopeChoiceList } from "./ScopePicker";
import { SkillChoiceList } from "./SkillChoiceList";

/** In the composer, a long Search scope shows this many chips, then "+N more". */
const CHIPS_SHOWN = 3;

/** From this many lines, the composer offers Expand (DESIGN.md, Composer › Height). */
const EXPAND_FROM_LINES = 4;

let measuring: CanvasRenderingContext2D | null = null;

/** The text's width on one line, in the font the field draws it in. */
function widthOf(text: string, field: HTMLElement): number {
  measuring ??= document.createElement("canvas").getContext("2d");
  if (!measuring) return 0;
  measuring.font = getComputedStyle(field).font;
  return measuring.measureText(text).width;
}

/** The "@" or "/" being typed: what it picks, what follows it, and where it is in the text. */
interface Picking {
  kind: "scope" | "skill";
  query: string;
  /** Where the "@" or "/" is. */
  from: number;
  /** The caret: the end of what was typed after it. */
  to: number;
}

/** An "@" or "/" at the start or after a space, and what follows it up to the caret. */
const TRIGGER = /(?:^|\s)([@/])([^\s@/]*)$/;

const isCommand = (event: { metaKey: boolean; ctrlKey: boolean }) =>
  isMacOS() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;

/** What the composer can be asked to do from outside: take the focus, with a Search scope. */
export interface ComposerHandle {
  /** Takes the focus; a `scope` (from the Library) becomes the Mind's Search scope. */
  focus(scope?: SearchScope | null): void;
}

/**
 * The one input at the foot of the Mind (DESIGN.md, Composer): "Ask, or say
 * what to change…". ⌘J focuses it from anywhere in the window; Enter asks
 * (Shift+Enter breaks the line); Esc closes its picker, or goes back to the
 * note. Asking puts the Question into the note as one grey line, at the cursor
 * (`cursorInMind`) or at the end of the Mind, and its Answer streams in below.
 *
 * In its row: a Skill chosen with "/", the text, the quiet model chip, the
 * Search scope (a chip for each Folder, Tag or Document, or "All Documents";
 * "@" adds more), and the Ask button (an up arrow). Why a Question couldn't
 * be asked shows on one line above them, with the one thing that fixes it.
 */
export function Composer({
  editor,
  doc,
  mindId,
  cursorInMind,
  ref,
}: {
  editor: Editor;
  doc: Y.Doc;
  mindId: string;
  /** Whether the Mind's cursor is where a Question goes: false sends it to the end. */
  cursorInMind: RefObject<boolean>;
  ref?: Ref<ComposerHandle>;
}) {
  const t = useT();
  const draft = useComposer((state) => state.drafts[mindId]) ?? EMPTY;
  const problem = useAnswers((state) => state.blocked[composerKey(mindId)]);
  const openSettings = useAppStore((state) => state.openSettings);
  const skills = useAppStore((state) => state.skills);
  const field = useRef<HTMLTextAreaElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<ChoiceListHandle>(null);
  const listId = useId();
  const row = useRef<HTMLDivElement>(null);
  const controls = useRef<HTMLDivElement>(null);
  /** The text runs past one line: it takes the full width and the controls go under it. */
  const [stacked, setStacked] = useState(false);
  /** How many lines the text takes, for Expand. */
  const [lines, setLines] = useState(1);
  const [expanded, setExpanded] = useState(false);
  /** The pasted text being read (View), and whether one couldn't be saved. */
  const [viewing, setViewing] = useState<Paste | null>(null);
  const [saveFailed, setSaveFailed] = useState(false);
  const pasteLabel = usePasteLabel();
  const [picking, setPicking] = useState<Picking | null>(null);
  const [option, setOption] = useState(-1);
  /** An "@" or "/" whose picker Esc closed: it stays text. */
  const dismissed = useRef<number | null>(null);
  /** Where the caret goes once the text drawn changes. */
  const caret = useRef<number | null>(null);
  const enabledSkills = skills.filter((skill) => skill.enabled);
  const skill = draft.skill ? skills.find((each) => each.name === draft.skill) : undefined;

  const update = (change: Partial<Draft>) => useComposer.getState().update(mindId, change);

  // Single-line text sits beside the controls; once it would run past one line, it takes the
  // full width and they move to a row under it. Decided from the width one line has beside the
  // controls, never from how the text wrapped, so the two layouts can't flip each other.
  const measureLayout = () => {
    const element = field.current;
    const rowElement = row.current;
    if (!element || !rowElement) return;
    const style = getComputedStyle(rowElement);
    const fieldStyle = getComputedStyle(element);
    const gap = Number.parseFloat(style.columnGap) || 0;
    const skillElement = rowElement.querySelector<HTMLElement>(".question-skill");
    const room =
      rowElement.clientWidth -
      Number.parseFloat(style.paddingLeft) -
      Number.parseFloat(style.paddingRight) -
      (skillElement ? skillElement.offsetWidth + gap : 0) -
      (controls.current?.offsetWidth ?? 0) -
      gap -
      Number.parseFloat(fieldStyle.paddingLeft);
    const runsPast = draft.text.includes("\n") || widthOf(draft.text, element) > room - 2;
    const lineHeight = Number.parseFloat(fieldStyle.lineHeight) || 20;
    const padding =
      Number.parseFloat(fieldStyle.paddingTop) + Number.parseFloat(fieldStyle.paddingBottom);
    setStacked(runsPast);
    setLines(Math.max(1, Math.round((element.scrollHeight - padding) / lineHeight)));
  };
  const measure = useRef(measureLayout);
  measure.current = measureLayout;
  // After every drawing: the text, a Skill or the controls may have changed its width.
  useLayoutEffect(measureLayout);
  useEffect(() => {
    const element = row.current;
    if (!element) return;
    const observer = new ResizeObserver(() => measure.current());
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const showExpand = lines >= EXPAND_FROM_LINES || expanded;

  useLayoutEffect(() => {
    const at = caret.current;
    if (at === null || !field.current) return;
    caret.current = null;
    field.current.setSelectionRange(at, at);
  });

  const pickingAt = (text: string, at: number): Picking | null => {
    const match = TRIGGER.exec(text.slice(0, at));
    if (!match) {
      dismissed.current = null;
      return null;
    }
    const query = match[2] ?? "";
    const from = at - query.length - 1;
    if (dismissed.current === from) return null;
    const kind = match[1] === "@" ? "scope" : "skill";
    // A slash is just a slash while there is no Skill to choose.
    if (kind === "skill" && enabledSkills.length === 0) return null;
    return { kind, query, from, to: at };
  };

  const focus = (end = true) => {
    const element = field.current;
    if (!element) return;
    element.focus();
    if (end) element.setSelectionRange(element.value.length, element.value.length);
  };

  useImperativeHandle(ref, () => ({
    focus(scope) {
      if (scope) update({ scope });
      cursorInMind.current = false;
      focus();
    },
  }));

  // ⌘J, anywhere in the window but a dialog: the composer takes the focus.
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.shiftKey || !isCommand(event)) return;
      if (event.code !== "KeyJ" || document.querySelector("dialog[open]")) return;
      event.preventDefault();
      const element = field.current;
      if (element && document.activeElement !== element) {
        element.focus();
        element.setSelectionRange(element.value.length, element.value.length);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Asked from elsewhere, e.g. a Skill chosen in the note's slash menu.
  const focusRequests = useComposer((state) => state.focusRequests);
  const seenRequests = useRef(focusRequests);
  useEffect(() => {
    if (focusRequests === seenRequests.current) return;
    seenRequests.current = focusRequests;
    field.current?.focus();
  }, [focusRequests]);

  const onChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const text = event.target.value;
    update({ text });
    setPicking(pickingAt(text, event.target.selectionStart));
  };

  /** The caret moved (arrows, a click): a picker opens or closes with it. */
  const onSelect = () => {
    const element = field.current;
    if (!element || element.selectionStart !== element.selectionEnd) return;
    const next = pickingAt(element.value, element.selectionStart);
    setPicking((current) =>
      current?.kind === next?.kind && current?.query === next?.query && current?.from === next?.from
        ? current
        : next,
    );
  };

  /** Takes the "@…" or "/…" being typed out of the text, the caret where it was. */
  const withoutTyped = (pick: Picking): string => {
    caret.current = pick.from;
    return draft.text.slice(0, pick.from) + draft.text.slice(pick.to);
  };

  const chooseScope = (choice: ScopeChoice) => {
    const ids = scopeIds(draft.scope, choice.kind);
    const scope = ids.includes(choice.id)
      ? draft.scope
      : { ...draft.scope, [SCOPE_LISTS[choice.kind]]: [...ids, choice.id] };
    update({ scope, ...(picking && { text: withoutTyped(picking) }) });
    setPicking(null);
    field.current?.focus();
  };

  const chooseSkill = (name: string) => {
    update({ skill: name, ...(picking && { text: withoutTyped(picking) }) });
    useAnswers.getState().dismiss(composerKey(mindId));
    setPicking(null);
    field.current?.focus();
  };

  const removeFromScope = (kind: ScopeKind, id: string) => {
    const left = scopeIds(draft.scope, kind).filter((each) => each !== id);
    update({ scope: { ...draft.scope, [SCOPE_LISTS[kind]]: left } });
  };

  /** A chip's name, or "All Documents": types the "@" that opens the picker. */
  const openScopePicker = () => {
    const element = field.current;
    const text = draft.text;
    const at = element ? element.selectionEnd : text.length;
    const space = at > 0 && !/\s/.test(text[at - 1] ?? "") ? " " : "";
    const next = `${text.slice(0, at)}${space}@${text.slice(at)}`;
    const from = at + space.length;
    dismissed.current = null;
    update({ text: next });
    caret.current = from + 1;
    setPicking({ kind: "scope", query: "", from, to: from + 1 });
    element?.focus();
  };

  /** A paste over 2,000 characters or 30 lines becomes a chip instead of going into the text. */
  const onPaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const pasted = event.clipboardData.getData("text/plain");
    if (!isLongPaste(pasted)) return;
    event.preventDefault();
    setSaveFailed(false);
    update({ pastes: [...draft.pastes, { id: crypto.randomUUID(), text: pasted }] });
  };

  const withoutPaste = (paste: Paste) => draft.pastes.filter((each) => each.id !== paste.id);

  const putBack = (paste: Paste) => {
    const element = field.current;
    const at = element ? element.selectionEnd : draft.text.length;
    const text = draft.text.slice(0, at) + paste.text + draft.text.slice(at);
    caret.current = at + paste.text.length;
    update({ text, pastes: withoutPaste(paste) });
    element?.focus();
  };

  /** Keeps the paste as a Document and adds it to the Search scope, so Answers can cite it. */
  const savePaste = async (paste: Paste) => {
    setSaveFailed(false);
    const first = paste.text.split("\n").find((line) => line.trim() !== "") ?? "";
    try {
      const saved = await core.saveTextAsDocument({
        name: first.trim().slice(0, 60) || t("composer.paste.name"),
        text: paste.text,
      });
      const now = useComposer.getState().draft(mindId);
      const ids = scopeIds(now.scope, "document");
      update({
        pastes: now.pastes.filter((each) => each.id !== paste.id),
        scope: ids.includes(saved.id)
          ? now.scope
          : { ...now.scope, documentIds: [...ids, saved.id] },
      });
    } catch {
      setSaveFailed(true);
    }
    field.current?.focus();
  };

  const ask = async () => {
    // As it is now, not as it was drawn: "Ask without it" has just taken the Skill off.
    const current = useComposer.getState().draft(mindId);
    const typed = current.text.trim();
    // What is pasted, in chips, goes with the words typed.
    const text = [typed, ...current.pastes.map((paste) => paste.text.trim())]
      .filter((part) => part !== "")
      .join("\n\n");
    if (!text || editor.isDestroyed) return;
    useAnswers.getState().dismiss(composerKey(mindId));
    const question: QuestionToAsk = {
      id: crypto.randomUUID(),
      text,
      model: mindModelOf(doc),
      scope: current.scope,
      skill: current.skill,
    };
    const selection = cursorInMind.current ? editor.state.selection : null;
    const placed = placeQuestion(editor, question, questionPlace(editor.state.doc, selection));
    if (!placed) return;
    update({ text: "", skill: null, pastes: [] });
    setExpanded(false);
    setViewing(null);
    setPicking(null);
    const line = editor.view.nodeDOM(placed.pos);
    if (line instanceof HTMLElement) line.scrollIntoView({ block: "nearest" });

    const result = await useAnswers.getState().askFromComposer(mindId, question.id);
    if (result?.asked) {
      // The next Question goes after this Answer; Esc comes back to write there. (Unless
      // another Mind is shown by then: this one's editor is gone, with its cursor.)
      const answered = await whenAnswerShown(editor, result.answerId);
      if (answered && !editor.isDestroyed && cursorBelowAnswer(editor, result.answerId)) {
        cursorInMind.current = true;
      }
      return;
    }
    // Not asked: the Question leaves the note, and its words come back here.
    if (!editor.isDestroyed) takeBackQuestion(editor, placed, text);
    const now = useComposer.getState().draft(mindId);
    if (now.text === "" && now.pastes.length === 0) {
      update({ text: typed, pastes: current.pastes, skill: now.skill ?? question.skill });
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Typing in an input method: its keys are its own.
    if (event.nativeEvent.isComposing || event.keyCode === 229) return;
    if (picking) {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        dismissed.current = picking.from;
        setPicking(null);
        return;
      }
      if (list.current?.onKeyDown(event.nativeEvent)) {
        event.preventDefault();
        return;
      }
    }
    if (event.key === "Enter" && !event.shiftKey && !event.altKey && !isCommand(event)) {
      event.preventDefault();
      void ask();
      return;
    }
    if (event.key === "Escape" && expanded) {
      // The tall editor goes back to its size; the next Esc goes to the note.
      event.preventDefault();
      setExpanded(false);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      // Back to the note, where the cursor was (or at its end), at once: what is typed next
      // goes there. (Tiptap's `focus` waits a frame, and the first keys would come here.)
      if (!cursorInMind.current) {
        editor.commands.command(({ tr }) => {
          tr.setSelection(Selection.atEnd(tr.doc));
          return true;
        });
      }
      editor.view.focus();
      editor.view.dispatch(editor.state.tr.scrollIntoView());
    }
  };

  const canAsk = draft.text.trim() !== "" || draft.pastes.length > 0;
  const activeOption = picking && option >= 0 ? `${listId}-${option}` : undefined;

  return (
    <div
      ref={root}
      data-testid="composer"
      data-expanded={expanded || undefined}
      className={expanded ? "composer composer--expanded" : "composer"}
    >
      {problem && (
        <ComposerProblem
          problem={problem}
          onSetUp={(page) => openSettings(page)}
          onAskWithoutSkill={() => {
            update({ skill: null });
            void ask();
          }}
        />
      )}
      {saveFailed && (
        <ProblemLine
          testId="composer-paste-save-failed"
          className="composer-problem"
          action={{ label: t("composer.paste.close"), onClick: () => setSaveFailed(false) }}
        >
          {t("composer.paste.saveFailed")}
        </ProblemLine>
      )}
      {viewing && (
        <section
          data-testid="composer-paste-view"
          aria-label={pasteLabel(viewing)}
          className="composer-view"
          onKeyDown={(event) => {
            if (event.key !== "Escape") return;
            event.preventDefault();
            event.stopPropagation();
            setViewing(null);
            field.current?.focus();
          }}
        >
          <header className="composer-view-head">
            <span className="truncate">{pasteLabel(viewing)}</span>
            <button
              type="button"
              // biome-ignore lint/a11y/noAutofocus: opened by the User's click, so reading starts here.
              autoFocus
              data-testid="composer-paste-view-close"
              onClick={() => {
                setViewing(null);
                field.current?.focus();
              }}
              className="composer-view-close"
            >
              {t("composer.paste.close")}
            </button>
          </header>
          <pre className="composer-view-text">{viewing.text}</pre>
        </section>
      )}
      <div ref={row} data-stacked={stacked || expanded || undefined} className="composer-row">
        {draft.pastes.length > 0 && (
          <div className="composer-pastes">
            {draft.pastes.map((paste) => (
              <PasteChip
                key={paste.id}
                paste={paste}
                onView={() => setViewing(paste)}
                onPutBack={() => putBack(paste)}
                onSave={() => void savePaste(paste)}
                onRemove={() => {
                  update({ pastes: withoutPaste(paste) });
                  field.current?.focus();
                }}
              />
            ))}
          </div>
        )}
        {draft.skill && (
          <SkillChip
            name={draft.skill}
            state={!skill ? "removed" : skill.enabled ? "enabled" : "disabled"}
            onRemove={() => {
              update({ skill: null });
              useAnswers.getState().dismiss(composerKey(mindId));
            }}
            testId="composer-skill"
          />
        )}
        <textarea
          ref={field}
          rows={1}
          data-testid="composer-input"
          aria-label={t("composer.label", { shortcut: SHORTCUT })}
          placeholder={t("composer.placeholder")}
          aria-autocomplete="list"
          aria-controls={picking ? listId : undefined}
          aria-activedescendant={activeOption}
          value={draft.text}
          onChange={onChange}
          onSelect={onSelect}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          onBlur={(event) => {
            if (!root.current?.contains(event.relatedTarget)) setPicking(null);
          }}
          className="composer-input"
        />
        <div ref={controls} className="composer-controls">
          <ModelChip doc={doc} onChosen={() => field.current?.focus()} />
          {hasSearchScope(draft.scope) ? (
            <ScopeChips
              scope={draft.scope}
              onRemove={removeFromScope}
              onChange={openScopePicker}
              shown={CHIPS_SHOWN}
              className="composer-scope"
              testId="composer-scope"
            />
          ) : (
            <button
              type="button"
              data-testid="composer-scope-all"
              aria-label={t("composer.scope.allLabel")}
              title={t("composer.scope.allTitle")}
              onClick={openScopePicker}
              className="composer-chip composer-scope-all"
            >
              <SearchIcon className="composer-chip-icon" />
              {t("composer.scope.all")}
            </button>
          )}
          <button
            type="button"
            data-testid="composer-ask"
            aria-label={t("composer.ask")}
            title={t("composer.ask")}
            disabled={!canAsk}
            // The text keeps the focus.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => void ask()}
            className="composer-ask"
          >
            <AskArrowIcon className="size-3" />
          </button>
        </div>
        {showExpand && (
          <button
            type="button"
            data-testid="composer-expand"
            aria-label={expanded ? t("composer.collapse") : t("composer.expand")}
            title={expanded ? t("composer.collapse") : t("composer.expand")}
            aria-pressed={expanded}
            // The text keeps the focus.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => {
              setExpanded(!expanded);
              field.current?.focus();
            }}
            className="composer-expand"
          >
            {expanded ? <CollapseIcon className="size-3.5" /> : <ExpandIcon className="size-3.5" />}
          </button>
        )}
      </div>
      {picking && (
        <div className="composer-picker">
          {picking.kind === "scope" ? (
            <ScopeChoiceList
              ref={list}
              id={listId}
              scope={draft.scope}
              query={picking.query}
              onChoose={chooseScope}
              onSelect={setOption}
            />
          ) : (
            <SkillChoiceList
              ref={list}
              id={listId}
              skills={enabledSkills}
              describe={(each) => skillDescription(each, t)}
              query={picking.query}
              onChoose={chooseSkill}
              onSelect={setOption}
            />
          )}
        </div>
      )}
      <AnswerAnnouncer mindId={mindId} />
    </div>
  );
}

const EMPTY: Draft = { text: "", scope: NO_SCOPE, skill: null, pastes: [] };

const SHORTCUT = isMacOS() ? "⌘J" : "Ctrl+J";

/** The draft's list of each kind of Search scope item. */
const SCOPE_LISTS: Record<ScopeKind, keyof SearchScope> = {
  folder: "folderIds",
  tag: "tagIds",
  document: "documentIds",
};

type AskProblem = NonNullable<ReturnType<typeof useAnswers.getState>["blocked"][string]>;

/** Why the last Question couldn't be asked: one line with the one thing that fixes it. */
function ComposerProblem({
  problem,
  onSetUp,
  onAskWithoutSkill,
}: {
  problem: AskProblem;
  onSetUp(page: ReturnType<typeof settingsPageFor>): void;
  onAskWithoutSkill(): void;
}) {
  const t = useT();
  if (problem.kind === "not-ready") {
    return (
      <ProblemLine
        role="status"
        testId="composer-not-ready"
        className="composer-problem"
        action={{
          label: t("question.setUp"),
          onClick: () => onSetUp(settingsPageFor(problem.readiness)),
        }}
      >
        <ReadinessExplanation readiness={problem.readiness} inLine />
      </ProblemLine>
    );
  }
  if (problem.kind === "skill-unavailable") {
    return (
      <ProblemLine
        testId="composer-skill-unavailable"
        className="composer-problem"
        action={{ label: t("skills.unavailable.drop"), onClick: onAskWithoutSkill }}
      >
        {t(`skills.unavailable.${problem.state}`, { name: problem.skill })}
      </ProblemLine>
    );
  }
  if (problem.kind === "error") {
    return (
      <ProblemLine testId="composer-error" className="composer-problem">
        {t("error.action", { message: problem.message })}
      </ProblemLine>
    );
  }
  return null;
}
