import { isMacOS } from "@tiptap/core";
import { PluginKey, TextSelection } from "@tiptap/pm/state";
import { type Editor, useEditorState } from "@tiptap/react";
import { BubbleMenu, type BubbleMenuProps } from "@tiptap/react/menus";
import {
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import type { MessageKey } from "../../../shared/i18n";
import { ChevronIcon } from "../components/icons";
import { useT } from "../i18n";
import { Popover } from "./Popover";
import {
  asOwnUndoStep,
  canTurnInto,
  currentTurnInto,
  inOneTextblock,
  linkTarget,
  selectionToMath,
  TURN_INTO,
} from "./selectionCommands";

/**
 * One action of the menu's AI section: Edit with AI, Find support and the
 * like. The section is hidden while there are none.
 */
export interface SelectionAction {
  /** A stable name; tests find the action by it (`selection-action-<id>`). */
  id: string;
  label: MessageKey;
  icon: ReactNode;
  /** The keys that do it too, as the menu shows them. */
  keys?: string;
  /** Whether it applies to the selection now; it isn't shown when it doesn't. */
  available?(editor: Editor): boolean;
  /** Choices it asks for first (Translate… into which language), in a menu beside it. */
  choices?: readonly SelectionChoice[];
  run(editor: Editor): void;
}

/** One choice of an AI action that asks for one. */
export interface SelectionChoice {
  /** A stable name; tests find the choice by it (`selection-choice-<id>`). */
  id: string;
  /** In words that need no translating (a language in its own name). */
  label: string;
  run(editor: Editor): void;
}

type Mark = "bold" | "italic" | "underline" | "strike" | "link" | "code" | "highlight";

const PLUGIN_KEY = new PluginKey("selectionMenu");

/** A shortcut as the menu shows it: ⌘⇧H on macOS, Ctrl+Shift+H elsewhere. */
const shortcut = (key: string, shift = false) =>
  isMacOS() ? `⌘${shift ? "⇧" : ""}${key}` : `Ctrl+${shift ? "Shift+" : ""}${key}`;

/** The format buttons in the order of the design: the glyph says what the mark does. */
const MARKS: readonly {
  mark: Mark;
  label: MessageKey;
  glyph: ReactNode;
  keys?: string;
  toggle(editor: Editor): boolean;
}[] = [
  {
    mark: "bold",
    label: "editor.format.bold",
    glyph: <span className="font-bold">B</span>,
    keys: shortcut("B"),
    toggle: (editor) => editor.chain().focus().toggleBold().run(),
  },
  {
    mark: "italic",
    label: "editor.format.italic",
    glyph: <span className="italic">I</span>,
    keys: shortcut("I"),
    toggle: (editor) => editor.chain().focus().toggleItalic().run(),
  },
  {
    mark: "underline",
    label: "editor.format.underline",
    glyph: <span className="underline">U</span>,
    keys: shortcut("U"),
    toggle: (editor) => editor.chain().focus().toggleUnderline().run(),
  },
  {
    mark: "strike",
    label: "editor.format.strike",
    glyph: <span className="line-through">S</span>,
    keys: shortcut("S", true),
    toggle: (editor) => editor.chain().focus().toggleStrike().run(),
  },
  {
    mark: "code",
    label: "editor.format.code",
    glyph: <span className="font-mono">{"</>"}</span>,
    keys: shortcut("E"),
    toggle: (editor) => editor.chain().focus().toggleCode().run(),
  },
  {
    mark: "highlight",
    label: "editor.format.highlight",
    glyph: <span className="format-highlight">H</span>,
    keys: shortcut("H", true),
    toggle: (editor) => editor.chain().focus().toggleHighlight().run(),
  },
];

const isComposing = (event: KeyboardEvent) =>
  event.nativeEvent.isComposing || event.key === "Process";

/** The menu's controls, in the order the arrow keys go through them. */
const controlsOf = (menu: HTMLElement | null) =>
  Array.from(
    menu?.querySelectorAll<HTMLElement>('[role^="menuitem"]:not([aria-disabled="true"])') ?? [],
  );

/** The selection the User closed the menu over, which keeps it closed until the selection changes, even for an update already waiting. */
const dismissedOver = new WeakMap<Editor, string>();
const selectionKey = (editor: Editor) =>
  `${editor.state.selection.from}-${editor.state.selection.to}`;

function hideMenu(editor: Editor) {
  dismissedOver.set(editor, selectionKey(editor));
  editor.view.dispatch(editor.state.tr.setMeta(PLUGIN_KEY, "hide"));
}

/** Shown over selected text, not over code or a selected Block, nor while a mouse button or an input method is working on the selection. */
function showsOver(editor: Editor, pressed: boolean): boolean {
  const { selection, doc } = editor.state;
  if (pressed || editor.view.composing || !editor.isEditable) return false;
  if (dismissedOver.get(editor) === selectionKey(editor)) return false;
  dismissedOver.delete(editor);
  if (selection.empty || !(selection instanceof TextSelection)) return false;
  if (editor.isActive("codeBlock")) return false;
  return doc.textBetween(selection.from, selection.to).length > 0;
}

/**
 * The menu under selected text: "Text ›" to turn Blocks into headings, lists
 * and quotes, the format buttons, and `ai`, the actions of the AI section.
 * The mouse and the keyboard (⌘/ moves into it from the text, arrows go
 * through it, Esc closes the innermost thing open) reach every action, and
 * none runs while an input method is composing.
 */
export function SelectionMenu({
  editor,
  ai = [],
}: {
  editor: Editor;
  ai?: readonly SelectionAction[];
}) {
  const t = useT();
  const menu = useRef<HTMLDivElement>(null);
  const turnRow = useRef<HTMLButtonElement>(null);
  const linkButton = useRef<HTMLButtonElement>(null);
  const shown = useRef(false);
  const pressed = useRef(false);
  const [turning, setTurning] = useState<"mouse" | "keyboard" | null>(null);
  const [linking, setLinking] = useState(false);
  /** The AI action whose choices are open, and how it was opened. */
  const [choosing, setChoosing] = useState<{
    action: SelectionAction;
    anchor: HTMLElement;
    focus: boolean;
  } | null>(null);
  const actions = useRef(ai);
  actions.current = ai;

  const state = useEditorState({
    editor,
    selector: ({ editor: current }) => ({
      bold: current.isActive("bold"),
      italic: current.isActive("italic"),
      underline: current.isActive("underline"),
      strike: current.isActive("strike"),
      code: current.isActive("code"),
      highlight: current.isActive("highlight"),
      link: current.isActive("link"),
      turnable: canTurnInto(current),
      turn: currentTurnInto(current).label,
      // A formula replaces text in a Note only: not in a Question or an Answer.
      oneLine: inOneTextblock(current) && canTurnInto(current),
      ai: actions.current
        .filter((action) => action.available?.(current) ?? true)
        .map((action) => action.id),
    }),
  });
  const shownAi = ai.filter((action) => state.ai.includes(action.id));

  const shouldShow = useCallback<NonNullable<BubbleMenuProps["shouldShow"]>>(
    ({ editor: current, element, view }) =>
      showsOver(current, pressed.current) &&
      (view.hasFocus() || element.contains(document.activeElement)),
    [],
  );

  // While a mouse button is down the menu waits: it opens under what the User
  // has selected once they let go, rather than chasing the pointer.
  useEffect(() => {
    const dom = editor.view.dom;
    const press = () => {
      pressed.current = true;
    };
    const release = () => {
      if (!pressed.current) return;
      pressed.current = false;
      if (editor.isDestroyed || !showsOver(editor, false)) return;
      editor.view.dispatch(editor.state.tr.setMeta(PLUGIN_KEY, "show"));
      editor.view.dispatch(editor.state.tr.setMeta(PLUGIN_KEY, "updatePosition"));
    };
    // ⌘/ moves into the menu; Esc, with the focus still in the text, closes it, ahead of the editor's own use of Esc.
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (!shown.current || event.isComposing || event.keyCode === 229) return;
      const modifier = isMacOS() ? event.metaKey : event.ctrlKey;
      if (event.key === "/" && modifier) {
        event.preventDefault();
        controlsOf(menu.current)[0]?.focus();
      } else if (event.key === "Escape" && !event.defaultPrevented) {
        event.preventDefault();
        event.stopImmediatePropagation();
        hideMenu(editor);
      }
    };
    // The menu doesn't update while an input method composes, and ProseMirror
    // ends the composition a moment after its last change: look again then.
    const afterComposition = () =>
      setTimeout(() => {
        if (!editor.isDestroyed && shown.current && !showsOver(editor, false)) hideMenu(editor);
      }, 50);
    dom.addEventListener("compositionend", afterComposition);
    dom.addEventListener("mousedown", press);
    document.addEventListener("mouseup", release);
    dom.addEventListener("keydown", onKey, true);
    return () => {
      dom.removeEventListener("compositionend", afterComposition);
      dom.removeEventListener("mousedown", press);
      document.removeEventListener("mouseup", release);
      dom.removeEventListener("keydown", onKey, true);
    };
  }, [editor]);

  const options = useRef({
    placement: "bottom-start" as const,
    offset: 8,
    onShow: () => {
      shown.current = true;
    },
    onHide: () => {
      shown.current = false;
      setTurning(null);
      setLinking(false);
      setChoosing(null);
    },
  }).current;

  /** Runs an action unless an input method is composing; the selection stays, so the next one can follow. */
  const act = (run: () => unknown) => {
    if (editor.view.composing) return;
    // Back in the text at once, so a key pressed right after the action reaches it.
    if (!editor.view.hasFocus()) editor.view.focus();
    asOwnUndoStep(editor, run);
  };

  const closeTurning = (refocus: boolean) => {
    setTurning(null);
    if (refocus) turnRow.current?.focus();
  };

  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (isComposing(event)) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      editor.commands.focus();
      hideMenu(editor);
      return;
    }
    const all = controlsOf(menu.current);
    const index = all.indexOf(document.activeElement as HTMLElement);
    if (event.key === "ArrowRight" && document.activeElement === turnRow.current) {
      event.preventDefault();
      setTurning("keyboard");
      return;
    }
    const withChoices = shownAi.find(
      (action) =>
        action.choices && document.activeElement?.getAttribute("data-action") === action.id,
    );
    if (
      event.key === "ArrowRight" &&
      withChoices &&
      document.activeElement instanceof HTMLElement
    ) {
      event.preventDefault();
      setChoosing({ action: withChoices, anchor: document.activeElement, focus: true });
      return;
    }
    const step =
      event.key === "ArrowDown" || event.key === "ArrowRight"
        ? 1
        : event.key === "ArrowUp" || event.key === "ArrowLeft"
          ? -1
          : 0;
    const next =
      event.key === "Home"
        ? all[0]
        : event.key === "End"
          ? all.at(-1)
          : step
            ? all[(index + step + all.length) % all.length]
            : undefined;
    if (!next) return;
    event.preventDefault();
    next.focus();
  };

  const turnInto = (run: () => boolean) => {
    act(() => {
      setTurning(null);
      run();
    });
  };

  const applyLink = (typed: string) => {
    setLinking(false);
    asOwnUndoStep(editor, () => {
      const chain = editor.chain().focus();
      if (typed.trim() === "") return chain.extendMarkRange("link").unsetLink().run();
      // Editing a link changes all of it, not just the part selected.
      if (editor.isActive("link")) chain.extendMarkRange("link");
      return chain.setLink({ href: linkTarget(typed) }).run();
    });
  };

  const keepSelection = (event: { preventDefault(): void }) => event.preventDefault();

  return (
    <BubbleMenu
      editor={editor}
      pluginKey={PLUGIN_KEY}
      updateDelay={50}
      shouldShow={shouldShow}
      options={options}
      className="selection-menu-frame"
    >
      <div
        ref={menu}
        role="menu"
        aria-label={t("editor.selection.label")}
        data-testid="selection-menu"
        onKeyDown={onMenuKeyDown}
        className="editor-menu selection-menu"
      >
        {state.turnable && (
          <>
            <button
              ref={turnRow}
              type="button"
              role="menuitem"
              aria-haspopup="menu"
              aria-expanded={turning !== null}
              data-testid="selection-turn-into"
              title={t("editor.turnInto.label")}
              onMouseDown={keepSelection}
              onClick={(event) =>
                setTurning(turning ? null : event.detail === 0 ? "keyboard" : "mouse")
              }
              className="editor-menu-item selection-turn"
            >
              <span className="editor-menu-icon">T</span>
              <span className="flex-1 truncate">{t(state.turn)}</span>
              <ChevronIcon className="size-4 text-ink-meta" />
            </button>
            <hr className="selection-rule" />
          </>
        )}
        {/* biome-ignore lint/a11y/useSemanticElements: a group of a menu's items, which a fieldset can't hold. */}
        <div role="group" aria-label={t("editor.format.label")} className="selection-formats">
          {MARKS.slice(0, 4).map((each) => (
            <FormatButton
              key={each.mark}
              {...each}
              on={state[each.mark]}
              editor={editor}
              act={act}
            />
          ))}
          <button
            ref={linkButton}
            type="button"
            role="menuitemcheckbox"
            aria-checked={state.link}
            aria-haspopup="dialog"
            data-testid="format-link"
            title={t("editor.format.link")}
            aria-label={t("editor.format.link")}
            onMouseDown={keepSelection}
            onClick={() => act(() => setLinking(true))}
            className="selection-format"
          >
            <span aria-hidden="true">↗</span>
          </button>
          {MARKS.slice(4, 5).map((each) => (
            <FormatButton
              key={each.mark}
              {...each}
              on={state[each.mark]}
              editor={editor}
              act={act}
            />
          ))}
          <button
            type="button"
            role="menuitem"
            aria-disabled={!state.oneLine}
            data-testid="format-math"
            title={t("editor.format.math")}
            aria-label={t("editor.format.math")}
            onMouseDown={keepSelection}
            onClick={() =>
              act(() => {
                const at = selectionToMath(editor);
                if (at !== null) editor.commands.editMath(at);
              })
            }
            className="selection-format"
          >
            <span aria-hidden="true">∑</span>
          </button>
          {MARKS.slice(5).map((each) => (
            <FormatButton
              key={each.mark}
              {...each}
              on={state[each.mark]}
              editor={editor}
              act={act}
            />
          ))}
          <button
            type="button"
            role="menuitem"
            data-testid="format-clear"
            title={t("editor.format.clear")}
            aria-label={t("editor.format.clear")}
            onMouseDown={keepSelection}
            onClick={() => act(() => editor.chain().focus().unsetAllMarks().run())}
            className="selection-format"
          >
            <span aria-hidden="true">T×</span>
          </button>
        </div>
        {shownAi.length > 0 && (
          <>
            <hr className="selection-rule" />
            {/* biome-ignore lint/a11y/useSemanticElements: a group of a menu's items, which a fieldset can't hold. */}
            <div role="group" aria-label={t("editor.selection.ai")}>
              <p aria-hidden="true" className="selection-section">
                {t("editor.selection.ai")}
              </p>
              {shownAi.map((action) => (
                <button
                  key={action.id}
                  type="button"
                  role="menuitem"
                  data-action={action.id}
                  data-testid={`selection-action-${action.id}`}
                  aria-haspopup={action.choices ? "menu" : undefined}
                  aria-expanded={action.choices ? choosing?.action.id === action.id : undefined}
                  aria-keyshortcuts={action.keys}
                  onMouseDown={keepSelection}
                  onClick={(event) => {
                    if (!action.choices) {
                      act(() => action.run(editor));
                      return;
                    }
                    const anchor = event.currentTarget;
                    setChoosing((open) =>
                      open?.action.id === action.id
                        ? null
                        : { action, anchor, focus: event.detail === 0 },
                    );
                  }}
                  className="editor-menu-item"
                >
                  <span className="editor-menu-icon">{action.icon}</span>
                  <span className="flex-1 truncate">{t(action.label)}</span>
                  {action.keys && <span className="editor-menu-shortcut">{action.keys}</span>}
                  {action.choices && <ChevronIcon className="size-4 text-ink-meta" />}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
      {turning && turnRow.current && (
        <TurnIntoMenu
          editor={editor}
          anchor={turnRow.current}
          focus={turning === "keyboard"}
          onChoose={turnInto}
          onClose={closeTurning}
        />
      )}
      {choosing?.action.choices && (
        <ChoiceMenu
          label={t(choosing.action.label)}
          choices={choosing.action.choices}
          anchor={choosing.anchor}
          focus={choosing.focus}
          testId={`selection-choices-${choosing.action.id}`}
          onChoose={(choice) => {
            setChoosing(null);
            act(() => choice.run(editor));
          }}
          onClose={(refocus) => {
            const anchor = choosing.anchor;
            setChoosing(null);
            if (refocus) anchor.focus();
          }}
        />
      )}
      {linking && linkButton.current && (
        <LinkField
          anchor={linkButton.current}
          current={editor.getAttributes("link").href as string | undefined}
          onApply={applyLink}
          onClose={() => {
            setLinking(false);
            editor.commands.focus();
          }}
        />
      )}
    </BubbleMenu>
  );
}

function FormatButton({
  mark,
  label,
  glyph,
  keys,
  toggle,
  on,
  editor,
  act,
}: {
  mark: Mark;
  label: MessageKey;
  glyph: ReactNode;
  keys?: string;
  toggle(editor: Editor): boolean;
  on: boolean;
  editor: Editor;
  act(run: () => unknown): void;
}) {
  const t = useT();
  return (
    <button
      type="button"
      role="menuitemcheckbox"
      aria-checked={on}
      data-testid={`format-${mark}`}
      title={keys ? `${t(label)} (${keys})` : t(label)}
      aria-label={t(label)}
      // Keep the selection in the editor.
      onMouseDown={(event) => event.preventDefault()}
      onClick={() => act(() => toggle(editor))}
      className="selection-format"
    >
      <span aria-hidden="true">{glyph}</span>
    </button>
  );
}

function TurnIntoMenu({
  editor,
  anchor,
  focus,
  onChoose,
  onClose,
}: {
  editor: Editor;
  anchor: HTMLElement;
  focus: boolean;
  onChoose(run: () => boolean): void;
  onClose(refocus: boolean): void;
}) {
  const t = useT();
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const current = TURN_INTO.findIndex((each) => each.active(editor));
  const chosen = useRef<HTMLButtonElement>(null);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (isComposing(event)) return;
    const all = items.current.filter((each): each is HTMLButtonElement => each !== null);
    const index = all.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "Escape" || event.key === "ArrowLeft") {
      event.preventDefault();
      event.stopPropagation();
      onClose(true);
      return;
    }
    const next =
      event.key === "ArrowDown"
        ? all[(index + 1) % all.length]
        : event.key === "ArrowUp"
          ? all[(index - 1 + all.length) % all.length]
          : event.key === "Home"
            ? all[0]
            : event.key === "End"
              ? all.at(-1)
              : undefined;
    if (!next) return;
    event.preventDefault();
    next.focus();
  };

  return (
    <Popover
      anchor={anchor}
      placement="right-start"
      initialFocus={focus ? chosen : undefined}
      role="menu"
      aria-label={t("editor.turnInto.label")}
      data-testid="selection-turn-into-menu"
      onDismiss={() => onClose(false)}
      onKeyDown={onKeyDown}
      className="editor-menu w-[200px]"
    >
      {TURN_INTO.map((kind, index) => (
        <button
          key={kind.id}
          ref={(element) => {
            items.current[index] = element;
            if (index === Math.max(current, 0)) chosen.current = element;
          }}
          type="button"
          role="menuitemradio"
          aria-checked={index === current}
          data-testid={`turn-into-${kind.id}`}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => onChoose(() => kind.run(editor))}
          className="editor-menu-item"
        >
          <span className="editor-menu-icon">{index === current ? "✓" : ""}</span>
          <span className="truncate">{t(kind.label)}</span>
        </button>
      ))}
    </Popover>
  );
}

/** The choices of an AI action that asks for one (Translate… into which language), beside it. */
function ChoiceMenu({
  label,
  choices,
  anchor,
  focus,
  testId,
  onChoose,
  onClose,
}: {
  label: string;
  choices: readonly SelectionChoice[];
  anchor: HTMLElement;
  focus: boolean;
  testId: string;
  onChoose(choice: SelectionChoice): void;
  onClose(refocus: boolean): void;
}) {
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const first = useRef<HTMLButtonElement>(null);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (isComposing(event)) return;
    const all = items.current.filter((each): each is HTMLButtonElement => each !== null);
    const index = all.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "Escape" || event.key === "ArrowLeft") {
      event.preventDefault();
      event.stopPropagation();
      onClose(true);
      return;
    }
    const next =
      event.key === "ArrowDown"
        ? all[(index + 1) % all.length]
        : event.key === "ArrowUp"
          ? all[(index - 1 + all.length) % all.length]
          : event.key === "Home"
            ? all[0]
            : event.key === "End"
              ? all.at(-1)
              : undefined;
    if (!next) return;
    event.preventDefault();
    next.focus();
  };

  return (
    <Popover
      anchor={anchor}
      placement="right-start"
      initialFocus={focus ? first : undefined}
      role="menu"
      aria-label={label}
      data-testid={testId}
      onDismiss={() => onClose(false)}
      onKeyDown={onKeyDown}
      className="editor-menu w-[200px]"
    >
      {choices.map((choice, index) => (
        <button
          key={choice.id}
          ref={(element) => {
            items.current[index] = element;
            if (index === 0) first.current = element;
          }}
          type="button"
          role="menuitem"
          data-testid={`selection-choice-${choice.id}`}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => onChoose(choice)}
          className="editor-menu-item"
        >
          <span className="editor-menu-icon" />
          <span className="truncate">{choice.label}</span>
        </button>
      ))}
    </Popover>
  );
}

function LinkField({
  anchor,
  current,
  onApply,
  onClose,
}: {
  anchor: HTMLElement;
  current: string | undefined;
  onApply(typed: string): void;
  onClose(): void;
}) {
  const t = useT();
  const field = useRef<HTMLInputElement>(null);
  const [typed, setTyped] = useState(current ?? "");

  return (
    <Popover
      anchor={anchor}
      initialFocus={field}
      aria-label={t("editor.link.label")}
      onDismiss={onClose}
      className="editor-menu link-editor"
    >
      <input
        ref={field}
        type="text"
        inputMode="url"
        data-testid="link-field"
        aria-label={t("editor.link.label")}
        placeholder={t("editor.link.placeholder")}
        value={typed}
        spellCheck={false}
        onChange={(event) => setTyped(event.target.value)}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing || event.key === "Process") return;
          if (event.key === "Enter") {
            event.preventDefault();
            onApply(typed);
          }
        }}
        className="math-input"
      />
      <p className="math-hint">{t(current ? "editor.link.hintEdit" : "editor.link.hint")}</p>
    </Popover>
  );
}
