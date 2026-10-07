import { shift } from "@floating-ui/dom";
import { type Editor, Extension, type Range } from "@tiptap/core";
import type { EditorState } from "@tiptap/pm/state";
import { PluginKey } from "@tiptap/pm/state";
import { ReactRenderer } from "@tiptap/react";
import { Suggestion, type SuggestionProps } from "@tiptap/suggestion";
import { type Ref, useEffect, useImperativeHandle, useRef, useState } from "react";
import { QUESTION_BLOCK, type SearchScope } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import {
  SCOPE_ATTRIBUTES,
  SCOPE_KINDS,
  type ScopeKind,
  scopeIds,
  searchScopeOf,
} from "../../../shared/searchScope";
import { DocumentIcon, FolderIcon, TagIcon } from "../components/icons";
import { useT } from "../i18n";
import { type ScopeChoice, scopeChoices } from "../scope";
import { useAppStore } from "../store";

interface ScopePickerHandle {
  onKeyDown(event: KeyboardEvent): boolean;
}

type ScopePickerListProps = SuggestionProps<ScopeChoice, ScopeChoice> & {
  ref?: Ref<ScopePickerHandle>;
};

const GROUP_LABELS: Record<ScopeKind, MessageKey> = {
  folder: "scope.picker.folders",
  tag: "scope.picker.tags",
  document: "scope.picker.documents",
};

/** The Search scope of the Question that holds position `pos`, or none outside a Question. */
function scopeAt(state: EditorState, pos: number): SearchScope {
  const { parent } = state.doc.resolve(Math.min(pos, state.doc.content.size));
  return searchScopeOf(parent.type.name === QUESTION_BLOCK ? parent.attrs : {});
}

/**
 * Adds a Folder, Tag or Document to the Search scope of the Question the "@"
 * was typed in, and removes the "@" and what was typed after it (`range`).
 */
function addToScope(editor: Editor, range: Range, choice: ScopeChoice): void {
  editor
    .chain()
    .focus()
    .deleteRange(range)
    .command(({ tr }) => {
      const $at = tr.doc.resolve(tr.mapping.map(range.from));
      const question = $at.parent;
      if (question.type.name !== QUESTION_BLOCK) return false;
      const ids = scopeIds(searchScopeOf(question.attrs), choice.kind);
      if (!ids.includes(choice.id)) {
        tr.setNodeMarkup($at.before(), undefined, {
          ...question.attrs,
          [SCOPE_ATTRIBUTES[choice.kind]]: [...ids, choice.id],
        });
      }
      return true;
    })
    .run();
}

/**
 * Typing `@` in a Question (at its start or after a space) opens a picker of
 * Folders, Tags and Documents, filtered by what is typed after it. Arrows
 * choose, Enter or Tab adds the choice to the Question's Search scope, where
 * it shows as a chip; Esc closes.
 */
export const ScopePicker = Extension.create({
  name: "scopePicker",
  // Before `QuestionCommands`: while the picker is open, Enter chooses instead of asking.
  priority: 1100,

  addProseMirrorPlugins() {
    return [
      Suggestion<ScopeChoice, ScopeChoice>({
        editor: this.editor,
        pluginKey: new PluginKey("scopePicker"),
        char: "@",
        // Only in a Question: elsewhere an "@" is just an "@".
        allow: ({ state, range }) =>
          state.doc.resolve(range.from).parent.type.name === QUESTION_BLOCK,
        // The list reads the Folders, Tags and Documents from the app store itself.
        items: () => [],
        command: ({ editor, range, props: choice }) => addToScope(editor, range, choice),
        floatingUi: { strategy: "fixed", middleware: [shift({ padding: 8 })] },
        render: () => {
          let renderer: ReactRenderer<ScopePickerHandle, ScopePickerListProps> | null = null;
          let unmount: (() => void) | null = null;
          return {
            onStart(props) {
              renderer = new ReactRenderer(ScopePickerList, { editor: props.editor, props });
              unmount = props.mount(renderer.element);
            },
            onUpdate(props) {
              renderer?.updateProps(props);
            },
            onKeyDown({ event }) {
              return renderer?.ref?.onKeyDown(event) ?? false;
            },
            onExit() {
              unmount?.();
              renderer?.destroy();
              renderer = null;
              unmount = null;
            },
          };
        },
      }),
    ];
  },
});

function ChoiceIcon({ choice }: { choice: ScopeChoice }) {
  if (choice.kind === "folder") return <FolderIcon className="size-4" />;
  if (choice.kind === "tag") return <TagIcon className="size-4 text-gray-500" />;
  return <DocumentIcon kind={choice.documentKind ?? "text"} className="size-4" />;
}

function ScopePickerList({ editor, range, query, command, ref }: ScopePickerListProps) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  const library = { folders, tags, documents };
  const shown = scopeChoices(library, scopeAt(editor.state, range.from), query);
  const nothing = folders.length + tags.length + documents.length === 0;
  // The choice belongs to a query: typing more starts again from the top.
  const [choice, setChoice] = useState({ query, index: 0 });
  const selected = choice.query === query ? Math.min(choice.index, shown.length - 1) : 0;
  const setSelected = (index: number) => setChoice({ query, index });
  const options = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    options.current[selected]?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  useImperativeHandle(ref, () => ({
    onKeyDown(event) {
      // Nothing to choose: Enter asks the Question, "@" and all.
      if (event.isComposing || shown.length === 0) return false;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        const step = event.key === "ArrowDown" ? 1 : -1;
        setSelected((selected + step + shown.length) % shown.length);
        return true;
      }
      if (event.key === "Enter" || event.key === "Tab") {
        const item = shown[selected];
        if (item) command(item);
        return true;
      }
      return false;
    },
  }));

  return (
    <div
      role="listbox"
      data-testid="scope-picker"
      aria-label={t("scope.picker.label")}
      className="editor-menu scope-picker"
    >
      {shown.length === 0 && (
        <p className="scope-picker-empty">
          {t(nothing ? "scope.picker.none" : "scope.picker.empty")}
        </p>
      )}
      {SCOPE_KINDS.map((kind) => {
        const ofKind = shown.filter((item) => item.kind === kind);
        if (ofKind.length === 0) return null;
        return (
          // biome-ignore lint/a11y/useSemanticElements: a group of a listbox's options, which a fieldset can't hold.
          <div key={kind} role="group" aria-label={t(GROUP_LABELS[kind])}>
            <p aria-hidden="true" className="scope-picker-heading">
              {t(GROUP_LABELS[kind])}
            </p>
            {ofKind.map((item) => {
              const index = shown.indexOf(item);
              return (
                <button
                  key={`${item.kind}:${item.id}`}
                  ref={(element) => {
                    options.current[index] = element;
                  }}
                  type="button"
                  role="option"
                  aria-selected={index === selected}
                  data-testid="scope-choice"
                  data-kind={item.kind}
                  data-id={item.id}
                  // Keep the cursor in the editor.
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setSelected(index)}
                  onClick={() => command(item)}
                  className="editor-menu-item"
                >
                  <span className="flex shrink-0 items-center">
                    <ChoiceIcon choice={item} />
                  </span>
                  <span className="truncate">{item.name}</span>
                  {item.path.length > 0 && (
                    <span className="scope-picker-path">{item.path.join(" / ")}</span>
                  )}
                </button>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
