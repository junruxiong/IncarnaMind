import { shift } from "@floating-ui/dom";
import { Extension } from "@tiptap/core";
import { PluginKey } from "@tiptap/pm/state";
import { ReactRenderer } from "@tiptap/react";
import { Suggestion, type SuggestionProps } from "@tiptap/suggestion";
import { type Ref, useEffect, useImperativeHandle, useRef, useState } from "react";
import { QUESTION_BLOCK } from "../../../core/api";
import { useT } from "../i18n";
import { matchSlashItems, noteSlashItems, type SlashItem } from "./slashItems";

export interface SlashMenuOptions {
  /** Every entry the menu can offer, read each time it opens or the query changes. */
  items: () => readonly SlashItem[];
}

interface SlashMenuHandle {
  onKeyDown(event: KeyboardEvent): boolean;
}

type SlashMenuListProps = SuggestionProps<SlashItem, SlashItem> & { ref?: Ref<SlashMenuHandle> };

/**
 * Typing `/` (at the start of a line or after a space) opens a menu of things
 * to insert, filtered by what is typed after it. Arrows choose, Enter inserts,
 * Esc closes.
 */
export const SlashMenu = Extension.create<SlashMenuOptions>({
  name: "slashMenu",

  addOptions() {
    return { items: () => noteSlashItems };
  },

  addProseMirrorPlugins() {
    return [
      Suggestion<SlashItem, SlashItem>({
        editor: this.editor,
        pluginKey: new PluginKey("slashMenu"),
        char: "/",
        // Not in code, where a slash is just a slash, nor in a Question, where Enter asks.
        allow: ({ state, range }) => {
          const { parent } = state.doc.resolve(range.from);
          return !parent.type.spec.code && parent.type.name !== QUESTION_BLOCK;
        },
        items: () => [...this.options.items()],
        command: ({ editor, range, props: item }) => item.run(editor, range),
        floatingUi: { strategy: "fixed", middleware: [shift({ padding: 8 })] },
        render: () => {
          let renderer: ReactRenderer<SlashMenuHandle, SlashMenuListProps> | null = null;
          let unmount: (() => void) | null = null;
          return {
            onStart(props) {
              renderer = new ReactRenderer(SlashMenuList, { editor: props.editor, props });
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

function SlashMenuList({ items, query, command, ref }: SlashMenuListProps) {
  const t = useT();
  const labelOf = (item: SlashItem) =>
    typeof item.label === "string" ? t(item.label) : item.label.text;
  const shown = matchSlashItems(items, query, labelOf);
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
      data-testid="slash-menu"
      aria-label={t("editor.slash.label")}
      className="editor-menu max-h-[300px] w-[220px] overflow-y-auto"
    >
      {shown.length === 0 && <p className="px-2 py-1 text-gray-500">{t("editor.slash.empty")}</p>}
      {shown.map((item, index) => (
        <button
          key={item.id}
          ref={(element) => {
            options.current[index] = element;
          }}
          type="button"
          role="option"
          aria-selected={index === selected}
          data-testid={`slash-item-${item.id}`}
          // Keep the cursor in the editor.
          onMouseDown={(event) => event.preventDefault()}
          onMouseEnter={() => setSelected(index)}
          onClick={() => command(item)}
          className="editor-menu-item"
        >
          <span className="editor-menu-icon">{item.icon}</span>
          <span className="truncate">{labelOf(item)}</span>
        </button>
      ))}
    </div>
  );
}
