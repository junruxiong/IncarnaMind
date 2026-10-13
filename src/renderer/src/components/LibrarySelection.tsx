import { useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";
import type { Document } from "../../../core/api";
import { buildMenu, type MenuItem } from "../commands/menus";
import { type CommandContext, commands, menuLabelOf } from "../commands/registry";
import { useT } from "../i18n";
import type { ObjectRef } from "../objects";
import { useSelectedOn, useSelection } from "../selection";
import { useAppStore } from "../store";
import { CloseLineIcon } from "./lineIcons";
import { ObjectMenu } from "./ObjectMenu";
import { SelectionTags } from "./SelectionTags";
import { buttonStyle } from "./ui";

const boxClass = "size-4 shrink-0 cursor-pointer accent-ink";

/** How many commands the bar names beside Tags… when it is wide; "⋯" holds them all. */
const NAMED_COMMANDS = 2;

const documentRef = (id: string): ObjectRef => ({ kind: "document", id });

/**
 * A Library row's checkbox, in the margin left of its name: shown when the
 * row is pointed at or focused, and on every row once two or more are
 * selected. It selects like ⌘-click, and with Shift like Shift-click.
 */
export function LibrarySelectBox({ document }: { document: Pick<Document, "id" | "name"> }) {
  const t = useT();
  const key = `document:${document.id}`;
  const checked = useSelection((state) => state.surface === "library" && state.keys.has(key));
  const selecting = useSelection((state) => state.surface === "library" && state.items.length > 1);
  return (
    <input
      type="checkbox"
      data-testid="library-select"
      aria-label={t("library.select", { name: document.name })}
      checked={checked}
      onChange={(event) => {
        const { toggle, extendTo } = useSelection.getState();
        const ref = documentRef(document.id);
        if ((event.nativeEvent as MouseEvent).shiftKey) extendTo("library", ref);
        else toggle("library", ref);
      }}
      className={`${boxClass} absolute top-3.5 -left-[21px] ${
        checked || selecting
          ? ""
          : "opacity-0 group-hover/row:opacity-100 focus-visible:opacity-100"
      }`}
    />
  );
}

/**
 * Above the rows while two or more are selected: select all or none, how
 * many, the commands that apply to all of them (the same ones the right-click
 * menu offers: "⋯" holds every one), and Esc to clear. Wide, it names the
 * first few; narrow, they go under "⋯" and the bar never wraps.
 */
export function LibrarySelectionBar({ shownIds }: { shownIds: readonly string[] }) {
  const t = useT();
  const items = useSelectedOn("library");
  const documents = useAppStore(
    useShallow((state) => {
      const ids = new Set(items.map((each) => each.id));
      return ids.size > 1 ? state.documents.filter((each) => ids.has(each.id)) : [];
    }),
  );
  const all = useRef<HTMLInputElement>(null);
  const every = items.length > 0 && items.length === shownIds.length;
  useEffect(() => {
    if (all.current) all.current.indeterminate = items.length > 0 && !every;
  });
  if (items.length < 2) return null;
  const last = items.at(-1) as ObjectRef;
  const ctx: CommandContext = { selection: items, focus: last, surface: "library" };
  const named = buildMenu("document", ctx)
    .filter((entry): entry is MenuItem => entry !== "rule" && entry.command.id !== "document.tags")
    .slice(0, NAMED_COMMANDS);
  const run = (item: MenuItem) => {
    if (item.enabled !== true) return;
    commands.run(item.command.id, ctx).catch(useAppStore.getState().reportError);
  };
  return (
    <div
      data-testid="library-selection"
      className="@container sticky top-0 z-10 mb-2 flex h-10 items-center gap-3 rounded-md bg-frame px-2 text-[13px]"
    >
      <input
        ref={all}
        type="checkbox"
        data-testid="library-select-all"
        aria-label={t("library.selectAll", { count: shownIds.length })}
        checked={every}
        onChange={() =>
          every ? useSelection.getState().clear("library") : useSelection.getState().all("library")
        }
        className={boxClass}
      />
      <span role="status" className="shrink-0 font-semibold text-ink">
        {t("library.selected", { count: items.length })}
      </span>
      <SelectionTags documents={documents} testId="library-selection-tags" />
      {named.map((item, index) => (
        <button
          key={item.command.id}
          type="button"
          data-testid="library-selection-action"
          data-command={item.command.id}
          aria-disabled={item.enabled === true ? undefined : true}
          onClick={() => run(item)}
          className={`${buttonStyle("secondary", "sm")} hidden ${
            index === 0 ? "@[34rem]:inline-flex" : "@[46rem]:inline-flex"
          }`}
        >
          {menuLabelOf(item.command, ctx, t)}
        </button>
      ))}
      <ObjectMenu
        target={last}
        selection={items}
        label={t("library.selected.more", { count: items.length })}
      />
      <span className="flex-1" />
      <button
        type="button"
        data-testid="library-selection-clear"
        aria-label={t("library.selected.clear")}
        title={t("library.selected.clear")}
        onClick={() => useSelection.getState().clear("library")}
        className="inline-flex h-7 shrink-0 items-center gap-1 rounded-md px-1.5 text-ink-meta hover:bg-rule hover:text-ink"
      >
        <CloseLineIcon className="size-3.5" />
        <span className="text-[12px]">Esc</span>
      </button>
    </div>
  );
}
