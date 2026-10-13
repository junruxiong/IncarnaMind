import { type KeyboardEvent, useRef, useState } from "react";
import type { Mind } from "../../../core/api";
import { mindName } from "../../../shared/naming";
import { useCitationStylePicker } from "../citationStylePicker";
import { startMoveDrag } from "../folderDrag";
import { useT } from "../i18n";
import { currentTabOf } from "../layoutSlice";
import { useMindStatus } from "../mindStatus";
import { selectedOn, useSelectable } from "../selection";
import { useAppStore } from "../store";
import { isMoveToKey, TREE_ROW } from "../treeKeys";
import { useRename } from "../useRename";
import { ExampleChip, useIsExample } from "./GettingStarted";
import { InlineName } from "./InlineName";
import { MindLineIcon } from "./lineIcons";
import { MoveToPicker } from "./MoveToPicker";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import {
  clearsActions,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowPadding,
} from "./sidebarRows";

/**
 * A Mind in the sidebar's tree, in its Folder (`depth` 1) or Not in a Folder
 * (0): its title, and an amber dot while one of its Answers waits for the
 * User's approval. Clicking or Enter opens it and selects just it; ⌘-click
 * and Shift-click select; a middle click opens it in a new tab. It renames in place: double-click it or press F2;
 * Enter saves, Esc cancels, and a blank title changes nothing.
 * It drags onto a Folder; its menu (⋯, or a right-click) and ⇧⌘M offer the
 * same as "Move to…".
 */
export function MindRow({ mind, depth }: { mind: Mind; depth: number }) {
  const t = useT();
  const isOpen = useAppStore((state) => {
    const shown = currentTabOf(state.layout)?.item;
    return shown?.kind === "mind" && shown.mindId === mind.id;
  });
  const openMind = useAppStore((state) => state.openMind);
  const renameMind = useAppStore((state) => state.renameMind);
  const waiting = useMindStatus(mind.id) === "waiting-for-approval";
  const isExample = useIsExample(mind.id);
  const rename = useRename<HTMLButtonElement>(`mind:${mind.id}`);
  const [moving, setMoving] = useState<HTMLElement | null>(null);
  const row = useRef<HTMLLIElement>(null);
  const title = mindName(mind) || t("mind.untitled");
  const objectMenu = useObjectMenu({ kind: "mind", id: mind.id });
  const select = useSelectable({ kind: "mind", id: mind.id }, "sidebar", () => openMind(mind.id));
  const lit = isOpen || select.selected;
  /** The Minds a drag or "Move to…" from this row takes: all those selected, if it is one of them. */
  const mindIdsHere = () => {
    const chosen = selectedOn("sidebar").flatMap((each) => (each.kind === "mind" ? [each.id] : []));
    return chosen.includes(mind.id) ? chosen : [mind.id];
  };

  if (rename.renaming) {
    return (
      <li className={rowClass(lit)} data-testid="mind-row" data-mind-id={mind.id}>
        <span
          className="flex h-full w-full min-w-0 items-center gap-2 pr-1"
          style={rowPadding(depth)}
        >
          <MindLineIcon className={rowIconClass(lit)} />
          <InlineName
            value={mind.title}
            label={t("mind.renameLabel", { title })}
            placeholder={title}
            testId="mind-rename"
            {...rename.field((name) => renameMind(mind.id, name))}
          />
        </span>
      </li>
    );
  }

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (isMoveToKey(event)) {
      event.preventDefault();
      setMoving(event.currentTarget);
    }
  };

  return (
    // biome-ignore lint/a11y/useFocusableInteractive: the row's own button takes the focus, and the arrow keys move between them.
    <li
      data-testid="mind-row"
      data-mind-id={mind.id}
      data-folder-id={mind.folderId ?? ""}
      role="treeitem"
      {...select.rowProps}
      draggable
      onDragStart={(event) => startMoveDrag(event, { mindIds: mindIdsHere() })}
      {...objectMenu.rowProps}
      ref={row}
      className={rowClass(lit)}
    >
      <button
        {...rename.nameProps}
        type="button"
        {...TREE_ROW}
        data-testid="mind-list-item"
        data-mind-id={mind.id}
        aria-current={isOpen ? "page" : undefined}
        title={title}
        // A click opens it; ⌘-click and Shift-click select (`useSelectable`). A middle click opens it to the side.
        onClick={select.onClick}
        onMouseDown={(event) => {
          if (event.button === 1) event.preventDefault(); // no autoscroll
        }}
        onAuxClick={(event) => {
          if (event.button === 1) openMind(mind.id, { side: true });
        }}
        onKeyDown={(event) => {
          rename.nameProps.onKeyDown(event);
          onKeyDown(event);
        }}
        className={`${rowButtonClass} ${clearsActions.menu}`}
        style={rowPadding(depth)}
      >
        <MindLineIcon className={rowIconClass(lit)} />
        <span
          data-testid="row-text"
          className={`min-w-0 flex-1 truncate ${mindName(mind) ? "" : "text-ink-meta"}`}
        >
          {title}
        </span>
        {isExample && <ExampleChip />}
        {waiting && (
          <span
            role="img"
            data-testid="mind-approval"
            aria-label={t("approvals.answer.waiting")}
            title={t("approvals.answer.waiting")}
            className="size-1.5 shrink-0 rounded-full bg-attention"
          />
        )}
      </button>
      <div className={rowActionsClass}>
        <ObjectMenu
          target={{ kind: "mind", id: mind.id }}
          label={t("mind.more", { title })}
          here={{
            "mind.rename": rename.start,
            "mind.citationStyle": () =>
              requestAnimationFrame(() =>
                useCitationStylePicker
                  .getState()
                  .show(
                    mind.id,
                    row.current?.querySelector<HTMLElement>("[data-object-menu]") ?? null,
                  ),
              ),
            // Once the menu has closed and given the focus back, so the picker opens beside the button.
            "mind.moveTo": () =>
              requestAnimationFrame(() =>
                setMoving(row.current?.querySelector<HTMLElement>("[data-object-menu]") ?? null),
              ),
          }}
        />
      </div>
      <MoveToPicker
        anchor={moving}
        name={
          moving && mindIdsHere().length > 1
            ? t("mind.count", { count: mindIdsHere().length })
            : title
        }
        current={moving && mindIdsHere().length > 1 ? undefined : mind.folderId}
        onChoose={(folderId) => {
          const mindIds = mindIdsHere();
          if (mindIds.length > 1 || folderId !== mind.folderId) {
            void useAppStore.getState().moveToFolder({ mindIds }, folderId);
          }
        }}
        onClose={() => setMoving(null)}
      />
    </li>
  );
}
