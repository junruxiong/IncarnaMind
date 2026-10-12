import { type KeyboardEvent, useRef, useState } from "react";
import type { Mind } from "../../../core/api";
import { mindName } from "../../../shared/naming";
import { useCitationStylePicker } from "../citationStylePicker";
import { startMoveDrag } from "../folderDrag";
import { useT } from "../i18n";
import { currentTabOf } from "../layoutSlice";
import { useMindStatus } from "../mindStatus";
import { useAppStore } from "../store";
import { isMoveToKey, TREE_ROW } from "../treeKeys";
import { useRename } from "../useRename";
import { ExampleChip, useIsExample } from "./GettingStarted";
import { InlineName } from "./InlineName";
import { MindLineIcon } from "./lineIcons";
import { MoveToPicker } from "./MoveToPicker";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import { rowActionsClass, rowButtonClass, rowClass, rowIconClass, rowPadding } from "./sidebarRows";

/**
 * A Mind in the sidebar's tree, in its Folder (`depth` 1) or Not in a Folder
 * (0): its title, and an amber dot while one of its Answers waits for the
 * User's approval. Clicking opens it (⌘-click or a middle click in a new
 * tab). Enter opens it. It renames in place: double-click it or press F2;
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

  if (rename.renaming) {
    return (
      <li className={rowClass(isOpen)} data-testid="mind-row" data-mind-id={mind.id}>
        <span
          className="flex h-full w-full min-w-0 items-center gap-2 pr-1"
          style={rowPadding(depth)}
        >
          <MindLineIcon className={rowIconClass(isOpen)} />
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
    <li
      data-testid="mind-row"
      data-mind-id={mind.id}
      data-folder-id={mind.folderId ?? ""}
      draggable
      onDragStart={(event) => startMoveDrag(event, { mindIds: [mind.id] })}
      {...objectMenu.rowProps}
      ref={row}
      className={rowClass(isOpen)}
    >
      <button
        {...rename.nameProps}
        type="button"
        {...TREE_ROW}
        data-testid="mind-list-item"
        data-mind-id={mind.id}
        aria-current={isOpen ? "page" : undefined}
        title={title}
        // ⌘-click (Ctrl-click) or a middle click opens it to the side, as in VS Code.
        onClick={(event) => openMind(mind.id, { side: event.metaKey || event.ctrlKey })}
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
        className={rowButtonClass}
        style={rowPadding(depth)}
      >
        <MindLineIcon className={rowIconClass(isOpen)} />
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
        name={title}
        current={mind.folderId}
        onChoose={(folderId) => {
          if (folderId !== mind.folderId) {
            void useAppStore.getState().moveToFolder({ mindIds: [mind.id] }, folderId);
          }
        }}
        onClose={() => setMoving(null)}
      />
    </li>
  );
}
