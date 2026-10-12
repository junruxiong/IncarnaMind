import { useEffect } from "react";
import type { DeletedItem } from "../../../core/api";
import { daysLeft, whenDeleted } from "../deletedItems";
import { useLanguage, useT } from "../i18n";
import { deletedName, restoreItems, useDeleted, watchDeleted } from "../recentlyDeleted";
import { useRequests } from "../requests";
import { useSelectable, useSelectedOn, useSelection, useSelectionOrder } from "../selection";
import { useAppStore } from "../store";
import {
  ConversationLineIcon,
  DocumentLineIcon,
  FolderLineIcon,
  MindLineIcon,
  TagLineIcon,
} from "./lineIcons";
import { ObjectMenu, useObjectMenu } from "./ObjectMenu";
import { rowActionButtonClass } from "./sidebarRows";
import { buttonStyle } from "./ui";

const iconClass = "size-4 shrink-0 text-ink-meta";
const boxClass = "size-4 shrink-0 cursor-pointer accent-ink";

function KindIcon({ item }: { item: DeletedItem }) {
  if (item.kind === "document") {
    return <DocumentLineIcon kind={item.documentKind ?? "text"} className={iconClass} />;
  }
  const Icon = {
    mind: MindLineIcon,
    conversation: ConversationLineIcon,
    folder: FolderLineIcon,
    tag: TagLineIcon,
  }[item.kind];
  return <Icon className={iconClass} />;
}

/**
 * Recently deleted (CONTEXT.md): a view of the Library tab. Each deleted Mind,
 * conversation, Document, Folder and Tag of the last 30 days, newest first,
 * with where it was, when it was deleted and the days left; Restore and Delete
 * Permanently work on one row, on a selection and from a row's menu.
 */
export function RecentlyDeleted() {
  const t = useT();
  const language = useLanguage();
  const items = useDeleted((state) => state.items);
  const groups = useAppStore((state) => state.library?.groups);
  const held = useSelectedOn("library");

  useEffect(() => {
    const report = useAppStore.getState().reportError;
    const reload = () => void useDeleted.getState().refresh().catch(report);
    reload();
    return watchDeleted(reload);
  }, []);
  // The selection keeps only what is still listed.
  useEffect(() => {
    if (!items) return;
    const listed = new Set(items.map((item) => item.id));
    useSelection.getState().keepOnly((ref) => ref.kind !== "deleted" || listed.has(ref.id));
  }, [items]);

  const rows = items ?? [];
  useSelectionOrder("library", () => rows.map((item) => ({ kind: "deleted", id: item.id })));
  const heldIds = new Set(held.map((ref) => ref.id));
  const chosen = rows.filter((item) => heldIds.has(item.id));
  const now = new Date();

  const wasIn = (item: DeletedItem): string => {
    if (item.kind === "folder") return t("deleted.wasIn.library");
    if (item.kind === "tag") return t("deleted.wasIn.tags");
    // A Folder deleted too is still where it was, until it is deleted permanently.
    const folder =
      groups?.find((group) => group.id === item.groupId) ??
      rows.find((each) => each.kind === "folder" && each.id === item.groupId);
    if (folder) return folder.name;
    return t(item.kind === "conversation" ? "deleted.wasIn.chats" : "deleted.wasIn.none");
  };

  const deletePermanently = (these: readonly DeletedItem[]) =>
    useRequests.getState().ask({ deletePermanently: { items: these, empty: false } });
  const report = useAppStore.getState().reportError;
  const restore = (these: readonly DeletedItem[]) => void restoreItems(these).catch(report);

  return (
    <section
      data-testid="recently-deleted"
      data-surface="library"
      role="tabpanel"
      id="mind-tabpanel"
      aria-labelledby="library-tab"
      className="library-pane flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-sheet text-ui text-ink"
    >
      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-[1040px]">
          <h1
            data-testid="deleted-heading"
            className="line-clamp-2 font-serif text-heading font-semibold [overflow-wrap:break-word]"
          >
            {t("deleted.title")}
          </h1>
          <p className="mt-1 text-[13px] text-ink-meta">{t("deleted.intro")}</p>
          <div className="mt-3 mb-5 flex flex-wrap items-center gap-2">
            <button
              type="button"
              data-testid="deleted-leave"
              className={buttonStyle("ghost", "sm")}
              onClick={() => useAppStore.getState().openLibrary()}
            >
              {t("deleted.back")}
            </button>
            {rows.length > 0 && (
              <button
                type="button"
                data-testid="deleted-empty"
                className={`${buttonStyle("ghost", "sm")} ml-auto`}
                onClick={() =>
                  useRequests.getState().ask({ deletePermanently: { items: rows, empty: true } })
                }
              >
                {t("deleted.emptyAction")}
              </button>
            )}
          </div>
          {items !== null && rows.length === 0 ? (
            <p data-testid="deleted-nothing" className="border-t border-rule py-10 text-ink-meta">
              {t("deleted.empty")}
            </p>
          ) : (
            <>
              {chosen.length > 0 && (
                <div
                  role="toolbar"
                  aria-label={t("deleted.selected", { count: chosen.length })}
                  data-testid="deleted-selection"
                  onKeyDown={(event) => {
                    if (event.key === "Escape") useSelection.getState().clear("library");
                  }}
                  className="sticky top-0 z-10 mb-2 flex min-h-10 flex-wrap items-center gap-x-3 gap-y-1 rounded-md bg-frame px-2 py-1 text-[13px]"
                >
                  <input
                    type="checkbox"
                    data-testid="deleted-select-all"
                    aria-label={t("deleted.selectAll", { count: rows.length })}
                    checked={chosen.length === rows.length}
                    ref={(box) => {
                      if (box) box.indeterminate = chosen.length < rows.length;
                    }}
                    onChange={() =>
                      chosen.length === rows.length
                        ? useSelection.getState().clear("library")
                        : useSelection.getState().all("library")
                    }
                    className={boxClass}
                  />
                  <span role="status" className="font-semibold text-ink">
                    {t("deleted.selected", { count: chosen.length })}
                  </span>
                  <button
                    type="button"
                    data-testid="deleted-restore-selected"
                    className={buttonStyle("secondary", "sm")}
                    onClick={() => restore(chosen)}
                  >
                    {t(chosen.length > 1 ? "deleted.restore.count" : "deleted.restore", {
                      count: chosen.length,
                    })}
                  </button>
                  <button
                    type="button"
                    data-testid="deleted-delete-selected"
                    className={buttonStyle("secondary", "sm")}
                    onClick={() => deletePermanently(chosen)}
                  >
                    {t(
                      chosen.length > 1
                        ? "deleted.deletePermanently.count"
                        : "deleted.deletePermanently",
                      { count: chosen.length },
                    )}
                  </button>
                  <button
                    type="button"
                    data-testid="deleted-clear"
                    className={`${buttonStyle("ghost", "sm")} ml-auto`}
                    onClick={() => useSelection.getState().clear("library")}
                  >
                    {t("deleted.clear")}
                  </button>
                </div>
              )}
              <div
                aria-hidden="true"
                className="deleted-columns deleted-column-labels border-y border-rule py-2 text-label font-semibold text-ink-meta"
              >
                <span>{t("deleted.name")}</span>
                <span className="deleted-meta">
                  <span>{t("deleted.wasIn")}</span>
                  <span>{t("deleted.when")}</span>
                  <span>{t("deleted.left")}</span>
                </span>
                <span />
              </div>
              {/* biome-ignore lint/a11y/useSemanticElements: a list of rows laid out as columns, not a table. */}
              <ul
                // biome-ignore lint/a11y/noNoninteractiveElementToInteractiveRole: the list is one grid whose rows select (aria-multiselectable).
                role="grid"
                aria-multiselectable="true"
                aria-label={t("deleted.list")}
              >
                {rows.map((item) => (
                  <DeletedRow
                    key={item.id}
                    item={item}
                    wasIn={wasIn(item)}
                    when={whenDeleted(item, now, language, t)}
                    left={daysLeft(item, now)}
                    selecting={chosen.length > 0}
                    onRestore={restore}
                    onDelete={deletePermanently}
                  />
                ))}
              </ul>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function DeletedRow({
  item,
  wasIn,
  when,
  left,
  selecting,
  onRestore,
  onDelete,
}: {
  item: DeletedItem;
  wasIn: string;
  when: string;
  left: number;
  selecting: boolean;
  onRestore(items: readonly DeletedItem[]): void;
  onDelete(items: readonly DeletedItem[]): void;
}) {
  const t = useT();
  const name = deletedName(item);
  const target = { kind: "deleted", id: item.id } as const;
  const objectMenu = useObjectMenu(target);
  // A click on the row selects just it; there is nothing to open.
  const select = useSelectable(target, "library", () => undefined);
  const selected = select.selected;
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the checkbox is the keyboard's way to select the row.
    // biome-ignore lint/a11y/useSemanticElements: the row is laid out as columns, not as a table row.
    // biome-ignore lint/a11y/useFocusableInteractive: the row's buttons and checkbox take the focus.
    <li
      // biome-ignore lint/a11y/noNoninteractiveElementToInteractiveRole: a row of the grid, which says whether it is selected.
      role="row"
      {...select.rowProps}
      data-testid="deleted-item"
      data-kind={item.kind}
      data-deleted-id={item.id}
      data-selected={selected ? "" : undefined}
      {...objectMenu.rowProps}
      onClick={(event) => {
        if (event.target instanceof Element && event.target.closest("button, input, [role=menu]"))
          return;
        select.onClick(event);
      }}
      className={`group/row deleted-columns relative border-b border-rule py-2.5 ${
        selected ? "bg-accent-wash" : ""
      }`}
    >
      <input
        type="checkbox"
        data-testid="deleted-select"
        aria-label={t("deleted.select", { name })}
        checked={selected}
        onChange={(event) => {
          const { toggle, extendTo } = useSelection.getState();
          if ((event.nativeEvent as MouseEvent).shiftKey) extendTo("library", target);
          else toggle("library", target);
        }}
        className={`${boxClass} absolute top-3.5 -left-[21px] ${
          selected || selecting
            ? ""
            : "opacity-0 group-hover/row:opacity-100 focus-visible:opacity-100"
        }`}
      />
      <span className="flex min-w-0 items-center gap-2">
        <KindIcon item={item} />
        <span
          data-testid="deleted-name"
          title={name}
          className="min-w-0 truncate font-semibold text-ink"
        >
          {name}
        </span>
        <span className="sr-only">{t(`deleted.kind.${item.kind}`)}</span>
      </span>
      <span className="deleted-meta text-ink-secondary">
        <span data-testid="deleted-was-in" className="min-w-0 truncate">
          {wasIn}
        </span>
        <span data-testid="deleted-when">{when}</span>
        <span data-testid="deleted-left" className="tabular-nums">
          {t(left === 1 ? "deleted.days.one" : "deleted.days", { count: left })}
        </span>
      </span>
      <span className="deleted-actions flex items-center justify-end gap-1">
        {!selected && (
          <>
            <button
              type="button"
              data-testid="deleted-restore"
              className={buttonStyle("ghost", "sm")}
              onClick={() => onRestore([item])}
            >
              {t("deleted.restore")}
            </button>
            <button
              type="button"
              data-testid="deleted-delete"
              className={buttonStyle("ghost", "sm")}
              onClick={() => onDelete([item])}
            >
              {t("deleted.deletePermanently")}
            </button>
          </>
        )}
        <ObjectMenu
          target={target}
          label={t("deleted.more", { name })}
          className={`${rowActionButtonClass} opacity-0 group-hover/row:opacity-100 group-has-[:focus-visible]/row:opacity-100 aria-expanded:opacity-100`}
        />
      </span>
    </li>
  );
}
