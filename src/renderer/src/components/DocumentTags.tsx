import type { Document, TaggingState } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { CheckIcon, CloseIcon, TagIcon } from "./icons";
import { menuClass, menuItemClass, menuTitleClass, usePopoverMenu } from "./usePopoverMenu";

/** What a ready Document's tagging line says; nothing once it is tagged. */
const taggingMessages: Partial<Record<TaggingState, MessageKey>> = {
  pending: "tags.state.pending",
  "waiting-for-provider": "tags.state.waiting-for-provider",
  tagging: "tags.state.tagging",
  failed: "tags.state.failed",
};

/**
 * Where automatic tagging is for a Document, under its status. Shown only
 * once the Document is ready: tagging never holds that up.
 */
export function TaggingStatus({ item }: { item: Document }) {
  const t = useT();
  const message = item.status === "ready" ? taggingMessages[item.tagging] : undefined;
  if (!message) return null;
  return (
    <span
      data-testid="document-tagging"
      title={item.taggingError?.message}
      className={`block truncate text-[11px] leading-4 ${
        item.tagging === "failed" ? "text-amber-700" : "text-gray-400"
      } ${item.tagging === "tagging" ? "animate-pulse" : ""}`}
    >
      {t(message)}
    </span>
  );
}

/**
 * A Document's Tags, as chips. Tags the User added are tinted; each chip's ×
 * takes the Tag off, and automatic tagging then leaves it off.
 */
export function DocumentTagChips({ item }: { item: Document }) {
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  const removeDocumentTag = useAppStore((state) => state.removeDocumentTag);
  const byId = new Map(tags.map((tag) => [tag.id, tag]));
  const shown = item.tags.flatMap((link) => {
    const tag = byId.get(link.tagId);
    return tag ? [{ link, tag }] : [];
  });
  if (shown.length === 0) return null;
  return (
    // Indented to line up with the Document's name, past its icon.
    <ul aria-label={t("tags.title")} className="mt-[3px] ml-[22px] flex flex-wrap gap-[3px]">
      {shown.map(({ link, tag }) => (
        <li
          key={tag.id}
          data-testid="document-tag"
          data-tag-id={tag.id}
          data-source={link.source}
          title={t(link.source === "user" ? "tags.chip.user" : "tags.chip.automatic", {
            tag: tag.name,
          })}
          className={`inline-flex max-w-full items-center gap-[1px] rounded-full py-[1px] pr-[2px] pl-[6px] text-[11px] leading-4 ${
            link.source === "user" ? "bg-indigo-50 text-indigo-700" : "bg-gray-100 text-gray-600"
          } ${link.needsReview ? "ring-1 ring-amber-400" : ""}`}
        >
          <span className="truncate">{tag.name}</span>
          <button
            type="button"
            data-testid="remove-document-tag"
            aria-label={t("tags.chip.remove", { tag: tag.name, name: item.name })}
            title={t("tags.chip.remove", { tag: tag.name, name: item.name })}
            onClick={() => void removeDocumentTag(item.id, tag.id)}
            className="shrink-0 rounded-full p-[1px] opacity-50 hover:bg-black/10 hover:opacity-100 focus-visible:opacity-100"
          >
            <CloseIcon className="size-[10px]" />
          </button>
        </li>
      ))}
    </ul>
  );
}

/**
 * A Document's Tags button and its menu: every Tag, checked when the Document
 * has it, to add or take off; then "Re-tag automatically" and "Manage Tags…".
 */
export function DocumentTagMenu({
  item,
  buttonClassName,
}: {
  item: Document;
  buttonClassName: string;
}) {
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  const addDocumentTag = useAppStore((state) => state.addDocumentTag);
  const removeDocumentTag = useAppStore((state) => state.removeDocumentTag);
  const retagDocuments = useAppStore((state) => state.retagDocuments);
  const openTagsDialog = useAppStore((state) => state.openTagsDialog);
  const menu = usePopoverMenu();
  const has = new Set(item.tags.map((link) => link.tagId));

  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="document-tags-menu"
        aria-label={t("tags.menu.open", { name: item.name })}
        title={t("tags.menu.open", { name: item.name })}
        className={buttonClassName}
      >
        <TagIcon className="size-[14px]" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("tags.menu.title")}
        data-testid="document-tags-popover"
        className={menuClass}
      >
        {/* Only while open: every Document has this menu, and Tag names in each would read as its own text. */}
        {menu.open && (
          <>
            <p className={menuTitleClass}>{t("tags.menu.title")}</p>
            {tags.length === 0 && (
              <p className="px-2 py-[5px] text-gray-400">{t("tags.menu.none")}</p>
            )}
            {tags.map((tag) => {
              const checked = has.has(tag.id);
              return (
                <button
                  key={tag.id}
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={checked}
                  data-testid="tag-menu-item"
                  data-tag-id={tag.id}
                  title={tag.description || undefined}
                  onClick={() =>
                    void (checked
                      ? removeDocumentTag(item.id, tag.id)
                      : addDocumentTag(item.id, tag.id))
                  }
                  className={`${menuItemClass} pl-2`}
                >
                  <CheckIcon
                    className={`size-[14px] shrink-0 ${checked ? "text-gray-700" : "invisible"}`}
                  />
                  <span className="truncate">{tag.name}</span>
                </button>
              );
            })}
            <div className="my-1 border-t border-gray-100" />
            {item.status === "ready" && (
              <button
                type="button"
                role="menuitem"
                data-testid="retag-document"
                onClick={() => {
                  menu.close();
                  void retagDocuments([item.id]);
                }}
                className={`${menuItemClass} pl-[28px]`}
              >
                {t("tags.menu.retag")}
              </button>
            )}
            <button
              type="button"
              role="menuitem"
              data-testid="open-tags-dialog"
              onClick={() => {
                menu.close();
                openTagsDialog();
              }}
              className={`${menuItemClass} pl-[28px]`}
            >
              {t("tags.menu.manage")}
            </button>
          </>
        )}
      </div>
    </>
  );
}

/**
 * The sidebar's Tag filter: a chip per Tag. Pressing one shows only the
 * Documents with it (in the Folder chosen above, if any); pressing it again
 * shows them all.
 */
export function TagFilter() {
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  const tagFilter = useAppStore((state) => state.tagFilter);
  const filterByTag = useAppStore((state) => state.filterByTag);
  if (tags.length === 0) return null;
  return (
    <nav
      aria-label={t("tags.filter.label")}
      data-testid="tag-filters"
      className="mx-3 mb-1 flex flex-wrap gap-[3px] border-b border-gray-200 px-1 pt-[2px] pb-[6px]"
    >
      {tags.map((tag) => {
        const selected = tag.id === tagFilter;
        return (
          <button
            key={tag.id}
            type="button"
            data-testid="tag-filter"
            data-tag-id={tag.id}
            aria-pressed={selected}
            title={tag.description || undefined}
            onClick={() => void filterByTag(selected ? null : tag.id)}
            className={`max-w-full truncate rounded-full px-[7px] py-[1px] text-[11px] leading-4 ${
              selected ? "bg-gray-700 text-white" : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {tag.name}
          </button>
        );
      })}
    </nav>
  );
}
