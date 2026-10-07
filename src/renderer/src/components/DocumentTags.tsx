import type { Document, DocumentTag, Tag } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { CheckLineIcon, CloseLineIcon, TagLineIcon } from "./lineIcons";
import { rowActionButtonClass, rowClass } from "./sidebarRows";
import { menuClass, menuItemClass, menuRuleClass, menuTitleClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/** A Tag's tooltip on a Document: who added it and, with Jev, how likely it is. */
function tagTitle(link: DocumentTag, tag: Tag, t: ReturnType<typeof useT>): string {
  if (link.source === "user") return t("tags.chip.user", { tag: tag.name });
  if (link.confidence === null) return t("tags.chip.automatic", { tag: tag.name });
  const percent = Math.round(link.confidence * 100);
  return t(link.needsReview ? "jev.chip.review" : "jev.chip.likely", { tag: tag.name, percent });
}

/**
 * A Document's Tags button and its menu, where its Tags are shown now that its
 * row is one line: every Tag, checked when the Document has it, to add or
 * take off (automatic tagging then leaves a removed one off). A Tag Jev wasn't
 * sure about says "needs review", with an item to confirm it (it becomes the
 * User's). Then "Re-tag automatically" and "Manage Tags…".
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
  const links = new Map(item.tags.map((link) => [link.tagId, link]));
  const taggingLine =
    item.status !== "ready"
      ? null
      : item.tagging === "pending"
        ? t("tags.state.pending")
        : item.tagging === "tagging"
          ? t("tags.state.tagging")
          : item.tagging === "failed"
            ? t("tags.state.failed")
            : null;

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
        <TagLineIcon className="size-[15px]" />
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
            {taggingLine && (
              <p
                data-testid="document-tagging"
                title={item.taggingError?.message}
                className={`px-2 pb-1 text-[12px] leading-[18px] ${
                  item.tagging === "failed" ? "text-danger" : "text-ink-meta"
                }`}
              >
                {taggingLine}
              </p>
            )}
            {tags.length === 0 && <p className="px-2 py-1 text-ink-meta">{t("tags.menu.none")}</p>}
            {tags.map((tag) => {
              const link = links.get(tag.id);
              return [
                <button
                  key={tag.id}
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={link !== undefined}
                  data-testid="tag-menu-item"
                  data-tag-id={tag.id}
                  data-source={link?.source}
                  data-needs-review={link?.needsReview ? "true" : undefined}
                  title={link ? tagTitle(link, tag, t) : tag.description || undefined}
                  onClick={() =>
                    void (link
                      ? removeDocumentTag(item.id, tag.id)
                      : addDocumentTag(item.id, tag.id))
                  }
                  className={`${menuItemClass} pl-2`}
                >
                  <CheckLineIcon
                    className={`size-3.5 shrink-0 ${link ? "text-ink" : "invisible"}`}
                  />
                  <span className="min-w-0 flex-1 truncate">{tag.name}</span>
                  {link?.needsReview && (
                    <span className="shrink-0 text-[12px] text-ink-meta">
                      {t("jev.chip.needsReview")}
                    </span>
                  )}
                </button>,
                link?.needsReview && (
                  <button
                    key={`${tag.id}-confirm`}
                    type="button"
                    role="menuitem"
                    data-testid="confirm-document-tag"
                    data-tag-id={tag.id}
                    aria-label={t("jev.chip.confirm", { tag: tag.name, name: item.name })}
                    onClick={() => void addDocumentTag(item.id, tag.id)}
                    className={`${menuItemClass} pl-[30px] text-ink-secondary`}
                  >
                    {t("tags.menu.confirm", { tag: tag.name })}
                  </button>
                ),
              ];
            })}
            <div className={menuRuleClass} />
            {item.status === "ready" && (
              <button
                type="button"
                role="menuitem"
                data-testid="retag-document"
                onClick={() => {
                  menu.close();
                  void retagDocuments([item.id]);
                }}
                className={`${menuItemClass} pl-[30px]`}
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
              className={`${menuItemClass} pl-[30px]`}
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
 * The Documents label's Tags button: a menu to show only the Documents with
 * a Tag (choosing it again shows them all), and "Manage Tags…".
 */
export function TagFilterMenu() {
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  const tagFilter = useAppStore((state) => state.tagFilter);
  const filterByTag = useAppStore((state) => state.filterByTag);
  const openTagsDialog = useAppStore((state) => state.openTagsDialog);
  const menu = usePopoverMenu();

  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="tag-filter-menu"
        aria-label={t("tags.title")}
        title={t("tags.filter.label")}
        className={rowActionButtonClass}
      >
        <TagLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("tags.title")}
        data-testid="tag-filters"
        className={menuClass}
      >
        {menu.open && (
          <>
            <p className={menuTitleClass}>{t("tags.filter.title")}</p>
            {tags.length === 0 && <p className="px-2 py-1 text-ink-meta">{t("tags.menu.none")}</p>}
            {tags.map((tag) => {
              const selected = tag.id === tagFilter;
              return (
                <button
                  key={tag.id}
                  type="button"
                  role="menuitemradio"
                  aria-checked={selected}
                  data-testid="tag-filter"
                  data-tag-id={tag.id}
                  title={tag.description || undefined}
                  onClick={() => {
                    menu.close();
                    void filterByTag(selected ? null : tag.id);
                  }}
                  className={`${menuItemClass} pl-2`}
                >
                  <CheckLineIcon
                    className={`size-3.5 shrink-0 ${selected ? "text-ink" : "invisible"}`}
                  />
                  <span className="truncate">{tag.name}</span>
                </button>
              );
            })}
            <div className={menuRuleClass} />
            <button
              type="button"
              role="menuitem"
              data-testid="manage-tags"
              onClick={() => {
                menu.close();
                openTagsDialog();
              }}
              className={`${menuItemClass} pl-[30px]`}
            >
              {t("tags.menu.manage")}
            </button>
          </>
        )}
      </div>
    </>
  );
}

/** While the Documents are filtered by a Tag: a row that says which, to clear it. */
export function ActiveTagFilter() {
  const t = useT();
  const tag = useAppStore((state) => state.tags.find((each) => each.id === state.tagFilter));
  const filterByTag = useAppStore((state) => state.filterByTag);
  if (!tag) return null;
  return (
    <div data-testid="tag-filter-active" data-tag-id={tag.id} className={rowClass(false)}>
      <div className="flex h-full w-full min-w-0 items-center gap-2 pr-1 pl-2">
        <TagLineIcon className="size-4 shrink-0 text-ink-meta" />
        <span className="min-w-0 flex-1 truncate">
          {t("tags.filter.active", { tag: tag.name })}
        </span>
        <button
          type="button"
          data-testid="tag-filter-clear"
          aria-label={t("tags.filter.clear")}
          title={t("tags.filter.clear")}
          onClick={() => void filterByTag(null)}
          className={rowActionButtonClass}
        >
          <CloseLineIcon className="size-3.5" />
        </button>
      </div>
    </div>
  );
}
