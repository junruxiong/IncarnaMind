import type { ReactNode } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { NEEDS_REVIEW } from "../libraryFilters";
import { useAppStore } from "../store";
import { CheckLineIcon, CloseLineIcon, TagLineIcon } from "./lineIcons";
import { rowActionButtonClass, rowClass } from "./sidebarRows";
import { ReviewRing, TagDot } from "./TagColour";
import { pickerLabel, TagPicker, tagPopoverClass, useTagPopover } from "./TagPicker";
import { menuClass, menuItemClass, menuRuleClass, menuTitleClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * A Document's Tags button and the Tag picker it opens (see `TagPicker`):
 * the sidebar's row offers it when pointed at. Under the picker, "Re-tag
 * automatically" (or Organize) and "Manage Tags…".
 */
export function DocumentTagMenu({
  item,
  buttonClassName,
  children,
}: {
  item: Document;
  buttonClassName: string;
  children?: ReactNode;
}) {
  const t = useT();
  const popover = useTagPopover();
  const label = pickerLabel(t, [item]);
  return (
    <>
      <button
        {...popover.buttonProps}
        type="button"
        data-testid="document-tags-menu"
        aria-label={label}
        title={label}
        className={buttonClassName}
      >
        {children ?? <TagLineIcon className="size-[15px]" />}
      </button>
      <div
        {...popover.popoverProps}
        role="dialog"
        aria-label={label}
        data-testid="document-tags-popover"
        className={tagPopoverClass}
      >
        {/* Only while open: every Document has this popover, and Tag names in each would read as its own text. */}
        {popover.open && (
          <TagPicker
            documentIds={[item.id]}
            label={label}
            onReposition={popover.reposition}
            footer={<DocumentTagFooter item={item} close={popover.close} />}
          />
        )}
      </div>
    </>
  );
}

/**
 * Under a Document's Tag picker: where automatic tagging is (waiting,
 * working, failed), "Re-tag automatically" (Organize, once Folders are in
 * use) and "Manage Tags…".
 */
export function DocumentTagFooter({ item, close }: { item: Document; close(): void }) {
  const t = useT();
  const library = useAppStore((state) => state.library);
  const organizing = !!library?.groups.length || !!library?.settings.classifier;
  const taggingLine =
    item.status !== "ready" || organizing
      ? null
      : item.tagging === "pending"
        ? t("tags.state.pending")
        : item.tagging === "tagging"
          ? t("tags.state.tagging")
          : item.tagging === "failed"
            ? t("tags.state.failed")
            : null;
  const footerButton =
    "h-7 shrink-0 rounded-md px-2 text-[12px] font-semibold text-ink-secondary hover:bg-hover hover:text-ink";
  return (
    <>
      {taggingLine && (
        <span
          data-testid="document-tagging"
          title={item.taggingError?.message}
          className={`max-w-28 truncate px-1 text-[12px] ${
            item.tagging === "failed" ? "text-danger" : "text-ink-meta"
          }`}
        >
          {taggingLine}
        </span>
      )}
      {item.status === "ready" && (
        <button
          type="button"
          data-testid="retag-document"
          onClick={() => {
            close();
            if (organizing && !library?.settings.classifier)
              useAppStore.getState().openSettings("organization");
            else void useAppStore.getState().retagDocuments([item.id]);
          }}
          className={footerButton}
        >
          {t(organizing ? "library.classify" : "tags.menu.retag")}
        </button>
      )}
      <button
        type="button"
        data-testid="open-tags-dialog"
        onClick={() => {
          close();
          useAppStore.getState().openTagsDialog();
        }}
        className={footerButton}
      >
        {t("tags.menu.manage")}
      </button>
    </>
  );
}

/** A Tag filter value in words: a Tag's name, or "Needs review". */
function useFilterName(): (value: string) => string {
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  return (value) =>
    value === NEEDS_REVIEW
      ? t("tags.filter.review")
      : (tags.find((tag) => tag.id === value)?.name ?? value);
}

/**
 * The Documents label's Tags button: a menu to show only the Documents with
 * any of the Tags chosen (choosing one again takes it out), and "Manage
 * Tags…". The Library's Tags filter is the same filter.
 */
export function TagFilterMenu() {
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  const tagFilter = useAppStore((state) => state.tagFilter);
  const reviewing = useAppStore((state) =>
    state.documents.some((item) => item.tags.some((link) => link.needsReview)),
  );
  const menu = usePopoverMenu();
  const values = [...(reviewing || tagFilter.includes(NEEDS_REVIEW) ? [NEEDS_REVIEW] : [])];
  const name = useFilterName();

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
            {[...values, ...tags.map((tag) => tag.id)].map((value) => {
              const selected = tagFilter.includes(value);
              const tag = tags.find((each) => each.id === value);
              return (
                <button
                  key={value}
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={selected}
                  data-testid="tag-filter"
                  data-tag-id={value}
                  title={tag?.description || undefined}
                  onClick={() => {
                    menu.close();
                    useAppStore.getState().toggleTagFilter(value);
                  }}
                  className={`${menuItemClass} pl-2`}
                >
                  <CheckLineIcon
                    className={`size-3.5 shrink-0 ${selected ? "text-ink" : "invisible"}`}
                  />
                  {tag ? <TagDot colour={tag.colour} /> : value === NEEDS_REVIEW && <ReviewRing />}
                  <span className="truncate">{name(value)}</span>
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
                useAppStore.getState().openTagsDialog();
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

/** While the Documents are filtered by Tags: a row that says which, to clear it. */
export function ActiveTagFilter() {
  const t = useT();
  const tagFilter = useAppStore((state) => state.tagFilter);
  const name = useFilterName();
  if (tagFilter.length === 0) return null;
  return (
    <div
      data-testid="tag-filter-active"
      data-tag-id={tagFilter.join(" ")}
      className={rowClass(false)}
    >
      <div className="flex h-full w-full min-w-0 items-center gap-2 pr-1 pl-2">
        <TagLineIcon className="size-4 shrink-0 text-ink-meta" />
        <span className="min-w-0 flex-1 truncate">
          {t("tags.filter.active", { tag: tagFilter.map(name).join(", ") })}
        </span>
        <button
          type="button"
          data-testid="tag-filter-clear"
          aria-label={t("tags.filter.clear")}
          title={t("tags.filter.clear")}
          onClick={() => useAppStore.getState().setTagFilter([])}
          className={rowActionButtonClass}
        >
          <CloseLineIcon className="size-3.5" />
        </button>
      </div>
    </div>
  );
}
