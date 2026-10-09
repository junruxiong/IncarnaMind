import type { Document, DocumentTag, Tag } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { chipsOf } from "../tagEditing";
import { DocumentTagFooter } from "./DocumentTags";
import {
  CheckLineIcon,
  CloseLineIcon,
  PencilLineIcon,
  SparkLineIcon,
  TagLineIcon,
} from "./lineIcons";
import { tagChipColour } from "./TagColour";
import {
  chipClass,
  chipTitle,
  pickerLabel,
  ReviewDot,
  TagPicker,
  tagPopoverClass,
  useTagPopover,
} from "./TagPicker";

/** How many chips a row, or the viewer's header, shows before "+N": they stay one line. */
const VISIBLE_CHIPS = { row: 3, header: 2 } as const;

/**
 * A Document's Tags on its Library row, on one line: the first few as chips
 * (a spark marks one tagging applied automatically, an amber dot one it
 * wasn't sure of, with ✓ and × to confirm or remove it), then "+N", then the
 * button that opens the Tag picker. Clicking a chip filters the Library by it.
 * In the viewer's header (`variant="header"`) the chips only show, the
 * picker's button always shows, and a narrow viewer keeps only the button.
 */
export function DocumentTagChips({
  document,
  variant = "row",
}: {
  document: Document;
  variant?: "row" | "header";
}) {
  const header = variant === "header";
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  const tagFilter = useAppStore((state) => state.tagFilter);
  const popover = useTagPopover();
  const chips = chipsOf(document.tags, tags);
  const shown = chips.slice(0, VISIBLE_CHIPS[variant]);
  const hidden = chips.slice(VISIBLE_CHIPS[variant]);
  const label = pickerLabel(t, [document]);
  return (
    <div
      data-testid="document-tags"
      data-variant={variant}
      className={`flex min-w-0 items-center gap-1 ${header ? "viewer-tags shrink" : "h-8"}`}
    >
      {shown.map(({ tag, link }) => (
        <TagChip
          key={tag.id}
          document={document}
          tag={tag}
          link={link}
          filtered={tagFilter.includes(tag.id)}
          filters={!header}
        />
      ))}
      {hidden.length > 0 && (
        <button
          type="button"
          data-testid="tag-chip-more"
          title={t("tags.chip.more", {
            count: hidden.length,
            names: hidden.map(({ tag }) => tag.name).join(", "),
          })}
          aria-label={t("tags.chip.more", {
            count: hidden.length,
            names: hidden.map(({ tag }) => tag.name).join(", "),
          })}
          onClick={() => popover.buttonProps.ref.current?.click()}
          className="h-5 shrink-0 rounded-sm px-1 text-[12px] text-ink-meta hover:bg-hover hover:text-ink"
        >
          +{hidden.length}
        </button>
      )}
      <button
        {...popover.buttonProps}
        type="button"
        data-testid="document-tags-menu"
        aria-label={t("tags.edit", { name: document.name })}
        title={t("tags.edit", { name: document.name })}
        className={
          header
            ? "viewer-icon-button"
            : chips.length === 0
              ? "h-6 shrink-0 rounded-sm px-1.5 text-[12px] text-ink-meta hover:bg-hover hover:text-ink"
              : "inline-flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-meta opacity-0 group-hover/row:opacity-100 hover:bg-hover hover:text-ink focus-visible:opacity-100 aria-expanded:opacity-100"
        }
      >
        {header ? (
          <TagLineIcon className="size-4" />
        ) : chips.length === 0 ? (
          t("library.addTags")
        ) : (
          <PencilLineIcon className="size-3.5" />
        )}
      </button>
      <div
        {...popover.popoverProps}
        role="dialog"
        aria-label={label}
        data-testid="document-tags-popover"
        className={tagPopoverClass}
      >
        {popover.open && (
          <TagPicker
            documentIds={[document.id]}
            label={label}
            onReposition={popover.reposition}
            footer={<DocumentTagFooter item={document} close={popover.close} />}
          />
        )}
      </div>
    </div>
  );
}

/**
 * One Tag on a row. Its name filters the Library by it (again: stops). One
 * awaiting review shows ✓ and × beside it, on hover or focus, to keep or
 * remove it. Without `filters` (the viewer's header) the name only shows, and
 * ✓ and × always do, so the keyboard reaches them.
 */
function TagChip({
  document,
  tag,
  link,
  filtered,
  filters,
}: {
  document: Document;
  tag: Tag;
  link: DocumentTag;
  filtered: boolean;
  filters: boolean;
}) {
  const t = useT();
  const review = link.needsReview;
  return (
    <span
      data-testid="tag-chip"
      data-tag-id={tag.id}
      data-source={link.source}
      data-needs-review={review ? "true" : undefined}
      data-filtered={filtered ? "true" : undefined}
      data-colour={tag.colour}
      title={chipTitle(link, tag, t)}
      className={`group/chip ${chipClass} ${tagChipColour(tag.colour)} min-w-0 shrink ${
        review ? "pr-0.5" : ""
      } ${filtered ? "shadow-[inset_0_0_0_1px_currentColor]" : ""}`}
    >
      {review && <ReviewDot />}
      {link.source === "automatic" && !review && (
        <SparkLineIcon aria-hidden="true" className="size-2.5 shrink-0 opacity-70" />
      )}
      {filters ? (
        <button
          type="button"
          data-testid="tag-chip-filter"
          aria-pressed={filtered}
          aria-label={t(filtered ? "tags.chip.unfilter" : "tags.chip.filter", { tag: tag.name })}
          onClick={() => useAppStore.getState().toggleTagFilter(tag.id)}
          className="min-w-0 truncate rounded-sm hover:underline focus-visible:outline-offset-0"
        >
          {tag.name}
        </button>
      ) : (
        <span className="min-w-0 truncate">{tag.name}</span>
      )}
      {review && (
        <span
          className={`shrink-0 items-center ${
            filters
              ? "hidden group-hover/chip:inline-flex group-focus-within/chip:inline-flex"
              : "inline-flex"
          }`}
        >
          <button
            type="button"
            data-testid="confirm-document-tag"
            aria-label={t("jev.chip.confirm", { tag: tag.name, name: document.name })}
            title={t("tags.review.confirm", { tag: tag.name })}
            onClick={() => void useAppStore.getState().addDocumentTag(document.id, tag.id)}
            className="inline-flex size-4 items-center justify-center rounded-sm hover:bg-sheet/70"
          >
            <CheckLineIcon className="size-3" />
          </button>
          <button
            type="button"
            data-testid="reject-document-tag"
            aria-label={t("tags.review.reject", { tag: tag.name })}
            title={t("tags.review.reject", { tag: tag.name })}
            onClick={() => void useAppStore.getState().removeDocumentTag(document.id, tag.id)}
            className="inline-flex size-4 items-center justify-center rounded-sm hover:bg-sheet/70"
          >
            <CloseLineIcon className="size-2.5" />
          </button>
        </span>
      )}
    </span>
  );
}
