import { type ReactNode, useMemo, useState } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { documentsByTag, NEEDS_REVIEW } from "../libraryFilters";
import { useAppStore } from "../store";
import { BrowseGroup } from "./LibraryFolders";
import { ReviewRing, TagDot } from "./TagColour";

const NONE: readonly Document[] = [];

/** A Tag's mark in the icon column: its 11px dot, as in Finder's Tags, or the review ring. */
const markBox = (mark: ReactNode) => (
  <span className="flex size-4 shrink-0 items-center justify-center">{mark}</span>
);

/**
 * The sidebar's Tags view, beside Folders and Source locations: "Needs
 * review" first while a Tag awaits review (as in the Tags filter), then every
 * Tag in name order, each with its dot and how many Documents carry it.
 * Clicking one shows its Documents in the Library, through the Tag filter the
 * Library and the sidebar share, and marks it chosen; clicking it again
 * there stops filtering, as a chip does. Its chevron lists its Documents
 * here, as a Folder's does; a Document with several Tags is under each. The
 * counts are of every Document, whatever the filter, as Manage Tags counts them.
 */
export function SidebarTags({
  renderDocument,
}: {
  renderDocument(document: Document, depth: number): ReactNode;
}) {
  const t = useT();
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  const tagFilter = useAppStore((state) => state.tagFilter);
  // One pass over the Documents, again only when they change.
  const byTag = useMemo(() => documentsByTag(documents), [documents]);
  // Folded at first: a Document with several Tags would be listed several times.
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());

  const group = (value: string, name: string, mark: ReactNode, title?: string) => (
    <BrowseGroup
      key={value}
      testId="sidebar-tag"
      data={{ "data-tag-id": value }}
      icon={markBox(mark)}
      name={name}
      title={title}
      documents={byTag.get(value) ?? NONE}
      selected={tagFilter.includes(value)}
      selectedAs="pressed"
      expanded={expanded.has(value)}
      onOpen={() => {
        const store = useAppStore.getState();
        const only = store.tagFilter.length === 1 && store.tagFilter[0] === value;
        if (only && store.libraryOpen) store.setTagFilter([]);
        else {
          store.setTagFilter([value]);
          store.openLibrary();
        }
      }}
      onToggle={() =>
        setExpanded((previous) => {
          const next = new Set(previous);
          if (!next.delete(value)) next.add(value);
          return next;
        })
      }
      renderDocument={renderDocument}
    />
  );

  return (
    <nav aria-label={t("tags.title")} data-testid="sidebar-tags">
      {tags.length === 0 ? (
        <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">{t("tags.menu.none")}</p>
      ) : (
        <ul>
          {(byTag.has(NEEDS_REVIEW) || tagFilter.includes(NEEDS_REVIEW)) &&
            group(NEEDS_REVIEW, t("tags.filter.review"), <ReviewRing />)}
          {tags.map((tag) =>
            group(
              tag.id,
              tag.name,
              <TagDot colour={tag.colour} size="md" />,
              tag.description || tag.name,
            ),
          )}
        </ul>
      )}
    </nav>
  );
}
