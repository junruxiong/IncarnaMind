import { type ReactNode, useMemo, useState } from "react";
import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { isShown } from "../layoutSlice";
import { documentsByTag, NEEDS_REVIEW } from "../libraryFilters";
import type { ObjectRef } from "../objects";
import { useAppStore } from "../store";
import { editTag } from "../tagChanges";
import { BrowseGroup } from "./BrowseGroup";
import { TagFilterMenu } from "./DocumentTags";
import { useFolds } from "./LibraryFolders";
import { SidebarSection } from "./SidebarSection";
import { ReviewRing, TagDot } from "./TagColour";

const NONE: readonly Document[] = [];

/** A Tag's mark in the icon column: its 11px dot, as in Finder's Tags, or the review ring. */
const markBox = (mark: ReactNode) => (
  <span className="flex size-4 shrink-0 items-center justify-center">{mark}</span>
);

/**
 * The sidebar's Tags section, last in the tree, once a Tag is on a Document:
 * "Needs review" first while a Tag awaits review (as in the Tags filter),
 * then each Tag in use, in name order, with its dot and how many Documents
 * carry it. Its heading folds it away, and offers the Tag filter. Clicking a
 * Tag shows its Documents in the Library, through the Tag filter the Library
 * and the sidebar share, and marks it chosen; clicking it again there stops
 * filtering, as a chip does. Its chevron lists its Documents here, as a
 * Folder's does; a Document with several Tags is under each. The counts are
 * of every Document, whatever the filter, as Manage Tags counts them.
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
  const folded = useFolds((state) => state.tagsFolded);

  const group = (
    value: string,
    name: string,
    mark: ReactNode,
    title?: string,
    objectOf?: ObjectRef,
    rename?: { label: string; commit(name: string): void | Promise<void> },
  ) => (
    <BrowseGroup
      key={value}
      testId="sidebar-tag"
      data={{ "data-tag-id": value }}
      icon={markBox(mark)}
      name={name}
      title={title}
      object={objectOf}
      documents={byTag.get(value) ?? NONE}
      selected={tagFilter.includes(value)}
      expanded={expanded.has(value)}
      onOpen={() => {
        const store = useAppStore.getState();
        const only = store.tagFilter.length === 1 && store.tagFilter[0] === value;
        if (only && isShown(store.layout, { kind: "library" })) store.setTagFilter([]);
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
      rename={rename}
      renderDocument={renderDocument}
    />
  );

  const used = tags.filter((tag) => byTag.has(tag.id));
  const reviewing = byTag.has(NEEDS_REVIEW) || tagFilter.includes(NEEDS_REVIEW);
  if (used.length === 0 && !reviewing) return null;
  return (
    <SidebarSection
      testId="sidebar-tags"
      title={t("tags.title")}
      headingTestId="tags-toggle"
      fold={{ open: !folded, toggle: () => useFolds.getState().toggleTags() }}
      actions={<TagFilterMenu />}
    >
      <ul role="none">
        {reviewing && group(NEEDS_REVIEW, t("tags.filter.review"), <ReviewRing />)}
        {used.map((tag) =>
          group(
            tag.id,
            tag.name,
            <TagDot colour={tag.colour} size="md" />,
            tag.description || tag.name,
            { kind: "tag", id: tag.id },
            {
              label: t("tag.renameLabel", { name: tag.name }),
              commit: (next) => editTag(tag.id, { name: next }),
            },
          ),
        )}
      </ul>
    </SidebarSection>
  );
}
