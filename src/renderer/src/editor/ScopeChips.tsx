import { useState } from "react";
import type { SearchScope } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import type { ScopeKind } from "../../../shared/searchScope";
import { DocumentLineIcon, FolderLineIcon, TagLineIcon } from "../components/lineIcons";
import { useT } from "../i18n";
import { type ScopeChip, scopeChips } from "../scope";
import { useAppStore } from "../store";
import { RemoveIcon } from "./icons";

const CHIP_TITLES: Record<ScopeKind, MessageKey> = {
  folder: "scope.chip.folder",
  tag: "scope.chip.tag",
  document: "scope.chip.document",
};

const DELETED_NAMES: Record<ScopeKind, MessageKey> = {
  folder: "scope.chip.deleted.folder",
  tag: "scope.chip.deleted.tag",
  document: "scope.chip.deleted.document",
};

/** A chip's icon: the sidebar's line icon of its kind, in its muted ink. */
function ChipIcon({ chip }: { chip: ScopeChip }) {
  const className = "scope-chip-icon";
  if (chip.kind === "folder") return <FolderLineIcon className={className} />;
  if (chip.kind === "tag") return <TagLineIcon className={className} />;
  return <DocumentLineIcon kind={chip.documentKind ?? "text"} className={className} />;
}

/**
 * A Search scope as chips: one for each Folder, Tag and Document (its title
 * says which), with a × that takes it out. One deleted since is struck
 * through (a Folder with its name, the others as "Deleted …"): the search
 * ignores it. A long scope shows its first chips and "+N more", which shows
 * the rest. In the composer (`onChange`), each chip shows its kind's icon and
 * its name opens the "@" picker, to change the scope; the Mind's own Folder
 * (`ownFolderId`) says how many Documents it searches, and that taking it
 * out searches everything.
 */
export function ScopeChips({
  scope,
  ownFolderId = null,
  onRemove,
  onChange,
  shown: most,
  className,
  testId,
}: {
  scope: SearchScope;
  /** The Folder of the Mind the composer is in: its chip is the Mind's own. */
  ownFolderId?: string | null;
  onRemove(kind: ScopeKind, id: string): void;
  /** Opens the picker from a chip's name, as "@" does. */
  onChange?(): void;
  /** Beyond this many chips (the Documents a Library filter showed, say), the rest fold behind "+N more". */
  shown: number;
  className: string;
  testId: string;
}) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const groups = useAppStore((state) => state.library?.groups);
  const deletedGroups = useAppStore((state) => state.library?.deletedGroups);
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  // How many Documents the Mind's own Folder holds now, for its chip's tooltip.
  const ownCount = useAppStore((state) =>
    ownFolderId === null
      ? 0
      : (state.library?.assignments.filter((each) => each.groupId === ownFolderId).length ?? 0),
  );
  const chips = scopeChips({ folders, groups, deletedGroups, tags, documents }, scope);
  const [expanded, setExpanded] = useState(false);
  const long = chips.length > most;
  const shown = long && !expanded ? chips.slice(0, most - 1) : chips;
  const more = chips.length - shown.length;
  return (
    <ul
      contentEditable={false}
      data-testid={testId}
      aria-label={t("scope.picker.label")}
      className={className}
    >
      {shown.map((chip) => {
        const { deleted } = chip;
        const name = chip.name ?? t(DELETED_NAMES[chip.kind]);
        const remove = t("scope.chip.remove", { name });
        const own = !deleted && chip.kind === "folder" && chip.id === ownFolderId;
        const title = deleted
          ? t("scope.chip.deleted")
          : own
            ? t(ownCount === 1 ? "scope.chip.own.one" : "scope.chip.own", { count: ownCount })
            : t(CHIP_TITLES[chip.kind], { name });
        const label = deleted ? (
          <>
            <s className="truncate">{name}</s>
            <span className="sr-only">{t("scope.chip.deleted")}</span>
          </>
        ) : (
          <span className="truncate">{name}</span>
        );
        return (
          <li
            key={`${chip.kind}:${chip.id}`}
            data-testid="scope-chip"
            data-kind={chip.kind}
            data-id={chip.id}
            data-deleted={deleted ? "true" : undefined}
            data-own={own ? "true" : undefined}
            title={title}
            className={`scope-chip ${deleted ? "scope-chip--deleted" : ""}`}
          >
            {onChange ? (
              <button
                type="button"
                data-testid="scope-chip-change"
                aria-label={t("composer.scope.change", { name: title })}
                onClick={onChange}
                className="scope-chip-name"
              >
                <ChipIcon chip={chip} />
                {label}
              </button>
            ) : (
              label
            )}
            <button
              type="button"
              data-testid="scope-chip-remove"
              aria-label={remove}
              title={remove}
              // Keep the cursor where it is.
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => onRemove(chip.kind, chip.id)}
              className="scope-chip-remove"
            >
              <RemoveIcon className="size-2.5" />
            </button>
          </li>
        );
      })}
      {long && (
        <li>
          <button
            type="button"
            data-testid="scope-chips-more"
            aria-expanded={expanded}
            aria-label={expanded ? undefined : t("scope.chip.moreLabel", { count: more })}
            // Keep the cursor where it is.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => setExpanded(!expanded)}
            className="scope-chip scope-chip-more"
          >
            {expanded ? t("scope.chip.fewer") : t("scope.chip.more", { count: more })}
          </button>
        </li>
      )}
    </ul>
  );
}
