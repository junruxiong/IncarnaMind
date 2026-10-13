import { type KeyboardEvent, type ReactNode, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { Document, LibraryGroup } from "../../../../core/api";
import { core } from "../../core";
import { useT } from "../../i18n";
import { useLibraryFolds, useRevealed } from "../../libraryFolds";
import { FIRST_ROWS, NOT_IN_A_FOLDER, type LibrarySection as Section } from "../../libraryPage";
import { useRequests } from "../../requests";
import { useAppStore } from "../../store";
import { TREE_ROW } from "../../treeKeys";
import { useRename } from "../../useRename";
import { InlineName } from "../InlineName";
import { FolderLineIcon } from "../lineIcons";
import { ObjectMenu, useObjectMenu } from "../ObjectMenu";
import { rowActionButtonClass } from "../sidebarRows";
import { FolderForm } from "./FolderForm";
import { LibraryRow } from "./LibraryRow";
import { SectionHead, sectionNameClass } from "./SectionHead";

/** A link-like button in a section's head or foot: quiet until pointed at. */
export const quietButtonClass =
  "shrink-0 rounded-sm text-[13px] font-semibold text-ink-secondary hover:text-ink hover:underline disabled:opacity-50 disabled:hover:no-underline";

/**
 * One section of the Library page: a Folder's Documents, or those in no
 * Folder, under a head that folds it and remembers it. A long section lists
 * its first Documents, then "Show all N"; while the page is narrowed it lists
 * every match.
 */
export function LibrarySection({
  section,
  narrowing,
  suggestedOf,
  notSortedOf,
}: {
  section: Section;
  narrowing: boolean;
  /** The Suggested Folder a Document waits in, by its id, if it waits in one. */
  suggestedOf(documentId: string): string | null;
  /** Why Organize couldn't sort a Document, by its id, if it couldn't. */
  notSortedOf(documentId: string): string | null;
}) {
  const t = useT();
  const open = useLibraryFolds((state) => !state.folded.has(section.key));
  const showingAll = useLibraryFolds((state) => state.showingAll.has(section.key));
  const element = useRevealed<HTMLDivElement>(section.key);
  const { documents } = section;
  const listed = narrowing || showingAll ? documents : documents.slice(0, FIRST_ROWS);
  return (
    // biome-ignore lint/a11y/useSemanticElements: a group of rows in the page's grid, under its head.
    <div
      ref={element}
      role="rowgroup"
      data-testid="library-section"
      data-section={section.key}
      data-count={documents.length}
      className="mt-[18px] first:mt-2"
    >
      {section.folder ? (
        <FolderHead folder={section.folder} count={documents.length} open={open} />
      ) : (
        <LooseHead documents={documents} open={open} />
      )}
      {open &&
        listed.map((doc) => (
          <LibraryRow
            key={doc.id}
            doc={doc}
            suggestedIn={suggestedOf(doc.id)}
            notSorted={notSortedOf(doc.id)}
          />
        ))}
      {open && listed.length < documents.length && (
        <div className="flex min-h-8 items-center border-b border-rule pl-14">
          <button
            type="button"
            data-testid="library-show-all"
            onClick={() => useLibraryFolds.getState().showAll(section.key)}
            className={quietButtonClass}
          >
            {t("library.showAll", { count: documents.length })}
          </button>
        </div>
      )}
    </div>
  );
}

/** ← folds a section and → opens it, as in the sidebar's tree. */
function foldKeys(key: string, open: boolean) {
  return (event: KeyboardEvent) => {
    if ((event.key === "ArrowRight" && !open) || (event.key === "ArrowLeft" && open)) {
      event.preventDefault();
      useLibraryFolds.getState().toggle(key);
    }
  };
}

/**
 * A Folder's head: its name, which folds the section and renames in place
 * (double-click or F2), how many it holds, its description and, pointed at,
 * its ⋯. Edit description… opens the Folder's form under it.
 */
function FolderHead({
  folder,
  count,
  open,
}: {
  folder: LibraryGroup;
  count: number;
  open: boolean;
}) {
  const t = useT();
  const toggle = () => useLibraryFolds.getState().toggle(folder.id);
  // The first click of a double-click folds or unfolds it: back as it was before renaming.
  const rename = useRename<HTMLButtonElement>(`library-folder:${folder.id}`, toggle);
  const objectMenu = useObjectMenu({ kind: "folder", id: folder.id });
  const [editing, setEditing] = useState(false);
  const editFolder = useRequests((state) => state.editFolder);
  useEffect(() => {
    if (editFolder !== folder.id) return;
    setEditing(true);
    useLibraryFolds.getState().setOpen(folder.id, true);
    useRequests.getState().clear("editFolder");
  }, [editFolder, folder.id]);
  const arrows = foldKeys(folder.id, open);
  let name: ReactNode;
  if (rename.renaming) {
    name = (
      <InlineName
        value={folder.name}
        label={t("folder.renameLabel", { name: folder.name })}
        testId="library-section-rename"
        className="h-7 w-full max-w-80 min-w-0 rounded-sm border border-accent bg-sheet px-1.5 text-[15px] font-semibold text-ink outline-1 outline-offset-0 outline-accent select-text"
        {...rename.field((next) => useAppStore.getState().renameFolder(folder.id, next))}
      />
    );
  } else {
    name = (
      <button
        {...rename.nameProps}
        type="button"
        {...TREE_ROW}
        data-testid="library-section-name"
        aria-expanded={open}
        title={folder.description ? `${folder.name}\n${folder.description}` : folder.name}
        onClick={(event) => {
          // The second click of a double-click renames instead.
          if (event.detail <= 1) toggle();
        }}
        onKeyDown={(event) => {
          rename.nameProps.onKeyDown(event);
          arrows(event);
        }}
        className={sectionNameClass}
      >
        {folder.name}
      </button>
    );
  }
  return (
    <>
      <SectionHead
        label={folder.name}
        open={open}
        onToggle={toggle}
        icon={<FolderLineIcon className="size-4" />}
        name={name}
        count={count}
        description={folder.description}
        testId="library-section-head"
        rowProps={rename.renaming ? {} : objectMenu.rowProps}
        end={
          !rename.renaming && (
            <ObjectMenu
              target={{ kind: "folder", id: folder.id }}
              label={t("folder.more", { name: folder.name })}
              here={{ "folder.rename": rename.start }}
              className={`${rowActionButtonClass} opacity-0 group-hover/head:opacity-100 group-has-[:focus-visible]/head:opacity-100 aria-expanded:opacity-100`}
            />
          )
        }
      />
      {editing && (
        <FolderForm group={folder} close={() => setEditing(false)} className="mt-3 mb-4 ml-14" />
      )}
    </>
  );
}

/**
 * Not in a Folder's head: how many are in no Folder and, at its end, Organize,
 * which suggests a Folder for each, or how it is going, or what went wrong.
 */
function LooseHead({ documents, open }: { documents: readonly Document[]; open: boolean }) {
  const t = useT();
  const label = t("library.unsorted");
  const toggle = () => useLibraryFolds.getState().toggle(NOT_IN_A_FOLDER);
  return (
    <SectionHead
      label={label}
      open={open}
      onToggle={toggle}
      icon={<FolderLineIcon className="size-4 opacity-60" />}
      name={
        <button
          type="button"
          {...TREE_ROW}
          data-testid="library-section-name"
          aria-expanded={open}
          onClick={toggle}
          onKeyDown={foldKeys(NOT_IN_A_FOLDER, open)}
          className={sectionNameClass}
        >
          {label}
        </button>
      }
      count={documents.length}
      testId="library-section-head"
      end={<Organize documents={documents} />}
    />
  );
}

/**
 * Organize, for the Documents in no Folder (its model is chosen in Settings ›
 * Models until #236 chooses it here), or how it is going, or what it couldn't
 * do, with one action.
 */
function Organize({ documents }: { documents: readonly Document[] }) {
  const t = useT();
  const { hasFolders, hasModel, pending, failed } = useAppStore(
    useShallow((state) => {
      let pending = 0;
      let failed = 0;
      for (const item of state.library?.assignments ?? []) {
        if (item.status === "pending" || item.status === "classifying") pending++;
        else if (item.status === "failed") failed++;
      }
      return {
        hasFolders: (state.library?.groups.length ?? 0) > 0,
        hasModel: Boolean(state.library?.settings.classifier),
        pending,
        failed,
      };
    }),
  );
  const run = (ids: readonly string[]) => {
    const app = useAppStore.getState();
    if (!hasModel) {
      app.openSettings("models");
      return;
    }
    core
      .classifyDocuments([...ids])
      .then(() => app.refreshLibrary())
      .catch(app.reportError);
  };
  if (pending > 0) {
    return (
      <span
        role="status"
        data-testid="library-organizing"
        className="shrink-0 text-[13px] text-ink-meta"
      >
        {t("library.organizing", { count: pending })}
      </span>
    );
  }
  if (failed > 0) {
    const retry = () =>
      run(
        (useAppStore.getState().library?.assignments ?? [])
          .filter((item) => item.status === "failed")
          .map((item) => item.documentId),
      );
    return (
      <span
        role="alert"
        data-testid="library-organize-failed"
        className="min-w-0 truncate text-[13px] text-ink-meta"
      >
        {t(failed === 1 ? "library.organize.failed.one" : "library.organize.failed", {
          count: failed,
        })}
        <span aria-hidden="true"> · </span>
        <button type="button" onClick={retry} className={quietButtonClass}>
          {t("library.organize.retry")}
        </button>
      </span>
    );
  }
  if (!hasFolders || documents.length === 0) return null;
  return (
    <button
      type="button"
      data-testid="library-organize"
      onClick={() => run(documents.map((doc) => doc.id))}
      className={quietButtonClass}
    >
      {t("library.classify")}
    </button>
  );
}
