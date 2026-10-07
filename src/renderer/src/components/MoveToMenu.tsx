import { type ReactNode, useMemo } from "react";
import type { Document } from "../../../core/api";
import { buildFolderTree, flattenFolderTree } from "../folders";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { useFolderTree } from "./FolderTree";
import { CheckLineIcon, FolderLineIcon, MoveToLineIcon } from "./lineIcons";
import { INDENT_PX } from "./sidebarRows";
import { menuClass, menuItemClass, menuTitleClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * A Document's "Move to…" button and its menu: "Unfiled", then every Folder,
 * indented as in the tree, with the Document's current place checked. The
 * menu is a popover, so it sits above everything, and a click outside or Esc
 * closes it. The Folder it goes to unfolds, so it stays in sight.
 */
export function MoveToMenu({ item, buttonClassName }: { item: Document; buttonClassName: string }) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const moveDocument = useAppStore((state) => state.moveDocument);
  const reveal = useFolderTree((state) => state.reveal);
  const options = useMemo(() => flattenFolderTree(buildFolderTree(folders)), [folders]);
  const menu = usePopoverMenu();

  const choose = (folderId: string | null) => {
    menu.close();
    if (folderId === item.folderId) return;
    reveal(folderId);
    void moveDocument(item.id, folderId);
  };

  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="move-document"
        aria-label={t("folders.moveTo", { name: item.name })}
        title={t("folders.moveTo", { name: item.name })}
        className={buttonClassName}
      >
        <MoveToLineIcon className="size-[15px]" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("folders.moveTo.title")}
        data-testid="move-to-menu"
        className={menuClass}
      >
        <p className={menuTitleClass}>{t("folders.moveTo.title")}</p>
        <MenuItem
          checked={item.folderId === null}
          depth={0}
          label={t("folders.unfiled")}
          onSelect={() => choose(null)}
        />
        {options.map(({ folder, depth }) => (
          <MenuItem
            key={folder.id}
            checked={item.folderId === folder.id}
            depth={depth}
            label={folder.name}
            icon={<FolderLineIcon className="size-4 shrink-0 text-ink-meta" />}
            onSelect={() => choose(folder.id)}
          />
        ))}
      </div>
    </>
  );
}

function MenuItem(props: {
  checked: boolean;
  depth: number;
  label: string;
  icon?: ReactNode;
  onSelect(): void;
}) {
  const { checked, depth, label, icon, onSelect } = props;
  return (
    <button
      type="button"
      role="menuitemradio"
      aria-checked={checked}
      onClick={onSelect}
      className={menuItemClass}
      style={{ paddingLeft: 8 + depth * INDENT_PX }}
    >
      <CheckLineIcon className={`size-3.5 shrink-0 ${checked ? "text-ink" : "invisible"}`} />
      {icon}
      <span className="truncate">{label}</span>
    </button>
  );
}
