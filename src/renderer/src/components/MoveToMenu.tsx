import { type KeyboardEvent, type ReactNode, useId, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import { buildFolderTree, flattenFolderTree } from "../folders";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { CheckIcon, FolderIcon, MoveToFolderIcon } from "./icons";

const GAP_PX = 4;
const EDGE_PX = 8;

/**
 * A Document's "Move to…" button and its menu: "Unfiled", then every Folder,
 * indented as in the tree, with the Document's current place checked. The
 * menu is a popover, so it sits above everything, and a click outside or Esc
 * closes it.
 */
export function MoveToMenu({ item, buttonClassName }: { item: Document; buttonClassName: string }) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const moveDocument = useAppStore((state) => state.moveDocument);
  const options = useMemo(() => flattenFolderTree(buildFolderTree(folders)), [folders]);
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  const items = () =>
    Array.from(menu.current?.querySelectorAll<HTMLElement>('[role="menuitemradio"]') ?? []);

  /** Below the button, or above it if the window is too short. */
  const place = () => {
    const anchor = button.current?.getBoundingClientRect();
    const element = menu.current;
    if (!anchor || !element) return;
    element.style.left = `${Math.max(EDGE_PX, anchor.left)}px`;
    const below = anchor.bottom + GAP_PX;
    const height = element.offsetHeight;
    element.style.top =
      below + height > window.innerHeight - EDGE_PX
        ? `${Math.max(EDGE_PX, anchor.top - GAP_PX - height)}px`
        : `${below}px`;
  };

  const choose = (folderId: string | null) => {
    menu.current?.hidePopover();
    if (folderId !== item.folderId) void moveDocument(item.id, folderId);
  };

  const moveFocus = (event: KeyboardEvent) => {
    const all = items();
    const index = all.indexOf(document.activeElement as HTMLElement);
    const next =
      event.key === "ArrowDown"
        ? all[(index + 1) % all.length]
        : event.key === "ArrowUp"
          ? all[(index - 1 + all.length) % all.length]
          : event.key === "Home"
            ? all[0]
            : event.key === "End"
              ? all.at(-1)
              : undefined;
    if (!next) return;
    event.preventDefault();
    next.focus();
  };

  return (
    <>
      <button
        ref={button}
        type="button"
        data-testid="move-document"
        popoverTarget={id}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t("folders.moveTo", { name: item.name })}
        title={t("folders.moveTo", { name: item.name })}
        className={buttonClassName}
      >
        <MoveToFolderIcon className="size-[14px]" />
      </button>
      <div
        ref={menu}
        id={id}
        popover="auto"
        role="menu"
        aria-label={t("folders.moveTo.title")}
        data-testid="move-to-menu"
        // Placed before it shows, then again once its height is known.
        onBeforeToggle={(event) => {
          if (event.newState === "open") place();
        }}
        onToggle={(event) => {
          const isOpen = event.newState === "open";
          setOpen(isOpen);
          if (!isOpen) return;
          place();
          const all = items();
          (all.find((each) => each.getAttribute("aria-checked") === "true") ?? all[0])?.focus();
        }}
        onKeyDown={moveFocus}
        className="inset-auto m-0 max-h-[60vh] w-max max-w-72 min-w-44 overflow-y-auto rounded-[9px] border-0 bg-white p-1 text-sm text-gray-700 shadow-custom-focus"
      >
        <p className="px-2 pt-1 pb-[2px] text-[11px] font-medium tracking-wide text-gray-400 uppercase">
          {t("folders.moveTo.title")}
        </p>
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
            icon={<FolderIcon className="size-4 shrink-0" />}
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
      className="flex w-full items-center gap-[6px] rounded-[6px] py-[5px] pr-2 text-left outline-none hover:bg-gray-100 focus-visible:bg-gray-100"
      style={{ paddingLeft: 8 + depth * 12 }}
    >
      <CheckIcon className={`size-[14px] shrink-0 ${checked ? "text-gray-700" : "invisible"}`} />
      {icon}
      <span className="truncate">{label}</span>
    </button>
  );
}
