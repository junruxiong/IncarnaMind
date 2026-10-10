import { type KeyboardEvent, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { CheckLineIcon, FolderLineIcon } from "./lineIcons";
import { menuClass, menuTitleClass } from "./ui";

const GAP_PX = 4;
const EDGE_PX = 8;

/**
 * "Move to…": a Folder picker that opens beside what is moved (`anchor`),
 * with a field to find a Folder by name. "Not in a Folder" first, then every
 * Folder by name; the one it is in now is ticked. Typing filters, the arrows
 * move, Enter moves it there, Esc closes. Shared by a Mind's menu and (#212)
 * the Documents, the Library's selection and ⌘⇧M.
 */
export function MoveToPicker({
  anchor,
  name,
  current,
  onChoose,
  onClose,
}: {
  /** What it opens beside; it takes the focus back once closed. Null: closed. */
  anchor: HTMLElement | null;
  /** What is moved: a Mind's title, or "3 Documents". */
  name: string;
  /** Where it is now (null: Not in a Folder), or undefined for things in different places. */
  current: string | null | undefined;
  onChoose(folderId: string | null): void;
  onClose(): void;
}) {
  const t = useT();
  const id = useId();
  const groups = useAppStore((state) => state.library?.groups) ?? [];
  const panel = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const open = anchor !== null;

  const wanted = query.trim().toLocaleLowerCase();
  const options: { id: string | null; name: string }[] = [
    { id: null, name: t("library.unsorted") },
    ...groups.map((group) => ({ id: group.id, name: group.name })),
  ].filter((option) => !wanted || option.name.toLocaleLowerCase().includes(wanted));
  const shown = Math.min(active, Math.max(0, options.length - 1));

  // Opens beside its anchor, below it or above if there is no room, and takes the focus.
  useLayoutEffect(() => {
    const element = panel.current;
    if (!open || !element || !anchor) return;
    setQuery("");
    setActive(0);
    element.showPopover();
    const box = anchor.getBoundingClientRect();
    const width = element.offsetWidth;
    const height = element.offsetHeight;
    element.style.left = `${Math.max(EDGE_PX, Math.min(box.left, window.innerWidth - width - EDGE_PX))}px`;
    const below = box.bottom + GAP_PX;
    element.style.top =
      below + height > window.innerHeight - EDGE_PX
        ? `${Math.max(EDGE_PX, box.top - GAP_PX - height)}px`
        : `${below}px`;
    field.current?.focus();
  }, [open, anchor]);

  // Closed from outside (its anchor went): hidden.
  useEffect(() => {
    if (!open && panel.current?.matches(":popover-open")) panel.current.hidePopover();
  }, [open]);

  const close = () => {
    if (panel.current?.matches(":popover-open")) panel.current.hidePopover();
    else onClose();
  };

  const choose = (folderId: string | null) => {
    close();
    onChoose(folderId);
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "Escape") {
      // Here, so the Document viewer's Esc doesn't close it too.
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (options.length === 0) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((shown + step + options.length) % options.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const option = options[shown];
      if (option) choose(option.id);
    }
  };

  // On the page's body: not inside a row that drags, where pressing in the field would start a drag.
  return createPortal(
    <div
      ref={panel}
      popover="auto"
      data-testid="move-to-picker"
      role="dialog"
      aria-label={t("move.title", { name })}
      onToggle={(event) => {
        if (event.newState === "closed") {
          onClose();
          anchor?.focus();
        }
      }}
      onKeyDown={onKeyDown}
      className={`${menuClass} w-64`}
    >
      {open && (
        <>
          <p className={`${menuTitleClass} truncate`} title={t("move.title", { name })}>
            {t("move.title", { name })}
          </p>
          <input
            ref={field}
            value={query}
            data-testid="move-to-search"
            role="combobox"
            aria-expanded="true"
            aria-controls={`${id}-list`}
            aria-activedescendant={options[shown] ? `${id}-${shown}` : undefined}
            aria-label={t("move.search")}
            placeholder={t("move.search")}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
            }}
            className="mx-1 mb-1 h-7 w-[calc(100%-8px)] rounded-sm border border-rule bg-sheet px-2 text-ui text-ink outline-none placeholder:text-ink-placeholder focus:border-accent"
          />
          <div id={`${id}-list`} role="listbox" aria-label={t("move.title", { name })}>
            {options.length === 0 ? (
              <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">{t("move.none")}</p>
            ) : (
              options.map((option, index) => {
                const here = option.id === current;
                return (
                  // biome-ignore lint/a11y/useKeyWithClickEvents: the field owns the keys (aria-activedescendant).
                  <div
                    key={option.id ?? "none"}
                    id={`${id}-${index}`}
                    role="option"
                    tabIndex={-1}
                    data-testid="move-to-option"
                    data-folder-id={option.id ?? ""}
                    aria-selected={index === shown}
                    aria-description={here ? t("move.here") : undefined}
                    data-current={here ? "true" : undefined}
                    // Choosing with the pointer keeps the focus in the field until it closes.
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => setActive(index)}
                    onClick={() => choose(option.id)}
                    className={`flex h-7 cursor-default items-center gap-2 rounded-md pr-2 pl-2 ${
                      index === shown ? "bg-hover" : ""
                    }`}
                  >
                    <FolderLineIcon
                      className={`size-4 shrink-0 text-ink-meta ${option.id === null ? "opacity-60" : ""}`}
                    />
                    <span className="min-w-0 flex-1 truncate" title={option.name}>
                      {option.name}
                    </span>
                    {here && (
                      <span title={t("move.here")} className="shrink-0 text-ink-meta">
                        <CheckLineIcon className="size-3.5" />
                      </span>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </>
      )}
    </div>,
    document.body,
  );
}
