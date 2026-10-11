import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { CheckLineIcon } from "./lineIcons";
import { menuClass, menuTitleClass } from "./ui";

const GAP_PX = 4;
const EDGE_PX = 8;

export interface PickerOption<Id> {
  id: Id;
  label: string;
  /** What else typing finds it by: other names, the other language's label. */
  keywords?: readonly string[];
  icon?: ReactNode;
  /** More attributes on its row, e.g. `data-folder-id`. */
  data?: Record<string, string>;
}

/**
 * A picker that opens beside `anchor`, with a field that filters its options as the User
 * types: the arrows move, Enter chooses, Esc closes, and the one chosen now is ticked. The
 * shared block behind "Move to…" (#212) and the citation style picker (#289).
 */
export function SearchPicker<Id>({
  anchor,
  title,
  searchLabel,
  noMatch,
  options,
  current,
  currentLabel,
  onChoose,
  onClose,
  testId,
  footer,
  className = "w-64",
}: {
  /** What it opens beside; it takes the focus back once closed. Null: closed. */
  anchor: HTMLElement | null;
  title: string;
  searchLabel: string;
  /** Said when nothing matches what was typed. */
  noMatch(query: string): string;
  options: readonly PickerOption<Id>[];
  /** The option chosen now, ticked; undefined for none. */
  current: Id | undefined;
  /** Says which option is the one chosen now, to a screen reader. */
  currentLabel: string;
  onChoose(id: Id): void;
  onClose(): void;
  /** Its test ids: `{testId}-picker`, `{testId}-search` and `{testId}-option`. */
  testId: string;
  /** A quiet line under the options. */
  footer?: ReactNode;
  className?: string;
}) {
  const id = useId();
  const panel = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const open = anchor !== null;

  const wanted = query.trim().toLocaleLowerCase();
  const shownOptions = options.filter(
    (option) =>
      !wanted ||
      [option.label, ...(option.keywords ?? [])].some((text) =>
        text.toLocaleLowerCase().includes(wanted),
      ),
  );
  const shown = Math.min(active, Math.max(0, shownOptions.length - 1));

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

  const choose = (option: PickerOption<Id>) => {
    close();
    onChoose(option.id);
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
      if (shownOptions.length === 0) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((shown + step + shownOptions.length) % shownOptions.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const option = shownOptions[shown];
      if (option) choose(option);
    }
  };

  // On the page's body: not inside a row that drags, where pressing in the field would start a
  // drag. Inside a modal dialog it opens from, though, or the dialog would make it inert.
  const container = anchor?.closest("dialog[open]") ?? document.body;
  return createPortal(
    <div
      ref={panel}
      popover="auto"
      data-testid={`${testId}-picker`}
      role="dialog"
      aria-label={title}
      onToggle={(event) => {
        if (event.newState === "closed") {
          onClose();
          anchor?.focus();
        }
      }}
      onKeyDown={onKeyDown}
      // A portal still hands React events to the row it is drawn from: not its drag and right-click.
      onDragStart={(event) => event.stopPropagation()}
      onContextMenu={(event) => event.stopPropagation()}
      className={`${menuClass} ${className}`}
    >
      {open && (
        <>
          <p className={`${menuTitleClass} truncate`} title={title}>
            {title}
          </p>
          <input
            ref={field}
            value={query}
            data-testid={`${testId}-search`}
            role="combobox"
            aria-expanded="true"
            aria-controls={`${id}-list`}
            aria-activedescendant={shownOptions[shown] ? `${id}-${shown}` : undefined}
            aria-label={searchLabel}
            placeholder={searchLabel}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
            }}
            className="mx-1 mb-1 h-7 w-[calc(100%-8px)] rounded-sm border border-rule bg-sheet px-2 text-ui text-ink outline-none placeholder:text-ink-placeholder focus:border-accent"
          />
          <div id={`${id}-list`} role="listbox" aria-label={title}>
            {shownOptions.length === 0 ? (
              <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">
                {noMatch(query.trim())}
              </p>
            ) : (
              shownOptions.map((option, index) => {
                const here = option.id === current;
                return (
                  // biome-ignore lint/a11y/useKeyWithClickEvents: the field owns the keys (aria-activedescendant).
                  <div
                    key={String(option.id ?? "none")}
                    id={`${id}-${index}`}
                    role="option"
                    tabIndex={-1}
                    data-testid={`${testId}-option`}
                    {...option.data}
                    aria-selected={index === shown}
                    aria-description={here ? currentLabel : undefined}
                    data-current={here ? "true" : undefined}
                    // Choosing with the pointer keeps the focus in the field until it closes.
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => setActive(index)}
                    onClick={() => choose(option)}
                    className={`flex h-7 cursor-default items-center gap-2 rounded-md pr-2 pl-2 ${
                      index === shown ? "bg-hover" : ""
                    }`}
                  >
                    {option.icon}
                    <span className="min-w-0 flex-1 truncate" title={option.label}>
                      {option.label}
                    </span>
                    {here && (
                      <span title={currentLabel} className="shrink-0 text-ink-meta">
                        <CheckLineIcon className="size-3.5" />
                      </span>
                    )}
                  </div>
                );
              })
            )}
          </div>
          {footer && (
            <p className="mx-2 mt-1 border-t border-rule pt-1.5 pb-1 text-[12px] leading-[18px] text-ink-meta">
              {footer}
            </p>
          )}
        </>
      )}
    </div>,
    container,
  );
}
