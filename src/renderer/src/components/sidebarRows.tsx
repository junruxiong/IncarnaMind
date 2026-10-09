import type { MouseEvent, ReactNode } from "react";
import { ChevronDownLineIcon, ChevronRightLineIcon } from "./lineIcons";

/*
 * The sidebar's rows (DESIGN.md, Sidebar): 28px, inset 8px by the sidebar's
 * padding and padded 8px, so every icon sits at x 16 and every text at x 40.
 * A row one level deeper (a Folder's child) moves both by `INDENT_PX`.
 */

/** One level deeper: a child's icon goes under its parent's text. */
export const INDENT_PX = 24;

/** The left padding of a row's button at a depth: 8px, plus one step per level. */
export const rowPadding = (depth: number) => ({ paddingLeft: 8 + depth * INDENT_PX });

/** The text of a row: the app's own rows, a Document's or Folder's name, or one still being processed. */
type RowTone = "app" | "item" | "muted";

const rowTones: Record<RowTone, string> = {
  app: "text-ink-secondary",
  item: "text-ink-strong",
  muted: "text-ink-meta",
};

/**
 * A row's container. Pointed at it's washed; selected it's on the sheet with
 * an inset rule and weight 600. Its actions (`rowActionsClass`) sit on its
 * right end and take its background. Names of Documents and Folders are a
 * shade darker than the app's own rows. A row's words can't be selected, so
 * a right-click (see `openRowMenu`) never selects them.
 */
export const rowClass = (selected: boolean, tone: RowTone = "app") =>
  `group relative flex h-7 shrink-0 items-center rounded-md text-ui select-none ${
    selected
      ? "bg-sheet font-semibold text-ink shadow-[inset_0_0_0_1px_var(--color-rule)]"
      : `${rowTones[tone]} hover:bg-hover has-[:focus-visible]:bg-hover`
  }`;

/**
 * A right-click on a row opens its "More" menu, as its ⋯ button (found by
 * `menuTestId`) does. A right-click inside the open menu leaves it be.
 */
export function openRowMenu(event: MouseEvent<HTMLElement>, menuTestId: string): void {
  if ((event.target as Element).closest('[role="menu"]')) return;
  const button = event.currentTarget.querySelector<HTMLButtonElement>(
    `[data-testid="${menuTestId}"]`,
  );
  if (!button) return;
  event.preventDefault();
  if (button.getAttribute("aria-expanded") === "true") return;
  // With a button still held, open once it's let go: letting go outside the menu would close it.
  if (event.buttons === 0) button.click();
  else {
    const open = () => {
      if (button.isConnected) button.click();
    };
    window.addEventListener("pointerup", () => setTimeout(open), { once: true });
  }
}

/** A name being typed in a row. */
export const rowInputClass =
  "h-6 w-full min-w-0 rounded-sm border border-accent bg-sheet px-1.5 text-ui text-ink outline-1 outline-offset-0 outline-accent select-text";

/** The button that fills a row: icon, 8px gap, text, and anything at its end. */
export const rowButtonClass =
  "flex h-full w-full min-w-0 items-center gap-2 rounded-md pr-2 pl-2 text-left focus-visible:-outline-offset-2";

/** A row's 16px icon: muted unless the row is selected. */
export const rowIconClass = (selected: boolean) =>
  `size-4 shrink-0 ${selected ? "text-ink" : "text-ink-meta"}`;

/**
 * A row's actions, over its right end: shown while it is pointed at, is
 * reached by keyboard, or has a menu open (a mouse click alone doesn't keep
 * them). They take the row's background, so its end (a status, a chevron)
 * doesn't show through, and sit 1px inside its edge, so a selected row's
 * outline runs unbroken behind them.
 */
export const rowActionsClass =
  "absolute inset-y-px right-px flex items-center gap-px rounded-r-[5px] bg-inherit pr-[3px] pl-1 opacity-0 group-hover:opacity-100 group-has-[:focus-visible]:opacity-100 has-[[aria-expanded=true]]:opacity-100";

/** A 24px icon button among a row's actions. */
export const rowActionButtonClass =
  "inline-flex size-6 shrink-0 items-center justify-center rounded-md text-ink-meta hover:bg-rule hover:text-ink focus-visible:outline-offset-0 aria-expanded:bg-rule aria-expanded:text-ink";

/**
 * A section's label ("Minds", "Documents"): 12/16 semibold, in the icon
 * column. With `onToggle`, it folds the section away and back.
 */
export function SectionLabel({
  id,
  children,
  folded,
  onToggle,
  toggleLabel,
}: {
  id?: string;
  children: ReactNode;
  folded?: boolean;
  onToggle?(): void;
  /** What the toggle does, for its tooltip and screen readers. */
  toggleLabel?: string;
}) {
  const label = "mt-3 flex h-7 shrink-0 items-end px-2 pb-1 text-label font-semibold text-ink-meta";
  if (!onToggle) {
    return (
      <h2 id={id} className={label}>
        {children}
      </h2>
    );
  }
  return (
    <h2 id={id} className={label}>
      <button
        type="button"
        aria-expanded={!folded}
        title={toggleLabel}
        onClick={onToggle}
        className="flex items-center gap-1 rounded-sm hover:text-ink-secondary"
      >
        {children}
        {folded ? (
          <ChevronRightLineIcon className="size-3" />
        ) : (
          <ChevronDownLineIcon className="size-3" />
        )}
      </button>
    </h2>
  );
}
