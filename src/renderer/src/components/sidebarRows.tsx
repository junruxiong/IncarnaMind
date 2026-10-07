import type { ReactNode } from "react";

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
 * shade darker than the app's own rows.
 */
export const rowClass = (selected: boolean, tone: RowTone = "app") =>
  `group relative flex h-7 shrink-0 items-center rounded-md text-ui ${
    selected
      ? "bg-sheet font-semibold text-ink shadow-[inset_0_0_0_1px_var(--color-rule)]"
      : `${rowTones[tone]} hover:bg-hover has-[:focus-visible]:bg-hover`
  }`;

/** A name being typed in a row. */
export const rowInputClass =
  "h-6 w-full min-w-0 rounded-sm border border-accent bg-sheet px-1.5 text-ui text-ink outline-1 outline-accent";

/** The button that fills a row: icon, 8px gap, text, and anything at its end. */
export const rowButtonClass =
  "flex h-full w-full min-w-0 items-center gap-2 rounded-md pr-2 pl-2 text-left outline-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent";

/** A row's 16px icon: muted unless the row is selected. */
export const rowIconClass = (selected: boolean) =>
  `size-4 shrink-0 ${selected ? "text-ink" : "text-ink-meta"}`;

/**
 * A row's actions, over its right end: shown while it is pointed at, is
 * reached by keyboard, or has a menu open (a mouse click alone doesn't keep
 * them). They take the row's background, so its end (a status, a chevron)
 * doesn't show through.
 */
export const rowActionsClass =
  "absolute inset-y-0 right-0 flex items-center gap-px rounded-r-md bg-inherit pr-1 pl-1 opacity-0 group-hover:opacity-100 group-has-[:focus-visible]:opacity-100 has-[[aria-expanded=true]]:opacity-100";

/** A 24px icon button among a row's actions. */
export const rowActionButtonClass =
  "inline-flex size-6 shrink-0 items-center justify-center rounded-md text-ink-meta outline-none hover:bg-rule hover:text-ink focus-visible:outline-2 focus-visible:outline-accent aria-expanded:bg-rule aria-expanded:text-ink";

/** A section's label ("Minds", "Documents"): 12/16 semibold, in the icon column. */
export function SectionLabel({ id, children }: { id?: string; children: ReactNode }) {
  return (
    <h2
      id={id}
      className="mt-3 flex h-7 shrink-0 items-end px-2 pb-1 text-label font-semibold text-ink-meta"
    >
      {children}
    </h2>
  );
}
