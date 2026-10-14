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
 * a right-click never selects them. `--row-bg` is the sheet while selected,
 * for a mark that rings itself in it (a Document's Tag dots, which step aside
 * while the row is washed).
 */
export const rowClass = (selected: boolean, tone: RowTone = "app") =>
  `group relative flex h-7 shrink-0 items-center rounded-md text-ui select-none ${
    selected
      ? "bg-sheet font-semibold text-ink shadow-[inset_0_0_0_1px_var(--color-rule)] [--row-bg:var(--color-sheet)]"
      : `${rowTones[tone]} hover:bg-hover has-[:focus-visible]:bg-hover`
  }`;

/** A name being typed in a row. */
export const rowInputClass =
  "h-6 w-full min-w-0 rounded-sm border border-accent bg-sheet px-1.5 text-ui text-ink outline-1 outline-offset-0 outline-accent select-text";

/** The button that fills a row: icon, 8px gap, text, and anything at its end. */
export const rowButtonClass =
  "flex h-full w-full min-w-0 items-center gap-2 rounded-md pr-2 pl-2 text-left focus-visible:-outline-offset-2";

/**
 * Added to a row's button, so that while the row's actions show its content
 * ends before them: the name gives way with its "…" instead of running under
 * them. The room is the actions' own (`rowActionsClass`): 24px a button, 1px
 * between, 4px and 3px of padding, and the 1px they sit in from the edge.
 */
export const clearsActions = {
  menu: "actions-shown:pr-[32px]",
  tagsAndMenu: "actions-shown:pr-[57px]",
  /** A Tag's row, whose menu sits 29px in from the edge, before its chevron. */
  menuBeforeChevron: "actions-shown:pr-[36px]",
} as const;

/** A row's 16px icon: muted unless the row is selected. */
export const rowIconClass = (selected: boolean) =>
  `size-4 shrink-0 ${selected ? "text-ink" : "text-ink-meta"}`;

/**
 * A row's actions, over its right end: shown while it is pointed at, is
 * reached by keyboard, or has a menu open (a mouse click alone doesn't keep
 * them), which is what the `actions-shown:` variant (styles.css) says; the
 * `row-actions` class is how the variant finds an open menu. They
 * take the row's background and sit 1px inside its edge, so a selected row's
 * outline runs unbroken behind them. The row's button keeps its content clear
 * of them (`clearsActions`).
 */
export const rowActionsClass =
  "row-actions absolute inset-y-px right-px flex items-center gap-px rounded-r-[5px] bg-inherit pr-[3px] pl-1 opacity-0 actions-shown:opacity-100";

/** A 24px icon button among a row's actions. */
export const rowActionButtonClass =
  "inline-flex size-6 shrink-0 items-center justify-center rounded-md text-ink-meta hover:bg-rule hover:text-ink focus-visible:outline-offset-0 aria-expanded:bg-rule aria-expanded:text-ink";
