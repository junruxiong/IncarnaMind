import type { HTMLAttributes, ReactNode } from "react";
import { useT } from "../../i18n";
import { ChevronDownLineIcon } from "../lineIcons";

/**
 * The head of a section of the Library page, as a toggle in a Notion page:
 * the chevron that folds it, its icon, its name, how many it holds, a line
 * about it and, at its end, what it offers. It rules the section off from
 * the next in the strong rule.
 */
export function SectionHead({
  label,
  open,
  onToggle,
  icon,
  name,
  count,
  description,
  end,
  testId,
  rowProps,
}: {
  /** Its name in words, for the chevron's label. */
  label: string;
  open: boolean;
  onToggle(): void;
  icon: ReactNode;
  /** The name as shown: a button that folds it too, or a field while it is renamed. */
  name: ReactNode;
  count: number;
  description?: string | undefined;
  end?: ReactNode;
  testId: string;
  rowProps?: HTMLAttributes<HTMLDivElement>;
}) {
  const t = useT();
  return (
    // biome-ignore lint/a11y/useSemanticElements: the head of a group of rows in a grid of sections, not a table.
    // biome-ignore lint/a11y/useFocusableInteractive: its name's button takes the focus.
    <div
      role="row"
      data-testid={testId}
      data-open={open ? "true" : "false"}
      {...rowProps}
      className="group/head flex h-9 items-center gap-2 rounded-t-md border-b border-rule-strong pr-1.5 pl-1 select-none hover:bg-frame"
    >
      <button
        type="button"
        tabIndex={-1}
        data-testid="library-section-toggle"
        aria-expanded={open}
        aria-label={t(open ? "library.collapse" : "library.expand", { name: label })}
        onClick={onToggle}
        className="inline-flex size-5 shrink-0 items-center justify-center rounded-sm text-ink-meta hover:bg-hover hover:text-ink"
      >
        <ChevronDownLineIcon className={`size-3.5 ${open ? "" : "-rotate-90"}`} />
      </button>
      <span className="flex shrink-0 text-ink-meta">{icon}</span>
      {name}
      <span
        data-testid="library-section-count"
        className="shrink-0 text-[12px] tabular-nums text-ink-meta"
      >
        {count}
      </span>
      <span
        title={description || undefined}
        className="min-w-0 flex-1 truncate text-[12px] text-ink-meta"
      >
        {description}
      </span>
      {end}
    </div>
  );
}

/** A section's name, the button that folds it: bold, cut with "…" when long. */
export const sectionNameClass =
  "min-w-0 shrink truncate rounded-sm text-left text-[15px] leading-5 font-semibold text-ink";
