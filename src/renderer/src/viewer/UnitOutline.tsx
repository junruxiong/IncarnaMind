import { useT } from "../i18n";
import { OutlineToggleIcon } from "./icons";

/** One entry of a deck's or a document's outline: a slide, or a heading. */
export interface OutlineEntry {
  /** What it scrolls to: a slide number, or a heading's index. */
  key: number;
  title: string;
  /** From 0: how far it is indented. */
  depth: number;
  /** Shown in front of the title, e.g. a slide's number. */
  prefix?: string;
}

/** The outline's indent per level, and a top-level entry's padding, in CSS pixels (as a PDF's). */
const INDENT = 16;
const INSET = 8;

/**
 * The outline beside a deck's slides or a document's sections, styled as a
 * PDF's: rows like the sidebar's, the entry in view on a sheet.
 */
export function UnitOutline({
  entries,
  current,
  onGo,
}: {
  entries: readonly OutlineEntry[];
  current: number | null;
  onGo(key: number): void;
}) {
  const t = useT();
  return (
    <nav
      data-testid="viewer-outline"
      aria-label={t("viewer.outline.label")}
      className="pdf-outline"
    >
      <ul>
        {entries.map((entry) => {
          const isCurrent = entry.key === current;
          const title = entry.title || t("viewer.outline.untitled");
          return (
            <li key={entry.key}>
              <div
                className={`pdf-outline-row ${entry.depth > 0 ? "pdf-outline-row--nested" : ""} ${
                  isCurrent ? "pdf-outline-row--current" : ""
                }`}
              >
                <button
                  type="button"
                  data-testid="viewer-outline-item"
                  data-key={entry.key}
                  aria-current={isCurrent ? "location" : undefined}
                  title={entry.prefix ? `${entry.prefix} ${title}` : title}
                  onClick={() => onGo(entry.key)}
                  className="pdf-outline-title"
                  style={{ paddingLeft: INSET + entry.depth * INDENT }}
                >
                  {entry.prefix && <span className="viewer-outline-prefix">{entry.prefix}</span>}
                  {title}
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

/** Shows and hides the outline. */
export function OutlineButton({ open, onToggle }: { open: boolean; onToggle(): void }) {
  const t = useT();
  const label = t(open ? "viewer.outline.hide" : "viewer.outline.show");
  return (
    <button
      type="button"
      data-testid="viewer-outline-button"
      aria-label={label}
      aria-pressed={open}
      title={label}
      onClick={onToggle}
      className="viewer-icon-button"
    >
      <OutlineToggleIcon className="size-4" />
    </button>
  );
}
