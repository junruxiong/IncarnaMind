import type { ReactNode } from "react";
import type { ComparedBlock, ComparedPart, MindComparison } from "../../../../core/api";
import { citationSource } from "../../../../shared/citations";
import { useT } from "../../i18n";

/** Each side's Citations and footnotes numbered together, in order, as the Mind numbers them. */
function numbersOf(comparison: MindComparison) {
  const count = { left: 0, right: 0 };
  return comparison.rows.map((row) => ({
    left: (row.left?.parts ?? []).map((part) => (part.kind === "text" ? 0 : ++count.left)),
    right: (row.right?.parts ?? []).map((part) => (part.kind === "text" ? 0 : ++count.right)),
  }));
}

/**
 * Two versions side by side, a row per Block: removed words struck on the
 * left, added words underlined on the right, both in grey, as nothing here
 * can be changed. Screen readers hear "Removed:" and "Added:" before them.
 */
export function ComparedRows({
  comparison,
  firstChange,
}: {
  comparison: MindComparison;
  firstChange: (row: HTMLElement | null) => void;
}) {
  const numbers = numbersOf(comparison);
  const first = comparison.rows.findIndex((row) => row.changed);
  return (
    <div className="history-rows">
      {comparison.rows.map((row, index) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: rows are a fresh comparison each time, in order.
          key={index}
          data-testid="history-row"
          data-changed={row.changed || undefined}
          ref={index === first ? firstChange : undefined}
          className="history-row"
        >
          <div data-side="left" className="history-cell">
            {row.left && <BlockView block={row.left} numbers={numbers[index]?.left ?? []} />}
          </div>
          <div data-side="right" className="history-cell">
            {row.right && <BlockView block={row.right} numbers={numbers[index]?.right ?? []} />}
          </div>
        </div>
      ))}
    </div>
  );
}

const HEADINGS = ["h2", "h3", "h4", "h5", "h6"] as const;

function BlockView({ block, numbers }: { block: ComparedBlock; numbers: readonly number[] }) {
  const t = useT();
  const parts = block.parts.map((part, index) => (
    // biome-ignore lint/suspicious/noArrayIndexKey: a Block's parts in order, read once.
    <Part key={index} part={part} number={numbers[index] ?? 0} />
  ));
  switch (block.kind) {
    case "heading": {
      // Under History's own title, the dialog's one h1.
      const Heading = HEADINGS[Math.min(block.level, HEADINGS.length) - 1] ?? "h2";
      return (
        <Heading data-level={block.level} className="history-block history-heading">
          {parts}
        </Heading>
      );
    }
    case "question":
      return <p className="history-block history-question">{parts}</p>;
    case "item":
      return (
        <p
          className="history-block history-item"
          style={{ paddingLeft: `${(block.level - 1) * 24 + 24}px` }}
        >
          {block.marker && (
            <span aria-hidden="true" className="history-marker">
              {block.marker}
            </span>
          )}
          {parts}
        </p>
      );
    case "quote":
      return <blockquote className="history-block history-quote">{parts}</blockquote>;
    case "code":
    case "math":
      return <pre className="history-block history-code">{parts}</pre>;
    case "rule":
      return <hr className="history-block history-rule" />;
    case "table":
      return <p className="history-block history-lines">{parts}</p>;
    case "chart":
    case "image":
      return (
        <p className="history-block history-lines">
          <span className="history-label">
            {t(block.kind === "chart" ? "history.chart" : "history.image")}
          </span>{" "}
          {parts}
        </p>
      );
    default:
      return <p className="history-block">{parts}</p>;
  }
}

function Part({ part, number }: { part: ComparedPart; number: number }) {
  const t = useT();
  let content: ReactNode;
  if (part.kind === "text") content = part.text;
  else {
    const source =
      part.kind === "citation"
        ? citationSource({ documentName: part.work.documentName, location: part.work.location }, t)
            .text
        : part.text;
    content = (
      <span title={source} className="history-citation">
        <span className="sr-only">{source}</span>
        <span aria-hidden="true">{number}</span>
      </span>
    );
  }
  if (part.change === "removed") {
    return (
      <del>
        <span className="sr-only">{t("history.removed")}</span>
        {content}
      </del>
    );
  }
  if (part.change === "added") {
    return (
      <ins>
        <span className="sr-only">{t("history.added")}</span>
        {content}
      </ins>
    );
  }
  return <>{content}</>;
}
