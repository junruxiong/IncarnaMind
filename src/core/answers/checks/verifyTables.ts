/**
 * A verified PDF's tables in its Mind (#1368): each one a table of cells, as the author laid it
 * out, so the figures being checked can be read; each figure a claim of its own, said with its
 * row's label and its column's heading.
 */
import * as Y from "yjs";
import type { DocumentText } from "../../api";
import type { PdfPagePart } from "../../documents/extract";
import { isFigure } from "../../documents/pdfTables";
import type { NodeJSON } from "../blocks";
import { paragraphsOf } from "./claims";
import { type SelectedSentence, textOf } from "./sentences";

/** The Blocks a PDF's text makes, each with its page: its paragraphs, and its tables where it has them. */
export function documentBlocks(
  pages: DocumentText["pages"],
  tables: ReadonlyMap<number, PdfPagePart[]>,
): { node: NodeJSON; page: number }[] {
  return pages.flatMap((unit) =>
    (tables.get(unit.page) ?? [{ kind: "text", text: unit.text }]).flatMap((part) =>
      part.kind === "table"
        ? [{ node: tableNode(part), page: unit.page }]
        : paragraphsOf([{ ...unit, text: part.text }]).map(({ text, heading }) => ({
            node: {
              type: heading ? "heading" : "paragraph",
              ...(heading && { attrs: { level: 2 } }),
              content: [{ type: "text", text }],
            },
            page: unit.page,
          })),
    ),
  );
}

const YEAR = /^(?:FY\s?)?(?:19|20)\d{2}$/i;

function tableNode({ rows, numeric }: Extract<PdfPagePart, { kind: "table" }>): NodeJSON {
  // A first row with no figure where its columns hold figures names them; a year names a column too.
  const header =
    numeric.some(Boolean) &&
    !rows[0]?.some((text, column) => numeric[column] && isFigure(text) && !YEAR.test(text.trim()));
  return {
    type: "table",
    content: rows.map((row, index) => ({
      type: "tableRow",
      content: row.map((text, column) => ({
        type: header && index === 0 ? "tableHeader" : "tableCell",
        attrs: { colspan: 1, rowspan: 1, ...(numeric[column] && { align: "right" }) },
        content: [{ type: "paragraph", ...(text !== "" && { content: [{ type: "text", text }] }) }],
      })),
    })),
  };
}

const cellOf = (element: Y.XmlElement) => {
  const cell = element.parent;
  const row = cell instanceof Y.XmlElement ? cell.parent : null;
  const table = row instanceof Y.XmlElement ? row.parent : null;
  return cell instanceof Y.XmlElement &&
    row instanceof Y.XmlElement &&
    table instanceof Y.XmlElement
    ? { cell, row, table }
    : null;
};

const cellText = (row: Y.XmlElement | undefined, column: number) => {
  const cell = row?.toArray()[column];
  const paragraph = cell instanceof Y.XmlElement ? cell.toArray()[0] : null;
  return paragraph instanceof Y.XmlElement ? textOf(paragraph).trim() : "";
};

/**
 * A table's row, as one claim on its first cell: "固定资产 — 期末余额: 9,470; 期初余额: 8,860",
 * with the table's caption (its date and units) as what its words refer to. One mark a row keeps
 * the margin's marks level with their rows. "not-a-claim" for the heading row and every other
 * cell; null for a sentence outside a table.
 */
export function tableClaim(
  sentence: SelectedSentence,
): { claim: string; context: string } | "not-a-claim" | null {
  const found = cellOf(sentence.element);
  if (found?.table.nodeName !== "table") return null;
  const { cell, row, table } = found;
  const cells = row.toArray();
  const texts = cells.map((_, column) => cellText(row, column));
  const anchor = texts.findIndex((text) => text !== "");
  if (cell.nodeName === "tableHeader" || cells.indexOf(cell) !== anchor) return "not-a-claim";
  const first = table.toArray().find((each) => each instanceof Y.XmlElement);
  const firstCell = first?.toArray()[0];
  const headed = firstCell instanceof Y.XmlElement && firstCell.nodeName === "tableHeader";
  const label = anchor === 0 ? texts[0] : "";
  const values = texts.flatMap((text, column) => {
    if (text === "" || (label && column === 0)) return [];
    const heading = headed && first ? cellText(first, column) : "";
    return [heading ? `${heading}: ${text}` : text];
  });
  if (values.length === 0) return "not-a-claim";
  const before = table.prevSibling;
  const caption = before instanceof Y.XmlElement ? textOf(before).trim() : "";
  const claim = `${label ? `${label} — ` : ""}${values.join("; ")}`;
  return { claim, context: caption || claim };
}
