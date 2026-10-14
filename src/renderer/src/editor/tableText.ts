import { getTextBetween, type TextSerializer } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { TableMap } from "@tiptap/pm/tables";

/**
 * As a spreadsheet pastes it: a cell's own line breaks become spaces so it stays one cell, and
 * a slot a merged cell covers is left empty so the columns keep their places.
 */
export function tableText(
  table: ProseMirrorNode,
  serializers: Record<string, TextSerializer>,
): string {
  const texts = new Map<number, string>();
  table.forEach((row, rowOffset) => {
    row.forEach((cell, cellOffset) => {
      texts.set(rowOffset + 1 + cellOffset, cellText(cell, serializers));
    });
  });
  const map = TableMap.get(table);
  const seen = new Set<number>();
  const lines: string[] = [];
  for (let row = 0; row < map.height; row++) {
    const cells: string[] = [];
    for (let column = 0; column < map.width; column++) {
      const offset = map.map[row * map.width + column] ?? 0;
      cells.push(seen.has(offset) ? "" : (texts.get(offset) ?? ""));
      seen.add(offset);
    }
    lines.push(cells.join("\t"));
  }
  return lines.join("\n");
}

function cellText(cell: ProseMirrorNode, serializers: Record<string, TextSerializer>): string {
  return getTextBetween(
    cell,
    { from: 0, to: cell.content.size },
    { blockSeparator: " ", textSerializers: serializers },
  )
    .replace(/\s*[\t\r\n]+\s*/g, " ")
    .trim();
}
