/**
 * Someone's changes from a Word file, read as changes (#734, see
 * `theirChangesReading`): what a model and the reader of an export are
 * shown. Inline content in stretches by the change each is part of, a
 * Block's change on its words, and Blocks with every change accepted.
 */
import type { Block, Inline, Pending } from "./mindText";

/** A change as shown: what it does to the words it is on, and who made it when. */
export interface ShownChange {
  kind: "insert" | "delete";
  pending: Pending;
}

/** A stretch of inline content one change is on, or none is. */
export interface ChangeSpan {
  change: ShownChange | null;
  content: Inline[];
}

const changeOf = (inline: Inline): ShownChange | null =>
  inline.deleted
    ? { kind: "delete", pending: inline.deleted }
    : inline.inserted
      ? { kind: "insert", pending: inline.inserted }
      : null;

const sameChange = (a: ShownChange | null, b: ShownChange | null) =>
  a === b || (a !== null && b !== null && a.kind === b.kind && a.pending.id === b.pending.id);

/** Inline content in stretches, each with the change it is part of, in order. */
export function changeSpans(content: readonly Inline[]): ChangeSpan[] {
  const spans: ChangeSpan[] = [];
  for (const inline of content) {
    const change = changeOf(inline);
    const last = spans.at(-1);
    if (last && sameChange(last.change, change)) last.content.push(inline);
    else spans.push({ change, content: [inline] });
  }
  return spans;
}

/** A Block's words, each with the change the Block is part of as a whole: a paragraph someone added. */
export function changedContent(block: Block & { content: Inline[] }): Inline[] {
  const { inserted, deleted } = block;
  if (!inserted && !deleted) return block.content;
  return block.content.map((inline) => ({
    ...inline,
    ...(inserted && { inserted }),
    ...(deleted && { deleted }),
  }));
}

/** What a change's note says it did: added words, removed them, or replaced some with others. */
export type NotedChange = Omit<ShownChange, "kind"> & { kind: ShownChange["kind"] | "replace" };

/**
 * The change whose note goes after the span at `at`, if it ends one there: the
 * last part of a change, a replacement's words removed and then added.
 */
export function changeEnding(spans: readonly ChangeSpan[], at: number): NotedChange | null {
  const change = spans[at]?.change;
  if (!change || spans[at + 1]?.change?.pending.id === change.pending.id) return null;
  const before = spans[at - 1]?.change;
  const replaced = before?.pending.id === change.pending.id && before.kind !== change.kind;
  return replaced ? { ...change, kind: "replace" } : change;
}

function acceptedInline(content: readonly Inline[]): Inline[] {
  return content.flatMap((inline): Inline[] => {
    if (inline.deleted) return [];
    const { inserted: _inserted, ...rest } = inline;
    return rest.kind === "footnote" ? [{ ...rest, content: acceptedInline(rest.content) }] : [rest];
  });
}

/** Blocks with every change in them accepted: what one adds kept as plain text, what one removes gone. */
export function withChangesAccepted(blocks: readonly Block[]): Block[] {
  return blocks.flatMap((block): Block[] => {
    if (block.deleted) return [];
    const { inserted: _inserted, ...rest } = block;
    switch (rest.kind) {
      case "paragraph":
      case "heading":
      case "question":
        return [{ ...rest, content: acceptedInline(rest.content) }];
      case "list":
        return [{ ...rest, items: rest.items.map(withChangesAccepted) }];
      case "quote":
        return [{ ...rest, content: withChangesAccepted(rest.content) }];
      case "table":
        return [
          {
            ...rest,
            rows: rest.rows.map((row) =>
              row.map((cell) => ({ ...cell, content: withChangesAccepted(cell.content) })),
            ),
          },
        ];
      case "chart":
        return [{ ...rest, sources: acceptedInline(rest.sources) }];
      default:
        return [rest];
    }
  });
}
