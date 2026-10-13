/**
 * Blocks on their way into a note from an Answer (#201, "Add to note"), as
 * the editor writes them in JSON: read from what the renderer sends, given
 * new ids, and with their Citations checked again where they land. Shared by
 * the core, which adds to a Mind that isn't open, and the renderer, whose
 * editor adds at its cursor.
 */
import {
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  type CitationRecheck,
  NOTE_BLOCK_TYPES,
  type RecheckCitationInput,
} from "../core/api";
import type { NodeJSON } from "./blockJson";

/** At most this many Blocks go into a note at once, and this deep. */
const MAX_BLOCKS = 2_000;
const MAX_DEPTH = 24;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function parseNode(value: unknown, depth: number): NodeJSON | null {
  if (!isRecord(value) || typeof value.type !== "string" || value.type === "") return null;
  if (depth > MAX_DEPTH) return null;
  const node: NodeJSON = { type: value.type };
  if (isRecord(value.attrs)) node.attrs = { ...value.attrs };
  if (typeof value.text === "string") node.text = value.text;
  if (Array.isArray(value.marks)) {
    const marks = value.marks.filter(
      (mark): mark is { type: string; attrs?: Record<string, unknown> } =>
        isRecord(mark) && typeof mark.type === "string",
    );
    if (marks.length > 0) node.marks = marks.map((mark) => ({ ...mark }));
  }
  if (Array.isArray(value.content)) {
    const content: NodeJSON[] = [];
    for (const child of value.content) {
      const parsed = parseNode(child, depth + 1);
      if (!parsed) return null;
      content.push(parsed);
    }
    node.content = content;
  }
  return node;
}

/**
 * The Blocks to add, read from input: Notes only at the top level
 * (`NOTE_BLOCK_TYPES`), with whatever they hold. Null when they aren't that.
 */
export function parseNoteBlocks(value: unknown): NodeJSON[] | null {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_BLOCKS) return null;
  const blocks: NodeJSON[] = [];
  for (const each of value) {
    const block = parseNode(each, 0);
    if (!block || !(NOTE_BLOCK_TYPES as readonly string[]).includes(block.type)) return null;
    blocks.push(block);
  }
  return blocks;
}

/** The Blocks without their ids, at any depth, so they are given new ones where they land. */
export function withoutIds(blocks: readonly NodeJSON[]): NodeJSON[] {
  return blocks.map((block) => {
    const { [BLOCK_ID_ATTRIBUTE]: _id, ...attrs } = block.attrs ?? {};
    return {
      ...block,
      ...(block.attrs && { attrs }),
      ...(block.content && { content: withoutIds(block.content) }),
    };
  });
}

/** What checking a Citation again gives, sync or async (the core's own, or over IPC). */
export type Recheck = (input: RecheckCitationInput) => CitationRecheck | Promise<CitationRecheck>;

const unit = (value: unknown): number | null =>
  typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;

/**
 * The Blocks with each Citation in them checked again against its Document's
 * current version, where they land: its check, and where it points, as the
 * check finds them now. A Citation whose Document is gone, or that can't be
 * checked again for any reason, keeps what it had.
 */
export async function recheckCitations(
  blocks: readonly NodeJSON[],
  recheck: Recheck,
): Promise<NodeJSON[]> {
  const visit = async (node: NodeJSON): Promise<NodeJSON> => {
    if (node.type === CITATION_NODE) return recheckOne(node, recheck);
    if (!node.content) return node;
    return { ...node, content: await Promise.all(node.content.map(visit)) };
  };
  return Promise.all(blocks.map(visit));
}

async function recheckOne(node: NodeJSON, recheck: Recheck): Promise<NodeJSON> {
  const attrs = node.attrs ?? {};
  const { documentId, quote } = attrs;
  if (typeof documentId !== "string" || documentId === "") return node;
  try {
    const pageFrom = unit(attrs.pageFrom);
    const checked = await recheck({
      documentId,
      quote: typeof quote === "string" ? quote : "",
      pageFrom,
      pageTo: unit(attrs.pageTo) ?? pageFrom,
    });
    // A deleted Document: the check made when it was cited stands, as the badge shows it.
    if (checked.contentHash === null) return node;
    return {
      ...node,
      attrs: {
        ...attrs,
        check: checked.check,
        checkReason: checked.checkReason,
        contentHash: checked.contentHash,
        passageId: checked.passageId,
        pageFrom: checked.pageFrom,
        pageTo: checked.pageTo,
        location: checked.location,
      },
    };
  } catch (error) {
    // The check made when it was cited stands; the Note still goes in.
    console.error("A Citation couldn't be checked again where it was added.", error);
    return node;
  }
}

/**
 * How much text the Blocks hold, for the record of AI use: as accepting a
 * Proposal counts it (src/shared/proposals.ts), a node with nothing in it
 * (a Citation, a formula) counts as one.
 */
export function textLength(blocks: readonly NodeJSON[]): number {
  const length = (node: NodeJSON): number =>
    node.type === "text"
      ? (node.text?.length ?? 0)
      : node.content && node.content.length > 0
        ? node.content.reduce((sum, child) => sum + length(child), 0)
        : 1;
  return blocks.reduce((sum, block) => sum + length(block), 0);
}

const textOf = (node: NodeJSON): string =>
  node.type === "text" ? (node.text ?? "") : (node.content ?? []).map(textOf).join("");

const holdsCitation = (node: NodeJSON): boolean =>
  node.type === CITATION_NODE || (node.content ?? []).some(holdsCitation);

/** A paragraph of Citations and nothing else but spaces and punctuation. */
const isCitationsOnly = (block: NodeJSON): boolean =>
  block.type === "paragraph" &&
  (block.content ?? []).some((child) => child.type === CITATION_NODE) &&
  (block.content ?? []).every(
    (child) =>
      child.type === CITATION_NODE ||
      (child.type === "text" && /^[\s.,;，。；]*$/u.test(child.text ?? "")),
  );

/**
 * A table's Citation, written under it as a paragraph of its own, goes onto the
 * table's cells that hold a figure or words after the first column, so the
 * Citation stays with the table (a chart made from it reads them there, and the
 * Mind doesn't count the table as having none). A table whose cells carry
 * Citations already is left as it is.
 */
export function citedTables(blocks: readonly NodeJSON[]): NodeJSON[] {
  const out: NodeJSON[] = [];
  for (const block of blocks) {
    const table = out.at(-1);
    if (table?.type !== "table" || !isCitationsOnly(block) || holdsCitation(table)) {
      out.push(block);
      continue;
    }
    const citations = (block.content ?? []).filter((child) => child.type === CITATION_NODE);
    out[out.length - 1] = {
      ...table,
      content: (table.content ?? []).map((row, rowIndex) => ({
        ...row,
        content: (row.content ?? []).map((cell, cellIndex) => {
          const [paragraph] = cell.content ?? [];
          if (rowIndex === 0 || cellIndex === 0 || cell.type !== "tableCell") return cell;
          if (!paragraph || paragraph.type !== "paragraph" || textOf(paragraph).trim() === "") {
            return cell;
          }
          return {
            ...cell,
            content: [
              { ...paragraph, content: [...(paragraph.content ?? []), ...citations] },
              ...(cell.content ?? []).slice(1),
            ],
          };
        }),
      })),
    };
  }
  return out;
}

const LEAD_IN =
  /^(?:(?:sure|certainly|of course|okay|ok)\b|here(?:'s| is| are)\b|below (?:is|are)\b|以下是|下面是|好的|当然|这是|这里是|如下)/iu;
const CLOSING_NOTE =
  /^(?:(?:note|notes|please note|n\.b\.|i hope|hope this|let me know|feel free)\b|注[：:]|备注|说明[：:]|希望|如需|如果需要|如有)/iu;
const CONTENT_BLOCKS = new Set(["table", "bulletList", "orderedList", "heading", "codeBlock"]);

/**
 * An Answer's content without its framing, for Add to note: the model's
 * lead-in before the content ("Here is a draft:", "以下是…：") and its closing
 * note after it ("Note: …", "注：…"), when they hold no Citation and the
 * Answer has more than the one paragraph. What the User selected is added as it
 * is: this is for a whole Answer.
 */
export function withoutFraming(blocks: readonly NodeJSON[]): NodeJSON[] {
  let kept = [...blocks];
  const first = kept[0];
  const next = kept[1];
  if (
    first?.type === "paragraph" &&
    next &&
    !holdsCitation(first) &&
    (LEAD_IN.test(textOf(first).trim()) ||
      (/[:：]\s*$/u.test(textOf(first)) && CONTENT_BLOCKS.has(next.type)))
  ) {
    kept = kept.slice(1);
  }
  const last = kept.at(-1);
  if (
    last?.type === "paragraph" &&
    kept.length > 1 &&
    !holdsCitation(last) &&
    CLOSING_NOTE.test(textOf(last).trim())
  ) {
    kept = kept.slice(0, -1);
  }
  return kept;
}
