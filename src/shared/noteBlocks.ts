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

/** A node as ProseMirror (and Tiptap) write it in JSON. */
export interface BlockJSON {
  type: string;
  attrs?: Record<string, unknown>;
  content?: BlockJSON[];
  /** Text nodes only. */
  text?: string;
  /** Text nodes only. */
  marks?: { type: string; attrs?: Record<string, unknown> }[];
}

/** At most this many Blocks go into a note at once, and this deep. */
const MAX_BLOCKS = 2_000;
const MAX_DEPTH = 24;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function parseNode(value: unknown, depth: number): BlockJSON | null {
  if (!isRecord(value) || typeof value.type !== "string" || value.type === "") return null;
  if (depth > MAX_DEPTH) return null;
  const node: BlockJSON = { type: value.type };
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
    const content: BlockJSON[] = [];
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
export function parseNoteBlocks(value: unknown): BlockJSON[] | null {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_BLOCKS) return null;
  const blocks: BlockJSON[] = [];
  for (const each of value) {
    const block = parseNode(each, 0);
    if (!block || !(NOTE_BLOCK_TYPES as readonly string[]).includes(block.type)) return null;
    blocks.push(block);
  }
  return blocks;
}

/** The Blocks without their ids, at any depth, so they are given new ones where they land. */
export function withoutIds(blocks: readonly BlockJSON[]): BlockJSON[] {
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
  blocks: readonly BlockJSON[],
  recheck: Recheck,
): Promise<BlockJSON[]> {
  const visit = async (node: BlockJSON): Promise<BlockJSON> => {
    if (node.type === CITATION_NODE) return recheckOne(node, recheck);
    if (!node.content) return node;
    return { ...node, content: await Promise.all(node.content.map(visit)) };
  };
  return Promise.all(blocks.map(visit));
}

async function recheckOne(node: BlockJSON, recheck: Recheck): Promise<BlockJSON> {
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
  } catch {
    return node;
  }
}

/** How many Citations the Blocks hold, at any depth. */
export function countCitations(blocks: readonly BlockJSON[]): number {
  let count = 0;
  const visit = (node: BlockJSON) => {
    if (node.type === CITATION_NODE) count++;
    node.content?.forEach(visit);
  };
  blocks.forEach(visit);
  return count;
}
