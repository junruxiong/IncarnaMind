/**
 * "Add to note" into a Mind that isn't open in the editor (#201): Blocks from
 * an Answer in a chat go at the end of the Mind's note, each Citation in them
 * checked again against its Document there, with a link back to the Answer
 * (`MIND_ORIGINS_FIELD`). Undo takes them out again. The open Mind's editor
 * adds at its cursor itself, by the same rules (src/shared/noteBlocks.ts).
 */
import * as Y from "yjs";
import { parseOrigin, removeOrigins, setOrigins } from "../shared/blockOrigins";
import { parseNoteBlocks, recheckCitations, withoutIds } from "../shared/noteBlocks";
import { createElement, type NodeJSON, textAttribute, topLevelBlocks } from "./answers/blocks";
import {
  type AddedToNote,
  BLOCK_ID_ATTRIBUTE,
  type CitationRecheck,
  MIND_CONTENT_FIELD,
  type RecheckCitationInput,
} from "./api";
import { InvalidInputError, isRecord } from "./errors";
import type { MindContent } from "./mindContent";

export interface NoteAddingOptions {
  content: MindContent;
  /** Throws unless the Mind exists and its note can be written to; returns its id. */
  requireNote(mindId: unknown): string;
  /** Checks a Citation again against its Document's current version. */
  recheck(input: RecheckCitationInput): CitationRecheck;
}

const isEmptyLine = (block: Y.XmlElement | undefined) =>
  block?.nodeName === "paragraph" && block.length === 0;

function parseBlockIds(value: unknown): string[] {
  if (
    !Array.isArray(value) ||
    !value.every((id): id is string => typeof id === "string" && id !== "")
  ) {
    throw new InvalidInputError("blockIds must be a list of Block ids.");
  }
  return value;
}

export function createNoteAdding(options: NoteAddingOptions) {
  const { content } = options;

  return {
    async add(input: unknown): Promise<AddedToNote> {
      if (!isRecord(input)) throw new InvalidInputError("addToNote expects an object.");
      const mindId = options.requireNote(input.mindId);
      const parsed = parseNoteBlocks(input.blocks);
      if (!parsed) {
        throw new InvalidInputError("addToNote takes a list of Notes, as the editor writes them.");
      }
      const origin =
        input.origin === null || input.origin === undefined ? null : parseOrigin(input.origin);
      if (input.origin && !origin) throw new InvalidInputError("That isn't a Block's origin.");
      const blocks = await recheckCitations(withoutIds(parsed), options.recheck);

      const blockIds = content.editDoc(mindId, (doc) => {
        const note = doc.getXmlFragment(MIND_CONTENT_FIELD);
        const elements = blocks.map((block) => createElement(block as NodeJSON));
        // At the end: in place of its empty last line, as a Question asked there would be.
        const children = topLevelBlocks(note);
        const last = children.at(-1);
        if (isEmptyLine(last) && note.get(note.length - 1) === last) {
          note.delete(note.length - 1, 1);
        }
        note.insert(note.length, elements);
        const ids = elements.map((element) => textAttribute(element, BLOCK_ID_ATTRIBUTE) ?? "");
        if (origin) setOrigins(doc, ids, origin);
        return ids;
      });
      return { mindId, blockIds };
    },

    undo(input: unknown): void {
      if (!isRecord(input)) throw new InvalidInputError("undoAddToNote expects an object.");
      const mindId = options.requireNote(input.mindId);
      const blockIds = new Set(parseBlockIds(input.blockIds));
      content.editDoc(mindId, (doc) => {
        const note = doc.getXmlFragment(MIND_CONTENT_FIELD);
        // From the end, so each index stays right.
        for (let index = note.length - 1; index >= 0; index--) {
          const block = note.get(index);
          if (
            block instanceof Y.XmlElement &&
            blockIds.has(textAttribute(block, BLOCK_ID_ATTRIBUTE) ?? "")
          ) {
            note.delete(index, 1);
          }
        }
        removeOrigins(doc, [...blockIds]);
      });
    },
  };
}
