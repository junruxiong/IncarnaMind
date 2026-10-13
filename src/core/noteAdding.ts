/**
 * "Add to note" into a Mind that isn't open in the editor (#201): Blocks from
 * an Answer in the assistant go at the end of the Mind's note, each Citation
 * in them checked again against its Document there, with a link back to the
 * Answer (`MIND_ORIGINS_FIELD`). What goes in at once is recorded as AI use
 * (#292); Blocks `proposed` (a new Mind made from an Answer) wait as changes
 * to accept or reject, each Block one, and are recorded once accepted. Their
 * words go in as AI text (`GENERATED_MARK`). Undo takes them out again. The
 * Mind open in the editor adds at its cursor itself, by the same rules
 * (src/shared/noteBlocks.ts).
 */
import { randomUUID } from "node:crypto";
import * as Y from "yjs";
import { asGenerated } from "../shared/aiText";
import { parseOrigin, removeOrigins, setOrigins } from "../shared/blockOrigins";
import { parseNoteBlocks, recheckCitations, textLength, withoutIds } from "../shared/noteBlocks";
import { pendingProposals, proposalRequestOf, recordProposalRequest } from "../shared/proposals";
import { createElement, type NodeJSON, textAttribute, topLevelBlocks } from "./answers/blocks";
import {
  type AddedToNote,
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type CitationRecheck,
  PROPOSAL_ATTRIBUTE,
  type ProposalAttribute,
  type ProposalRequest,
  type RecheckCitationInput,
  type RecordAiUseInput,
} from "./api";
import { InvalidInputError, isRecord } from "./errors";
import type { MindContent } from "./mindContent";
import { isChatModelChoice } from "./settings";

export interface NoteAddingOptions {
  content: MindContent;
  /** Throws unless the Mind exists and can be written to; returns its id. */
  requireMind(mindId: unknown): string;
  recheck(input: RecheckCitationInput): CitationRecheck;
  /** Keeps what the AI put into a Mind in the record of AI use. */
  recordAiUse(inputs: RecordAiUseInput[]): void;
  now(): Date;
}

const isEmptyLine = (block: Y.XmlElement | undefined) =>
  block?.nodeName === "paragraph" && block.length === 0;

/** Blocks as changes to accept, each its own, all one request's. */
const asProposals = (blocks: readonly NodeJSON[], request: string): NodeJSON[] =>
  blocks.map(
    (block) =>
      ({
        ...block,
        attrs: {
          ...block.attrs,
          [PROPOSAL_ATTRIBUTE]: {
            id: randomUUID(),
            request,
            kind: "insert",
          } satisfies ProposalAttribute,
        },
      }) as NodeJSON,
  );

/** Puts Blocks at the note's end, in place of its empty last line, as a Question asked there would be; returns their ids. */
function appendBlocks(note: Y.XmlFragment, blocks: readonly NodeJSON[]): string[] {
  const elements = blocks.map((block) => createElement(block));
  const last = topLevelBlocks(note).at(-1);
  if (isEmptyLine(last) && note.get(note.length - 1) === last) note.delete(note.length - 1, 1);
  note.insert(note.length, elements);
  return elements.map((element) => textAttribute(element, BLOCK_ID_ATTRIBUTE) ?? "");
}

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
      const mindId = options.requireMind(input.mindId);
      const parsed = parseNoteBlocks(input.blocks);
      if (!parsed) {
        throw new InvalidInputError("addToNote takes a list of Notes, as the editor writes them.");
      }
      const origin =
        input.origin === null || input.origin === undefined ? null : parseOrigin(input.origin);
      if (input.origin && !origin) throw new InvalidInputError("That isn't a Block's origin.");
      const model: ChatModelChoice | null = isChatModelChoice(input.model) ? input.model : null;
      const proposed = input.proposed === true;
      const blocks = asGenerated(await recheckCitations(withoutIds(parsed), options.recheck));
      const request = randomUUID();

      const blockIds = content.edit(mindId, (note) => {
        const ids = appendBlocks(
          note,
          proposed ? asProposals(blocks as NodeJSON[], request) : (blocks as NodeJSON[]),
        );
        const doc = note.doc;
        if (doc && origin) setOrigins(doc, ids, origin);
        if (doc && proposed && model && !proposalRequestOf(doc, request)) {
          recordProposalRequest(doc, request, {
            source: "answer-added",
            providerId: model.providerId,
            modelId: model.modelId,
            createdAt: options.now().toISOString(),
          });
        }
        return ids;
      });
      // A change to accept is recorded once it is accepted, as Edit with AI's are.
      if (!proposed) {
        options.recordAiUse([
          {
            mindId,
            kind: "answer-added",
            providerId: model?.providerId ?? null,
            modelId: model?.modelId ?? null,
            characters: textLength(blocks),
            blockIds,
          },
        ]);
      }
      return { mindId, blockIds };
    },

    /**
     * Puts a Task's cited Blocks at the Mind's end as changes to accept, one
     * a Block, under `request`, their Citations checked again. Once only: a
     * Task that goes on after a quit finds them there and adds nothing.
     * Returns the first change's id and how many there are.
     */
    async propose(
      mindId: string,
      given: readonly NodeJSON[],
      request: string,
      made: ProposalRequest,
    ): Promise<{ proposalId: string; changes: number }> {
      const blocks = asProposals(
        asGenerated(await recheckCitations(withoutIds(given), options.recheck)),
        request,
      );
      return content.edit(mindId, (note) => {
        const doc = note.doc;
        if (doc && proposalRequestOf(doc, request)) {
          const mine = pendingProposals(note).filter((each) => each.request === request);
          return { proposalId: mine[0]?.id ?? "", changes: mine.length };
        }
        appendBlocks(note, blocks);
        if (doc) recordProposalRequest(doc, request, made);
        const first = blocks[0]?.attrs?.[PROPOSAL_ATTRIBUTE] as ProposalAttribute | undefined;
        return { proposalId: first?.id ?? "", changes: blocks.length };
      });
    },

    undo(input: unknown): void {
      if (!isRecord(input)) throw new InvalidInputError("undoAddToNote expects an object.");
      const mindId = options.requireMind(input.mindId);
      const blockIds = new Set(parseBlockIds(input.blockIds));
      content.edit(mindId, (note) => {
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
        if (note.doc) removeOrigins(note.doc, [...blockIds]);
      });
    },
  };
}
