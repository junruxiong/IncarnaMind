/**
 * Compare Documents (#98): a comparison table in the Mind, one row per
 * Document and one column per Question, whose cells are cited Answers.
 *
 * - The table arrives as one Proposal (#93): a whole Block the User accepts
 *   or rejects. Rejecting it, or deleting it, stops its fill.
 * - Each cell is an Answer from the Answer engine (../answers), scoped to its
 *   row's Document alone and offered the Documents' search and `cite` only (a
 *   check Run), its Citations checked as any Answer's. Documents IncarnaMind
 *   wrote are left out of its evidence as they are of every check's (#80).
 * - The cells are filled as items of one run (../runs/stop): up to 4 at a
 *   time on a model off this computer, one at a time on one on it; a provider
 *   limiting requests makes the run pause and try again; the monthly cap or
 *   a plan's limit stops it, and every cell left says it wasn't checked.
 * - Each cell is written as it finishes, with its problem when it has no
 *   Answer. A fill reads the cells to fill from the table each time, so a
 *   column added meanwhile, or the cells a quit left, are filled by the same
 *   call: what a Task (#126) will run in the background.
 */
import { randomUUID } from "node:crypto";
import * as Y from "yjs";
import {
  cellToFill,
  comparisonCellOf,
  comparisonQuestionsOf,
  MAX_COMPARISON_DOCUMENTS,
  MAX_COMPARISON_QUESTIONS,
  MAX_COMPARISON_TEXT,
} from "../../shared/comparisons";
import { recordProposalRequest, targetHash } from "../../shared/proposals";
import type { CellAnswer } from "../answers";
import {
  createElement,
  findBlock,
  type NodeJSON,
  syncContent,
  topLevelBlocks,
} from "../answers/blocks";
import type { QuestionContext } from "../answers/context";
import {
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type ChatReadiness,
  COMPARISON_CELL_NODE,
  COMPARISON_ROW_NODE,
  COMPARISON_TABLE_NODE,
  type ComparisonCellAttributes,
  type ComparisonProblem,
  type ComparisonResult,
  type ComparisonTableRef,
  type CoreEvents,
  PROPOSAL_ATTRIBUTE,
  type ProposalAttribute,
} from "../api";
import { ChatNotReadyError, InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { MindContent } from "../mindContent";
import { plainText } from "../mindText";
import { classifyProviderError } from "../providers/providerErrors";
import { type ItemOutcome, RunItemError, runItems } from "../runs/stop";
import { cellContext, isNotFoundReply, saysNotFound } from "./prompt";

/** Cells written at once on a model off this computer. */
export const CLOUD_CELLS_AT_ONCE = 4;

export interface ComparisonsOptions {
  content: Pick<MindContent, "read" | "edit">;
  /** Throws unless the Mind exists and this version may write to it; returns its id. */
  requireMind(mindId: unknown): string;
  mindExists(mindId: string): boolean;
  answers: {
    answerCell(input: {
      context: QuestionContext;
      documentIds: string[];
      model: ChatModelChoice;
      signal: AbortSignal;
    }): Promise<CellAnswer>;
  };
  chat: {
    providerExists(providerId: string): boolean;
    readiness(choice: ChatModelChoice | undefined): Promise<ChatReadiness>;
    /** The model, once the User has allowed its data flow (asking them if needed). */
    prepareModel(choice: ChatModelChoice): Promise<unknown>;
    /** Whether the model runs on this computer, which serves one request at a time. */
    runsLocally(choice: ChatModelChoice): boolean;
  };
  documents: {
    /** The live Documents among `ids`, in that order. */
    live(ids: readonly string[]): { id: string; name: string }[];
    /** How many of `ids` have Passages to search. */
    searchableCount(ids: readonly string[]): number;
  };
  /** The Documents among `ids` that checks may use as evidence (#80). */
  evidence(ids: readonly string[]): string[];
  emit<E extends "comparison.filling">(event: E, payload: CoreEvents[E]): void;
  reportError(error: unknown): void;
}

/** One cell to fill, with what its Answer is asked from. */
interface Cell {
  documentId: string;
  questionId: string;
  question: string;
}

/** What a cell came to: its Answer, or why it has none. */
type Written =
  | { kind: "answer"; blocks: NodeJSON[] }
  | { kind: "problem"; problem: ComparisonProblem; message?: string };

interface Fill {
  mindId: string;
  model: ChatModelChoice;
  controller: AbortController;
}

/** What a cell's attributes are set to, and its content (none: one empty paragraph). */
type CellPatch = Partial<Omit<ComparisonCellAttributes, "questionId">> & { blocks?: NodeJSON[] };

const keyOf = (cell: Pick<Cell, "documentId" | "questionId">) =>
  `${cell.documentId}\n${cell.questionId}`;

const EMPTY_CELL: NodeJSON[] = [{ type: "paragraph" }];

function oneLine(value: unknown, what: string): string {
  if (typeof value !== "string") throw new InvalidInputError(`${what} must be text.`);
  const line = value.replace(/\s+/g, " ").trim();
  if (line.length > MAX_COMPARISON_TEXT) {
    throw new InvalidInputError(`${what} is too long: at most ${MAX_COMPARISON_TEXT} characters.`);
  }
  return line;
}

function parseModel(value: unknown): ChatModelChoice | null {
  if (value === undefined || value === null) return null;
  if (
    !isRecord(value) ||
    typeof value.providerId !== "string" ||
    typeof value.modelId !== "string" ||
    value.providerId === "" ||
    value.modelId === ""
  ) {
    throw new InvalidInputError("model must be a provider and a model.");
  }
  return { providerId: value.providerId, modelId: value.modelId };
}

function parseRef(input: unknown, what: string): ComparisonTableRef {
  if (!isRecord(input) || typeof input.tableId !== "string" || input.tableId === "") {
    throw new InvalidInputError(`${what} needs a Mind and a comparison table.`);
  }
  return { mindId: String(input.mindId ?? ""), tableId: input.tableId };
}

/** A comparison table's Block, by its id: null once it was deleted or rejected. */
const tableIn = (blocks: Y.XmlFragment, tableId: string) =>
  findBlock(blocks, COMPARISON_TABLE_NODE, tableId)?.element ?? null;

const elements = (parent: Y.XmlElement) =>
  parent.toArray().filter((child): child is Y.XmlElement => child instanceof Y.XmlElement);

/** The cells to fill in a table, row by row, each row's in its columns' order. */
function cellsToFill(table: Y.XmlElement): Cell[] {
  const questions = comparisonQuestionsOf(table.getAttributes());
  const cells: Cell[] = [];
  for (const row of elements(table)) {
    if (row.nodeName !== COMPARISON_ROW_NODE) continue;
    const documentId = String(row.getAttribute("documentId") ?? "");
    const byQuestion = new Map(
      elements(row).map((cell) => [String(cell.getAttribute("questionId") ?? ""), cell]),
    );
    for (const question of questions) {
      const cell = byQuestion.get(question.id);
      if (!cell || !cellToFill(comparisonCellOf(cell.getAttributes()))) continue;
      cells.push({ documentId, questionId: question.id, question: question.text });
    }
  }
  return cells;
}

/** The JSON of a cell, as the editor stores it. */
const cellJSON = (questionId: string): NodeJSON => ({
  type: COMPARISON_CELL_NODE,
  attrs: { questionId, status: "waiting" },
  content: EMPTY_CELL,
});

export function createComparisons(options: ComparisonsOptions) {
  const { content } = options;
  /** The fills going, by table. */
  const fills = new Map<string, Fill>();

  /** The model a table's cells are written with, ready and allowed; or why it can't be. */
  async function readyModel(
    picked: ChatModelChoice | null,
  ): Promise<{ model: ChatModelChoice } | Exclude<ComparisonResult, { started: true }>> {
    const choice = picked && options.chat.providerExists(picked.providerId) ? picked : undefined;
    const readiness = await options.chat.readiness(choice);
    if (!readiness.ready) return { started: false, reason: "not-ready", readiness };
    const model = { providerId: readiness.provider.id, modelId: readiness.modelId };
    try {
      await options.chat.prepareModel(model);
    } catch (error) {
      if (error instanceof ChatNotReadyError) {
        return { started: false, reason: "not-ready", readiness: error.readiness };
      }
      if (classifyProviderError(error).kind === "consent-declined") {
        return { started: false, reason: "consent-declined" };
      }
      throw error;
    }
    return { model };
  }

  /**
   * Sets a cell's state and content. "table": the table is gone (rejected or
   * deleted), or its Mind; "cell": the cell is (its column taken back).
   */
  function setCell(
    fill: Fill,
    tableId: string,
    cell: Pick<Cell, "documentId" | "questionId">,
    patch: CellPatch,
  ): "written" | "table" | "cell" {
    return setCells(fill, tableId, [cell], patch);
  }

  /** `setCell` for several cells, in one change: "cell" when any of them is gone. */
  function setCells(
    fill: Fill,
    tableId: string,
    cells: readonly Pick<Cell, "documentId" | "questionId">[],
    { blocks, ...patch }: CellPatch,
  ): "written" | "table" | "cell" {
    if (!options.mindExists(fill.mindId)) return "table";
    return content.edit(fill.mindId, (all) => {
      const table = tableIn(all, tableId);
      if (!table) return "table";
      const rows = new Map(
        elements(table).map((row) => [String(row.getAttribute("documentId") ?? ""), row]),
      );
      let result: "written" | "cell" = "written";
      for (const cell of cells) {
        const row = rows.get(cell.documentId);
        const element = row
          ? elements(row).find((each) => each.getAttribute("questionId") === cell.questionId)
          : undefined;
        if (!element) {
          result = "cell";
          continue;
        }
        for (const [name, value] of Object.entries(patch)) {
          if (value === null || value === undefined) element.removeAttribute(name);
          else element.setAttribute(name, value);
        }
        if (blocks) syncContent(element, blocks.length > 0 ? blocks : EMPTY_CELL);
      }
      return result;
    });
  }

  /** Writes a cell's outcome as it comes. A stopped cell stays as it was: filling again writes it. */
  function record(fill: Fill, tableId: string, cell: Cell, outcome: ItemOutcome<Written>): void {
    let patch: CellPatch;
    switch (outcome.status) {
      case "stopped":
        return;
      case "done":
        patch =
          outcome.value.kind === "answer"
            ? {
                status: "done",
                problem: null,
                message: null,
                resetsAt: null,
                blocks: outcome.value.blocks,
              }
            : {
                status: "failed",
                problem: outcome.value.problem,
                message: outcome.value.message ?? null,
                resetsAt: null,
                blocks: EMPTY_CELL,
              };
        break;
      case "failed":
        patch = {
          status: "failed",
          problem: outcome.error.kind,
          message: outcome.error.message || null,
          resetsAt: outcome.error.resetsAt ?? null,
          blocks: EMPTY_CELL,
        };
        break;
      case "not-checked":
        patch = {
          status: "failed",
          problem: outcome.stop.reason,
          message: null,
          resetsAt: outcome.stop.reason === "plan-limit" ? outcome.stop.resetsAt : null,
          blocks: EMPTY_CELL,
        };
        break;
    }
    try {
      if (setCell(fill, tableId, cell, patch) === "table") fill.controller.abort();
    } catch (error) {
      options.reportError(error);
      fill.controller.abort();
    }
  }

  /** Writes one cell: its Answer from its row's Document alone, or why there is none. */
  async function write(fill: Fill, tableId: string, cell: Cell): Promise<Written> {
    const { signal } = fill.controller;
    const placed = setCell(fill, tableId, cell, { status: "writing" });
    if (placed !== "written") {
      if (placed === "table") fill.controller.abort();
      throw new DOMException("The cell is gone.", "AbortError");
    }
    const [document] = options.documents.live([cell.documentId]);
    if (!document) return { kind: "problem", problem: "source-removed" };
    const documentIds = options.evidence([document.id]);
    if (documentIds.length === 0) return { kind: "problem", problem: "left-out" };
    if (options.documents.searchableCount(documentIds) === 0) {
      return { kind: "problem", problem: "nothing-to-search" };
    }
    const instruction = content.read(fill.mindId, (all) => {
      const value = tableIn(all, tableId)?.getAttribute("instruction");
      return typeof value === "string" && value !== "" ? value : null;
    });
    const answer = await options.answers.answerCell({
      context: cellContext(cell.question, document.name, instruction),
      documentIds,
      model: fill.model,
      signal,
    });
    if (answer.status === "stopped") throw new DOMException("Stopped.", "AbortError");
    if (answer.status === "failed" && answer.error) throw new RunItemError(answer.error);
    if (saysNotFound(answer.markdown, answer.citations)) {
      // What the model said instead, if anything but NOT_FOUND, for the cell's tooltip.
      const said = isNotFoundReply(answer.markdown) ? "" : answer.markdown.trim();
      return { kind: "problem", problem: "not-found", ...(said && { message: said }) };
    }
    return { kind: "answer", blocks: answer.blocks };
  }

  /** Fills the table's cells to fill, again while a column added meanwhile leaves more. */
  async function run(fill: Fill, tableId: string): Promise<void> {
    const tried = new Set<string>();
    const { signal } = fill.controller;
    for (;;) {
      if (signal.aborted || !options.mindExists(fill.mindId)) return;
      const cells = content.read(fill.mindId, (all) => {
        const table = tableIn(all, tableId);
        return table ? cellsToFill(table) : null;
      });
      const next = (cells ?? []).filter((cell) => !tried.has(keyOf(cell)));
      if (next.length === 0) return;
      for (const cell of next) tried.add(keyOf(cell));
      // Each shows it waits again: a failure from before isn't its outcome now.
      const waiting = setCells(fill, tableId, next, {
        status: "waiting",
        problem: null,
        message: null,
        resetsAt: null,
        blocks: EMPTY_CELL,
      });
      if (waiting === "table") return;
      const { stop } = await runItems(next, (cell) => write(fill, tableId, cell), {
        concurrency: options.chat.runsLocally(fill.model) ? 1 : CLOUD_CELLS_AT_ONCE,
        signal,
        onOutcome: (cell, outcome) => record(fill, tableId, cell, outcome),
      });
      if (stop) {
        // A column added meanwhile isn't checked either: the same limit stands.
        const left = content.read(fill.mindId, (all) => {
          const table = tableIn(all, tableId);
          return table ? cellsToFill(table).filter((cell) => !tried.has(keyOf(cell))) : [];
        });
        for (const cell of left) record(fill, tableId, cell, { status: "not-checked", stop });
        return;
      }
    }
  }

  /** Starts filling a table, unless a fill is going: that one takes the cells up. */
  function startFill(mindId: string, tableId: string, model: ChatModelChoice): void {
    if (fills.has(tableId)) return;
    const fill: Fill = { mindId, model, controller: new AbortController() };
    fills.set(tableId, fill);
    options.emit("comparison.filling", { mindId, tableId, filling: true });
    run(fill, tableId)
      .catch((error: unknown) => {
        if (!fill.controller.signal.aborted) options.reportError(error);
      })
      .finally(() => {
        fills.delete(tableId);
        options.emit("comparison.filling", { mindId, tableId, filling: false });
      });
  }

  /** Where the table goes: after the Block named, or before it when it is an empty paragraph. */
  function placeOf(blocks: Y.XmlFragment, afterBlockId: string | null): number {
    if (!afterBlockId) return blocks.length;
    const children = topLevelBlocks(blocks);
    const index = children.findIndex(
      (each) => each.getAttribute(BLOCK_ID_ATTRIBUTE) === afterBlockId,
    );
    if (index === -1) return blocks.length;
    const block = children[index] as Y.XmlElement;
    const empty =
      block.nodeName === "paragraph" &&
      plainText(block) === "" &&
      !block.toArray().some((child) => child instanceof Y.XmlElement);
    return blocks.toArray().indexOf(block) + (empty ? 0 : 1);
  }

  return {
    async compareDocuments(input: unknown): Promise<ComparisonResult> {
      if (!isRecord(input)) throw new InvalidInputError("compareDocuments expects an object.");
      const mindId = options.requireMind(input.mindId);
      if (!Array.isArray(input.questions)) throw new InvalidInputError("questions must be a list.");
      const questions = input.questions
        .map((each) => oneLine(each, "A Question"))
        .filter((each) => each !== "");
      if (questions.length === 0) throw new InvalidInputError("Add a Question to compare by.");
      if (questions.length > MAX_COMPARISON_QUESTIONS) {
        throw new InvalidInputError(
          `A comparison table has at most ${MAX_COMPARISON_QUESTIONS} Questions.`,
        );
      }
      if (
        !Array.isArray(input.documentIds) ||
        !input.documentIds.every((id) => typeof id === "string")
      ) {
        throw new InvalidInputError("documentIds must be a list of Document ids.");
      }
      const counted = new Set(options.evidence([...new Set(input.documentIds as string[])]));
      const documents = options.documents
        .live([...new Set(input.documentIds as string[])])
        .filter((document) => counted.has(document.id));
      if (documents.length === 0) throw new InvalidInputError("There are no Documents to compare.");
      if (documents.length > MAX_COMPARISON_DOCUMENTS) {
        throw new InvalidInputError(
          `A comparison table has at most ${MAX_COMPARISON_DOCUMENTS} Documents.`,
        );
      }
      const instruction =
        input.instruction === undefined || input.instruction === null
          ? ""
          : oneLine(input.instruction, "What else the table should show");
      const afterBlockId = typeof input.afterBlockId === "string" ? input.afterBlockId : null;

      const ready = await readyModel(parseModel(input.model));
      if (!("model" in ready)) return ready;
      const { model } = ready;

      const tableId = randomUUID();
      const request = randomUUID();
      const columns = questions.map((text) => ({ id: randomUUID(), text }));
      const proposal: ProposalAttribute = {
        id: randomUUID(),
        request,
        target: targetHash(""),
        kind: "insert",
      };
      const table = createElement({
        type: COMPARISON_TABLE_NODE,
        attrs: {
          [BLOCK_ID_ATTRIBUTE]: tableId,
          questions: columns,
          providerId: model.providerId,
          modelId: model.modelId,
          instruction: instruction || null,
          [PROPOSAL_ATTRIBUTE]: proposal,
        },
        content: documents.map((document) => ({
          type: COMPARISON_ROW_NODE,
          attrs: { documentId: document.id, documentName: document.name },
          content: columns.map((column) => cellJSON(column.id)),
        })),
      });
      content.edit(mindId, (blocks) => {
        blocks.insert(placeOf(blocks, afterBlockId), [table]);
        if (blocks.doc) {
          recordProposalRequest(blocks.doc, request, {
            source: "comparison-table",
            tableId,
            providerId: model.providerId,
            modelId: model.modelId,
            createdAt: new Date().toISOString(),
          });
        }
      });
      startFill(mindId, tableId, model);
      return { started: true, tableId };
    },

    async fillComparisonTable(input: unknown): Promise<ComparisonResult> {
      const ref = parseRef(input, "fillComparisonTable");
      const mindId = options.requireMind(ref.mindId);
      const picked = content.read(mindId, (blocks) => {
        const table = tableIn(blocks, ref.tableId);
        if (!table) throw new NotFoundError("That comparison table isn't in this Mind.");
        const providerId = table.getAttribute("providerId");
        const modelId = table.getAttribute("modelId");
        return typeof providerId === "string" && typeof modelId === "string"
          ? { providerId, modelId }
          : null;
      });
      const going = fills.get(ref.tableId);
      if (going) return { started: true, tableId: ref.tableId };
      const ready = await readyModel(picked);
      if (!("model" in ready)) return ready;
      startFill(mindId, ref.tableId, ready.model);
      return { started: true, tableId: ref.tableId };
    },

    stop(input: unknown): void {
      const ref = parseRef(input, "stopComparisonTable");
      const fill = fills.get(ref.tableId);
      if (fill && fill.mindId === ref.mindId) fill.controller.abort();
    },

    filling(): ComparisonTableRef[] {
      return [...fills].map(([tableId, fill]) => ({ mindId: fill.mindId, tableId }));
    },

    /** A Mind changed: a fill whose table has gone (rejected, or deleted) stops. */
    mindChanged(mindId: string): void {
      for (const [tableId, fill] of fills) {
        if (fill.mindId !== mindId || fill.controller.signal.aborted) continue;
        const there =
          options.mindExists(mindId) &&
          content.read(mindId, (blocks) => !!tableIn(blocks, tableId));
        if (!there) fill.controller.abort();
      }
    },

    /** Stops every fill, e.g. when the app quits: the cells left are filled next time. */
    stopAll(): void {
      for (const fill of fills.values()) fill.controller.abort();
    },
  };
}
