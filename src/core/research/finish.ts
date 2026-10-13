/**
 * How a Research Task ends (#130), once its steps are done: the web pages it
 * saved are filed into a new Folder named for the research, with Tags its
 * model chose; then it writes a cited draft into the Mind it was told to, as
 * changes to accept. Each is a step of its journal (`save`, `draft`); what
 * each makes (the Folder, a new Mind) is kept as a record (`~folder`,
 * `~mind`), so going on after a quit uses it rather than making another.
 */
import { generateText } from "ai";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import { cutTitle } from "../../shared/naming";
import type { CellAnswer } from "../answers";
import type { NodeJSON } from "../answers/blocks";
import type { QuestionContext } from "../answers/context";
import type { ChatModelChoice, ProposalRequest, ResearchPlan, ResearchTarget } from "../api";
import { isRecord } from "../errors";
import type { PreparedChatModel } from "../providers/chat";
import { RECORD_PREFIX } from "../runs/journal";
import type { TaskEnd, TaskRun } from "../runs/tasks";
import { draftContext, readTags, TAG_INSTRUCTIONS, tagPrompt } from "./prompt";

/** A Tag request that takes longer than this fails: the pages are filed without Tags. */
const TAG_TIMEOUT_MS = 60_000;

export const SAVE_STEP = "save";
export const DRAFT_STEP = "draft";
const FOLDER_RECORD = `${RECORD_PREFIX}folder`;
const MIND_RECORD = `${RECORD_PREFIX}mind`;

export interface FinishOptions {
  mindExists(mindId: string): boolean;
  /** The Folder a Mind is in; null for none. */
  folderOfMind(mindId: string): string | null;
  /** Makes a Mind in a Folder (or none), as New Mind does; returns its id. */
  createMind(title: string, folderId: string | null): string;
  documents: {
    /** Of these Documents, the live web pages added since `since` (ISO 8601), with their names. */
    webPagesSince(ids: readonly string[], since: string): { id: string; name: string }[];
    /** Of these Documents, those still there. */
    live(ids: readonly string[]): string[];
  };
  library: {
    folderExists(folderId: string): boolean;
    folderNames(): string[];
    createFolder(name: string, description: string): string;
    /** Files Documents into a Folder as the User's choice. */
    file(documentIds: readonly string[], folderId: string): void;
  };
  tags: {
    list(): { id: string; name: string }[];
    create(name: string, description: string): string;
    /** Adds Tags to a Document as IncarnaMind's, beside those it has. */
    apply(documentId: string, tagIds: readonly string[]): void;
  };
  notes: {
    propose(
      mindId: string,
      blocks: readonly NodeJSON[],
      request: string,
      made: ProposalRequest,
    ): Promise<{ proposalId: string; changes: number }>;
  };
  say(key: MessageKey, params?: MessageParams): string;
  now(): Date;
  reportError(error: unknown): void;
}

/** What a finished `save` step says: where the pages went, and how many. */
export interface SavedSources {
  folderId: string;
  folder: string;
  documents: number;
  tags: string[];
}

/** What a finished `draft` step says: where the changes are, and how many. */
export interface WrittenDraft {
  mindId: string;
  proposalId: string;
  changes: number;
}

export function savedOf(value: unknown): SavedSources | null {
  if (!isRecord(value) || typeof value.folder !== "string" || typeof value.documents !== "number") {
    return null;
  }
  return {
    folderId: typeof value.folderId === "string" ? value.folderId : "",
    folder: value.folder,
    documents: value.documents,
    tags: Array.isArray(value.tags) ? value.tags.filter((tag) => typeof tag === "string") : [],
  };
}

export function writtenOf(value: unknown): WrittenDraft | null {
  if (
    !isRecord(value) ||
    typeof value.mindId !== "string" ||
    typeof value.proposalId !== "string" ||
    typeof value.changes !== "number"
  ) {
    return null;
  }
  return { mindId: value.mindId, proposalId: value.proposalId, changes: value.changes };
}

/** A Folder's name no other Folder has, ignoring case: the name, else "name 2", "name 3"… */
function freeName(name: string, taken: readonly string[]): string {
  const names = new Set(taken.map((each) => each.toLocaleLowerCase()));
  if (!names.has(name.toLocaleLowerCase())) return name;
  for (let number = 2; ; number++) {
    const next = `${name} ${number}`;
    if (!names.has(next.toLocaleLowerCase())) return next;
  }
}

const idOf = (value: unknown, field: string): string | null =>
  isRecord(value) && typeof value[field] === "string" ? (value[field] as string) : null;

/** The Folder an earlier try made, if it is still there; else a new one, named for the research. */
function folderFor(options: FinishOptions, task: TaskRun, title: string, topic: string) {
  const earlier = idOf(task.step(FOLDER_RECORD)?.result, "folderId");
  const name = idOf(task.step(FOLDER_RECORD)?.result, "name");
  if (earlier && name && options.library.folderExists(earlier)) return { id: earlier, name };
  const chosen = freeName(
    cutTitle(title) || options.say("research.folder.untitled"),
    options.library.folderNames(),
  );
  const id = options.library.createFolder(
    chosen,
    options.say("research.folder.description", { topic: cutTitle(topic) }),
  );
  task.started(FOLDER_RECORD);
  task.settled(FOLDER_RECORD, "completed", { folderId: id, name: chosen });
  return { id, name: chosen };
}

/** Tags for the pages, as the research's model chooses them, the User's own reused by name. */
async function tagPages(
  options: FinishOptions,
  prepared: PreparedChatModel,
  title: string,
  pages: readonly { id: string; name: string }[],
  signal: AbortSignal,
): Promise<string[]> {
  const existing = options.tags.list();
  let reply: string;
  try {
    const result = await generateText({
      model: prepared.model,
      instructions: TAG_INSTRUCTIONS,
      prompt: tagPrompt(
        title,
        pages.map((page) => page.name),
        existing.map((tag) => tag.name),
      ),
      maxOutputTokens: 256,
      maxRetries: 1,
      abortSignal: AbortSignal.any([signal, AbortSignal.timeout(TAG_TIMEOUT_MS)]),
    });
    reply = result.text;
  } catch (error) {
    if (!signal.aborted) options.reportError(error);
    return [];
  }
  const applied: string[] = [];
  for (const tag of readTags(reply, pages.length)) {
    const known = existing.find(
      (each) => each.name.toLocaleLowerCase() === tag.name.toLocaleLowerCase(),
    );
    let tagId: string;
    try {
      tagId =
        known?.id ??
        options.tags.create(tag.name, options.say("research.tag.description", { title }));
    } catch (error) {
      options.reportError(error);
      continue;
    }
    for (const number of tag.pages) {
      const page = pages[number - 1];
      if (page) options.tags.apply(page.id, [tagId]);
    }
    applied.push(known?.name ?? tag.name);
  }
  return applied;
}

/**
 * Files the web pages the Task saved into a new Folder with Tags: the `save`
 * step. Pages the User had saved before it started stay where they are.
 */
export async function fileSources(
  options: FinishOptions,
  task: TaskRun,
  input: { topic: string; plan: ResearchPlan },
  documents: readonly string[],
  prepared: PreparedChatModel,
): Promise<SavedSources> {
  task.started(SAVE_STEP);
  const pages = options.documents.webPagesSince(documents, task.startedAt);
  if (pages.length === 0) {
    const none: SavedSources = { folderId: "", folder: "", documents: 0, tags: [] };
    task.settled(SAVE_STEP, "completed", none);
    return none;
  }
  const folder = folderFor(options, task, input.plan.title, input.topic);
  options.library.file(
    pages.map((page) => page.id),
    folder.id,
  );
  const tags = await tagPages(options, prepared, input.plan.title, pages, task.signal);
  const saved: SavedSources = {
    folderId: folder.id,
    folder: folder.name,
    documents: pages.length,
    tags,
  };
  task.settled(SAVE_STEP, "completed", saved);
  return saved;
}

/** The Mind the draft goes into: the one chosen, else the new one an earlier try made, else a new one. */
export function draftMind(
  options: FinishOptions,
  task: TaskRun,
  target: ResearchTarget,
  title: string,
  folderId: string | null,
): string {
  if ("mindId" in target && options.mindExists(target.mindId)) return target.mindId;
  const earlier = idOf(task.step(MIND_RECORD)?.result, "mindId");
  if (earlier && options.mindExists(earlier)) return earlier;
  const home = folderId ?? (task.mindId ? options.folderOfMind(task.mindId) : null);
  const mindId = options.createMind(cutTitle(title), home);
  task.started(MIND_RECORD);
  task.settled(MIND_RECORD, "completed", { mindId });
  return mindId;
}

/** What writing the draft needs of the Task's Answers. */
export type AnswerStep = (
  run: (signal: AbortSignal) => Promise<CellAnswer>,
) => Promise<{ answer: CellAnswer } | { failed: string } | { end: TaskEnd }>;

/**
 * Writes the cited draft from the Documents the steps used, as changes to
 * accept in the Mind: the `draft` step. A quit after the changes went in
 * finds them there and proposes nothing again.
 */
export async function writeDraft(
  options: FinishOptions,
  task: TaskRun,
  input: { topic: string; plan: ResearchPlan; draftRequest: string; writeInto: ResearchTarget },
  found: { notes: readonly string[]; documents: readonly string[]; folderId: string | null },
  model: ChatModelChoice,
  answer: AnswerStep,
  answerCell: (input: {
    context: QuestionContext;
    documentIds: string[];
    model: ChatModelChoice;
    signal: AbortSignal;
  }) => Promise<CellAnswer>,
): Promise<{ written: WrittenDraft } | { failed: string } | { end: TaskEnd }> {
  task.started(DRAFT_STEP);
  const documentIds = options.documents.live(found.documents);
  if (documentIds.length === 0) {
    task.settled(DRAFT_STEP, "failed", { problem: "no-sources" });
    return { failed: "no-sources" };
  }
  const result = await answer((signal) =>
    answerCell({
      context: draftContext({ topic: input.topic, plan: input.plan, notes: found.notes }),
      documentIds,
      model,
      signal,
    }),
  );
  if (!("answer" in result)) {
    if ("failed" in result) task.settled(DRAFT_STEP, "failed", { problem: result.failed });
    return result;
  }
  if (result.answer.blocks.length === 0) {
    task.settled(DRAFT_STEP, "failed", { problem: "no-answer" });
    return { failed: "no-answer" };
  }
  const mindId = draftMind(options, task, input.writeInto, input.plan.title, found.folderId);
  const proposed = await options.notes.propose(mindId, result.answer.blocks, input.draftRequest, {
    source: "research",
    taskId: task.id,
    providerId: model.providerId,
    modelId: model.modelId,
    createdAt: options.now().toISOString(),
  });
  const written: WrittenDraft = { mindId, ...proposed };
  task.settled(DRAFT_STEP, "completed", written);
  return { written };
}
