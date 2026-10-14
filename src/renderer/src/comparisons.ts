/**
 * Compare Documents (#98), the window's side: the start sheet each Mind may
 * have open (its Documents, Questions and what else the table should show,
 * kept until it is made or closed), the comparison tables being filled, the
 * one table filling the card (Expand), and why a table couldn't be made or
 * filled, by Mind or table, until the next try.
 */
import { create } from "zustand";
import type { ComparisonResult, ComparisonTableRef, Document } from "../../core/api";
import { MAX_COMPARISON_DOCUMENTS } from "../../shared/comparisons";
import { leftOutOfEvidence } from "../../shared/evidence";
import { core } from "./core";
import { currentMindIdOf } from "./layoutSlice";
import { useAppStore } from "./store";

/** Where a sheet takes its Documents from: a Folder, or every Document ("all"). */
export type SheetSource = string | "all";

export interface CompareSheet {
  /** The Block the table goes after (where the cursor was); null: the Mind's end. */
  afterBlockId: string | null;
  source: SheetSource;
  /** The Documents ticked, by id. */
  checked: readonly string[];
  questions: readonly string[];
  instruction: string;
  /** Asking the core: the button waits, and a second press does nothing. */
  making: boolean;
}

/** Why nothing was made or filled: the model can't be used yet, or the User didn't allow sending to it. */
export type StartProblem = Exclude<ComparisonResult, { started: true }>;

interface ComparisonsStore {
  /** The open start sheets, by Mind. */
  sheets: Readonly<Record<string, CompareSheet>>;
  /** The tables whose cells are being filled, by id. */
  filling: ReadonlySet<string>;
  /** The table filling the card, if any. */
  expanded: string | null;
  /** Why a sheet's table, or a table's cells, couldn't be started: by Mind, or by table. */
  problems: Readonly<Record<string, StartProblem>>;
}

export const useComparisons = create<ComparisonsStore>()(() => ({
  sheets: {},
  filling: new Set(),
  expanded: null,
  problems: {},
}));

const report = (error: unknown) => useAppStore.getState().reportError(error);

const setProblem = (key: string, problem: StartProblem | null) =>
  useComparisons.setState((state) => {
    const { [key]: _old, ...rest } = state.problems;
    return { problems: problem ? { ...rest, [key]: problem } : rest };
  });

/** Opens a Mind's start sheet, its Documents those of `source` ticked, or keeps the one open. */
export function openCompareSheet(
  mindId: string,
  {
    afterBlockId,
    source,
    documentIds,
  }: { afterBlockId: string | null; source: SheetSource; documentIds: readonly string[] },
): void {
  useComparisons.setState((state) => {
    const open = state.sheets[mindId];
    const sheet: CompareSheet = open
      ? { ...open, afterBlockId: afterBlockId ?? open.afterBlockId }
      : {
          afterBlockId,
          source,
          checked: documentIds,
          questions: [],
          instruction: "",
          making: false,
        };
    return { sheets: { ...state.sheets, [mindId]: sheet } };
  });
  setProblem(mindId, null);
}

/** The Documents a sheet offers: those of its Folder (or every one), less those checks leave out (#80). */
export function sheetDocuments(source: SheetSource): Document[] {
  const { documents, library, settings } = useAppStore.getState();
  const folderOf = new Map(library?.assignments.map((each) => [each.documentId, each.groupId]));
  return documents.filter(
    (document) =>
      (source === "all" || folderOf.get(document.id) === source) &&
      !leftOutOfEvidence(document, settings, library),
  );
}

/**
 * Compare Documents… (⌘K, "/", the + menu): opens the sheet in `mindId`, or
 * in the Mind in front, or in a new Mind when none is, its table to go after
 * `afterBlockId`. It starts from the Mind's Folder, every Document ticked;
 * from All Documents for a Mind in none, ticked only when there are few.
 */
export async function startCompareDocuments(
  mindId: string | null,
  afterBlockId: string | null,
): Promise<void> {
  const store = useAppStore.getState();
  let target = mindId ?? currentMindIdOf(store.layout);
  if (!target) {
    await store.createMind();
    target = currentMindIdOf(useAppStore.getState().layout);
  }
  if (!target) return;
  const folderId = useAppStore.getState().minds.find((mind) => mind.id === target)?.folderId;
  const source: SheetSource = folderId ?? "all";
  const offered = sheetDocuments(source);
  const ticked = source !== "all" || offered.length <= MAX_COMPARISON_DOCUMENTS ? offered : [];
  openCompareSheet(target, {
    afterBlockId,
    source,
    documentIds: ticked.slice(0, MAX_COMPARISON_DOCUMENTS).map((document) => document.id),
  });
}

export function updateCompareSheet(mindId: string, change: Partial<CompareSheet>): void {
  useComparisons.setState((state) => {
    const open = state.sheets[mindId];
    return open ? { sheets: { ...state.sheets, [mindId]: { ...open, ...change } } } : {};
  });
}

export function closeCompareSheet(mindId: string): void {
  useComparisons.setState((state) => {
    const { [mindId]: _closed, ...sheets } = state.sheets;
    return { sheets };
  });
  setProblem(mindId, null);
}

/**
 * Makes the table the sheet describes, its rows `documentIds` in the order
 * the sheet lists them: it goes into the Mind as a change to accept and
 * starts filling, and the sheet closes. Resolves with the table's id, or null
 * when it wasn't made (the sheet then says why).
 */
export async function makeComparisonTable(
  mindId: string,
  documentIds: readonly string[],
  model: { providerId: string; modelId: string } | null,
): Promise<string | null> {
  const sheet = useComparisons.getState().sheets[mindId];
  if (!sheet || sheet.making) return null;
  updateCompareSheet(mindId, { making: true });
  setProblem(mindId, null);
  try {
    const result = await core.compareDocuments({
      mindId,
      documentIds: [...documentIds],
      questions: sheet.questions.map((question) => question.trim()).filter(Boolean),
      instruction: sheet.instruction.trim() || null,
      model,
      afterBlockId: sheet.afterBlockId,
    });
    if (!result.started) {
      setProblem(mindId, result);
      return null;
    }
    closeCompareSheet(mindId);
    return result.tableId;
  } catch (error) {
    report(error);
    return null;
  } finally {
    updateCompareSheet(mindId, { making: false });
  }
}

/** Fills a table's cells that have no Answer: a column just added, or those a fill left. */
export async function fillComparisonTable(ref: ComparisonTableRef): Promise<void> {
  setProblem(ref.tableId, null);
  try {
    const result = await core.fillComparisonTable(ref);
    if (!result.started) setProblem(ref.tableId, result);
  } catch (error) {
    report(error);
  }
}

export function stopComparisonTable(ref: ComparisonTableRef): void {
  core.stopComparisonTable(ref).catch(report);
}

export const expandComparisonTable = (tableId: string | null) =>
  useComparisons.setState({ expanded: tableId });

const setFilling = (tableId: string, filling: boolean) =>
  useComparisons.setState((state) => {
    const next = new Set(state.filling);
    if (filling) next.add(tableId);
    else next.delete(tableId);
    return { filling: next };
  });

core.on("comparison.filling", ({ tableId, filling }) => setFilling(tableId, filling));

let fillingAsked = false;

/** Once, when the window first draws a table: those a fill was already going on when it opened. */
export function loadFillingComparisons(): void {
  if (fillingAsked) return;
  fillingAsked = true;
  core
    .listFillingComparisons()
    .then((tables) => {
      for (const table of tables) setFilling(table.tableId, true);
    })
    .catch(report);
}
