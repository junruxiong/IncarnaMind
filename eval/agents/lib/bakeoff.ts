/**
 * The bake-off's manifest, eval/agents/bakeoff.json: which Documents go in
 * which Folder, the ten tasks with their expected sources, the prompt each
 * contestant gets and the canaries.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type EvalDocument, loadEvaluationSet } from "../../lib/evaluationSet";

export type FolderKey = "in-scope" | "private";

export interface BakeoffDocument extends EvalDocument {
  folder: FolderKey;
}

export interface ExpectedSource {
  /** A Document key: of an evaluation set, or of the manifest's own Documents. */
  document: string;
  /** The Units (pages) the cited Passage must cover; null for any. */
  pages: [number, number] | null;
  quote: string;
}

export interface Task {
  id: string;
  kind: string;
  language: "en" | "zh";
  /** The evaluation set's Question it is taken from, if any. */
  from?: string;
  /** The words put into the prompt's `{task}`. */
  task: string;
  /** The expected answer, for the judge. */
  answer: string;
  /** "any" (default): one expected source cited is a hit; "all": every one must be. */
  expectedRule?: "any" | "all";
  expected: ExpectedSource[];
  /** The Library can't answer it: a Citation is an invented one. */
  unanswerable?: boolean;
  /** Its answer is only outside the Search scope (the scope canary). */
  outOfScope?: boolean;
  /** Its best source is the injection Document. */
  injection?: boolean;
  /** Citations the text should have at least (a cite-heavy paragraph). */
  minCitations?: number;
}

export interface Bakeoff {
  folders: Record<FolderKey, string>;
  documents: BakeoffDocument[];
  prompts: Record<"en" | "zh", string>;
  canaries: {
    scope: string[];
    injection: { port: number; file: string; search: string; word: string };
  };
  tasks: Task[];
}

export const MANIFEST = "eval/agents/bakeoff.json";

interface RawManifest {
  folders: Record<FolderKey, string>;
  library: {
    sets: { set: string; folder: FolderKey }[];
    documents: Record<string, { path: string; folder: FolderKey }>;
  };
  prompts: Record<"en" | "zh", string>;
  canaries: Bakeoff["canaries"];
  tasks: Task[];
}

export function loadBakeoff(root: string): Bakeoff {
  const raw = JSON.parse(readFileSync(join(root, MANIFEST), "utf8")) as RawManifest;
  const documents: BakeoffDocument[] = [];
  for (const { set, folder } of raw.library.sets) {
    for (const document of loadEvaluationSet(root, set).documents) {
      documents.push({ ...document, folder });
    }
  }
  for (const [key, { path, folder }] of Object.entries(raw.library.documents)) {
    const absolute = join(root, path);
    if (!existsSync(absolute)) throw new Error(`${MANIFEST}: ${key}: ${path} doesn't exist.`);
    documents.push({ key, path: absolute, folder });
  }
  const keys = new Set<string>();
  for (const document of documents) {
    if (keys.has(document.key)) throw new Error(`${MANIFEST}: ${document.key} is listed twice.`);
    keys.add(document.key);
  }
  for (const task of raw.tasks) {
    for (const expected of task.expected) {
      if (!keys.has(expected.document)) {
        throw new Error(
          `${MANIFEST}: ${task.id} expects ${expected.document}, which isn't in the Library.`,
        );
      }
    }
  }
  return {
    folders: raw.folders,
    documents,
    prompts: raw.prompts,
    canaries: raw.canaries,
    tasks: raw.tasks,
  };
}

/** The words a contestant gets for a task. */
export function promptFor(bakeoff: Bakeoff, task: Task): string {
  return bakeoff.prompts[task.language].replace("{task}", task.task);
}
