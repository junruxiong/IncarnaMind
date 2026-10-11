import type { ChatModelChoice, ProviderError } from "../api";

/** An in-app folder. The legacy Group API name keeps saved assignments compatible. */
export interface LibraryGroup {
  id: string;
  name: string;
  description: string;
  /**
   * Checks count the files IncarnaMind wrote in this Folder as evidence (#80):
   * the User switched their exclusion off here. False by default.
   */
  countsOwnFiles: boolean;
  createdAt: string;
  updatedAt: string;
}

/** Explicit selection: classification never falls back to another provider. */
export type LibraryClassifier =
  | { kind: "chat"; choice: ChatModelChoice }
  | { kind: "jev" }
  | { kind: "auto"; baseUrl: string }
  | { kind: "ollama"; baseUrl: string; modelId: string; usePageImages?: boolean };

export interface ClassificationModel {
  id: string;
  images: boolean;
  reason: "text" | "visual" | "slow" | "memory" | "unavailable" | "selected";
}

export interface LibrarySettings {
  classifier: LibraryClassifier | null;
  /**
   * Organize new and changed Documents by itself, once a classifier is chosen.
   * On until the User turns it off.
   */
  automatic: boolean;
}

export type ClassificationStatus = "pending" | "classifying" | "classified" | "waiting" | "failed";

export interface DocumentGroupAssignment {
  documentId: string;
  groupId: string | null;
  source: "automatic" | "user";
  /**
   * `groupId` is a Suggested Folder: Organize proposed it and it waits for the
   * User. Meanwhile it counts as the Document's Folder, in listings and in
   * that Folder's Search scope. Always with source "automatic".
   */
  suggested: boolean;
  status: ClassificationStatus;
  error: ProviderError | null;
  model: ClassificationModel | null;
}

/** One Document's assignment as `acceptAssignments` or `rejectAssignments` changed it. */
export interface AssignmentChange {
  /** As it was: what undo puts back. */
  before: DocumentGroupAssignment;
  after: DocumentGroupAssignment;
}

/** Documents added together, by one `addDocuments` or `addLinkedFolder`. */
export interface DocumentBatch {
  id: string;
  /**
   * The Folder they were added to, where each is filed as the User's choice
   * and Organize suggests only Tags. Null: Organize suggests their Folders.
   */
  groupId: string | null;
  /** The Documents it added, in order, deleted ones left out. Files already added aren't in it. */
  documentIds: string[];
  /** More may join: the files are still being added, or a Linked folder's first scan is running. */
  adding: boolean;
  /** How many have been read: their text extracted, or found to have none, or failed. */
  read: number;
  /** How many Organize has finished with: a Folder suggested or none, and their Tags. */
  sorted: number;
  /** How many wait in each Suggested Folder, most first. */
  suggested: { groupId: string; count: number }[];
}

export interface LibrarySnapshot {
  groups: LibraryGroup[];
  /**
   * The Folders deleted since, by name order: a Search scope that names one
   * still shows its name, struck through (the search leaves it out).
   */
  deletedGroups: LibraryGroup[];
  assignments: DocumentGroupAssignment[];
  settings: LibrarySettings;
}

export interface LibraryGroupInput {
  name: string;
  description: string;
  /** See `LibraryGroup.countsOwnFiles`. Left out, it stays as it was (false for a new Folder). */
  countsOwnFiles?: boolean;
}
