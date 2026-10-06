/**
 * The core's public interface: the only API the UI uses.
 *
 * The preload script exposes it to the renderer over IPC, and the tests drive it
 * directly. Everything that crosses it is plain data, so it survives IPC.
 *
 * This file must stay free of imports with side effects: the preload script and
 * the renderer import it.
 */
import type { Language, LanguagePreference } from "./language";

export interface Mind {
  /** A random UUID generated on this device. */
  id: string;
  /** May be empty: the UI shows an "Untitled" placeholder. */
  title: string;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface CreateMindInput {
  title?: string;
}

/**
 * The name of the Y.XmlFragment that holds a Mind's Blocks in its Yjs document
 * (ADR-0003). The editor binds to it, and the core reads it.
 */
export const MIND_CONTENT_FIELD = "blocks";

/** A Mind opened for editing. */
export interface OpenedMind {
  mind: Mind;
  /** The Mind's whole Yjs document, encoded as one update: apply it to an empty `Y.Doc`. */
  state: Uint8Array;
}

/** A change to one Mind's Yjs document. */
export interface MindUpdate {
  mindId: string;
  /** A Yjs update (the default v1 encoding). */
  update: Uint8Array;
}

/** Settings that belong to the User and will sync across their devices (ADR-0003). */
export interface UserSettings {
  language: LanguagePreference;
}

/** Settings that belong to this device and never sync (ADR-0003). */
export interface DeviceSettings {
  /** Width of the left sidebar, in CSS pixels. */
  sidebarWidth: number;
  /** Width of the right Document viewer pane, in CSS pixels. */
  viewerWidth: number;
}

export interface Settings {
  user: UserSettings;
  device: DeviceSettings;
  /** The interface language in effect: the User's choice, or the OS language for "system". */
  language: Language;
}

export interface SettingsPatch {
  user?: Partial<UserSettings>;
  device?: Partial<DeviceSettings>;
}

/** The kinds of file that can be added as Documents. */
export type DocumentKind = "pdf" | "text" | "markdown";

/**
 * Where a Document is in processing: "queued", then "extracting", then "ready".
 * The other end states are "failed" (see `failure`) and "no-text": the file has
 * no text to extract, e.g. a scan without a text layer.
 */
export type DocumentStatus = "queued" | "extracting" | "ready" | "failed" | "no-text";

export type DocumentFailureReason =
  /** The file isn't a valid PDF, or a text file holds binary data. */
  | "unreadable"
  | "password-protected"
  /** The copy in the data folder has gone. */
  | "file-missing"
  /** Anything else, e.g. the processing worker crashed. */
  | "processing-error";

export interface DocumentFailure {
  reason: DocumentFailureReason;
  /** Technical detail in English, for logs and tooltips. */
  message: string;
}

export interface Document {
  /** A random UUID generated on this device. */
  id: string;
  /** The display name. It starts as the file name without its extension and can be renamed. */
  name: string;
  kind: DocumentKind;
  /**
   * SHA-256 of the file's bytes, in hex. The same file always gives the same hash,
   * so it is recognised as one Document (ADR-0003). It also names the stored copy.
   */
  contentHash: string;
  /** In bytes. */
  size: number;
  /** PDFs only, once their text has been extracted. */
  pageCount: number | null;
  status: DocumentStatus;
  /** Set when `status` is "failed". */
  failure: DocumentFailure | null;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface SkippedFile {
  path: string;
  /** "unsupported-type": not PDF, TXT or Markdown. "unreadable": missing, a folder, or not readable. */
  reason: "unsupported-type" | "unreadable";
}

export interface AddDocumentsResult {
  /** One per added file, in the order given. A file already added gives its existing Document. */
  documents: Document[];
  /** Files that were not added. */
  skipped: SkippedFile[];
}

export interface PassageSearchResult {
  passageId: string;
  documentId: string;
  documentName: string;
  /** The first page the Passage covers, from 1. Null for Documents without pages (TXT, Markdown). */
  pageFrom: number | null;
  /** The last page the Passage covers. A Passage can cross a page break. */
  pageTo: number | null;
  /** The Passage's position in its Document, from 0. */
  position: number;
  text: string;
}

export interface CoreApi {
  createMind(input?: CreateMindInput): Promise<Mind>;
  /** Minds that are not deleted, most recently updated first. Editing a Mind's content updates it. */
  listMinds(): Promise<Mind[]>;
  /** Changes a Mind's title (trimmed; empty means "Untitled") and returns the Mind. */
  renameMind(mindId: string, title: string): Promise<Mind>;
  /** Soft-deletes a Mind: it leaves the list, and its rows stay, marked deleted (ADR-0003). */
  deleteMind(mindId: string): Promise<void>;
  /**
   * Returns a Mind's content for editing. A client applies `state` to an empty
   * `Y.Doc`, sends its own changes with `applyMindUpdate`, and applies every
   * `"mind.update"` event for the Mind (these include its own changes, which Yjs ignores).
   */
  openMind(mindId: string): Promise<OpenedMind>;
  /** Applies a client's Yjs update to the Mind, stores it, and pushes it to every client as `"mind.update"`. */
  applyMindUpdate(mindId: string, update: Uint8Array): Promise<void>;
  /**
   * Tells the core a client stopped editing the Mind, so it can compact the
   * Mind's stored updates and free its memory. Editing the Mind again is fine.
   */
  closeMind(mindId: string): Promise<void>;
  getSettings(): Promise<Settings>;
  /** Changes only the fields given and returns the settings now in effect. */
  updateSettings(patch: SettingsPatch): Promise<Settings>;
  /**
   * Adds PDF, TXT and Markdown files, given their absolute paths. Each file is
   * copied into the data folder and queued for processing; "document.status"
   * events report its progress.
   */
  addDocuments(paths: string[]): Promise<AddDocumentsResult>;
  /** Documents that are not deleted, most recently added first. */
  listDocuments(): Promise<Document[]>;
  /** Returns the renamed Document. */
  renameDocument(id: string, name: string): Promise<Document>;
  /**
   * Soft-deletes a Document and its Passages, so search ignores them. The stored
   * file is removed once no Document uses it.
   */
  deleteDocument(id: string): Promise<void>;
  /** Keyword search over the Passages of all Documents, best match first. `limit` defaults to 20. */
  searchPassages(query: string, limit?: number): Promise<PassageSearchResult[]>;
}

/**
 * Events the core pushes to the UI, by name, with their payloads. Tickets that
 * need to push something (processing progress, Answer streams) add their
 * events here. Payloads are plain data (Uint8Array included), so they survive IPC.
 */
export interface CoreEvents {
  /** The settings in effect changed, e.g. the interface language. */
  "settings.changed": Settings;
  /** Minds were created, renamed, deleted or edited: the list as `listMinds` now returns it. */
  "minds.changed": Mind[];
  /** A Mind's content changed. Clients editing that Mind apply the update to their `Y.Doc`. */
  "mind.update": MindUpdate;
  /** A Document was added or its processing status changed. Carries the whole Document. */
  "document.status": Document;
}

export type CoreEventName = keyof CoreEvents;

export type CoreEventListener<E extends CoreEventName> = (payload: CoreEvents[E]) => void;

/** Stops a listener. Safe to call more than once. */
export type Unsubscribe = () => void;

/** The push half of the core's public interface. */
export interface CoreEventSource {
  on<E extends CoreEventName>(event: E, listener: CoreEventListener<E>): Unsubscribe;
}

/** What the renderer gets on `window.incarnamind`: every method plus events. */
export type CoreBridge = CoreApi & CoreEventSource;

export type CoreApiMethod = keyof CoreApi;

// A Record over every method name: adding a method to CoreApi without listing it here fails the type-check.
const methods: Record<CoreApiMethod, true> = {
  createMind: true,
  listMinds: true,
  renameMind: true,
  deleteMind: true,
  openMind: true,
  applyMindUpdate: true,
  closeMind: true,
  getSettings: true,
  updateSettings: true,
  addDocuments: true,
  listDocuments: true,
  renameDocument: true,
  deleteDocument: true,
  searchPassages: true,
};

/** Every method of CoreApi, used to wire the IPC bridge. */
export const coreApiMethods = Object.keys(methods) as CoreApiMethod[];
