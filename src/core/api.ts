/**
 * The core's public interface: the only API the UI uses.
 *
 * The preload script exposes it to the renderer over IPC, and the tests drive it
 * directly. Everything that crosses it is plain data, so it survives IPC.
 *
 * This file must stay free of imports with side effects: the preload script and
 * the renderer import it.
 */
import type { UnitKind, UnitLabel } from "../shared/units";
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

/** The attribute every Block keeps its UUID in, in the Mind's Yjs document. */
export const BLOCK_ID_ATTRIBUTE = "id";

/**
 * The node types of a Note: any top-level Block that isn't a Question or an
 * Answer. Answers hold the same types, nested.
 */
export const NOTE_BLOCK_TYPES = [
  "paragraph",
  "heading",
  "codeBlock",
  "blockMath",
  "bulletList",
  "orderedList",
  "blockquote",
  "horizontalRule",
] as const;

/** The node type of a Question Block. Its content is the User's text (inline). */
export const QUESTION_BLOCK = "question";

/** The node type of an Answer Block. Its content is rich text (Note node types), which the User can edit. */
export const ANSWER_BLOCK = "answer";

/**
 * A Note's "include in Question context" flag. Absent (or null) means on, the
 * default; `false` means the User switched the Note off, so Questions don't see it.
 */
export const INCLUDE_IN_CONTEXT_ATTRIBUTE = "includeInContext";

/**
 * A Question's Search scope (CONTEXT.md): the Folders, Tags and individual
 * Documents its Document search is limited to. It covers every Document in
 * one of the Folders or their sub-Folders (at any depth), every Document with
 * one of the Tags, and each of the Documents. Those deleted since are ignored.
 * With all three empty there is no Search scope: every Document is searched.
 */
export interface SearchScope {
  folderIds: string[];
  tagIds: string[];
  documentIds: string[];
}

/**
 * Attributes of a Question Block, as stored in the Mind's Yjs document.
 */
export interface QuestionAttributes {
  id: string | null;
  /** The model picked for this Question, overriding the default. Both null: the default model. */
  providerId: string | null;
  modelId: string | null;
  /**
   * Its Search scope (see `SearchScope`), as lists of ids: the User types "@"
   * in the Question to choose them. Null (or empty) for none of that kind;
   * all three null: no Search scope.
   */
  scopeFolderIds: string[] | null;
  scopeTagIds: string[] | null;
  scopeDocumentIds: string[] | null;
  /**
   * The name of the Skill the User forced on this Question from the slash
   * menu: its instructions are loaded up front. Null: the model picks Skills
   * itself. A name rather than an id, so a Skill removed and imported again
   * still matches.
   */
  forcedSkill: string | null;
}

/** Where an Answer is: being written, finished, stopped by the User, or failed (see `errorKind`). */
export type AnswerStatus = "streaming" | "done" | "stopped" | "failed";

/**
 * How the model behind an Answer gives the records of its Citations:
 * - "tools": through the `cite` Tool, in a Tool-calling loop that searches the Documents;
 * - "structured-output": the model can't call Tools, so the core searches once with the
 *   Question's text and the model returns the Answer and its records as one JSON object;
 * - "none": the model can do neither, so its Answers have no Citations, and say so.
 */
export type CitationSupport = "tools" | "structured-output" | "none";

/** Attributes of an Answer Block, as stored in the Mind's Yjs document. */
export interface AnswerAttributes {
  id: string | null;
  /** The Question it answers. */
  questionId: string | null;
  /** The model that wrote it. */
  providerId: string | null;
  modelId: string | null;
  status: AnswerStatus;
  /** Why it failed, when `status` is "failed". */
  errorKind: ProviderErrorKind | null;
  /** The provider's own message about the failure, for details. */
  errorMessage: string | null;
  /** A fingerprint of the content as it was generated: it differs once the User edits the Answer. */
  generatedHash: string | null;
  /**
   * How the model could give Citations (see `CitationSupport`). Null until the
   * model has answered, when there were no Documents to search, and on Answers
   * written before Citations.
   */
  citationSupport: CitationSupport | null;
  /** The Tools the Answer called, in order: a JSON array of `AnswerToolCall`. Null when none. */
  toolCalls: string | null;
}

/** A Tool the model called while writing an Answer, e.g. a search of the User's Documents. */
export interface AnswerToolCall {
  /** Unique within its Answer. */
  id: string;
  /**
   * "search_documents": the document-search Tool. "use_skill": loading a
   * Skill's instructions (`{ name }`). "read_skill_file": reading one of a
   * Skill's files (`{ skill, path }`). "run_skill_script": running one of a
   * Skill's scripts (`{ skill, script, args }`, see `script`). A Connector's
   * Tool: its name as the Connector gives it, e.g. "search_issues".
   */
  tool: string;
  /**
   * Where the Tool comes from: "documents" is IncarnaMind's own document
   * search; "skill", the Skill Tools; "connector", one of the User's
   * Connectors (see `connector`).
   */
  source: "documents" | "skill" | "connector";
  /** A Connector's Tool: which Connector, with its name when the call was made. */
  connector?: { id: string; name: string };
  /** What the model asked, e.g. `{ query }` for a search, or the arguments sent to a Connector. */
  input: Record<string, unknown>;
  /**
   * "failed" also covers a call the User didn't allow (see `approval`): it
   * never ran. A Skill script is "done" only when it ran and exited with 0.
   */
  status: "running" | "done" | "failed";
  /**
   * A call that asked the User first (see `ApprovalRequest`): a Connector's
   * Tool, or a Skill script. "waiting" while the Answer waits for them;
   * "allowed" once they allowed it (once, or always); "denied" when they
   * denied it, or the Answer stopped (or IncarnaMind closed) before they
   * decided, so it never ran. Absent when it didn't ask: its Connector says
   * it only reads, the User always allows the Tool, or always runs the Skill's scripts.
   */
  approval?: ToolCallApproval;
  /** A search: how many Passages it gave the model. Null otherwise, and while running. */
  resultCount: number | null;
  /**
   * "use_skill" only: the Question forced this Skill, so the core loaded it up
   * front instead of the model calling the Tool. Absent otherwise.
   */
  forced?: boolean;
  /**
   * Not a call: the note that a remote Connector waits for the User to sign
   * in again, so its Tools weren't offered to this Answer (`tool` is
   * "sign_in", `status` "failed"). Absent otherwise.
   */
  signInRequired?: boolean;
  /**
   * "run_skill_script" only: how the run went once it ended, or why the
   * script couldn't run. Absent while it runs, and when it never did (denied).
   */
  script?: SkillScriptRun;
}

/** Where the User's approval of a Tool call stands (see `AnswerToolCall.approval`). */
export type ToolCallApproval = "waiting" | "allowed" | "denied";

/**
 * A run of a Skill script, as its Tool-call card shows it: what the model was
 * given back. Each output is cut to `SKILL_SCRIPT_LIMITS.maxOutputBytes`: the
 * start of the standard output, the end of the error output.
 */
export interface SkillScriptRun {
  /** Its exit code; null when it couldn't start (see `error`), or was stopped. */
  exitCode: number | null;
  /** It ran longer than the timeout, so it was stopped with every process it started. */
  timedOut: boolean;
  stdout: string;
  stderr: string;
  /** It wrote more to its standard output than is kept. */
  stdoutTruncated: boolean;
  /** It wrote more to its error output than is kept. */
  stderrTruncated: boolean;
  /**
   * Why it didn't run, in plain language: e.g. its interpreter isn't
   * installed ("python3 not found: install Python 3."), it is a shell script
   * on Windows, or a kind of script IncarnaMind can't run. Null when it ran.
   */
  error: string | null;
}

/**
 * The node type of a Citation: an inline node anchored in the text of an
 * Answer (or of a Note it was copied into), so it moves with edits and goes
 * when its text is deleted.
 */
export const CITATION_NODE = "citation";

/**
 * Where a Citation's check stands. The check compares the model's quote with
 * the text of the cited Units (its Location: one or two pages, slides,
 * sections, blocks of rows or of lines; see src/shared/units.ts), once, when
 * the Answer finishes. "found" means the quote is there, not that it supports
 * the sentence.
 * - "checking": the Answer is still being written.
 * - "found", "not-found": see `checkReason` for why it wasn't found.
 * - "cant-check": there is no text to look in (see `checkReason`).
 */
export type CitationCheck = "checking" | "found" | "not-found" | "cant-check";

export type CitationCheckReason =
  /** Not found: the quote isn't in the text of the cited Units (pages, slides…). */
  | "quote-not-on-pages"
  /** Not found: the cited Units aren't all among the Units of the cited Passage. */
  | "pages-outside-passage"
  /** Not found: a Citation covers one Unit, or two consecutive ones of one sheet, at most. */
  | "too-many-pages"
  /** Can't check: the cited Units have no text, e.g. scanned pages. */
  | "no-text"
  /** Can't check: the Document was deleted. */
  | "document-removed";

/**
 * What to re-check a Citation with against its Document's current version
 * (see `CoreApi.recheckCitation`): its Document, cited Units and quote.
 */
export interface RecheckCitationInput {
  documentId: string;
  /** The cited Units (pages, slides…), from 1; null for a whole TXT or Markdown file. */
  pageFrom: number | null;
  pageTo: number | null;
  quote: string;
}

/**
 * A Citation checked again against its Document's current version: the
 * attributes to store on its node in place of the old ones. The quote is
 * looked for in the cited Units first, then in any Unit (or two consecutive
 * ones) of the current version, so `pageFrom`, `pageTo` and `location` may change.
 */
export interface CitationRecheck {
  check: Exclude<CitationCheck, "checking">;
  checkReason: CitationCheckReason | null;
  /** The version checked: the Document's current `contentHash`; null when it can't be checked. */
  contentHash: string | null;
  /** A Passage of the current version that holds the Units, or null. */
  passageId: string | null;
  pageFrom: number | null;
  pageTo: number | null;
  location: CitationLocation | null;
}

/**
 * Text kept of a Document that left the index with its Linked folder: the
 * Units of one version that Citations pointed to when the folder was
 * unlinked. Kept so the Citations that quote it can still be checked (see
 * `citationState` in src/shared/citations.ts), until no Citation quotes it.
 */
export interface KeptCitationText {
  documentId: string;
  /** The version it is of. */
  contentHash: string;
  /** The numbers of the Units kept (pages, slides…), from 1, in order. */
  units: number[];
}

/** At most this many consecutive Units (pages, slides…) per Citation (the Location rule). */
export const MAX_CITED_PAGES = 2;

/**
 * Where in a Document a Citation points, as its short label shows it
 * (ADR-0011), stored with it so the label still reads after the Document is
 * gone. Language-neutral: the label is worded when shown ("p. 4", "slide 4",
 * "Revenue, rows 12–14", "§ 2.1 Sensitivity", "lines 120–134").
 * - "page", "slide": the cited pages or slides.
 * - "rows": the rows the quote covers in a sheet (a CSV's has no name), or
 *   the cited block's when it wasn't found.
 * - "section": the heading of the section the quote sits under; null before
 *   the first heading. `notes`: a Word file's footnotes and endnotes.
 * - "lines": the lines the quote covers, or the cited block's.
 */
export type CitationLocation =
  | { kind: "page" | "slide"; from: number; to: number }
  | { kind: "rows"; sheet: string | null; from: number; to: number }
  | { kind: "lines"; from: number; to: number }
  | { kind: "section"; heading: string | null; notes?: boolean };

/**
 * A Citation's attributes, as stored on its node in the Mind's Yjs document.
 * Everything a footnote or a message needs is stored here, so it still reads
 * after the Document is deleted.
 */
export interface CitationAttributes {
  passageId: string | null;
  documentId: string | null;
  /** The Document's display name when the Citation was made. */
  documentName: string | null;
  /**
   * The version of the Document the Citation quotes: the SHA-256 of its file
   * as it was then (`Document.contentHash`). The check reads that version's
   * text, which is kept for as long as a Citation quotes it. When it differs
   * from the Document's `contentHash` now, the Document changed after it was
   * cited (see `citationState` in src/shared/citations.ts).
   */
  contentHash: string | null;
  /**
   * The cited Units, from 1: pages of a PDF, slides of a deck, and otherwise
   * the Units' numbers in the version quoted (see src/shared/units.ts). Both
   * null for a whole TXT or Markdown file, cited before Units.
   */
  pageFrom: number | null;
  pageTo: number | null;
  /**
   * Where the Citation points, for its label (see `CitationLocation`). Null
   * for Citations made before Locations: their label comes from the pages.
   */
  location: CitationLocation | null;
  /** The quote the model gave, which it was asked to copy word for word from the Passage. */
  quote: string | null;
  check: CitationCheck;
  checkReason: CitationCheckReason | null;
}

/** A Citation, as the Answer event stream reports it. */
export interface Citation {
  passageId: string;
  documentId: string;
  documentName: string;
  contentHash: string;
  pageFrom: number | null;
  pageTo: number | null;
  location: CitationLocation | null;
  quote: string;
  check: CitationCheck;
  checkReason: CitationCheckReason | null;
}

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
  /** The default model for Answers, or null before a chat provider is set up. */
  chatModel: ChatModelChoice | null;
}

/** Settings that belong to this device and never sync (ADR-0003). */
export interface DeviceSettings {
  /** Width of the left sidebar, in CSS pixels. */
  sidebarWidth: number;
  /** Width of the right Document viewer pane, in CSS pixels. */
  viewerWidth: number;
  /**
   * The Minds open as tabs in the Mind pane, by ID, in their order, so they
   * open again after a restart. IDs of Minds deleted since are ignored.
   */
  openMinds: string[];
  /** The tab shown, one of `openMinds`, or null when none is open. */
  activeMind: string | null;
  /** The User chose "set up later" on the first-run chat setup screen. */
  chatSetupDismissed: boolean;
  /**
   * Skill scripts may run on this device: on by default, and each run still
   * asks first unless the Skill's scripts always run. Off, Answers aren't
   * offered `run_skill_script` at all, scripts running are stopped, and runs
   * waiting for approval are denied.
   */
  skillScriptsEnabled: boolean;
  /**
   * How long a Skill script may run, in seconds, before it is stopped with
   * every process it started: from `SKILL_SCRIPT_LIMITS.minTimeoutSeconds` to
   * `maxTimeoutSeconds`, `defaultTimeoutSeconds` by default.
   */
  skillScriptTimeoutSeconds: number;
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

/**
 * The kinds of file that can be added as Documents (ADR-0011): PDF, plain
 * text, Markdown, Word (.docx), PowerPoint (.pptx), Excel (.xlsx) and CSV.
 */
export type DocumentKind = "pdf" | "text" | "markdown" | "docx" | "pptx" | "xlsx" | "csv";

/**
 * Where a Document is in processing: "queued", then "extracting" its text, then
 * "embedding" its Passages with the embedding model (the built-in one unless
 * the User chose another), then "ready". Documents are embedded one at a
 * time: one waiting its turn is "queued" again. Before the model has been
 * downloaded, or while the chosen provider can't be used, a Document waits
 * after extracting as "waiting-for-model", and carries on by itself once it
 * can; keyword search already finds its Passages. Switching the embedding
 * model takes every Document back through "embedding" (see `EmbeddingRebuild`).
 * The other end states are "failed" (see `failure`) and "no-text": the file
 * has no text to extract, e.g. a scan without a text layer. When a new
 * version of a file fails (a sync client wrote it half-way, it is corrupt),
 * the last good version stays indexed: its text is still searched and read,
 * `contentHash` is still its, and the Document is "failed" until a later
 * change of the file, or a retry, is read.
 */
export type DocumentStatus =
  | "queued"
  | "extracting"
  | "waiting-for-model"
  | "embedding"
  | "ready"
  | "failed"
  | "no-text";

export type DocumentFailureReason =
  /** The file isn't valid (a broken PDF or Office file), or a text file holds binary data. */
  | "unreadable"
  | "password-protected"
  /**
   * No longer given: a file that has gone is a "missing" Document (see
   * `DocumentFileStatus`), not a failed one. Kept so older data still reads.
   */
  | "file-missing"
  /** Anything else, e.g. the processing worker crashed. */
  | "processing-error";

export interface DocumentFailure {
  reason: DocumentFailureReason;
  /** Technical detail in English, for logs and tooltips. */
  message: string;
}

/**
 * Where automatic tagging is for a Document. It runs once the Document is
 * "ready", and never holds that up: a Document is searchable as soon as it is
 * embedded, tagged or not. The tagger is Jev when a Jev key is set up on this
 * device (see `getJevSettings`), otherwise the default chat model.
 * - "pending": tagged once processing finishes, or about to be.
 * - "waiting-for-provider": no tagger can be used yet (no chat model is set
 *   up, a key or sign-in is missing, or the User declined sending excerpts to
 *   the tagger's service). Tagging resumes by itself once one can.
 * - "tagging": the tagger is deciding the Document's Tags.
 * - "tagged": its automatic Tags are up to date.
 * - "failed": the tagger's provider failed (see `taggingError`); a re-tag, a
 *   change of model or a restart tries again.
 * - "skipped": the Document has no text to tag (processing failed, or found none).
 */
export type TaggingState =
  | "pending"
  | "waiting-for-provider"
  | "tagging"
  | "tagged"
  | "failed"
  | "skipped";

/** Who put a Tag on a Document: automatic tagging, or the User. */
export type TagSource = "automatic" | "user";

/** A Tag on a Document. */
export interface DocumentTag {
  tagId: string;
  /**
   * "automatic": automatic tagging applied it, and a re-tag may take it away.
   * "user": the User added it; automatic tagging never changes it.
   */
  source: TagSource;
  /**
   * How likely automatic tagging found it that the Tag applies, from 0 to 1,
   * when its tagger says (Jev does). Null for the chat model and for the User's Tags.
   */
  confidence: number | null;
  /**
   * Automatic tagging wasn't sure (Jev's probability fell in the review band),
   * so the Tag is applied and marked for the User to check: `addDocumentTag`
   * confirms it, making it theirs, and `removeDocumentTag` takes it off.
   */
  needsReview: boolean;
}

/**
 * Where a Document's file is (ADR-0010), separately from `status`:
 * - "available": at its path, as last seen.
 * - "missing": gone from its path, and no file elsewhere matched it as moved.
 *   Its Linked folder (or, for a file added on its own, the folder it was in)
 *   is still there. It keeps its text, Passages, Tags and Citations, but new
 *   searches leave it out. If the file comes back, it isn't missing any more.
 * - "unavailable": can't be reached right now: its Linked folder (an
 *   unplugged drive, an unmounted share), or the folder a single file was in,
 *   is gone, or the file can't be read. It stays searchable from its stored
 *   text; only opening the file fails. Checked again later.
 */
export type DocumentFileStatus = "available" | "missing" | "unavailable";

export interface Document {
  /** A random UUID generated on this device. */
  id: string;
  /** The display name. It starts as the file name without its extension and can be renamed. */
  name: string;
  kind: DocumentKind;
  /**
   * The version whose text is indexed: the SHA-256 of the file's bytes as
   * they were read, in hex. When the file changes, the new version is
   * processed, and this changes once its text is indexed (not if it fails:
   * see `DocumentStatus`). A file found at a
   * new path with the hash of a Document whose file went is that Document,
   * moved. The same content at two paths is two Documents.
   */
  contentHash: string;
  /** The file, where the User keeps it: an absolute path. IncarnaMind never changes it. */
  path: string;
  /** Where the file is: see `DocumentFileStatus`. */
  fileStatus: DocumentFileStatus;
  /** The Linked folder the file is in, or null for a file added on its own ("Other Documents"). */
  linkedFolderId: string | null;
  /** In bytes. */
  size: number;
  /** PDFs only, once their text has been extracted. */
  pageCount: number | null;
  status: DocumentStatus;
  /** While `status` is "embedding": the share of its Passages embedded so far, from 0 to 1. Null otherwise. */
  progress: number | null;
  /** Set when `status` is "failed": why the latest version read failed. */
  failure: DocumentFailure | null;
  /**
   * The Folder its file is in: the Linked folder's own Folder, or one inside
   * it. Null for a file added on its own.
   */
  folderId: string | null;
  /** The Tags on the Document, in Tag name order (ignoring case). */
  tags: DocumentTag[];
  /** Where automatic tagging is, separately from `status`. */
  tagging: TaggingState;
  /** Set when `tagging` is "failed". */
  taggingError: ProviderError | null;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface ListDocumentsOptions {
  /** Only Documents in this Folder. Omitted: every Document, in a Folder or not. */
  folderId?: string;
  /** With `folderId`, also Documents in its sub-Folders, at any depth. Defaults to false. */
  includeSubfolders?: boolean;
  /** Only Documents that carry this Tag. Combines with `folderId`: both must match. */
  tagId?: string;
  /**
   * Only the Documents of this Linked folder, or, with null, only the files
   * added on their own (the sidebar's "Other Documents"). Omitted: all.
   */
  linkedFolderId?: string | null;
}

/** The text IncarnaMind kept of a Document's current version, Unit by Unit. */
export interface DocumentText {
  documentId: string;
  /** The version the text is of. */
  contentHash: string;
  fileStatus: DocumentFileStatus;
  /**
   * Its Units (pages, slides, sections, blocks of rows or lines) in order,
   * `page` being the Unit's number, from 1. Empty if no text is indexed.
   */
  pages: { page: number; text: string; kind: UnitKind; label: UnitLabel | null }[];
}

/**
 * A label with a short description that Documents can carry. IncarnaMind
 * applies Tags automatically, choosing them by their names and descriptions,
 * and the User can add or remove them.
 */
export interface Tag {
  /** A random UUID generated on this device. */
  id: string;
  /** Never empty; unique among Tags, ignoring case. */
  name: string;
  /** What the Tag means, for the User and for automatic tagging. May be empty. */
  description: string;
  /** Created on first run as one of the preset Tags, rather than by the User. Editing one keeps it a preset. */
  preset: boolean;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface CreateTagInput {
  /** Trimmed; must not be empty, nor another Tag's name (ignoring case). */
  name: string;
  /** Trimmed. Defaults to empty. */
  description?: string;
}

/** The fields to change; the others are kept. */
export interface UpdateTagInput {
  name?: string;
  description?: string;
}

/**
 * A folder inside a Linked folder, as it is on disk (CONTEXT.md), or the
 * Linked folder itself. Folders follow the disk: the User doesn't create,
 * rename or move them in IncarnaMind.
 */
export interface Folder {
  /**
   * Derived from its Linked folder and `relativePath`, so the same folder has
   * the same id at every scan. A folder renamed on disk is a new Folder.
   */
  id: string;
  /** The folder's name on disk. */
  name: string;
  /** The Folder this one is in, or null for a Linked folder's own Folder. */
  parentId: string | null;
  linkedFolderId: string;
  /** Relative to the Linked folder, "/" between names: "" for the Linked folder itself. */
  relativePath: string;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

/**
 * How the sidebar shows a Linked folder: "tree", with its folders as
 * Folders, or "flat", every Document in one list. Flat suits folders such as
 * Zotero's storage, where nearly every folder holds one file.
 */
export type LinkedFolderLayout = "tree" | "flat";

/**
 * - "scanning": being compared with the index, or indexed for the first time.
 * - "watching": up to date, and watched for changes while the app runs.
 * - "paused": the User paused it: nothing in it is read or processed until resumed.
 * - "unavailable": the folder can't be reached (an unplugged drive, an
 *   unmounted share). Its Documents stay searchable; it is checked again later.
 */
export type LinkedFolderStatus = "scanning" | "watching" | "paused" | "unavailable";

/**
 * A folder on the User's computer that IncarnaMind keeps in sync, read only:
 * every supported file in it, at any depth, is a Document indexed where it
 * is. Hidden files and folders, .git and node_modules are left out, and cloud
 * placeholders (online-only files) aren't read unless the User asks.
 */
export interface LinkedFolder {
  /** A random UUID generated on this device. */
  id: string;
  /** Absolute, symbolic links resolved. */
  path: string;
  /** Its own Folder, the root of its Folders. */
  folderId: string;
  status: LinkedFolderStatus;
  layout: LinkedFolderLayout;
  progress: {
    /** Supported files in it that are or will be Documents (missing ones and online-only files not counted). */
    files: number;
    /** Of those, the ones processed to the end: ready, with no text, or failed. */
    indexed: number;
  };
  /**
   * Cloud placeholders found in it (iCloud Drive, Dropbox, Google Drive or
   * OneDrive files not downloaded), which aren't indexed: reading one would
   * download it. `downloadOnlineOnlyFiles` downloads and indexes them.
   */
  onlineOnly: { files: number; bytes: number; downloading: boolean };
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

/**
 * What linking a folder would take, so the User can confirm first: found
 * from the files' metadata, nothing read.
 */
export interface LinkedFolderPreview {
  /** Absolute, symbolic links resolved: the path `addLinkedFolder` would link. */
  path: string;
  /** Supported files that would be indexed now, and their total size in bytes. */
  files: number;
  bytes: number;
  /** Cloud placeholders, left out until the User asks to download them. */
  onlineOnly: { files: number; bytes: number };
  /** A rough guess at how long indexing takes on this computer, in seconds. */
  estimatedSeconds: number;
  /** The layout it would get. */
  layout: LinkedFolderLayout;
  /** The folder is inside a Linked folder already: linking it adds nothing. Its id, or null. */
  insideLinkedFolderId: string | null;
  /** Linked folders inside it, which linking it would merge into it (their Documents keep their ids). */
  containsLinkedFolderIds: string[];
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

/**
 * How `searchPassages` finds Passages (ADR-0009):
 * - "keyword": FTS5 over the words of each Passage and its Document's name,
 *   ranked by BM25. Finds nothing without a word in common.
 * - "vector": the current embedding model's vectors (see `EmbeddingSettings`),
 *   by cosine similarity. Always ranks every Passage embedded with that model,
 *   however unrelated; Passages embedded with another model are never compared.
 * - "hybrid": both, each list's top 50 fused by reciprocal rank fusion
 *   (k = 60). Keyword only until the embedding model is ready.
 */
export type SearchMode = "hybrid" | "keyword" | "vector";

export interface SearchPassagesOptions {
  /** Defaults to "hybrid". */
  mode?: SearchMode;
  /** The most results to return, from 1 to 200. Defaults to 20. */
  limit?: number;
  /** Only Passages of these Documents (a Search scope). Omitted: every Document. */
  documentIds?: string[];
}

export interface PassageSearchResult {
  passageId: string;
  documentId: string;
  documentName: string;
  /**
   * The first Unit the Passage covers, from 1: a PDF's page, a deck's slide,
   * otherwise a Unit's number (see src/shared/units.ts). Null only for
   * Passages built before Units, of a TXT or Markdown file.
   */
  pageFrom: number | null;
  /** The last Unit the Passage covers. A Passage can cross from one Unit to the next. */
  pageTo: number | null;
  /** The Passage's position in its Document, from 0. */
  position: number;
  text: string;
}

// ---------------------------------------------------------------------------
// The built-in embedding model (ADR-0009)

/**
 * The built-in embedding model, which Document search uses on this computer.
 * Its files are downloaded once into the data folder (about 135 MB). The
 * download sends nothing of the User's, so it needs no consent.
 * - "not-downloaded": nothing has asked for it yet.
 * - "downloading": see the byte counts.
 * - "ready": downloaded and checked. Embedding works offline from here on.
 * - "failed": see `error`; `downloadEmbeddingModel` tries again.
 */
export type EmbeddingModelState = "not-downloaded" | "downloading" | "ready" | "failed";

export interface EmbeddingModelError {
  /**
   * "network": the download stopped, e.g. offline; a retry resumes it.
   * "integrity": a downloaded file didn't match its recorded size and SHA-256, so it was thrown away.
   * "storage": the files couldn't be written to the data folder, e.g. the disk is full.
   * "load": the files are there, but the model couldn't start on this computer.
   */
  kind: "network" | "integrity" | "storage" | "load";
  /** Technical detail in English, for logs and tooltips. */
  message: string;
}

export interface EmbeddingModelStatus {
  /** The model's name, e.g. "multilingual-e5-small". */
  name: string;
  /** Where the files are downloaded from, e.g. "huggingface.co": traffic that carries no User content. */
  host: string;
  state: EmbeddingModelState;
  /** Bytes downloaded and checked so far. */
  downloadedBytes: number;
  totalBytes: number;
  /** Set when `state` is "failed". */
  error: EmbeddingModelError | null;
}

// ---------------------------------------------------------------------------
// Embedding providers (ADR-0005)

/**
 * Where Passages and search queries are embedded. "built-in" is the model
 * above, on this computer: the default. The others trade privacy for quality:
 * a cloud provider receives the text of every Document, and every search
 * query. "ollama" is Ollama's embeddings, normally on this computer too.
 */
export const embeddingProviderKinds = [
  "built-in",
  "openai",
  "google",
  "openai-compatible",
  "ollama",
] as const;

export type EmbeddingProviderKind = (typeof embeddingProviderKinds)[number];

/**
 * Prefilled when a provider is picked; any embedding model the account has
 * can be typed instead. Ollama's is multilingual (bge-m3, the best model in
 * ADR-0009's comparison); it must be pulled first.
 */
export const SUGGESTED_EMBEDDING_MODELS: Readonly<
  Record<Exclude<EmbeddingProviderKind, "built-in">, string>
> = {
  openai: "text-embedding-3-small",
  google: "gemini-embedding-001",
  "openai-compatible": "",
  ollama: "bge-m3",
};

/** The embedding model search uses on this device. */
export interface EmbeddingProvider {
  kind: EmbeddingProviderKind;
  /** The server's URL for "openai-compatible" and "ollama"; null for the others. */
  baseUrl: string | null;
  /** The model, e.g. "text-embedding-3-small"; for "built-in", the built-in model's name. */
  modelId: string;
  /** Whether an API key is stored for it. Keys live in the keychain, never in the database. */
  hasApiKey: boolean;
  /** How many numbers the model's vectors have: known once it has made one (384 for the built-in model). */
  dimensions: number | null;
  /** Where Document text and search queries go, or null when they stay on this computer. */
  service: ExternalService | null;
}

export interface SaveEmbeddingProviderInput {
  kind: EmbeddingProviderKind;
  /** Required for "openai-compatible"; optional for "ollama" (Ollama's local port); not allowed otherwise. */
  baseUrl?: string;
  /** A new API key. Leave it out to keep the one stored for the same provider and server. */
  apiKey?: string;
  /** The embedding model. Required, except for "built-in", which takes none. */
  modelId?: string;
}

/** Settings to test, which need not be saved. Without a key, the one stored for the same provider and server. */
export type TestEmbeddingConnectionInput = SaveEmbeddingProviderInput;

export type EmbeddingConnectionTestResult =
  /** The model made a vector of this many numbers. */
  { ok: true; dimensions: number } | { ok: false; error: ProviderError };

/**
 * Documents being embedded again after the embedding model changed. Each one
 * goes through the usual statuses ("embedding", then "ready"); until it is
 * done, keyword search still finds its Passages, but vector search doesn't:
 * vectors from different models are never compared.
 */
export interface EmbeddingRebuild {
  /**
   * "provider-changed": the User chose another embedding model.
   * "local-mode": local mode switched a cloud provider back to the built-in model.
   */
  reason: "provider-changed" | "local-mode";
  /** Documents with Passages to search. */
  total: number;
  /** Of those, the ones whose Passages are all embedded with the current model. */
  done: number;
}

/** Document search's embedding model on this device, and local mode. */
export interface EmbeddingSettings {
  provider: EmbeddingProvider;
  /**
   * "Keep everything on this computer": only embedding providers on this
   * computer can be chosen, and search results aren't reranked by a cloud
   * service. Turned on in Settings, or by choosing local models with one
   * click (`selectOllama`).
   */
  localOnly: boolean;
  /** Set while Documents are embedded again after a switch. */
  rebuild: EmbeddingRebuild | null;
  /**
   * Why the provider can't embed right now, e.g. its key was refused or the
   * User declined sending it data; null when it can. Documents wait
   * ("waiting-for-model") until `retryEmbedding`, and search finds Passages by
   * their words meanwhile. The built-in model reports its download in
   * `getEmbeddingModel` instead.
   */
  error: ProviderError | null;
}

// ---------------------------------------------------------------------------
// Rerank (ADR-0005)

/** Reranking services. With a key for one, document search reranks its best matches. */
export const rerankProviderKinds = ["cohere", "voyage"] as const;

export type RerankProviderKind = (typeof rerankProviderKinds)[number];

/** The model each provider reranks with unless the User names another: both are multilingual. */
export const DEFAULT_RERANK_MODELS: Readonly<Record<RerankProviderKind, string>> = {
  cohere: "rerank-v3.5",
  voyage: "rerank-2.5",
};

/** Rerank on this device. Without a key, search is as before: nothing is reranked. */
export interface RerankSettings {
  /** A Cohere or Voyage key is set up on this device: document search reranks its candidates. */
  enabled: boolean;
  kind: RerankProviderKind | null;
  /** The model requests name, e.g. "rerank-v3.5"; null when rerank isn't set up. */
  modelId: string | null;
  /** Whether its key can be read on this device. Keys live in the keychain, never in the database. */
  hasApiKey: boolean;
  /** Where the Question and candidate Passages go, while rerank is set up. */
  service: ExternalService | null;
  /** Local mode is on: nothing is reranked, though the settings are kept. */
  paused: boolean;
}

export interface SaveRerankSettingsInput {
  kind: RerankProviderKind;
  /** A new key. Leave it out to keep the stored one (for the same kind); the first save needs one. */
  apiKey?: string;
  /** Null or empty: the provider's default model (`DEFAULT_RERANK_MODELS`). Left out: unchanged. */
  modelId?: string | null;
}

/** Settings to test, which need not be saved; those left out are the saved ones. */
export interface TestRerankConnectionInput {
  kind?: RerankProviderKind;
  apiKey?: string;
  modelId?: string | null;
}

// ---------------------------------------------------------------------------
// Chat providers (ADR-0005)

/**
 * Where Answers can come from. "ollama" is an OpenAI-compatible server run by
 * Ollama, normally on this computer; it also offers one-click model downloads.
 * "chatgpt" is the User's ChatGPT plan, used through a browser sign-in instead
 * of an API key; it is experimental (see `ChatGptPlanStatus`).
 */
export const chatProviderKinds = [
  "openai",
  "anthropic",
  "google",
  "openai-compatible",
  "ollama",
  "chatgpt",
] as const;

export type ChatProviderKind = (typeof chatProviderKinds)[number];

/** A service outside this computer that IncarnaMind can send data to. */
export interface ExternalService {
  /** The API's origin, e.g. "https://api.openai.com". Consent is recorded per service. */
  id: string;
  /** What the User sees, e.g. "OpenAI" or "api.deepseek.com". */
  name: string;
}

export interface ChatProvider {
  /** A random UUID generated on this device. */
  id: string;
  kind: ChatProviderKind;
  /** The server's URL for "openai-compatible" and "ollama"; null for the others. */
  baseUrl: string | null;
  /** Whether an API key is stored for it. Keys live in the keychain, never in the database. */
  hasApiKey: boolean;
  /** Where Questions go, or null when the server runs on this computer and nothing leaves it. */
  service: ExternalService | null;
}

export interface SaveChatProviderInput {
  kind: ChatProviderKind;
  /** Required for "openai-compatible"; optional for "ollama" (Ollama's local port); not allowed otherwise. */
  baseUrl?: string;
  /** A new API key. Leave it out to keep the stored one; null removes it. */
  apiKey?: string | null;
  /** The model to make the default, e.g. "gpt-5.4-mini". */
  modelId: string;
}

export interface TestChatConnectionInput {
  kind: ChatProviderKind;
  baseUrl?: string;
  /** The key to test. Leave it out to test the key stored for the same provider. */
  apiKey?: string;
  modelId: string;
}

/** Why a request to a model provider failed, so the UI can say what to fix. */
export type ProviderErrorKind =
  | "auth"
  | "model"
  | "rate-limit"
  | "network"
  | "provider"
  | "consent-declined"
  /** ChatGPT plan: there is no sign-in on this device, or it expired: sign in again. */
  | "not-signed-in"
  /** ChatGPT plan: the plan's usage limit is reached, or the plan doesn't include this use. */
  | "plan-limit"
  /** ChatGPT plan: OpenAI refused a valid sign-in (401 or 403), e.g. it blocked this integration. */
  | "blocked"
  /**
   * A local model: the request doesn't fit its context window, even with the
   * oldest Question context left out. Nothing is cut silently.
   */
  | "too-long"
  | "unknown";

export interface ProviderError {
  kind: ProviderErrorKind;
  /** The provider's own message, for details. */
  message: string;
}

export type ConnectionTestResult = { ok: true } | { ok: false; error: ProviderError };

/** A model on a saved chat provider. */
export interface ChatModelChoice {
  providerId: string;
  modelId: string;
}

/**
 * Whether Questions can be asked, and if not, what the User has to do.
 *
 * When `consent` is "needed", the first Question asks the User to accept the
 * chat data flow before anything is sent.
 */
export type ChatReadiness =
  | {
      ready: true;
      provider: ChatProvider;
      modelId: string;
      consent: "accepted" | "needed" | "not-required";
    }
  | { ready: false; reason: "no-provider" }
  | {
      ready: false;
      /**
       * "missing-api-key": the provider needs a key and none is stored.
       * "consent-declined": the User declined sending data to its service.
       * "sign-in-required": the provider uses a sign-in (the ChatGPT plan) and
       * this device has none, e.g. the User signed out or it expired.
       */
      reason: "missing-api-key" | "consent-declined" | "sign-in-required";
      provider: ChatProvider;
      modelId: string;
    };

/** A saved chat provider and the models a Question's model picker offers on it. */
export interface ChatModelGroup {
  provider: ChatProvider;
  /**
   * Model ids: the models the provider lists, when it can be asked, and always
   * the default model if it is on this provider. The default comes first.
   */
  models: string[];
}

// ---------------------------------------------------------------------------
// Questions and Answers (ADR-0007)

export interface AskQuestionInput {
  mindId: string;
  /** The Question Block's id. */
  questionId: string;
  /** Replace the Question's Answer even if the User has edited it. Defaults to false. */
  discardEdits?: boolean;
}

export interface RegenerateAnswerInput {
  mindId: string;
  answerId: string;
  /** Replace the Answer even if the User has edited it. Defaults to false. */
  discardEdits?: boolean;
}

export interface StopAnswerInput {
  mindId: string;
  answerId: string;
}

/** What asking a Question (or regenerating its Answer) did. */
export type AskResult =
  /** The Answer is being written into the Mind, right after its Question. */
  | { asked: true; answerId: string }
  /** Nothing was sent: Questions can't be asked yet, and `readiness` says why. */
  | { asked: false; reason: "not-ready"; readiness: Extract<ChatReadiness, { ready: false }> }
  /** Nothing was sent: the User has edited the Answer. Ask again with `discardEdits` to replace it. */
  | { asked: false; reason: "edited"; answerId: string }
  /**
   * Nothing was sent: the Question forces a Skill (`skill`, its name) that is
   * turned off or no longer there. Turn it on, or take it off the Question.
   */
  | { asked: false; reason: "skill-unavailable"; skill: string; state: SkillAvailability };

/** An Answer started: it is in the Mind with status "streaming". */
export interface AnswerStarted {
  mindId: string;
  answerId: string;
  questionId: string;
  /** The model writing it. */
  model: ChatModelChoice;
}

/** More of an Answer's text arrived, as the model wrote it (Markdown). */
export interface AnswerDelta {
  mindId: string;
  answerId: string;
  text: string;
}

/**
 * What an Answer being written is doing, for its meta line:
 * "waiting-for-consent", the User is asked whether to send the Question to
 * the model's service, and nothing is sent until they say; "loading", Ollama
 * is loading the local model; "searching", the User's Documents are being
 * searched; "writing", the model is at work.
 */
export type AnswerPhase = "waiting-for-consent" | "loading" | "searching" | "writing";

/** An Answer being written moved to another phase. */
export interface AnswerPhaseEvent {
  mindId: string;
  answerId: string;
  phase: AnswerPhase;
}

/** A Tool call of an Answer started, or finished (see `call.status`). */
export interface AnswerToolCallEvent {
  mindId: string;
  answerId: string;
  call: AnswerToolCall;
}

/**
 * The model gave the record for a Citation marker in the Answer: the marker
 * becomes this Citation, "checking" until the Answer finishes.
 */
export interface AnswerCitationAdded {
  mindId: string;
  answerId: string;
  /** The number of the marker the model wrote in the text, e.g. 1 for "[^1]". */
  marker: number;
  citation: Citation;
}

/** An Answer is complete ("done") or the User stopped it ("stopped"), keeping what was written. */
export interface AnswerFinished {
  mindId: string;
  answerId: string;
  status: "done" | "stopped";
  /** The Answer's Citations, checked, in the order of their markers in the text, each once. */
  citations: Citation[];
  /** Markers the model wrote with no valid record: removed, leaving their sentences uncited. */
  droppedMarkers: number;
  /** Records the model gave for no marker in the text, or naming no Passage it was given: dropped. */
  droppedRecords: number;
  /** Markers the model left out of its text for records it gave, which the engine put in. */
  placedMarkers: number;
  /** How the model could give Citations; null when there were no Documents to search. */
  citationSupport: CitationSupport | null;
}

/** An Answer failed; the Answer shows the error by kind. */
export interface AnswerFailed {
  mindId: string;
  answerId: string;
  error: ProviderError;
}

// ---------------------------------------------------------------------------
// Skills (CONTEXT.md: Skill)

/**
 * A Skill: a packaged description of how to do a particular task, in the
 * standard `SKILL.md` format (https://agentskills.io/specification), with
 * optional reference files and scripts. Its files are stored in the data
 * folder, under `skills/<id>/`. The system prompt of every Answer lists the
 * enabled Skills' names and descriptions; the model loads a Skill's full
 * instructions when it needs them, or the User forces one on a Question.
 *
 * Built-in Skills ship with the app and are installed on first run. A newer
 * version of the app updates them; whether each is on, or removed, is kept.
 */
export interface Skill {
  /** A random UUID generated on this device. Importing a Skill of the same name again keeps it. */
  id: string;
  /** From the frontmatter: lowercase letters, digits and hyphens. Unique among Skills. */
  name: string;
  /** From the frontmatter: what the Skill does and when to use it. */
  description: string;
  /** From the frontmatter, if given: a licence name, or the name of a bundled licence file. */
  license: string | null;
  /** From the frontmatter, if given: what the Skill needs, e.g. "Requires Python 3.14+". */
  compatibility: string | null;
  /** Turned on, Answers can use it. New Skills start on. */
  enabled: boolean;
  /**
   * Ships with the app. Its files can't be replaced by importing a Skill of
   * its name; `duplicateSkill` makes a copy that is the User's own.
   */
  builtIn: boolean;
  /** Every file in the Skill: SKILL.md first, then the others in path order. */
  files: SkillFile[];
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface SkillFile {
  /** Relative to the Skill's folder, with "/" between folders, e.g. "references/guide.md". */
  path: string;
  /** In bytes. */
  size: number;
  /**
   * A script: in the Skill's `scripts/` folder, or a script's file type. The
   * model can read it and, unless the User turned Skill scripts off, run it
   * with `run_skill_script`, which asks the User first. Python (.py),
   * JavaScript (.js, .mjs, .cjs) and shell (.sh, .bash; not on Windows)
   * scripts run; any other kind gives an error saying so.
   */
  script: boolean;
}

/** Whether a Skill can be used: on, turned off, or not there (removed, or never imported). */
export type SkillAvailability = "enabled" | "disabled" | "removed";

/** What importing a Skill would add, for the User to check before importing it. */
export interface SkillImportPreview {
  /** Pass to `importSkill` to import exactly this, or to `cancelSkillImport`. */
  importId: string;
  /** The folder or zip it was read from. */
  source: string;
  name: string;
  description: string;
  license: string | null;
  compatibility: string | null;
  /** SKILL.md first, then the others in path order. */
  files: SkillFile[];
  /** The files' sizes added up, in bytes. */
  totalBytes: number;
  /** The Skill of the same name that importing replaces (keeping its id and whether it's on), or null. */
  replaces: Skill | null;
}

/**
 * Why a folder or zip can't be imported as a Skill.
 * - "unreadable": it (or a file in it, see `path`) can't be read.
 * - "not-a-skill": there is no SKILL.md at its top, nor in its one top-level folder.
 * - "invalid-zip": not a zip file, or one IncarnaMind can't read (encrypted, ZIP64, damaged).
 * - "invalid-frontmatter": SKILL.md's frontmatter is missing or breaks the format (see `field`).
 * - "path-traversal": a zip entry's path leads outside the Skill (see `path`).
 * - "link-outside": a symbolic link points outside the Skill (see `path`).
 * - "too-large": its files add up to more than `SKILL_LIMITS.maxBytes`, or SKILL.md
 *   is over `SKILL_LIMITS.maxInstructionsBytes`.
 * - "too-many-files": more than `SKILL_LIMITS.maxFiles` files.
 * - "built-in-name": a built-in Skill has its name, and a built-in Skill's files
 *   can't be replaced (`field` is "name").
 */
export type SkillImportErrorKind =
  | "unreadable"
  | "not-a-skill"
  | "invalid-zip"
  | "invalid-frontmatter"
  | "path-traversal"
  | "link-outside"
  | "too-large"
  | "too-many-files"
  | "built-in-name";

export interface SkillImportError {
  kind: SkillImportErrorKind;
  /** The file at fault, relative to the Skill (or as named in the zip), when there is one. */
  path: string | null;
  /** "invalid-frontmatter": the field at fault, e.g. "name", when there is one. */
  field: string | null;
  /** Technical detail in English, for logs and tooltips. */
  message: string;
}

export type SkillImportCheck =
  | { ok: true; preview: SkillImportPreview }
  | { ok: false; error: SkillImportError };

/** How Skill scripts run. */
export const SKILL_SCRIPT_LIMITS = {
  /** How long a script may run when the User hasn't chosen (`DeviceSettings.skillScriptTimeoutSeconds`). */
  defaultTimeoutSeconds: 60,
  minTimeoutSeconds: 1,
  maxTimeoutSeconds: 3600,
  /** Of each of the standard output and the error output: what the model and the card get. */
  maxOutputBytes: 20_000,
  /** Arguments one run may have, and characters in all of them together. */
  maxArgs: 100,
  maxArgsChars: 20_000,
} as const;

/** How big a Skill may be. Checked before anything is imported. */
export const SKILL_LIMITS = {
  /** All its files together, uncompressed; a zip file itself may be no larger either. */
  maxBytes: 20 * 1024 * 1024,
  maxFiles: 1000,
  /** SKILL.md alone: its instructions are sent whole to the model. */
  maxInstructionsBytes: 256 * 1024,
} as const;

// ---------------------------------------------------------------------------
// ChatGPT plan (experimental)

/**
 * A model the ChatGPT plan's endpoint accepts. The list is fixed in the app
 * and follows the Codex CLI's own model catalog.
 */
export interface ChatGptPlanModel {
  id: string;
  /** What the User sees, e.g. "GPT-5.5". */
  name: string;
}

/** The ChatGPT sign-in on this device. */
export type ChatGptAccount =
  | { state: "signed-out" }
  /** Refreshing the sign-in failed, so it was removed: the User has to sign in again. */
  | { state: "expired" }
  | {
      state: "signed-in";
      /** From the sign-in's ID token, when it carries one. */
      email: string | null;
      /** The ChatGPT plan, e.g. "plus" or "pro", when the ID token says. */
      plan: string | null;
    };

/**
 * The experimental "ChatGPT plan (via Codex sign-in)" provider. It signs in
 * with the sign-in OpenAI's Codex CLI uses, which OpenAI hasn't approved for
 * other apps and may block. It is off until the User turns it on in Settings.
 */
export interface ChatGptPlanStatus {
  /** The User turned the experimental provider on, on this device. Off by default. */
  enabled: boolean;
  account: ChatGptAccount;
  /** A browser sign-in is waiting for the User. */
  signingIn: boolean;
  /** The models Questions can use with the plan, the default first. */
  models: ChatGptPlanModel[];
}

export type ChatGptSignInErrorKind =
  /** The sign-in's fixed local port is taken, e.g. the Codex CLI is signing in at the same time. */
  | "port-in-use"
  /** The User didn't finish in the browser within a few minutes. */
  | "timed-out"
  /** The User cancelled in IncarnaMind. */
  | "cancelled"
  /** OpenAI refused the sign-in, e.g. the User declined in the browser. */
  | "denied"
  /** This device can't store the sign-in securely (see `getSecretStorage`). */
  | "secret-storage"
  | "failed";

export type ChatGptSignInResult =
  | { ok: true; status: ChatGptPlanStatus }
  | { ok: false; error: { kind: ChatGptSignInErrorKind; message: string } };

// ---------------------------------------------------------------------------
// Secrets

/**
 * How API keys are protected on this device.
 * - "os": encrypted with a key held by the OS secret store.
 * - "plain-text": Linux with no keyring running (safeStorage's "basic_text"
 *   backend). Keys would be stored effectively in plain text.
 * - "unavailable": keys can't be encrypted at all.
 */
export type SecretProtection = "os" | "plain-text" | "unavailable";

export interface SecretStorageStatus {
  protection: SecretProtection;
  /** The User accepted storing keys without keyring protection on this device. */
  plainTextAccepted: boolean;
  /** Whether keys can be saved now. */
  canSave: boolean;
}

// ---------------------------------------------------------------------------
// Ollama

export type OllamaStatus =
  | { running: false; baseUrl: string }
  | {
      running: true;
      baseUrl: string;
      /** Models already pulled, e.g. "qwen3:4b". */
      models: string[];
      /** The model one click pulls and selects. */
      recommendedModel: string;
    };

export interface SelectOllamaInput {
  /** Defaults to Ollama's local port. */
  baseUrl?: string;
  /** Defaults to the recommended model. */
  model?: string;
}

export interface OllamaPullProgress {
  model: string;
  /** Ollama's status line, e.g. "pulling manifest" or "success". */
  status: string;
  /** Bytes of the current layer, when Ollama reports them. */
  completed: number | null;
  total: number | null;
}

// ---------------------------------------------------------------------------
// Data-flow consent

/**
 * Kinds of data a flow can send. The UI describes each one
 * (`consent.data.<kind>`). Later tickets add theirs.
 */
export const dataKinds = [
  "blocks",
  "passages",
  "tool-results",
  "tags",
  "document-excerpts",
  "tool-arguments",
  "document-text",
  "queries",
] as const;

export type DataKind = (typeof dataKinds)[number];

/**
 * External data flows. The UI names each one (`consent.flow.<id>`). Later tickets add theirs.
 * - "chat": Questions, to the chat provider they are asked with.
 * - "tagging": automatic tagging, to Jev's service when a Jev key is set up,
 *   otherwise to the default chat model's provider.
 * - "connectors": the arguments of the Tool calls an Answer makes, to the
 *   Connector it calls: one service per local Connector (it runs on this
 *   computer but can reach the internet itself), and one per server origin
 *   for remote ones, e.g. "https://mcp.example.com".
 * - "embeddings": every Document's text and every search query, to a cloud
 *   embedding provider, when the User chose one instead of the built-in model.
 * - "rerank": each search query and its candidate Passages, to Cohere or
 *   Voyage, when a rerank key is set up.
 */
export const dataFlowIds = ["chat", "tagging", "connectors", "embeddings", "rerank"] as const;

export type DataFlowId = (typeof dataFlowIds)[number];

/** Data leaving this computer: what one flow sends to one service. */
export interface DataFlow {
  id: DataFlowId;
  service: ExternalService;
  /** Everything the flow sends to the service. */
  sends: DataKind[];
}

/** The core is waiting for the User to accept or decline a data flow. Nothing is sent until they do. */
export interface ConsentRequest {
  requestId: string;
  flow: DataFlow;
  /** What the User hasn't accepted yet: everything the first time, only the new kinds when a flow starts sending more. */
  newKinds: DataKind[];
}

export interface DataFlowStatus {
  flow: DataFlow;
  /** "not-asked" also covers a flow that started sending a new kind of data since the User accepted it. */
  consent: "accepted" | "declined" | "not-asked";
  /**
   * When the User decided, ISO 8601, UTC. Null when never asked; set with
   * "not-asked" when they allowed the flow before it started sending more.
   */
  decidedAt: string | null;
}

/** A registered external data flow, for the Privacy page: what it sends, and where. */
export interface RegisteredDataFlow {
  id: DataFlowId;
  /** Everything the flow sends. */
  sends: DataKind[];
  /**
   * The flow to each service it currently goes to and each service the User
   * has decided on, with that decision. Empty when nothing is sent on it now,
   * e.g. chat with a model on this computer.
   */
  services: DataFlowStatus[];
}

// ---------------------------------------------------------------------------
// Privacy

/**
 * Network traffic that carries no User content: it needs no consent, and the
 * Privacy page lists it. The UI names and describes each one
 * (`privacy.traffic.<id>`). Features register theirs; later tickets add ids.
 * - "update-check": asks GitHub Releases for a newer version, when IncarnaMind starts.
 * - "embedding-model": downloads the built-in embedding model's files, once.
 * - "ollama-pull": Ollama downloads a model from its registry, when the User picks local models.
 * - "chatgpt-sign-in": the experimental ChatGPT plan's sign-in, and refreshing it.
 * - "remote-connectors": connecting to each remote Connector that is on, and
 *   signing in to it: its server, and the authorization server it names
 *   (discovery, registration, tokens). One entry per server. Tool calls, which
 *   carry the User's content, are the "connectors" data flow.
 */
export const networkTrafficIds = [
  "update-check",
  "embedding-model",
  "ollama-pull",
  "chatgpt-sign-in",
  "remote-connectors",
] as const;

export type NetworkTrafficId = (typeof networkTrafficIds)[number];

export interface NetworkTraffic {
  id: NetworkTrafficId;
  /** Where it goes. */
  service: ExternalService;
  /** False while the User has turned it off, e.g. automatic update checks. */
  enabled: boolean;
}

/** The privacy choices on this device. IncarnaMind collects no usage data, whatever they are. */
export interface PrivacySettings {
  crashReports: {
    /**
     * Whether this copy of IncarnaMind can send crash reports: only a build
     * made with a crash-report address (a Sentry DSN) can. Without one, they
     * aren't offered.
     */
    available: boolean;
    /** The User opted in to sending scrubbed crash reports. Off by default. */
    enabled: boolean;
  };
  /** IncarnaMind checks GitHub Releases for a new version when it starts. On by default. */
  automaticUpdateChecks: boolean;
}

/** The choices to change; the others are kept. */
export interface PrivacySettingsPatch {
  crashReports?: boolean;
  automaticUpdateChecks?: boolean;
}

// ---------------------------------------------------------------------------
// TypeSafe Jev, the optional tagger (ADR-0005)

/** The model Jev requests name unless the User changes it: TypeSafe's latest stable Jev. */
export const JEV_DEFAULT_MODEL = "jev-latest";

/**
 * Where Jev's probability that a Tag applies counts as unsure. Below `low` the
 * Tag isn't applied; from `low` up to (not including) `high` it is applied and
 * marked "needs review"; from `high` it is applied. 0 < low ≤ high ≤ 1.
 */
export interface JevReviewBand {
  low: number;
  high: number;
}

export const JEV_DEFAULT_REVIEW_BAND: Readonly<JevReviewBand> = { low: 0.35, high: 0.65 };

/**
 * TypeSafe Jev on this device: a hosted classifier that decides each Tag with
 * a probability, faster and cheaper than a chat model. Optional: without it,
 * the chat model tags Documents. Its settings and key stay on this device.
 */
export interface JevSettings {
  /** Jev is set up on this device: automatic tagging uses it instead of the chat model. */
  enabled: boolean;
  /**
   * Whether its key can be read on this device. Keys live in the keychain,
   * never in the database. Enabled without a key, Documents wait for one.
   */
  hasApiKey: boolean;
  /** A Jev-compatible server's base URL, or null for TypeSafe's hosted Jev. */
  endpoint: string | null;
  /** The model each request names, e.g. "jev-latest". */
  model: string;
  reviewBand: JevReviewBand;
  /** Where tagging requests go, or null when the server runs on this computer. */
  service: ExternalService | null;
}

export interface SaveJevSettingsInput {
  /** A new key. Leave it out to keep the stored one; the first save needs one. */
  apiKey?: string;
  /**
   * A Jev-compatible server's base URL, e.g. "https://jev.example.com"
   * (requests go to its `/v1/systemone`). Null or empty: TypeSafe's hosted
   * Jev. Left out: unchanged.
   */
  endpoint?: string | null;
  /** Null or empty: "jev-latest". Left out: unchanged. */
  model?: string | null;
  /** Left out: unchanged (at first, `JEV_DEFAULT_REVIEW_BAND`). */
  reviewBand?: JevReviewBand;
}

/** Settings to test, which need not be saved; those left out are the saved ones. */
export interface TestJevConnectionInput {
  apiKey?: string;
  endpoint?: string | null;
  model?: string | null;
}

// ---------------------------------------------------------------------------
// Connectors (MCP servers)

/**
 * Where a Connector is:
 * - "off": the User turned it off, so its process isn't running (or, for a
 *   remote one, IncarnaMind isn't connected to it).
 * - "connecting": its process is starting, or IncarnaMind is connecting to it.
 * - "signing-in": remote: the User is signing in to it in their browser.
 * - "needs-sign-in": remote: the server wants a sign-in IncarnaMind doesn't
 *   have on this device: the User hasn't signed in, signed out, or renewing
 *   the sign-in failed (`signIn.expired`). Answers skip its Tools.
 * - "ready": connected; Answers can use its Tools (see `ApprovalPolicy` for which ask first).
 * - "error": it couldn't start or be reached, or it stopped (see `error`).
 */
export type ConnectorState =
  | "off"
  | "connecting"
  | "signing-in"
  | "needs-sign-in"
  | "ready"
  | "error";

export type ConnectorErrorKind =
  /**
   * Its command, or the runtime the command needs, isn't installed or isn't
   * on the PATH of the User's login shell (see `command` and `install`).
   */
  | "missing-command"
  /** Its environment variables are kept in the keychain, and can't be read on this device. */
  | "missing-secrets"
  /** Its process started but didn't answer in time; or a remote server didn't. */
  | "timed-out"
  /** Its process exited, or closed the connection. */
  | "stopped"
  /** Remote: its server can't be reached, e.g. no network, or nothing answers at its URL. */
  | "unreachable"
  /** Anything else, e.g. it doesn't speak MCP. */
  | "failed";

export interface ConnectorError {
  kind: ConnectorErrorKind;
  /** "missing-command": the command that wasn't found, e.g. "npx". Null otherwise. */
  command: string | null;
  /** "missing-command": what to install to get it, when known, e.g. "Node.js". */
  install: string | null;
  /** Technical detail in English, e.g. the last lines the process wrote to its error output. */
  message: string;
  /** IncarnaMind will start it again by itself shortly, waiting longer after each failure. */
  retrying: boolean;
}

/** A Tool a Connector offers. */
export interface ConnectorTool {
  /** As the Connector names it, e.g. "search_issues". */
  name: string;
  /** A display name, when the Connector gives one. */
  title: string | null;
  description: string;
  /**
   * The Connector says the Tool only reads and changes nothing (its
   * `readOnlyHint` annotation). That is the Connector's claim, a hint, not
   * something IncarnaMind can check. Answers call such a Tool without asking
   * unless the User switched it to "ask"; every other Tool asks first, unless
   * the User always allows it (see `ApprovalPolicy`).
   */
  readOnly: boolean;
}

interface ConnectorBase {
  /** A random UUID generated on this device. */
  id: string;
  /** Unique among Connectors, ignoring case. Answers see its Tools under this name. */
  name: string;
  /** The User turned it on. Off, it isn't running or connected, and Answers don't use it. */
  enabled: boolean;
  state: ConnectorState;
  /** Set when `state` is "error". */
  error: ConnectorError | null;
  /** Its Tools, as of when it last connected; null before it has. Null while it is off. */
  tools: ConnectorTool[] | null;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

/**
 * A local Connector: a program on this computer, started with the User's
 * login-shell environment and spoken to over its standard input and output.
 */
export interface LocalConnector extends ConnectorBase {
  transport: "stdio";
  /** The program to run, e.g. "npx", found on the PATH of the User's login shell. */
  command: string;
  args: string[];
  /**
   * The names of the environment variables it is started with, e.g. API keys.
   * Their values are kept in the keychain, never in the database, and never shown.
   */
  env: string[];
}

export type ConnectorSignInErrorKind =
  /** Another program took the local port the browser comes back to. */
  | "port-in-use"
  /** The User didn't finish in the browser within a few minutes. */
  | "timed-out"
  /** The User cancelled in IncarnaMind. */
  | "cancelled"
  /** The service refused the sign-in, e.g. the User declined in the browser. */
  | "denied"
  /**
   * The service can't register IncarnaMind by itself: the User registers an
   * OAuth app with it and enters its client ID (`setConnectorClient`).
   */
  | "client-id-required"
  /** This device can't store the sign-in securely (see `getSecretStorage`). */
  | "secret-storage"
  | "failed";

export interface ConnectorSignInError {
  kind: ConnectorSignInErrorKind;
  /** Technical detail in English. */
  message: string;
}

/** A remote Connector's sign-in on this device. Its tokens are kept in the keychain, never the database. */
export interface RemoteConnectorSignIn {
  /** Tokens for it are stored on this device. */
  signedIn: boolean;
  /** Renewing the last sign-in failed, so its tokens were removed: the User signs in again. */
  expired: boolean;
  /** Why the last browser sign-in didn't work; null once one works or starts again. */
  error: ConnectorSignInError | null;
}

/**
 * A remote Connector: an MCP server reached by its URL over Streamable HTTP.
 * If it requires a sign-in, the User signs in in their browser (OAuth 2.1
 * with PKCE, as the MCP authorization spec describes).
 */
export interface RemoteConnector extends ConnectorBase {
  transport: "http";
  /** Its MCP endpoint, e.g. "https://mcp.example.com/mcp". */
  url: string;
  /**
   * The client ID of an OAuth app the User registered with the service, for
   * one that can't register IncarnaMind by itself. Null otherwise.
   */
  clientId: string | null;
  signIn: RemoteConnectorSignIn;
}

/** An external service the User has connected: an MCP server, local or remote. */
export type Connector = LocalConnector | RemoteConnector;

export interface AddLocalConnectorInput {
  /** Trimmed; must not be empty, nor another Connector's name (ignoring case). */
  name: string;
  /** The program to run, e.g. "npx", or a full path. */
  command: string;
  /** Defaults to none. */
  args?: string[];
  /** Environment variables to start it with, e.g. API keys. Their values go to the keychain, never the database. */
  env?: Record<string, string>;
}

/** The OAuth app a User registered with a service that can't register IncarnaMind by itself. */
export interface ConnectorClientInput {
  clientId: string;
  /** Only if the service gave the app one. It goes to the keychain, never the database. */
  clientSecret?: string;
}

export interface AddRemoteConnectorInput {
  /** Trimmed; must not be empty, nor another Connector's name (ignoring case). */
  name: string;
  /**
   * Its MCP endpoint: an https URL, or http on this computer (e.g.
   * "http://127.0.0.1:8000/mcp"). Its sign-in, if any, starts with `signInToConnector`.
   */
  url: string;
  /** Only for a service that can't register IncarnaMind by itself (see `ConnectorClientInput`). */
  client?: ConnectorClientInput;
}

/** A local Connector by its command, or a remote one by its `url`. */
export type AddConnectorInput = AddLocalConnectorInput | AddRemoteConnectorInput;

export type ConnectorSignInResult =
  | { ok: true; connector: Connector }
  | { ok: false; error: ConnectorSignInError };

/**
 * One server of an `mcpServers` configuration (from Claude Desktop or
 * Cursor), and what importing it would do:
 * - "add": it is added as a Connector;
 * - "exists": a Connector with its name already exists (or the configuration
 *   lists the name twice), so it is skipped;
 * - "remote": it is a remote server IncarnaMind can't connect to: one that
 *   uses the older SSE transport, or needs custom headers;
 * - "invalid": it has neither a command nor a URL, or its arguments,
 *   environment or URL aren't valid.
 */
export interface ConnectorImportEntry {
  name: string;
  /** A local server's command; null for a remote one. */
  command: string | null;
  args: string[];
  /** The names of its environment variables. Their values go to the keychain. */
  env: string[];
  /** A remote server's URL. Absent for a local one. */
  url?: string;
  action: "add" | "exists" | "remote" | "invalid";
}

export interface ConnectorImportResult {
  /** The Connectors added, in the configuration's order. They start right away. */
  added: Connector[];
  /** The servers that weren't added, with why. */
  skipped: ConnectorImportEntry[];
}

// ---------------------------------------------------------------------------
// Approvals (#38)

/**
 * What an approval policy is about:
 * - "tool": one Tool of one Connector;
 * - "skill-script": the scripts of one Skill.
 */
export type ApprovalSubject =
  | {
      kind: "tool";
      connectorId: string;
      /** The Tool's name as the Connector gives it. */
      tool: string;
    }
  | { kind: "skill-script"; skillId: string };

export type ApprovalSubjectKind = ApprovalSubject["kind"];

/**
 * What the User chose for a subject, replacing the default:
 * - "always": it runs without asking ("always allow" a Tool; "always run" a Skill's scripts).
 * - "ask": it asks every time, even a Tool its Connector says only reads.
 *
 * Without a policy, a Tool asks first unless its Connector says it only reads
 * (`ConnectorTool.readOnly`), and a Skill script always asks.
 */
export type ApprovalPolicyValue = "always" | "ask";

/** One of the User's approval policies, as the approvals page lists it. */
export interface ApprovalPolicy {
  /** A random UUID generated on this device. */
  id: string;
  subject: ApprovalSubject;
  policy: ApprovalPolicyValue;
  /** What the subject belongs to, as named now: the Connector (for a Tool) or the Skill. */
  ownerName: string;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface SetApprovalPolicyInput {
  /** A Connector that exists, or a Skill that exists. */
  subject: ApprovalSubject;
  /** Null goes back to the default, removing the User's policy. A Skill script takes "always" or null. */
  policy: ApprovalPolicyValue | null;
  /**
   * "Always run" a Skill's scripts only: the User has seen the warning that
   * its scripts will then run on this computer without asking, with no
   * sandbox, and confirmed it. Without it, that policy is refused.
   */
  riskAccepted?: boolean;
}

/**
 * What the User decides about a Tool call that asks first:
 * - "allow-once": this call runs;
 * - "always-allow": this call runs, and so will every later call of the Tool
 *   without asking (the policy becomes "always"). For a Skill script this is
 *   "always run": every script of the Skill runs from now on without asking,
 *   which needs `ApprovalResponseOptions.riskAccepted`;
 * - "deny": the call doesn't run; the model is told the User denied it, and
 *   the Answer carries on without it.
 */
export type ApprovalDecision = "allow-once" | "always-allow" | "deny";

export interface ApprovalResponseOptions {
  /**
   * "always-allow" of a Skill script ("always run") only: the User has seen
   * the warning that the Skill's scripts will then run on this computer
   * without asking, with no sandbox, and confirmed it. Without it, that
   * decision is refused and the request keeps waiting.
   */
  riskAccepted?: boolean;
}

/**
 * A call waiting for the User's approval: of a Connector's Tool, or a run of
 * a Skill script (`subject.kind` tells which). Its Answer is paused until they
 * decide; stopping the Answer, or closing IncarnaMind, denies it.
 */
export type ApprovalRequest = ToolApprovalRequest | SkillScriptApprovalRequest;

interface ApprovalRequestBase {
  requestId: string;
  mindId: string;
  answerId: string;
  /** The call's id among the Answer's `toolCalls`. */
  toolCallId: string;
}

/**
 * A run of a Skill script waiting for the User's approval: which Skill, which
 * of its scripts, and its arguments. Scripts run on this computer with no
 * sandbox (v1): they can do whatever the User can.
 */
export interface SkillScriptApprovalRequest extends ApprovalRequestBase {
  subject: Extract<ApprovalSubject, { kind: "skill-script" }>;
  /** The Skill the script belongs to, as named when the run was asked for. */
  skill: { id: string; name: string };
  tool: "run_skill_script";
  /** The script's path inside the Skill, e.g. "scripts/convert.py". */
  script: string;
  /** The arguments it would run with, in order. */
  args: string[];
}

/** A call of a Connector's Tool waiting for the User's approval. */
export interface ToolApprovalRequest extends ApprovalRequestBase {
  subject: Extract<ApprovalSubject, { kind: "tool" }>;
  /** The Connector the Tool belongs to, as named when the call was made. */
  connector: { id: string; name: string };
  /** The Tool's name as its Connector gives it, e.g. "create_issue". */
  tool: string;
  /** The Tool's display name, when the Connector gives one. */
  title: string | null;
  /** The arguments the model wants to send. */
  input: Record<string, unknown>;
  /**
   * The Connector says this Tool only reads: it asks only because the User
   * switched it to "ask". False: the Tool may change something.
   */
  readOnly: boolean;
}

// ---------------------------------------------------------------------------
// Export

/** What a Mind exports to: Markdown, for an archive, or Word (.docx), for the deliverable. */
export type ExportFormat = "markdown" | "docx";

export interface ExportMindOptions {
  format: ExportFormat;
  /**
   * Whether Questions are exported, marked as Questions. Defaults to true for
   * Markdown, an archive of everything, and to false for .docx: the deliverable
   * leaves the working material out.
   */
  includeQuestions?: boolean;
}

/** What exporting a Mind will write, for the User to see before the file is written. */
export interface MindExportPreview {
  /** A file name for the export, from the Mind's title, e.g. "Tides.docx". */
  fileName: string;
  /**
   * The Citations in the exported text. Each becomes its own footnote, even
   * when several cite the same page.
   */
  citations: number;
  /**
   * Of those, the ones not shown as "Quote found": the quote wasn't found on
   * the cited pages, or it can't be checked (the pages have no text, or the
   * Document was deleted), or it is still being checked. Their footnotes carry
   * an "[unverified]" marker.
   */
  unverifiedCitations: number;
  /** The Question Blocks in the Mind, whether the export includes them or not. */
  questions: number;
}

/** A Mind exported as a file, for the host to save where the User chooses. */
export interface MindExport extends MindExportPreview {
  /** The file's contents: UTF-8 text for Markdown, a ZIP package for .docx. */
  data: Uint8Array;
}

// ---------------------------------------------------------------------------

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
   * Adds PDF, TXT and Markdown files on their own, given their absolute
   * paths. Each is indexed where it is, never copied, and queued for
   * processing; "document.status" events report its progress. A file already
   * indexed at that path gives its Document; a file inside a Linked folder
   * is that Linked folder's Document; a file with the content of a missing
   * Document is that Document, moved.
   */
  addDocuments(paths: string[]): Promise<AddDocumentsResult>;
  /**
   * Documents that are not deleted, missing ones included, most recently
   * added first. With a Folder, only the Documents in it, and in its
   * sub-Folders if asked; with a Tag, only the Documents carrying it; with a
   * Linked folder (or null), only its Documents (or the files added on their own).
   */
  listDocuments(options?: ListDocumentsOptions): Promise<Document[]>;
  /** Returns the renamed Document. Only its name in IncarnaMind changes, never the file. */
  renameDocument(id: string, name: string): Promise<Document>;
  /**
   * Removes a Document from the index: soft-deletes it, its Passages and its
   * Tags, so search ignores them. The file isn't touched. A Document in a
   * Linked folder whose file is still there stays out of the index after
   * later scans, until the Linked folder is removed and linked again.
   */
  deleteDocument(id: string): Promise<void>;
  /**
   * Processes a Document that failed again, from its file as it is now, e.g.
   * once the User has fixed it. Returns it, queued; "document.status" events
   * report its progress. A last good version it kept stays searched meanwhile. Throws InvalidInputError for a Document that didn't
   * fail, and NotFoundError for an unknown one or one whose file isn't there.
   */
  retryDocument(id: string): Promise<Document>;
  /**
   * The text IncarnaMind kept of a Document's current version: what the
   * viewer shows when the file is missing or can't be reached. Throws
   * NotFoundError for an unknown or deleted Document.
   */
  readDocumentText(documentId: string): Promise<DocumentText>;
  /**
   * What linking a folder would take (see `LinkedFolderPreview`), from the
   * files' metadata, without reading or changing anything. Throws for a path
   * that isn't a folder.
   */
  previewLinkedFolder(path: string): Promise<LinkedFolderPreview>;
  /**
   * Links a folder: every supported file in it, at any depth, becomes a
   * Document indexed where it is, newest first, and the folder is watched
   * for changes while the app runs. Returns at once, "scanning";
   * "linkedFolders.changed" and "document.status" events report progress.
   * A folder inside a Linked folder is already linked: that Linked folder is
   * returned. Linked folders inside this one are merged into it, their
   * Documents keeping their ids. Files added on their own that are inside it
   * become its Documents.
   */
  addLinkedFolder(path: string): Promise<LinkedFolder>;
  /** The Linked folders, in path order. */
  listLinkedFolders(): Promise<LinkedFolder[]>;
  /**
   * Stops syncing a Linked folder and removes its Documents and Folders from
   * the index. Nothing on disk is touched. Of their stored text, only the
   * Units the Citations in Minds point to are kept, so those Citations can
   * still be checked: "keptCitationTexts.changed" reports them.
   */
  removeLinkedFolder(linkedFolderId: string): Promise<void>;
  /** The text kept of Documents unlinked with their Linked folder, for the Citations that quote it. */
  listKeptCitationTexts(): Promise<KeptCitationText[]>;
  /** Shows a Linked folder as a tree of Folders or as a flat list. Returns it. */
  setLinkedFolderLayout(linkedFolderId: string, layout: LinkedFolderLayout): Promise<LinkedFolder>;
  /**
   * Pauses a Linked folder's indexing: nothing in it is read, extracted or
   * embedded until it is resumed, which scans it again. Returns it.
   */
  setLinkedFolderPaused(linkedFolderId: string, paused: boolean): Promise<LinkedFolder>;
  /**
   * Downloads the Linked folder's online-only files (reading one makes its
   * sync service download it) and indexes them. Returns at once.
   */
  downloadOnlineOnlyFiles(linkedFolderId: string): Promise<LinkedFolder>;
  /**
   * Checks a Citation again against its Document's current version (see
   * `CitationRecheck`), e.g. after the Document changed. Nothing is stored:
   * the editor writes the result onto the Citation's node.
   */
  recheckCitation(input: RecheckCitationInput): Promise<CitationRecheck>;
  /**
   * Searches the Passages of live Documents, best match first: hybrid
   * (keyword and vector) search by default, over every Document or only the
   * given ones. A "vector" search throws EmbeddingModelNotReadyError while the
   * embedding model isn't ready.
   */
  searchPassages(query: string, options?: SearchPassagesOptions): Promise<PassageSearchResult[]>;
  /** The built-in embedding model and its download. */
  getEmbeddingModel(): Promise<EmbeddingModelStatus>;
  /**
   * Starts downloading the built-in embedding model, or tries again after a
   * failure, resuming what was already downloaded. Returns at once;
   * "embeddingModel.status" events report progress. The core also starts the
   * download by itself as soon as a Document needs the model.
   */
  downloadEmbeddingModel(): Promise<EmbeddingModelStatus>;

  /** The embedding model document search uses, local mode, and any rebuild under way. */
  getEmbeddingSettings(): Promise<EmbeddingSettings>;
  /**
   * Switches document search to another embedding model (its key goes to the
   * keychain, never the database). A cloud provider's "embeddings" flow needs
   * consent first: if the User declines, nothing changes. A different model
   * means every Document is embedded again: each goes back to "embedding",
   * and "embedding.changed" events report the rebuild. Saving the same model
   * again (e.g. with a new key) re-embeds nothing. Refused for a cloud
   * provider while local mode is on.
   */
  saveEmbeddingProvider(input: SaveEmbeddingProviderInput): Promise<EmbeddingSettings>;
  /**
   * Embeds one fixed text, nothing of the User's. A cloud provider's
   * "embeddings" flow needs consent first, as for a chat provider's test.
   */
  testEmbeddingConnection(
    input: TestEmbeddingConnectionInput,
  ): Promise<EmbeddingConnectionTestResult>;
  /** Tries the embedding provider again after an error: Documents waiting for it carry on. */
  retryEmbedding(): Promise<EmbeddingSettings>;
  /**
   * Turns local mode ("keep everything on this computer") on or off on this
   * device. Turning it on switches a cloud embedding provider back to the
   * built-in model, which embeds every Document again (an embedding provider
   * on this computer, e.g. Ollama, is kept), and stops reranking.
   */
  setLocalOnly(enabled: boolean): Promise<EmbeddingSettings>;

  /** Rerank on this device: whether a Cohere or Voyage key is set up. */
  getRerankSettings(): Promise<RerankSettings>;
  /**
   * Sets up rerank, or changes it: from then on document search reranks its
   * candidates. The "rerank" flow to the service needs consent first: if the
   * User declines, nothing changes. The key goes to the keychain.
   */
  saveRerankSettings(input: SaveRerankSettingsInput): Promise<RerankSettings>;
  /** Removes rerank's key and settings from this device: search is as before. */
  removeRerankSettings(): Promise<RerankSettings>;
  /**
   * Reranks two fixed texts against a fixed query, nothing of the User's,
   * with the given settings or the saved ones. The "rerank" flow needs consent first.
   */
  testRerankConnection(input?: TestRerankConnectionInput): Promise<ConnectionTestResult>;

  listChatProviders(): Promise<ChatProvider[]>;
  /**
   * Saves a chat provider (its key goes to the keychain, never the database)
   * and makes `modelId` on it the default chat model. Saving the same kind and
   * server again updates the existing provider.
   */
  saveChatProvider(input: SaveChatProviderInput): Promise<ChatProvider>;
  /** Removes a provider and its key. If it held the default model, there is none afterwards. */
  deleteChatProvider(id: string): Promise<void>;
  /** Makes one small real request. A cloud provider's data flow needs consent first. */
  testChatConnection(input: TestChatConnectionInput): Promise<ConnectionTestResult>;
  getChatReadiness(): Promise<ChatReadiness>;
  /**
   * The models a Question's model picker offers: for each saved provider, the
   * models it lists and its default model. Providers are asked without sending
   * any User content, and a cloud one only once the User has allowed the chat
   * flow to it. A provider that isn't asked, or can't be reached, offers its default.
   */
  listChatModels(): Promise<ChatModelGroup[]>;

  /**
   * Asks a Question: its Answer is written into the Mind right after it, as the
   * model streams it ("answer.*" events follow its progress). Asking a Question
   * that already has an Answer replaces that Answer in place. Nothing is sent
   * when Questions can't be asked yet, or when the Answer has edits the User
   * hasn't agreed to lose.
   */
  askQuestion(input: AskQuestionInput): Promise<AskResult>;
  /** Writes an Answer again, in place, from its Question. Same rules as `askQuestion`. */
  regenerateAnswer(input: RegenerateAnswerInput): Promise<AskResult>;
  /**
   * Stops an Answer being written. It keeps what was written so far and is
   * marked "stopped". Stopping a finished Answer does nothing.
   */
  stopAnswer(input: StopAnswerInput): Promise<void>;

  getSecretStorage(): Promise<SecretStorageStatus>;
  /** The User accepts storing keys without keyring protection on this device. */
  acceptPlainTextSecretStorage(): Promise<SecretStorageStatus>;

  /** Looks for Ollama, on its default local port unless another URL is given. */
  detectOllama(input?: { baseUrl?: string }): Promise<OllamaStatus>;
  /**
   * One click "use local models": pulls the model if needed (progress arrives as
   * "ollama.pullProgress" events), saves Ollama as a provider and makes the model the default.
   */
  selectOllama(input?: SelectOllamaInput): Promise<ChatProvider>;

  /** The experimental ChatGPT plan provider: whether it's on, the sign-in, and its models. */
  getChatGptPlan(): Promise<ChatGptPlanStatus>;
  /**
   * Turns the experimental ChatGPT plan provider on or off on this device.
   * Turning it off signs out (deleting the tokens) and removes the provider.
   */
  setChatGptPlanEnabled(enabled: boolean): Promise<ChatGptPlanStatus>;
  /**
   * Opens the ChatGPT sign-in in the User's browser and waits, for a few
   * minutes at most, until they finish. Starting again cancels a sign-in
   * still waiting. The provider must be turned on.
   */
  signInToChatGpt(): Promise<ChatGptSignInResult>;
  /** Stops waiting for a browser sign-in. */
  cancelChatGptSignIn(): Promise<void>;
  /** Deletes the ChatGPT tokens from this device. A saved ChatGPT provider then needs a new sign-in. */
  signOutOfChatGpt(): Promise<ChatGptPlanStatus>;

  /**
   * Every registered external data flow, to each service it currently goes to
   * and each service the User has decided on, with that decision.
   */
  listDataFlows(): Promise<DataFlowStatus[]>;
  /** Consent requests still waiting for an answer, e.g. for a window that opened after they were raised. */
  listConsentRequests(): Promise<ConsentRequest[]>;
  respondToConsent(requestId: string, accept: boolean): Promise<void>;
  /** Forgets the User's decision: the next request on the flow asks again. */
  revokeConsent(flowId: DataFlowId, serviceId: string): Promise<void>;
  /**
   * Every registered external data flow, each with the services it goes to,
   * including a flow that sends nothing now: the Privacy page lists them all.
   */
  listRegisteredDataFlows(): Promise<RegisteredDataFlow[]>;
  /**
   * The User allows a flow to one of its services from Settings, without
   * waiting to be asked, e.g. after declining it. Everything the flow sends
   * counts as accepted, and a request waiting for that flow and service is
   * answered too.
   */
  allowDataFlow(flowId: DataFlowId, serviceId: string): Promise<void>;

  /** Network traffic that carries nothing of the User's, such as update checks and model downloads. */
  listNetworkTraffic(): Promise<NetworkTraffic[]>;
  getPrivacySettings(): Promise<PrivacySettings>;
  /**
   * Changes the privacy choices on this device and returns them. Turning crash
   * reports on starts them; turning them off stops them at once. Refused if
   * this copy can't send crash reports.
   */
  updatePrivacySettings(patch: PrivacySettingsPatch): Promise<PrivacySettings>;
  /**
   * The Folders of every Linked folder, as a flat list in name order (ignoring
   * case). Build the tree from each Folder's `parentId`; siblings keep the
   * list's order. Each Linked folder's own Folder has no parent.
   */
  listFolders(): Promise<Folder[]>;

  /**
   * Tags that are not deleted, in name order (ignoring case). The preset Tags
   * are created on first run, in the interface language of the time.
   */
  listTags(): Promise<Tag[]>;
  createTag(input: CreateTagInput): Promise<Tag>;
  /**
   * Changes a Tag's name or description and returns the Tag. Documents keep
   * their Tags; "Re-tag" applies the new definition to the automatic ones.
   */
  updateTag(tagId: string, patch: UpdateTagInput): Promise<Tag>;
  /** Soft-deletes a Tag, and takes it off every Document. */
  deleteTag(tagId: string): Promise<void>;
  /**
   * The User puts a Tag on a Document. From then on automatic tagging leaves
   * that Tag on that Document alone. Confirming a Tag automatic tagging
   * applied (e.g. one marked "needs review") is the same: it becomes the
   * User's. Returns the Document.
   */
  addDocumentTag(documentId: string, tagId: string): Promise<Document>;
  /**
   * The User takes a Tag off a Document. The removal is kept, so automatic
   * tagging never puts it back. Returns the Document.
   */
  removeDocumentTag(documentId: string, tagId: string): Promise<Document>;
  /**
   * Recomputes the automatic Tags of these Documents, or of every Document,
   * e.g. after Tag definitions changed. Tags the User added or removed are
   * kept as they are. Returns once the Documents are queued; "documents.tagged"
   * events report progress. Documents still being processed are tagged when they finish.
   */
  retagDocuments(documentIds?: string[]): Promise<void>;

  /** TypeSafe Jev, the optional tagger: whether it is set up on this device, and how. */
  getJevSettings(): Promise<JevSettings>;
  /**
   * Sets up Jev on this device, or changes its settings. From then on
   * automatic tagging uses Jev instead of the chat model; Documents already
   * tagged keep their Tags until re-tagged. The key goes to the keychain,
   * never the database.
   */
  saveJevSettings(input: SaveJevSettingsInput): Promise<JevSettings>;
  /** Removes Jev's key and settings from this device: automatic tagging uses the chat model again. */
  removeJevSettings(): Promise<JevSettings>;
  /**
   * Asks Jev one question about a fixed text, nothing of the User's. Like a
   * chat provider's test, a cloud service's flow (here "tagging") needs
   * consent first.
   */
  testJevConnection(input?: TestJevConnectionInput): Promise<ConnectionTestResult>;

  /** Connectors that are not deleted, in name order (ignoring case), each with its state. */
  listConnectors(): Promise<Connector[]>;
  /**
   * Adds a Connector, turned on, and starts it: "connectors.changed" events
   * follow its state. A local one's environment values, and a remote one's
   * client secret, go to the keychain; if they can't be stored safely,
   * nothing is added. A remote one that requires a sign-in ends up
   * "needs-sign-in": nothing opens in the browser until `signInToConnector`.
   */
  addConnector(input: AddConnectorInput): Promise<Connector>;
  /**
   * Changes a local Connector's name, command, arguments and environment (an
   * empty `env` keeps the saved values), and starts it again with them. Its
   * Tools' approvals and its data-flow decision stay.
   */
  editConnector(connectorId: string, input: AddLocalConnectorInput): Promise<Connector>;
  /** Turns a Connector on (starting it) or off (stopping its process, or disconnecting). */
  setConnectorEnabled(connectorId: string, enabled: boolean): Promise<Connector>;
  /** Starts a Connector that is on again, e.g. after an error. */
  restartConnector(connectorId: string): Promise<Connector>;
  /**
   * Soft-deletes a Connector: stops it, removes its environment, client
   * secret and sign-in from the keychain, and forgets the User's consent
   * decision for it (for a remote one, unless another Connector uses the same server).
   */
  deleteConnector(connectorId: string): Promise<void>;
  /**
   * Signs in to a remote Connector that is on: discovers its authorization
   * server, registers IncarnaMind with it if needed, opens the sign-in in the
   * User's browser and waits for them, a few minutes at most. Its tokens go
   * to the keychain. Starting again cancels a sign-in still waiting.
   */
  signInToConnector(connectorId: string): Promise<ConnectorSignInResult>;
  /** Stops waiting for a remote Connector's browser sign-in. */
  cancelConnectorSignIn(connectorId: string): Promise<void>;
  /** Deletes a remote Connector's tokens from this device and disconnects it. */
  signOutOfConnector(connectorId: string): Promise<Connector>;
  /**
   * Sets the OAuth app a remote Connector signs in with, for a service that
   * can't register IncarnaMind by itself; null goes back to registering
   * automatically. Either way it signs out first.
   */
  setConnectorClient(connectorId: string, client: ConnectorClientInput | null): Promise<Connector>;
  /**
   * What importing an `mcpServers` configuration (Claude Desktop's or
   * Cursor's JSON, pasted or read from a file) would add, without adding anything.
   */
  previewConnectorImport(json: string): Promise<ConnectorImportEntry[]>;
  /** Adds every server of an `mcpServers` configuration that the preview marks "add". */
  importConnectors(json: string): Promise<ConnectorImportResult>;

  /**
   * The User's approval policies, for the approvals page: every Tool always
   * allowed, every Tool switched to "ask", and every Skill whose scripts
   * always run. Policies of deleted Connectors and Skills aren't listed.
   */
  listApprovalPolicies(): Promise<ApprovalPolicy[]>;
  /**
   * Sets a policy, or with `policy: null` removes it (back to the default).
   * Returns the policy now in effect, or null for the default. "Always run"
   * for a Skill's scripts needs `riskAccepted`.
   */
  setApprovalPolicy(input: SetApprovalPolicyInput): Promise<ApprovalPolicy | null>;
  /** Removes a policy by its id: its subject goes back to the default. Revoking one that's gone does nothing. */
  revokeApprovalPolicy(policyId: string): Promise<void>;
  /** Tool calls still waiting for the User's approval, e.g. for a window that opened after they were raised. */
  listApprovalRequests(): Promise<ApprovalRequest[]>;
  /**
   * Answers an approval request. Answering one that's no longer waiting (decided
   * in another window, or its Answer stopped) does nothing. "always-allow" of a
   * Skill script ("always run") is refused unless `options.riskAccepted`: the
   * User must have seen and confirmed the risk warning first.
   */
  respondToApproval(
    requestId: string,
    decision: ApprovalDecision,
    options?: ApprovalResponseOptions,
  ): Promise<void>;

  /** Skills that are not removed, in name order. */
  listSkills(): Promise<Skill[]>;
  /**
   * Reads a Skill folder, or a zip holding one, given its absolute path, and
   * checks it: SKILL.md's frontmatter, paths and links that stay inside the
   * Skill, and its size (`SKILL_LIMITS`). Nothing is imported yet: the preview
   * says what would be, and `importSkill` imports exactly what was read.
   */
  previewSkillImport(path: string): Promise<SkillImportCheck>;
  /**
   * Imports a previewed Skill into the data folder, turned on. A Skill of the
   * same name is replaced, keeping its id and whether it's on. Previews are
   * kept in memory only: after a restart, or a few newer previews, preview again.
   */
  importSkill(importId: string): Promise<Skill>;
  /** Forgets a preview without importing it. Forgetting one that's gone does nothing. */
  cancelSkillImport(importId: string): Promise<void>;
  /** Turns a Skill on or off. Answers list and load only Skills that are on. Returns the Skill. */
  setSkillEnabled(skillId: string, enabled: boolean): Promise<Skill>;
  /**
   * Removes a Skill: soft-deleted in the database (ADR-0003). Its files are
   * deleted once nothing uses them, e.g. after Answers still being written with it finish.
   * A removed built-in Skill stays removed, through updates of the app too,
   * until `restoreBuiltInSkills`.
   */
  removeSkill(skillId: string): Promise<void>;
  /**
   * Copies a Skill as the User's own, e.g. a built-in one, whose files can't
   * be replaced: the same files, named "<name>-copy" (or "<name>-copy-2" and
   * on), turned on, not built-in. Returns the copy.
   */
  duplicateSkill(skillId: string): Promise<Skill>;
  /**
   * The names of the built-in Skills the User removed, which
   * `restoreBuiltInSkills` would install again. Empty when they are all there.
   */
  listRemovedBuiltInSkills(): Promise<string[]>;
  /**
   * Installs again, turned on, the built-in Skills the User removed (those
   * `listRemovedBuiltInSkills` names), with this version's files. Returns them.
   */
  restoreBuiltInSkills(): Promise<Skill[]>;

  /**
   * What `exportMind` will write with these options: the file name, and how
   * many of the exported Citations are unverified, so the User sees that
   * before the file is written.
   */
  previewMindExport(mindId: string, options: ExportMindOptions): Promise<MindExportPreview>;
  /**
   * Exports a Mind as a file, which the host saves: its title, its Notes
   * (those switched out of Question context too), its Answers and, if
   * included, its Questions, in order. Each Citation becomes its own footnote
   * naming its Document and pages, e.g. "Tides, p. 12–13", marked
   * "[unverified]" unless its quote was found. Math stays LaTeX.
   */
  exportMind(mindId: string, options: ExportMindOptions): Promise<MindExport>;
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
  /** A Document was added or its processing status (or embedding progress) changed. Carries the whole Document. */
  "document.status": Document;
  /** The built-in embedding model's state changed, or its download made progress. */
  "embeddingModel.status": EmbeddingModelStatus;
  /**
   * The embedding model search uses changed, or local mode, or a rebuild made
   * progress (a Document finished) or finished, or the provider failed or recovered.
   */
  "embedding.changed": EmbeddingSettings;
  /** Rerank was set up, changed or removed on this device, or paused by local mode. */
  "rerank.changed": RerankSettings;
  /** Whether Questions can be asked may have changed. */
  "chatReadiness.changed": ChatReadiness;
  /** A data flow needs the User's consent before anything is sent. */
  "consent.requested": ConsentRequest;
  /** A consent request was answered, here or in another window. */
  "consent.resolved": { requestId: string; accepted: boolean };
  /** The User allowed or revoked a data flow in Settings. */
  "dataFlows.changed": RegisteredDataFlow[];
  /** The privacy choices changed, e.g. crash reports were turned on. */
  "privacy.changed": PrivacySettings;
  /** Progress of a model download through Ollama. */
  "ollama.pullProgress": OllamaPullProgress;
  /**
   * Documents whose file moved to another Folder on disk, or into or out of
   * a Linked folder. Carries each moved Document whole.
   */
  "documents.moved": Document[];
  /** Documents removed from the index, by the User or with their Linked folder: their ids. */
  "documents.removed": string[];
  /** Folders appeared or went with the folders on disk: the list as `listFolders` now returns it. */
  "folders.changed": Folder[];
  /**
   * Linked folders were added, removed, paused or resumed, or one's status,
   * layout, progress or online-only files changed: the list as
   * `listLinkedFolders` now returns it.
   */
  "linkedFolders.changed": LinkedFolder[];
  /**
   * A Linked folder was unlinked, and the text its Citations quote kept: the
   * list as `listKeptCitationTexts` now returns it.
   */
  "keptCitationTexts.changed": KeptCitationText[];
  /** Tags were created (including the presets), edited or deleted: the list as `listTags` now returns it. */
  "tags.changed": Tag[];
  /**
   * Documents' Tags changed, or where automatic tagging is for them: the User
   * added or removed a Tag, automatic tagging started, waited, finished or
   * failed, or a deleted Tag left them. Carries each Document whole.
   */
  "documents.tagged": Document[];
  /** The ChatGPT plan provider was turned on or off, or its sign-in changed (including expiring). */
  "chatGptPlan.changed": ChatGptPlanStatus;
  /** Jev was set up, changed or removed on this device. */
  "jev.changed": JevSettings;
  /**
   * Skills were imported, duplicated, turned on or off, removed or restored:
   * the list as `listSkills` now returns it.
   */
  "skills.changed": Skill[];
  /**
   * Connectors were added, turned on or off, or deleted, or one's state or
   * sign-in changed (connecting, signing in, ready, needs sign-in, error):
   * the list as `listConnectors` now returns it.
   */
  "connectors.changed": Connector[];
  /** Approval policies were set or revoked: the list as `listApprovalPolicies` now returns it. */
  "approvals.changed": ApprovalPolicy[];
  /** A Tool call is waiting for the User's approval: its Answer shows an approval card, in every window. */
  "approval.requested": ApprovalRequest;
  /**
   * An approval request was decided, here or in another window, or its Answer
   * stopped ("deny"): every window takes its card away.
   */
  "approval.resolved": { requestId: string; decision: ApprovalDecision };
  /**
   * The Answer event stream. The core writes each Answer into its Mind's Yjs
   * document as it streams (its text, Citations and Tool calls), so every
   * window sees it; these events are for UI state, e.g. the stop button, and
   * for the evaluation.
   */
  "answer.started": AnswerStarted;
  /** The Answer is loading its model, searching, or writing: its meta line says which. */
  "answer.phase": AnswerPhaseEvent;
  "answer.delta": AnswerDelta;
  "answer.toolCallStarted": AnswerToolCallEvent;
  "answer.toolCallFinished": AnswerToolCallEvent;
  "answer.citationAdded": AnswerCitationAdded;
  "answer.finished": AnswerFinished;
  "answer.failed": AnswerFailed;
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
  retryDocument: true,
  readDocumentText: true,
  previewLinkedFolder: true,
  addLinkedFolder: true,
  listLinkedFolders: true,
  removeLinkedFolder: true,
  listKeptCitationTexts: true,
  setLinkedFolderLayout: true,
  setLinkedFolderPaused: true,
  downloadOnlineOnlyFiles: true,
  recheckCitation: true,
  searchPassages: true,
  getEmbeddingModel: true,
  downloadEmbeddingModel: true,
  getEmbeddingSettings: true,
  saveEmbeddingProvider: true,
  testEmbeddingConnection: true,
  retryEmbedding: true,
  setLocalOnly: true,
  getRerankSettings: true,
  saveRerankSettings: true,
  removeRerankSettings: true,
  testRerankConnection: true,
  listChatProviders: true,
  saveChatProvider: true,
  deleteChatProvider: true,
  testChatConnection: true,
  getChatReadiness: true,
  listChatModels: true,
  askQuestion: true,
  regenerateAnswer: true,
  stopAnswer: true,
  getSecretStorage: true,
  acceptPlainTextSecretStorage: true,
  detectOllama: true,
  selectOllama: true,
  getChatGptPlan: true,
  setChatGptPlanEnabled: true,
  signInToChatGpt: true,
  cancelChatGptSignIn: true,
  signOutOfChatGpt: true,
  listDataFlows: true,
  listConsentRequests: true,
  respondToConsent: true,
  revokeConsent: true,
  listRegisteredDataFlows: true,
  allowDataFlow: true,
  listNetworkTraffic: true,
  getPrivacySettings: true,
  updatePrivacySettings: true,
  listFolders: true,
  listTags: true,
  createTag: true,
  updateTag: true,
  deleteTag: true,
  addDocumentTag: true,
  removeDocumentTag: true,
  retagDocuments: true,
  getJevSettings: true,
  saveJevSettings: true,
  removeJevSettings: true,
  testJevConnection: true,
  listConnectors: true,
  addConnector: true,
  editConnector: true,
  setConnectorEnabled: true,
  restartConnector: true,
  deleteConnector: true,
  signInToConnector: true,
  cancelConnectorSignIn: true,
  signOutOfConnector: true,
  setConnectorClient: true,
  previewConnectorImport: true,
  importConnectors: true,
  listApprovalPolicies: true,
  setApprovalPolicy: true,
  revokeApprovalPolicy: true,
  listApprovalRequests: true,
  respondToApproval: true,
  listSkills: true,
  previewSkillImport: true,
  importSkill: true,
  cancelSkillImport: true,
  setSkillEnabled: true,
  removeSkill: true,
  duplicateSkill: true,
  listRemovedBuiltInSkills: true,
  restoreBuiltInSkills: true,
  previewMindExport: true,
  exportMind: true,
};

/** Every method of CoreApi, used to wire the IPC bridge. */
export const coreApiMethods = Object.keys(methods) as CoreApiMethod[];
