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
 * Attributes of a Question Block, as stored in the Mind's Yjs document. Later
 * tickets add its Search scope (#30) and a forced Skill.
 */
export interface QuestionAttributes {
  id: string | null;
  /** The model picked for this Question, overriding the default. Both null: the default model. */
  providerId: string | null;
  modelId: string | null;
}

/** Where an Answer is: being written, finished, stopped by the User, or failed (see `errorKind`). */
export type AnswerStatus = "streaming" | "done" | "stopped" | "failed";

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
  /** The User chose "set up later" on the first-run chat setup screen. */
  chatSetupDismissed: boolean;
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
 * Where a Document is in processing: "queued", then "extracting" its text, then
 * "embedding" its Passages with the built-in model, then "ready". Before the
 * model has been downloaded, a Document waits after extracting as
 * "waiting-for-model", and carries on by itself once the download finishes;
 * keyword search already finds its Passages. The other end states are
 * "failed" (see `failure`) and "no-text": the file has no text to extract,
 * e.g. a scan without a text layer.
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

/**
 * Where automatic tagging is for a Document. It runs once the Document is
 * "ready", and never holds that up: a Document is searchable as soon as it is
 * embedded, tagged or not.
 * - "pending": tagged once processing finishes, or about to be.
 * - "waiting-for-provider": no chat model can be used yet (none is set up, its
 *   key or sign-in is missing, or the User declined sending excerpts to its
 *   service). Tagging resumes by itself once one can.
 * - "tagging": the chat model is choosing the Document's Tags.
 * - "tagged": its automatic Tags are up to date.
 * - "failed": the chat model's provider failed (see `taggingError`); a re-tag,
 *   a change of chat model or a restart tries again.
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
  /** How sure automatic tagging was, from 0 to 1, when its model says. Null for the chat model and for the User's Tags. */
  confidence: number | null;
  /** Automatic tagging wasn't sure, so the User may want to check it. */
  needsReview: boolean;
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
  /** While `status` is "embedding": the share of its Passages embedded so far, from 0 to 1. Null otherwise. */
  progress: number | null;
  /** Set when `status` is "failed". */
  failure: DocumentFailure | null;
  /** The Folder the Document is filed in, or null if it is unfiled. A Document is in at most one Folder. */
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
  /** Only Documents filed in this Folder. Omitted: every Document, filed or not. */
  folderId?: string;
  /** With `folderId`, also Documents filed in its sub-Folders, at any depth. Defaults to false. */
  includeSubfolders?: boolean;
  /** Only Documents that carry this Tag. Combines with `folderId`: both must match. */
  tagId?: string;
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

/** A place where the User files Documents by hand. Folders nest, with no depth limit. */
export interface Folder {
  /** A random UUID generated on this device. */
  id: string;
  /** Never empty. */
  name: string;
  /** The Folder this one is in, or null at the top level. */
  parentId: string | null;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface CreateFolderInput {
  /** Trimmed; must not be empty. */
  name: string;
  /** The Folder to create it in. Omitted or null: at the top level. */
  parentId?: string | null;
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
 * - "vector": the built-in embedding model's vectors, by cosine similarity.
 *   Always ranks every embedded Passage, however unrelated.
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
  /** The first page the Passage covers, from 1. Null for Documents without pages (TXT, Markdown). */
  pageFrom: number | null;
  /** The last page the Passage covers. A Passage can cross a page break. */
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
  | { asked: false; reason: "edited"; answerId: string };

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

/** An Answer is complete ("done") or the User stopped it ("stopped"), keeping what was written. */
export interface AnswerFinished {
  mindId: string;
  answerId: string;
  status: "done" | "stopped";
}

/** An Answer failed; the Answer shows the error by kind. */
export interface AnswerFailed {
  mindId: string;
  answerId: string;
  error: ProviderError;
}

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
] as const;

export type DataKind = (typeof dataKinds)[number];

/**
 * External data flows. The UI names each one (`consent.flow.<id>`). Later tickets add theirs.
 * - "chat": Questions, to the chat provider they are asked with.
 * - "tagging": automatic tagging, to the default chat model's provider.
 */
export const dataFlowIds = ["chat", "tagging"] as const;

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
  /** ISO 8601, UTC; null when not asked. */
  decidedAt: string | null;
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
   * Adds PDF, TXT and Markdown files, given their absolute paths. Each file is
   * copied into the data folder and queued for processing; "document.status"
   * events report its progress.
   */
  addDocuments(paths: string[]): Promise<AddDocumentsResult>;
  /**
   * Documents that are not deleted, most recently added first. With a Folder,
   * only the Documents filed in it, and in its sub-Folders if asked; with a
   * Tag, only the Documents carrying it.
   */
  listDocuments(options?: ListDocumentsOptions): Promise<Document[]>;
  /** Returns the renamed Document. */
  renameDocument(id: string, name: string): Promise<Document>;
  /**
   * Soft-deletes a Document and its Passages, so search ignores them. The stored
   * file is removed once no Document uses it.
   */
  deleteDocument(id: string): Promise<void>;
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
   * Files a Document in a Folder, or takes it out to unfiled with `null`. It
   * leaves any Folder it was in. Returns the Document.
   */
  moveDocument(documentId: string, folderId: string | null): Promise<Document>;
  createFolder(input: CreateFolderInput): Promise<Folder>;
  /**
   * Folders that are not deleted, as a flat list in name order (ignoring case).
   * Build the tree from each Folder's `parentId`; siblings keep the list's order.
   */
  listFolders(): Promise<Folder[]>;
  /** Changes a Folder's name (trimmed; must not be empty) and returns the Folder. */
  renameFolder(folderId: string, name: string): Promise<Folder>;
  /**
   * Moves a Folder, with everything in it, into another Folder, or to the top
   * level with `null`. Moving a Folder into itself or one of its own sub-Folders
   * is refused. Returns the Folder.
   */
  moveFolder(folderId: string, parentId: string | null): Promise<Folder>;
  /**
   * Soft-deletes a Folder and all its sub-Folders. The Documents filed in them
   * are kept, and become unfiled: Documents are never deleted with a Folder.
   */
  deleteFolder(folderId: string): Promise<void>;

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
   * that Tag on that Document alone. Returns the Document.
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
  /** Whether Questions can be asked may have changed. */
  "chatReadiness.changed": ChatReadiness;
  /** A data flow needs the User's consent before anything is sent. */
  "consent.requested": ConsentRequest;
  /** A consent request was answered, here or in another window. */
  "consent.resolved": { requestId: string; accepted: boolean };
  /** Progress of a model download through Ollama. */
  "ollama.pullProgress": OllamaPullProgress;
  /**
   * Documents moved into a Folder or out to unfiled, including those unfiled
   * because their Folder was deleted. Carries each moved Document whole.
   */
  "documents.moved": Document[];
  /** Folders were created, renamed, moved or deleted: the list as `listFolders` now returns it. */
  "folders.changed": Folder[];
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
  /**
   * The Answer event stream. The core writes each Answer into its Mind's Yjs
   * document as it streams, so every window sees it; these events are for UI
   * state, e.g. the stop button. #30 adds Tool calls and Citations.
   */
  "answer.started": AnswerStarted;
  "answer.delta": AnswerDelta;
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
  searchPassages: true,
  getEmbeddingModel: true,
  downloadEmbeddingModel: true,
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
  moveDocument: true,
  createFolder: true,
  listFolders: true,
  renameFolder: true,
  moveFolder: true,
  deleteFolder: true,
  listTags: true,
  createTag: true,
  updateTag: true,
  deleteTag: true,
  addDocumentTag: true,
  removeDocumentTag: true,
  retagDocuments: true,
};

/** Every method of CoreApi, used to wire the IPC bridge. */
export const coreApiMethods = Object.keys(methods) as CoreApiMethod[];
