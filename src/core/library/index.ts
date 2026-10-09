import { randomUUID } from "node:crypto";
import { translate } from "../../shared/i18n";
import { supportsLibraryImages } from "../../shared/libraryModels";
import { libraryPresetKeys } from "../../shared/libraryPresets";
import type { DocumentKind, ProviderError } from "../api";
import type { BackgroundQueue } from "../backgroundQueue";
import {
  ChatNotReadyError,
  ConsentDeclinedError,
  InvalidInputError,
  isRecord,
  NotFoundError,
  TaggingNotReadyError,
} from "../errors";
import { normalizeBaseUrl, serviceForUrl } from "../providers/kinds";
import { classifyProviderError } from "../providers/providerErrors";
import { isChatModelChoice, type SettingsStore } from "../settings";
import type { Database } from "../storage";
import { sameName, type TagsStore } from "../tags";
import { excerptFromPassages } from "../tags/classify";
import type { GroupClassifier } from "./classifier";
import type { DocumentPageImage } from "./pdfImages";
import { pdfNeedsPageImages } from "./routing";
import type {
  ClassificationModel,
  ClassificationStatus,
  DocumentGroupAssignment,
  LibraryGroup,
  LibrarySettings,
  LibrarySnapshot,
} from "./types";

interface AssignmentRow {
  document_id: string;
  group_id: string | null;
  source: "automatic" | "user";
  status: ClassificationStatus;
  content_hash: string | null;
  request_id: string;
  error_kind: ProviderError["kind"] | null;
  error_message: string | null;
  classification_model: string | null;
}
interface DocumentRow {
  id: string;
  name: string;
  kind: DocumentKind;
  page_count: number | null;
  content_hash: string | null;
  status: string;
}
const GROUP_COLUMNS = "id, name, description, created_at AS createdAt, updated_at AS updatedAt";

/** One flat group per Document; user choices, including Unsorted, always win. */
export function createLibrary(options: {
  db: Database;
  now(): string;
  settings: SettingsStore;
  background: BackgroundQueue;
  changed(): void;
  tags: TagsStore;
  tagged(ids: string[]): void;
  prepare(settings: LibrarySettings): Promise<GroupClassifier>;
  providerExists(id: string): boolean;
  pageImages(id: string, contentHash: string, signal: AbortSignal): Promise<DocumentPageImage[]>;
}) {
  const { db, now, settings, background, changed } = options;
  const lifetime = new AbortController();
  const queued = new Set<string>();
  let revision = 0;
  const getSettings = (): LibrarySettings =>
    (settings.readDeviceValue("library") as LibrarySettings | null) ?? {
      classifier: null,
      automatic: false,
    };
  const groups = () =>
    db.all<LibraryGroup>(
      `SELECT ${GROUP_COLUMNS} FROM library_groups WHERE deleted_at IS NULL ORDER BY name COLLATE NOCASE, id`,
    );
  const group = (id: unknown) => {
    if (typeof id !== "string") throw new InvalidInputError("A group id must be text.");
    const value = groups().find((item) => item.id === id);
    if (!value) throw new NotFoundError("That group no longer exists.");
    return value;
  };
  const document = (id: unknown) => {
    if (typeof id !== "string") throw new InvalidInputError("A Document id must be text.");
    const value = db.get<DocumentRow>(
      "SELECT id, name, kind, page_count, content_hash, status FROM documents WHERE id = ? AND deleted_at IS NULL",
      [id],
    );
    if (!value) throw new NotFoundError("That Document no longer exists.");
    return value;
  };
  const assignment = (id: string) =>
    db.get<AssignmentRow>(
      "SELECT * FROM document_groups WHERE document_id = ? AND deleted_at IS NULL",
      [id],
    );
  const hasContent = (doc: DocumentRow) => {
    const classifier = getSettings().classifier;
    const canReadPages =
      doc.kind === "pdf" &&
      (classifier?.kind === "auto" ||
        (classifier?.kind === "ollama" &&
          supportsLibraryImages(classifier.modelId) &&
          classifier.usePageImages === true));
    return (
      (canReadPages && doc.status === "no-text") ||
      (["ready", "waiting-for-model", "embedding"].includes(doc.status) &&
        !!db.get("SELECT 1 FROM passages WHERE document_id = ? AND deleted_at IS NULL LIMIT 1", [
          doc.id,
        ]))
    );
  };
  const needsPageImages = (doc: DocumentRow) =>
    doc.kind === "pdf" &&
    pdfNeedsPageImages(
      db.all<{ page: number; text: string }>(
        "SELECT page, substr(text, 1, 4000) AS text FROM document_pages WHERE document_id = ? AND content_hash = ? AND deleted_at IS NULL AND page BETWEEN 1 AND 12 ORDER BY page LIMIT 12",
        [doc.id, doc.content_hash],
      ),
      doc.page_count,
    );
  const tagState = (id: string, status: ClassificationStatus, error?: ProviderError) => {
    const tagging = {
      pending: "pending",
      classifying: "tagging",
      classified: "tagged",
      waiting: "waiting-for-provider",
      failed: "failed",
    }[status];
    db.run(
      "UPDATE documents SET tagging_status = ?, tagging_error_kind = ?, tagging_error_message = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
      [tagging, error?.kind ?? null, error?.message ?? null, now(), id],
    );
  };
  const state = (id: string, status: ClassificationStatus, error?: ProviderError) => {
    db.run(
      "UPDATE document_groups SET status = ?, error_kind = ?, error_message = ?, updated_at = ? WHERE document_id = ? AND deleted_at IS NULL",
      [status, error?.kind ?? null, error?.message ?? null, now(), id],
    );
    tagState(id, status, error);
    options.tagged([id]);
    changed();
  };
  const write = (
    doc: DocumentRow,
    groupId: string | null,
    source: "automatic" | "user",
    status: ClassificationStatus,
    model: ClassificationModel | null = null,
  ) => {
    const at = now();
    const existing = assignment(doc.id);
    if (existing) {
      db.run(
        "UPDATE document_groups SET group_id = ?, source = ?, status = ?, content_hash = ?, request_id = ?, error_kind = NULL, error_message = NULL, updated_at = ?, classification_model = ? WHERE document_id = ? AND deleted_at IS NULL",
        [
          groupId,
          source,
          status,
          doc.content_hash,
          randomUUID(),
          at,
          model ? JSON.stringify(model) : null,
          doc.id,
        ],
      );
    } else {
      db.run(
        "INSERT INTO document_groups (id, document_id, group_id, source, status, content_hash, request_id, created_at, updated_at, classification_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
          randomUUID(),
          doc.id,
          groupId,
          source,
          status,
          doc.content_hash,
          randomUUID(),
          at,
          at,
          model ? JSON.stringify(model) : null,
        ],
      );
    }
  };

  function enqueue(id: string): void {
    if (lifetime.signal.aborted || queued.has(id)) return;
    if (!getSettings().classifier || groups().length === 0) {
      if (assignment(id)?.status === "pending") state(id, "waiting");
      return;
    }
    let doc: DocumentRow;
    try {
      doc = document(id);
    } catch {
      return;
    }
    if (!hasContent(doc) || !assignment(id)) return;
    queued.add(id);
    background.add({
      kind: "classification",
      async run(call) {
        const initial = assignment(id);
        const version = revision;
        const stale = () =>
          lifetime.signal.aborted ||
          call.signal.aborted ||
          version !== revision ||
          assignment(id)?.request_id !== initial?.request_id;
        try {
          if (!initial || initial.status === "classified") return;
          const current = document(id);
          if (!hasContent(current)) {
            state(id, "waiting");
            return;
          }
          const definitions = groups();
          const tagDefinitions = options.tags.list();
          if (!getSettings().classifier || definitions.length === 0) {
            state(id, "waiting");
            return;
          }
          state(id, "classifying");
          const classifier = await options.prepare(getSettings());
          if (classifier.local) call.runsLocally();
          if (stale()) return;
          const passages = db.all<{ text: string }>(
            "SELECT text FROM passages WHERE document_id = ? AND deleted_at IS NULL ORDER BY position LIMIT 6",
            [id],
          );
          const signal = AbortSignal.any([lifetime.signal, call.signal]);
          const images =
            classifier.pageImages &&
            current.kind === "pdf" &&
            (classifier.pageImages !== "auto" || needsPageImages(current))
              ? await options.pageImages(id, current.content_hash ?? "", signal)
              : [];
          if (stale()) return;
          const result = await classifier.organize(
            definitions,
            tagDefinitions,
            {
              name: current.name,
              kind: current.kind,
              pageCount: current.page_count,
              text: excerptFromPassages(passages.map((row) => row.text)),
            },
            signal,
            images,
          );
          if (stale()) return;
          const latest = document(id);
          if (!hasContent(latest) || latest.content_hash !== current.content_hash) {
            state(id, "pending");
            return;
          }
          // Definition changes make the complete result stale, including tags.
          if (JSON.stringify(tagDefinitions) !== JSON.stringify(options.tags.list())) {
            state(id, "pending");
            return;
          }
          const { groupId } = result;
          if (groupId !== null && !definitions.some((item) => item.id === groupId))
            throw new Error("The classifier returned an unknown group.");
          db.transaction(() => {
            write(
              latest,
              initial.source === "user" ? initial.group_id : groupId,
              initial.source,
              "classified",
              classifier.model
                ? { ...classifier.model, images: images.length > 0 && classifier.model.images }
                : null,
            );
            options.tags.applyAutomatic(id, result.tags);
            tagState(id, "classified");
          });
          options.tagged([id]);
          changed();
        } catch (error) {
          if (!stale()) {
            if (error instanceof NotFoundError) return;
            const waiting =
              error instanceof ChatNotReadyError ||
              error instanceof ConsentDeclinedError ||
              error instanceof TaggingNotReadyError;
            state(id, waiting ? "waiting" : "failed", classifyProviderError(error));
          }
        } finally {
          if (!lifetime.signal.aborted) {
            if (assignment(id)?.status === "classifying") state(id, "pending");
            if (!call.gaveWay) {
              queued.delete(id);
              if (assignment(id)?.status === "pending") enqueue(id);
            }
          }
        }
      },
    });
  }

  function resume(): void {
    if (lifetime.signal.aborted) return;
    const waiting = db.all<{ document_id: string }>(
      "SELECT g.document_id FROM document_groups g JOIN documents d ON d.id = g.document_id WHERE g.deleted_at IS NULL AND d.deleted_at IS NULL AND g.status IN ('pending', 'waiting')",
    );
    // Existing batches run text first, then PDFs needing images, avoiding model churn.
    const ordered =
      getSettings().classifier?.kind === "auto"
        ? waiting
            .map((row) => ({ ...row, visual: needsPageImages(document(row.document_id)) }))
            .sort((a, b) => Number(a.visual) - Number(b.visual))
        : waiting;
    for (const row of ordered) enqueue(row.document_id);
  }

  function create(input: unknown): LibraryGroup {
    const value = parseGroup(input);
    const at = now();
    const id = randomUUID();
    db.run(
      "INSERT INTO library_groups (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
      [id, value.name, value.description, at, at],
    );
    revision++;
    changed();
    resume();
    return group(id);
  }

  function parseGroup(input: unknown, exceptId?: string) {
    if (!isRecord(input) || typeof input.name !== "string" || typeof input.description !== "string")
      throw new InvalidInputError("Enter a folder name and description.");
    const name = input.name.trim();
    const description = input.description.trim();
    if (!name || name.length > 100 || description.length > 500)
      throw new InvalidInputError(
        "Use a name of 1–100 characters and a description of at most 500 characters.",
      );
    if (groups().some((item) => item.id !== exceptId && sameName(item.name, name)))
      throw new InvalidInputError("A folder with this name already exists.");
    if (!exceptId && groups().length >= 100)
      throw new InvalidInputError("The Library supports up to 100 folders.");
    return { name, description };
  }

  return {
    settings: getSettings,
    documentIds(folderId: string): string[] {
      return db
        .all<{ document_id: string }>(
          "SELECT a.document_id FROM document_groups a JOIN library_groups g ON g.id = a.group_id JOIN documents d ON d.id = a.document_id WHERE g.id = ? AND g.deleted_at IS NULL AND a.deleted_at IS NULL AND d.deleted_at IS NULL",
          [folderId],
        )
        .map((row) => row.document_id);
    },
    snapshot(): LibrarySnapshot {
      const assignments: DocumentGroupAssignment[] = db
        .all<AssignmentRow>(
          "SELECT g.* FROM document_groups g JOIN documents d ON d.id = g.document_id WHERE g.deleted_at IS NULL AND d.deleted_at IS NULL",
        )
        .map((row) => ({
          documentId: row.document_id,
          groupId: row.group_id,
          source: row.source,
          status: row.status,
          error: row.error_kind ? { kind: row.error_kind, message: row.error_message ?? "" } : null,
          model: row.classification_model
            ? (JSON.parse(row.classification_model) as ClassificationModel)
            : null,
        }));
      return { groups: groups(), assignments, settings: getSettings() };
    },
    create,
    update(id: unknown, input: unknown) {
      const existing = group(id);
      const value = parseGroup(input, existing.id);
      db.run("UPDATE library_groups SET name = ?, description = ?, updated_at = ? WHERE id = ?", [
        value.name,
        value.description,
        now(),
        existing.id,
      ]);
      revision++;
      changed();
      return group(existing.id);
    },
    delete(id: unknown) {
      const existing = group(id);
      db.transaction(() => {
        db.run("UPDATE library_groups SET deleted_at = ?, updated_at = ? WHERE id = ?", [
          now(),
          now(),
          existing.id,
        ]);
        db.run(
          "UPDATE document_groups SET group_id = NULL, updated_at = ? WHERE group_id = ? AND deleted_at IS NULL",
          [now(), existing.id],
        );
      });
      revision++;
      changed();
    },
    addStarters(input: unknown) {
      if (!Array.isArray(input) || !input.every((key) => libraryPresetKeys.includes(key)))
        throw new InvalidInputError("Choose valid starter groups.");
      const language = settings.get().language;
      // Validate the complete input before changing anything; repeated selections are harmless.
      db.transaction(() => {
        for (const key of new Set(input as (typeof libraryPresetKeys)[number][])) {
          const name = translate(language, `library.preset.${key}`);
          if (!groups().some((item) => sameName(item.name, name)))
            create({ name, description: translate(language, `library.preset.${key}.description`) });
        }
      });
    },
    saveSettings(input: unknown) {
      if (!isRecord(input) || typeof input.automatic !== "boolean")
        throw new InvalidInputError("Choose classification settings.");
      const selected = input.classifier;
      if (selected !== null) {
        if (!isRecord(selected)) throw new InvalidInputError("Choose a classifier.");
        if (selected.kind === "chat") {
          if (
            !isChatModelChoice(selected.choice) ||
            !options.providerExists(selected.choice.providerId)
          )
            throw new InvalidInputError("Choose a connected provider and a model.");
        } else if (selected.kind === "auto") {
          if (typeof selected.baseUrl !== "string")
            throw new InvalidInputError("Enter the local Ollama server.");
          selected.baseUrl = normalizeBaseUrl(selected.baseUrl);
          if (serviceForUrl(selected.baseUrl as string))
            throw new InvalidInputError("Automatic classification must run on this computer.");
        } else if (selected.kind === "ollama") {
          if (
            typeof selected.baseUrl !== "string" ||
            typeof selected.modelId !== "string" ||
            !selected.modelId.trim() ||
            selected.modelId.length > 200
          )
            throw new InvalidInputError("Enter the local server and decision model.");
          if (selected.usePageImages !== undefined && typeof selected.usePageImages !== "boolean")
            throw new InvalidInputError("Choose whether to include PDF page images.");
          selected.modelId = selected.modelId.trim();
          if (selected.usePageImages === true && !supportsLibraryImages(selected.modelId as string))
            throw new InvalidInputError("PDF page images currently require local Clef-Flash.");
          selected.baseUrl = normalizeBaseUrl(selected.baseUrl);
          if (serviceForUrl(selected.baseUrl as string))
            throw new InvalidInputError("The Ollama decision model must run on this computer.");
        } else if (selected.kind !== "jev")
          throw new InvalidInputError("Choose a supported classifier.");
      }
      if (input.automatic && selected === null)
        throw new InvalidInputError(
          "Choose a classifier before enabling automatic classification.",
        );
      settings.writeDeviceValue("library", { classifier: selected, automatic: input.automatic });
      if (selected === null) {
        db.run(
          "UPDATE document_groups SET status = 'waiting', updated_at = ? WHERE status IN ('pending', 'classifying') AND deleted_at IS NULL",
          [now()],
        );
      }
      revision++;
      changed();
      resume();
    },
    classify(input: unknown) {
      if (!getSettings().classifier)
        throw new InvalidInputError("Choose a classification model first.");
      if (groups().length === 0) throw new InvalidInputError("Create or choose folders first.");
      if (
        input !== undefined &&
        (!Array.isArray(input) || !input.every((id) => typeof id === "string" && id !== ""))
      )
        throw new InvalidInputError("Choose valid Documents.");
      const ids =
        input === undefined
          ? db
              .all<{ id: string }>("SELECT id FROM documents WHERE deleted_at IS NULL")
              .map((row) => row.id)
          : [...new Set(input as string[])];
      const docs = ids.map(document);
      db.transaction(() => {
        for (const doc of docs) {
          const existing = assignment(doc.id);
          write(
            doc,
            existing?.group_id ?? null,
            existing?.source ?? "automatic",
            hasContent(doc) ? "pending" : "waiting",
          );
          tagState(doc.id, hasContent(doc) ? "pending" : "waiting");
        }
      });
      options.tagged(docs.map((doc) => doc.id));
      changed();
      resume();
    },
    assign(id: unknown, groupId: unknown) {
      const doc = document(id);
      if (groupId !== null) group(groupId);
      write(doc, groupId as string | null, "user", "classified");
      // A manual correction cancels an in-flight suggestion, including its busy indicator.
      db.run(
        "UPDATE documents SET tagging_status = 'waiting-for-provider' WHERE id = ? AND tagging_status IN ('pending', 'tagging')",
        [doc.id],
      );
      options.tagged([doc.id]);
      changed();
    },
    documentChanged(id: string) {
      if (lifetime.signal.aborted) return;
      const doc = document(id);
      const existing = assignment(id);
      if (!hasContent(doc)) return;
      if (
        getSettings().automatic &&
        groups().length > 0 &&
        (!existing || existing.content_hash !== doc.content_hash)
      ) {
        write(doc, existing?.group_id ?? null, existing?.source ?? "automatic", "pending");
        changed();
      }
      const status = assignment(id)?.status;
      if (status === "pending" || status === "waiting") enqueue(id);
    },
    resume,
    start() {
      db.run(
        "UPDATE document_groups SET status = 'pending' WHERE status = 'classifying' AND deleted_at IS NULL",
      );
      resume();
    },
    close() {
      lifetime.abort();
    },
  };
}
