/**
 * Conversations (CONTEXT.md, ADR-0017): the assistant's own history, kept in
 * SQLite beside the Minds, never as Minds. A conversation is made with its first
 * Question and belongs to the Folder of the Mind that Question was asked from;
 * its messages, Questions and Answers, are rows in order. An Answer is the
 * Markdown the model wrote, with a marker where each Citation goes, and its
 * Citations (with their markers), written as it streams by the Answer code
 * (../answers). Deleting marks the conversation deleted (ADR-0003); deleting it
 * permanently is Recently deleted's (../recentlyDeleted).
 */
import { randomUUID } from "node:crypto";
import { conversationName, cutTitle } from "../../shared/naming";
import {
  addCitedUnit,
  addCitedVersion,
  type CitationFields,
  type CitedUnits,
  type CitedVersions,
} from "../answers/citedVersions";
import type {
  AnswerStatus,
  AnswerToolCall,
  AskedFrom,
  ChatModelChoice,
  CitationSupport,
  Conversation,
  ConversationAnswer,
  ConversationContent,
  ConversationMatch,
  ConversationMessage,
  ConversationQuestion,
  MarkedCitation,
  MovedItem,
  ProviderError,
  ProviderErrorKind,
  SearchScope,
} from "../api";
import { InvalidInputError, isRecord, NotFoundError } from "../errors";
import { isChatModelChoice } from "../settings";
import type { Database, SqlValue } from "../storage";
import { excerptOf, findTerms, indexText, plainText, searchTerms, termQuery } from "./search";

/** A Question to add, read from `askInConversation`'s input (see `parseQuestion`). */
export interface QuestionInput {
  /** The conversation to add it to; null starts one. */
  conversationId: string | null;
  text: string;
  /** The Mind in focus, and its Folder: a new conversation's. Null: none. */
  mind: { id: string; folderId: string | null } | null;
  scope: SearchScope | null;
  model: ChatModelChoice | null;
  forcedSkill: string | null;
}

/** An Answer as it stands, to write (see `writeAnswer`); `outcome` once it has ended. */
export interface AnswerWrite {
  text: string;
  citations: MarkedCitation[];
  toolCalls: AnswerToolCall[];
  support: CitationSupport | null;
  outcome?: { status: Exclude<AnswerStatus, "streaming">; error: ProviderError | null };
}

/** A Question with its Answer, being written, and the messages before them (its Question context). */
export interface Turn {
  conversation: Conversation;
  question: ConversationQuestion;
  answer: ConversationAnswer;
  earlier: ConversationMessage[];
}

interface ConversationRow {
  seq: number;
  id: string;
  title: string;
  auto_title: string;
  folder_id: string | null;
  created_at: string;
  updated_at: string;
}

interface MessageRow {
  seq: number;
  id: string;
  conversation_id: string;
  position: number;
  kind: string;
  question_id: string | null;
  text: string;
  asked_from_mind_id: string | null;
  scope: string | null;
  provider_id: string | null;
  model_id: string | null;
  forced_skill: string | null;
  status: string | null;
  error_kind: string | null;
  error_message: string | null;
  citation_support: string | null;
  citations: string;
  tool_calls: string;
  created_at: string;
  updated_at: string;
  /** Joined in: the asked-from Mind's name, and 1 if it is deleted or not on this computer. */
  mind_name: string | null;
  mind_gone: number | null;
}

const CONVERSATION_COLUMNS = "seq, id, title, auto_title, folder_id, created_at, updated_at";

/** A Mind's name as `mindName` gives it: its title, else its first line. */
const MIND_NAME = "CASE WHEN mind.title <> '' THEN mind.title ELSE mind.auto_title END";
const MIND_GONE = "mind.id IS NULL OR mind.deleted_at IS NOT NULL";

const MESSAGE_SELECT = `
  SELECT msg.*, ${MIND_NAME} AS mind_name, ${MIND_GONE} AS mind_gone
  FROM conversation_messages msg
  LEFT JOIN minds mind ON mind.id = msg.asked_from_mind_id`;

/** The order of a conversation's messages; a sync that brings two at one position orders them by time, then id. */
const MESSAGE_ORDER = "msg.position, msg.created_at, msg.id";

const IN = "IN (SELECT value FROM json_each(?))";

const STATUSES: readonly AnswerStatus[] = ["streaming", "done", "stopped", "failed"];
const SUPPORTS: readonly CitationSupport[] = ["tools", "structured-output", "none"];
const ERROR_KINDS: Record<ProviderErrorKind, true> = {
  auth: true,
  model: true,
  "rate-limit": true,
  network: true,
  provider: true,
  "consent-declined": true,
  "not-signed-in": true,
  "plan-limit": true,
  blocked: true,
  "too-long": true,
  unknown: true,
};
const isErrorKind = (value: string | null): value is ProviderErrorKind =>
  value !== null && Object.hasOwn(ERROR_KINDS, value);

/** A conversation's name: the title the User typed, else its own. */
const nameOf = (row: { title: string; auto_title: string }) => row.title || row.auto_title;

/** The records in a list stored as JSON; none when it isn't one (written by a newer version, say). */
function storedList<T>(value: string): T[] {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? (parsed.filter(isRecord) as T[]) : [];
  } catch {
    return [];
  }
}

function parseIdList(value: unknown, what: string): string[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || !value.every((id) => typeof id === "string" && id !== "")) {
    throw new InvalidInputError(`${what} must be a list of ids.`);
  }
  return [...new Set(value as string[])];
}

/** A Search scope from input; one that names nothing is none (null). */
function parseScope(value: unknown): SearchScope | null {
  if (value === undefined || value === null) return null;
  if (!isRecord(value)) throw new InvalidInputError("A Search scope must be an object.");
  const scope = {
    folderIds: parseIdList(value.folderIds, "A Search scope's Folders"),
    tagIds: parseIdList(value.tagIds, "A Search scope's Tags"),
    documentIds: parseIdList(value.documentIds, "A Search scope's Documents"),
  };
  const empty = scope.folderIds.length + scope.tagIds.length + scope.documentIds.length === 0;
  return empty ? null : scope;
}

function storedScope(value: string | null): SearchScope | null {
  if (value === null) return null;
  try {
    return parseScope(JSON.parse(value));
  } catch {
    return null;
  }
}

const modelOf = (row: Pick<MessageRow, "provider_id" | "model_id">): ChatModelChoice | null =>
  row.provider_id && row.model_id ? { providerId: row.provider_id, modelId: row.model_id } : null;

const askedFromOf = (row: MessageRow): AskedFrom | null =>
  row.asked_from_mind_id === null
    ? null
    : { mindId: row.asked_from_mind_id, name: row.mind_name ?? "", deleted: row.mind_gone !== 0 };

function toQuestion(row: MessageRow): ConversationQuestion {
  return {
    kind: "question",
    id: row.id,
    conversationId: row.conversation_id,
    text: row.text,
    askedFrom: askedFromOf(row),
    scope: storedScope(row.scope),
    model: modelOf(row),
    forcedSkill: row.forced_skill,
    createdAt: row.created_at,
  };
}

function toAnswer(row: MessageRow): ConversationAnswer {
  const status = STATUSES.find((each) => each === row.status) ?? "stopped";
  return {
    kind: "answer",
    id: row.id,
    conversationId: row.conversation_id,
    questionId: row.question_id ?? "",
    text: row.text,
    citations: storedList<MarkedCitation>(row.citations),
    model: modelOf(row),
    status,
    errorKind:
      status !== "failed" ? null : isErrorKind(row.error_kind) ? row.error_kind : "unknown",
    errorMessage: status === "failed" ? row.error_message : null,
    citationSupport: SUPPORTS.find((each) => each === row.citation_support) ?? null,
    toolCalls: storedList<AnswerToolCall>(row.tool_calls),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

/** A stored message as the public interface gives it; null for a kind this version doesn't know. */
const toMessage = (row: MessageRow): ConversationMessage | null =>
  row.kind === "question" ? toQuestion(row) : row.kind === "answer" ? toAnswer(row) : null;

const messagesOf = (rows: readonly MessageRow[]) =>
  rows.map(toMessage).filter((message): message is ConversationMessage => message !== null);

export function createConversations(db: Database, now: () => string) {
  /** Each conversation's Minds asked from, each once, in the order first asked from. */
  function askedFromIn(ids: readonly string[]): Map<string, AskedFrom[]> {
    const rows = db.all<{
      conversation_id: string;
      mind_id: string;
      name: string | null;
      gone: number;
    }>(
      `SELECT msg.conversation_id, msg.asked_from_mind_id AS mind_id, min(msg.position) AS first,
         ${MIND_NAME} AS name, ${MIND_GONE} AS gone
       FROM conversation_messages msg
       LEFT JOIN minds mind ON mind.id = msg.asked_from_mind_id
       WHERE msg.conversation_id ${IN} AND msg.kind = 'question'
         AND msg.asked_from_mind_id IS NOT NULL AND msg.deleted_at IS NULL
       GROUP BY msg.conversation_id, msg.asked_from_mind_id
       ORDER BY msg.conversation_id, first`,
      [JSON.stringify(ids)],
    );
    const minds = new Map<string, AskedFrom[]>();
    for (const row of rows) {
      const list = minds.get(row.conversation_id) ?? [];
      list.push({ mindId: row.mind_id, name: row.name ?? "", deleted: row.gone !== 0 });
      minds.set(row.conversation_id, list);
    }
    return minds;
  }

  function toConversations(rows: readonly ConversationRow[]): Conversation[] {
    const minds = askedFromIn(rows.map((row) => row.id));
    return rows.map((row) => ({
      id: row.id,
      title: row.title,
      autoTitle: row.auto_title,
      folderId: row.folder_id,
      askedFrom: minds.get(row.id) ?? [],
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    }));
  }

  /** A conversation that isn't deleted, by id; throws otherwise. */
  const liveRow = (conversationId: unknown): ConversationRow => {
    if (typeof conversationId !== "string" || conversationId === "") {
      throw new InvalidInputError("A conversation id must be a non-empty string.");
    }
    const row = db.get<ConversationRow>(
      `SELECT ${CONVERSATION_COLUMNS} FROM conversations WHERE id = ? AND deleted_at IS NULL`,
      [conversationId],
    );
    if (!row) throw new NotFoundError("That conversation doesn't exist or has been deleted.");
    return row;
  };

  const get = (conversationId: unknown): Conversation => {
    const [conversation] = toConversations([liveRow(conversationId)]);
    if (!conversation) throw new NotFoundError("That conversation doesn't exist.");
    return conversation;
  };

  const messageRows = (conversationId: string, where = "", params: readonly SqlValue[] = []) =>
    db.all<MessageRow>(
      `${MESSAGE_SELECT} WHERE msg.conversation_id = ? AND msg.deleted_at IS NULL ${where}
       ORDER BY ${MESSAGE_ORDER}`,
      [conversationId, ...params],
    );

  /** A message that isn't deleted, of a conversation that isn't, by id. */
  const liveMessage = (messageId: unknown) =>
    typeof messageId !== "string"
      ? undefined
      : db.get<MessageRow>(
          `${MESSAGE_SELECT} JOIN conversations c ON c.id = msg.conversation_id
           WHERE msg.id = ? AND msg.deleted_at IS NULL AND c.deleted_at IS NULL`,
          [messageId],
        );

  const storedMessage = (messageId: string): MessageRow => {
    const row = db.get<MessageRow>(`${MESSAGE_SELECT} WHERE msg.id = ?`, [messageId]);
    if (!row) throw new Error("The message wasn't saved.");
    return row;
  };

  /** Indexes a conversation's name (again). */
  const indexName = (seq: number, name: string) => {
    db.run("DELETE FROM conversations_fts WHERE rowid = ?", [BigInt(seq)]);
    db.run("INSERT INTO conversations_fts (rowid, text) VALUES (?, ?)", [
      BigInt(seq),
      indexText(name),
    ]);
  };

  /** Indexes a message's text (again): its plain text's words. */
  const indexMessage = (seq: number, text: string) => {
    db.run("DELETE FROM conversation_messages_fts WHERE rowid = ?", [BigInt(seq)]);
    db.run("INSERT INTO conversation_messages_fts (rowid, text) VALUES (?, ?)", [
      BigInt(seq),
      indexText(plainText(text)),
    ]);
  };

  /** Adds a message at the end of a conversation; returns its row. Run in a transaction. */
  function insertMessage(
    conversationId: string,
    fields: {
      kind: "question" | "answer";
      text: string;
      questionId?: string;
      askedFrom?: string | null;
      scope?: SearchScope | null;
      model: ChatModelChoice | null;
      forcedSkill?: string | null;
      status?: AnswerStatus;
    },
    at: string,
  ): MessageRow {
    const id = randomUUID();
    db.run(
      `INSERT INTO conversation_messages (id, conversation_id, position, kind, question_id, text,
         asked_from_mind_id, scope, provider_id, model_id, forced_skill, status, created_at, updated_at)
       VALUES (?, ?, (SELECT coalesce(max(position), 0) + 1 FROM conversation_messages
                      WHERE conversation_id = ?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        id,
        conversationId,
        conversationId,
        fields.kind,
        fields.questionId ?? null,
        fields.text,
        fields.askedFrom ?? null,
        fields.scope ? JSON.stringify(fields.scope) : null,
        fields.model?.providerId ?? null,
        fields.model?.modelId ?? null,
        fields.forcedSkill ?? null,
        fields.status ?? null,
        at,
        at,
      ],
    );
    return storedMessage(id);
  }

  /** The Question, its Answer as stored now, and the messages before the Question. */
  const turnOf = (question: MessageRow, answerId: string): Turn => ({
    conversation: get(question.conversation_id),
    question: toQuestion(question),
    answer: toAnswer(storedMessage(answerId)),
    earlier: messagesOf(
      messageRows(question.conversation_id, "AND msg.position < ?", [BigInt(question.position)]),
    ),
  });

  /** Answers left being written by an earlier run of the app: nothing writes them any more. */
  db.transaction(() => {
    for (const row of db.all<{ seq: number; text: string }>(
      `UPDATE conversation_messages SET status = 'stopped', updated_at = ?
       WHERE kind = 'answer' AND status = 'streaming' RETURNING seq, text`,
      [now()],
    )) {
      indexMessage(row.seq, row.text);
    }
  });

  /** Calls `add` with every Citation kept by an Answer of a conversation not deleted permanently. */
  function eachCitation(add: (citation: CitationFields) => void): void {
    const rows = db.all<{ citations: string }>(
      `SELECT msg.citations FROM conversation_messages msg
       JOIN conversations c ON c.id = msg.conversation_id
       WHERE msg.kind = 'answer' AND msg.deleted_at IS NULL AND c.purged_at IS NULL
         AND msg.citations <> '[]'`,
    );
    for (const row of rows)
      for (const citation of storedList<CitationFields>(row.citations)) add(citation);
  }

  return {
    get,

    /** See `CoreApi.listConversations`. */
    list(options: unknown): Conversation[] {
      if (options !== undefined && !isRecord(options)) {
        throw new InvalidInputError("listConversations expects an object.");
      }
      const folderId = options?.folderId;
      if (
        folderId !== undefined &&
        folderId !== null &&
        (typeof folderId !== "string" || folderId === "")
      ) {
        throw new InvalidInputError("A Folder id must be a non-empty string, or null for none.");
      }
      const where =
        folderId === undefined
          ? ""
          : folderId === null
            ? "AND folder_id IS NULL"
            : "AND folder_id = ?";
      return toConversations(
        db.all<ConversationRow>(
          `SELECT ${CONVERSATION_COLUMNS} FROM conversations WHERE deleted_at IS NULL ${where}
           ORDER BY updated_at DESC, seq DESC`,
          typeof folderId === "string" ? [folderId] : [],
        ),
      );
    },

    /** See `CoreApi.getConversation`. */
    content(conversationId: unknown): ConversationContent {
      const conversation = get(conversationId);
      return { conversation, messages: messagesOf(messageRows(conversation.id)) };
    },

    /**
     * Reads `askInConversation`'s input: the Question's text, its conversation (not
     * deleted), the Mind in focus (not deleted, with its Folder), its Search scope, model
     * and forced Skill. Throws, keeping nothing, for anything else.
     */
    parseQuestion(input: unknown): QuestionInput {
      if (!isRecord(input)) throw new InvalidInputError("askInConversation expects an object.");
      const { text, conversationId, mindId, model, forcedSkill } = input;
      if (typeof text !== "string" || text.trim() === "") {
        throw new InvalidInputError("Write the Question before asking it.");
      }
      let mind: QuestionInput["mind"] = null;
      if (mindId !== undefined && mindId !== null) {
        if (typeof mindId !== "string" || mindId === "") {
          throw new InvalidInputError("A Mind id must be a non-empty string, or null for none.");
        }
        const row = db.get<{ folder_id: string | null }>(
          "SELECT folder_id FROM minds WHERE id = ? AND deleted_at IS NULL",
          [mindId],
        );
        if (!row) throw new NotFoundError("That Mind doesn't exist or has been deleted.");
        mind = { id: mindId, folderId: row.folder_id };
      }
      if (model !== undefined && model !== null && !isChatModelChoice(model)) {
        throw new InvalidInputError("A model must be { providerId, modelId }.");
      }
      if (
        forcedSkill !== undefined &&
        forcedSkill !== null &&
        (typeof forcedSkill !== "string" || forcedSkill === "")
      ) {
        throw new InvalidInputError("A forced Skill must be a Skill's name.");
      }
      return {
        conversationId:
          conversationId === undefined || conversationId === null
            ? null
            : liveRow(conversationId).id,
        text: text.trim(),
        mind,
        scope: parseScope(input.scope),
        model: model ?? null,
        forcedSkill: typeof forcedSkill === "string" ? forcedSkill : null,
      };
    },

    /**
     * Adds a Question, starting its conversation when it has none, and its Answer, being
     * written with `answerModel`, all in one go. A new conversation is named by the
     * Question and belongs to the Folder of the Mind in focus, or to none.
     */
    add(input: QuestionInput, answerModel: ChatModelChoice | null): Turn {
      return db.transaction(() => {
        const at = now();
        let conversationId = input.conversationId;
        if (conversationId === null) {
          conversationId = randomUUID();
          const autoTitle = cutTitle(input.text);
          const created = db.get<{ seq: number }>(
            `INSERT INTO conversations (id, auto_title, folder_id, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?) RETURNING seq`,
            [conversationId, autoTitle, input.mind?.folderId ?? null, at, at],
          );
          if (!created) throw new Error("The conversation wasn't saved.");
          indexName(created.seq, autoTitle);
        } else {
          db.run("UPDATE conversations SET updated_at = ? WHERE id = ?", [
            at,
            liveRow(conversationId).id,
          ]);
        }
        const question = insertMessage(
          conversationId,
          {
            kind: "question",
            text: input.text,
            askedFrom: input.mind?.id ?? null,
            scope: input.scope,
            model: input.model,
            forcedSkill: input.forcedSkill,
          },
          at,
        );
        indexMessage(question.seq, question.text);
        const answer = insertMessage(
          conversationId,
          {
            kind: "answer",
            text: "",
            questionId: question.id,
            model: answerModel,
            status: "streaming",
          },
          at,
        );
        return turnOf(question, answer.id);
      });
    },

    /** The Question of an Answer, both live, to write the Answer again. */
    questionOf(input: unknown): { answerId: string; question: ConversationQuestion } {
      if (!isRecord(input)) {
        throw new InvalidInputError("Expected { conversationId, answerId }.");
      }
      const conversation = liveRow(input.conversationId);
      const answer = liveMessage(input.answerId);
      if (answer?.kind !== "answer" || answer.conversation_id !== conversation.id) {
        throw new NotFoundError("That Answer isn't in this conversation.");
      }
      const question = liveMessage(answer.question_id);
      if (question?.kind !== "question") {
        throw new NotFoundError("The Question of this Answer has been deleted.");
      }
      return { answerId: answer.id, question: toQuestion(question) };
    },

    /** Empties an Answer, where it is, to write it again with `answerModel`. */
    restartAnswer(answerId: string, answerModel: ChatModelChoice | null): Turn {
      return db.transaction(() => {
        const answer = liveMessage(answerId);
        const question = liveMessage(answer?.question_id);
        if (!answer || !question)
          throw new NotFoundError("That Answer isn't in this conversation.");
        db.run(
          `UPDATE conversation_messages SET text = '', citations = '[]', tool_calls = '[]',
             status = 'streaming', error_kind = NULL, error_message = NULL,
             citation_support = NULL, provider_id = ?, model_id = ?, updated_at = ?
           WHERE id = ?`,
          [answerModel?.providerId ?? null, answerModel?.modelId ?? null, now(), answerId],
        );
        db.run("DELETE FROM conversation_messages_fts WHERE rowid = ?", [BigInt(answer.seq)]);
        return turnOf(question, answerId);
      });
    },

    /**
     * Writes an Answer as it stands, and indexes it once it has ended. Returns it, or
     * null when it is gone: its conversation was deleted.
     */
    writeAnswer(answerId: string, write: AnswerWrite): ConversationAnswer | null {
      const row = liveMessage(answerId);
      if (row?.kind !== "answer") return null;
      const { outcome } = write;
      db.transaction(() => {
        db.run(
          `UPDATE conversation_messages SET text = ?, citations = ?, tool_calls = ?,
             citation_support = ?, status = ?, error_kind = ?, error_message = ?, updated_at = ?
           WHERE id = ?`,
          [
            write.text,
            JSON.stringify(write.citations),
            JSON.stringify(write.toolCalls),
            write.support,
            outcome?.status ?? "streaming",
            outcome?.error?.kind ?? null,
            outcome?.error?.message ?? null,
            now(),
            answerId,
          ],
        );
        if (outcome) indexMessage(row.seq, write.text);
      });
      return toAnswer(storedMessage(answerId));
    },

    /** See `CoreApi.renameConversation`. */
    rename(conversationId: unknown, title: unknown): Conversation {
      const row = liveRow(conversationId);
      if (typeof title !== "string") {
        throw new InvalidInputError("A conversation's title must be text.");
      }
      const typed = title.trim();
      db.transaction(() => {
        db.run("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", [
          typed,
          now(),
          row.id,
        ]);
        indexName(row.seq, nameOf({ title: typed, auto_title: row.auto_title }));
      });
      return get(row.id);
    },

    /** Puts these conversations in a Folder (checked by the caller), or in none; returns where each was. */
    move(conversationIds: readonly string[], folderId: string | null): MovedItem[] {
      return db.transaction(() => {
        const moved = conversationIds.map((id) => ({ id, from: liveRow(id).folder_id }));
        const at = now();
        for (const item of moved) {
          if (item.from === folderId) continue;
          db.run("UPDATE conversations SET folder_id = ?, updated_at = ? WHERE id = ?", [
            folderId,
            at,
            item.id,
          ]);
        }
        return moved;
      });
    },

    /** Marks a conversation deleted at `at`; its messages stay as they are, for `restore`. */
    delete(conversationId: unknown, at: string): Conversation {
      const conversation = get(conversationId);
      db.run("UPDATE conversations SET deleted_at = ?, updated_at = ? WHERE id = ?", [
        at,
        at,
        conversation.id,
      ]);
      return conversation;
    },

    /** Brings back a deleted conversation with its messages (the caller checks it may). */
    restore(conversationId: string): void {
      db.run("UPDATE conversations SET deleted_at = NULL, updated_at = ? WHERE id = ?", [
        now(),
        conversationId,
      ]);
    },

    /** A Folder was deleted: its conversations go to Not in a Folder, deleted ones too. Returns them. */
    leaveFolder(folderId: string): { id: string; live: boolean }[] {
      return db
        .all<{ id: string; live: number }>(
          `UPDATE conversations SET folder_id = NULL, updated_at = ? WHERE folder_id = ?
           RETURNING id, deleted_at IS NULL AS live`,
          [now(), folderId],
        )
        .map((row) => ({ id: row.id, live: row.live === 1 }));
    },

    /** A deleted Folder was restored: these go back into it, those still in none. Returns whether a live one did. */
    returnToFolder(conversationIds: readonly string[], folderId: string): boolean {
      return db
        .all<{ live: number }>(
          `UPDATE conversations SET folder_id = ?, updated_at = ?
           WHERE folder_id IS NULL AND id ${IN}
           RETURNING deleted_at IS NULL AS live`,
          [folderId, now(), JSON.stringify(conversationIds)],
        )
        .some((row) => row.live === 1);
    },

    /** See `CoreApi.searchConversations`. */
    search(query: unknown): ConversationMatch[] {
      if (typeof query !== "string") throw new InvalidInputError("A search must be text.");
      const terms = searchTerms(query);
      if (terms.length === 0) return [];
      // Every word somewhere in the conversation: in its name, or in one of its messages.
      const withEach = terms.map((term) => {
        const match = termQuery(term);
        return db
          .all<{ id: string }>(
            `SELECT c.id FROM conversations c
             WHERE c.deleted_at IS NULL AND (
               c.seq IN (SELECT rowid FROM conversations_fts WHERE conversations_fts MATCH ?)
               OR c.id IN (
                 SELECT msg.conversation_id FROM conversation_messages msg
                 WHERE msg.deleted_at IS NULL AND msg.seq IN (
                   SELECT rowid FROM conversation_messages_fts
                   WHERE conversation_messages_fts MATCH ?)))`,
            [match, match],
          )
          .map((row) => row.id);
      });
      const found = (withEach[0] ?? []).filter((id) => withEach.every((list) => list.includes(id)));
      if (found.length === 0) return [];
      const ids = JSON.stringify(found);
      const messages = db.all<MessageRow>(
        `${MESSAGE_SELECT}
         WHERE msg.conversation_id ${IN} AND msg.deleted_at IS NULL AND msg.seq IN (
           SELECT rowid FROM conversation_messages_fts WHERE conversation_messages_fts MATCH ?)
         ORDER BY ${MESSAGE_ORDER}`,
        [ids, terms.map(termQuery).join(" OR ")],
      );
      const rows = db.all<ConversationRow>(
        `SELECT ${CONVERSATION_COLUMNS} FROM conversations WHERE id ${IN}
         ORDER BY updated_at DESC, seq DESC`,
        [ids],
      );
      return toConversations(rows).map((conversation) => ({
        conversation,
        titleHighlights: findTerms(conversationName(conversation), terms),
        messages: messages
          .filter((row) => row.conversation_id === conversation.id)
          .flatMap((row) => {
            const message = toMessage(row);
            if (!message) return [];
            return [
              {
                messageId: message.id,
                kind: message.kind,
                ...excerptOf(plainText(row.text), terms),
              },
            ];
          }),
      }));
    },

    /**
     * Where a message of a conversation is now: there, or in a conversation in Recently
     * deleted (with its name); null when it is gone (deleted permanently, the message
     * removed, or never here).
     */
    locate(
      conversationId: string,
      messageId: string,
    ): { name: string; deletedAt: string | null } | null {
      const row = db.get<{
        title: string;
        auto_title: string;
        deleted_at: string | null;
        purged_at: string | null;
      }>("SELECT title, auto_title, deleted_at, purged_at FROM conversations WHERE id = ?", [
        conversationId,
      ]);
      const message = db.get(
        "SELECT 1 FROM conversation_messages WHERE id = ? AND conversation_id = ? AND deleted_at IS NULL",
        [messageId, conversationId],
      );
      if (!row || row.purged_at !== null || !message) return null;
      return { name: nameOf(row), deletedAt: row.deleted_at };
    },

    /**
     * Adds the versions the Citations of conversations quote to `cited`: those in Recently
     * deleted too, so a conversation restored keeps Citations that can be checked.
     */
    addCitedVersions(cited: CitedVersions): void {
      eachCitation((citation) => addCitedVersion(citation, cited));
    },

    /** Adds the Units the Citations of conversations point to in these Documents to `cited`. */
    addCitedUnits(documentIds: ReadonlySet<string>, cited: CitedUnits[]): void {
      eachCitation((citation) => addCitedUnit(citation, documentIds, cited));
    },
  };
}

export type Conversations = ReturnType<typeof createConversations>;
