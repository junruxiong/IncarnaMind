/**
 * What each command the assistant may run does (#1110): checked first, with what it would do
 * outside IncarnaMind worked out from the arguments (`CheckedCommand.ask`), then run. Each run
 * says what it changed and how to take it back, from what the core returned. A command that
 * changes several things changes all of them or none.
 */
import { lstat, readdir, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join } from "node:path";
import { ASSISTANT_COMMANDS, type AssistantCommandId } from "../../../shared/commands";
import { displayName } from "../../../shared/documentNames";
import { mindName } from "../../../shared/naming";
import { tagLinks } from "../../../shared/tagLinks";
import type {
  CommandAsk,
  CommandDone,
  CommandItem,
  CoreApi,
  Document,
  Effect,
  FolderMove,
  Mind,
  SavedFile,
} from "../../api";
import { RECENTLY_DELETED_DAYS } from "../../api";
import { isInside, kindOf } from "../../documents/files";
import { InvalidInputError, NotFoundError } from "../../errors";
import type { CommandStep } from "./steps";

/** The core's calls the commands make: the same the window's own clicks make. */
export type CommandsCore = Pick<
  CoreApi,
  | "listDocuments"
  | "listMinds"
  | "getLibrary"
  | "listTags"
  | "renameDocument"
  | "renameMind"
  | "moveToFolder"
  | "restoreAssignment"
  | "createLibraryGroup"
  | "deleteLibraryGroup"
  | "restore"
  | "createTag"
  | "deleteTag"
  | "addTagToDocuments"
  | "revertDocumentTags"
  | "deleteDocument"
  | "deleteMind"
  | "addDocuments"
  | "previewMindExport"
>;

export const COMMAND_EXPORT_FORMATS = ["docx", "pdf", "markdown", "html"] as const;
export type CommandExportFormat = (typeof COMMAND_EXPORT_FORMATS)[number];

export interface CommandsHost {
  core: CommandsCore;
  /** The Output location, links followed; null while none is chosen. */
  outputFolder(): Promise<string | null>;
  /**
   * Saves a Mind as a file with the User's export choices for the format: in the Output location
   * (`folder` null), in a folder inside it, or, `approved`, in a folder the User allowed for this
   * one file. Throws, saying why, when nothing was saved.
   */
  exportMind(
    mindId: string,
    format: CommandExportFormat,
    folder: string | null,
    approved: boolean,
  ): Promise<SavedFile>;
}

/** A call's arguments, as the model gives them, every one checked. */
export interface CommandInput {
  command: AssistantCommandId;
  ids: string[];
  name: string | null;
  folder: string | null;
  tag: string | null;
  format: CommandExportFormat;
  kinds: string[];
}

/** What a command changed, how to take it back, and what the model is told. */
export interface Ran {
  done: Omit<CommandDone, "command" | "run">;
  step: CommandStep | null;
  said: string;
}

/** A command checked at the gate: what it can do, what it asks the User first, and the run. */
export interface CheckedCommand {
  command: AssistantCommandId;
  effects: Effect[];
  /** Null: it runs without asking. */
  ask: CommandAsk | null;
  run(): Promise<Ran>;
}

const IN_APP: Effect[] = [{ action: "write", scope: { kind: "library" } }];

const isCommand = (value: unknown): value is AssistantCommandId =>
  typeof value === "string" && Object.hasOwn(ASSISTANT_COMMANDS, value);

const text = (value: unknown): string | null =>
  typeof value === "string" && value.trim() !== "" ? value.trim() : null;

const texts = (value: unknown): string[] =>
  Array.isArray(value) ? value.flatMap((each) => text(each) ?? []) : [];

export function parseCommandInput(input: Record<string, unknown>): CommandInput {
  const { command } = input;
  if (!isCommand(command)) {
    throw new InvalidInputError({
      key: "error.commands.unknown",
      values: { command: String(command) },
    });
  }
  return {
    command,
    ids: texts(input.ids),
    name: text(input.name),
    folder: text(input.folder),
    tag: text(input.tag),
    format: COMMAND_EXPORT_FORMATS.find((each) => each === input.format) ?? "docx",
    kinds: texts(input.kinds).map((kind) => kind.replace(/^\./, "").toLowerCase()),
  };
}

/**
 * What a call can do, from its command's own metadata: a change inside IncarnaMind, a file
 * saved in the Output location or in the folder named, or a folder read and its files added.
 */
export function commandEffects(input: Record<string, unknown>): Effect[] {
  if (!isCommand(input.command)) return [];
  const folder = text(input.folder);
  switch (ASSISTANT_COMMANDS[input.command].does) {
    case "app":
      return IN_APP;
    case "export":
      return [
        { action: "write", scope: folder ? { kind: "folder", path: folder } : { kind: "output" } },
      ];
    case "add":
      return [{ action: "read", scope: { kind: "folder", path: folder ?? "" } }, ...IN_APP];
  }
}

const plural = (count: number, one: string, many = `${one}s`) =>
  `${count} ${count === 1 ? one : many}`;

const same = (a: string, b: string) =>
  a.normalize("NFKC").trim().toLowerCase() === b.normalize("NFKC").trim().toLowerCase();

type Missing =
  | "error.commands.noName"
  | "error.commands.noFolderName"
  | "error.commands.noTag"
  | "error.commands.noFolderPath";

function needs<T>(value: T | null, missing: Missing): T {
  if (value === null) throw new InvalidInputError({ key: missing });
  return value;
}

async function documentsOf(core: CommandsCore, ids: readonly string[]): Promise<Document[]> {
  if (ids.length === 0) throw new InvalidInputError({ key: "error.commands.noIds" });
  const all = await core.listDocuments();
  return [...new Set(ids)].map((id) => {
    const document = all.find((each) => each.id === id);
    if (!document) throw new NotFoundError({ key: "error.commands.noDocument", values: { id } });
    return document;
  });
}

async function mindsOf(core: CommandsCore, ids: readonly string[]): Promise<Mind[]> {
  if (ids.length === 0) throw new InvalidInputError({ key: "error.commands.noIds" });
  const all = await core.listMinds();
  return [...new Set(ids)].map((id) => {
    const mind = all.find((each) => each.id === id);
    if (!mind) throw new NotFoundError({ key: "error.commands.noMind", values: { id } });
    return mind;
  });
}

function onlyOne<T>(items: readonly T[]): T {
  const [item] = items;
  if (!item || items.length > 1) throw new InvalidInputError({ key: "error.commands.oneOnly" });
  return item;
}

async function folderOnDisk(given: string): Promise<string> {
  const path = given === "~" ? homedir() : given.replace(/^~(?=[\\/])/, homedir());
  if (!isAbsolute(path)) {
    throw new InvalidInputError({ key: "error.commands.notAbsolute", values: { folder: given } });
  }
  let real: string;
  try {
    real = await realpath(path);
  } catch {
    throw new NotFoundError({ key: "error.commands.noFolder", values: { folder: given } });
  }
  if (!(await stat(real)).isDirectory()) {
    throw new InvalidInputError({ key: "error.path.notFolder" });
  }
  return real;
}

async function filesIn(folder: string, kinds: readonly string[]): Promise<string[]> {
  const names: string[] = [];
  for (const entry of await readdir(folder, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) continue;
    const path = join(folder, entry.name);
    const extension = entry.name.split(".").pop()?.toLowerCase() ?? "";
    if (kindOf(path) === undefined || (kinds.length > 0 && !kinds.includes(extension))) continue;
    // A link that leads out of the folder isn't read.
    const real = entry.isSymbolicLink() ? await realpath(path).catch(() => null) : path;
    if (!real || !isInside(folder, real) || !(await lstat(real)).isFile()) continue;
    names.push(entry.name);
  }
  return names.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

const documentItem = (document: Document, renamed?: Document): CommandItem => ({
  kind: "document",
  id: document.id,
  name: displayName(document),
  ...(renamed && { renamed: displayName(renamed) }),
});

const mindItem = (mind: Mind, renamed?: Mind): CommandItem => ({
  kind: "mind",
  id: mind.id,
  name: mindName(mind),
  ...(renamed && { renamed: mindName(renamed) }),
});

type Checker = (input: CommandInput, host: CommandsHost) => Promise<CheckedCommand>;

const inApp = (command: AssistantCommandId, run: () => Promise<Ran>): CheckedCommand => ({
  command,
  effects: IN_APP,
  ask: null,
  run,
});

async function folderNamed(core: CommandsCore, name: string) {
  const existing = (await core.getLibrary()).groups.find((group) => same(group.name, name));
  if (existing) return { folder: existing, made: false };
  return { folder: await core.createLibraryGroup({ name, description: "" }), made: true };
}

async function moveBack(core: CommandsCore, move: FolderMove): Promise<void> {
  for (const document of move.documents) {
    if (document.from !== move.folderId) {
      await core.restoreAssignment(document.id, document.previous);
    }
  }
  const back = new Map<string | null, string[]>();
  for (const { id, from } of move.minds) {
    if (from !== move.folderId) back.set(from, [...(back.get(from) ?? []), id]);
  }
  for (const [folderId, mindIds] of back) await core.moveToFolder({ mindIds, folderId });
}

function moveTo(command: "document.moveTo" | "mind.moveTo"): Checker {
  const kind = ASSISTANT_COMMANDS[command].on;
  return async (input, { core }) => {
    const name = needs(input.folder, "error.commands.noFolderName");
    const documents = kind === "document" ? await documentsOf(core, input.ids) : [];
    const minds = kind === "mind" ? await mindsOf(core, input.ids) : [];
    const items = {
      documentIds: documents.map((each) => each.id),
      mindIds: minds.map((each) => each.id),
    };
    return inApp(command, async () => {
      const { folder, made } = await folderNamed(core, name);
      let move: FolderMove;
      try {
        move = await core.moveToFolder({ ...items, folderId: folder.id });
      } catch (error) {
        if (made) await core.deleteLibraryGroup(folder.id);
        throw error;
      }
      const moved = new Set(
        [...move.documents, ...move.minds]
          .filter((each) => each.from !== folder.id)
          .map((each) => each.id),
      );
      const changed = [
        ...documents.filter((each) => moved.has(each.id)).map((each) => documentItem(each)),
        ...minds.filter((each) => moved.has(each.id)).map((each) => mindItem(each)),
      ];
      const already = documents.length + minds.length - changed.length;
      return {
        done: { items: changed, into: { kind: "folder", id: folder.id, name: folder.name, made } },
        step: {
          undo: async () => {
            await moveBack(core, move);
            if (made) await core.deleteLibraryGroup(folder.id);
          },
          redo: async () => {
            if (made) await core.restore([{ kind: "folder", id: folder.id }]);
            move = await core.moveToFolder({ ...items, folderId: folder.id });
          },
        },
        said: `Moved ${plural(changed.length, kind === "document" ? "Document" : "Mind")} to the Folder “${folder.name}”${made ? ", which this made" : ""}.${already > 0 ? ` ${plural(already, "was", "were")} in it already.` : ""}`,
      };
    });
  };
}

async function deleteAll(
  refs: readonly { kind: "document" | "mind"; id: string }[],
  remove: (id: string) => Promise<unknown>,
  restore: CommandsCore["restore"],
): Promise<void> {
  const done: { kind: "document" | "mind"; id: string }[] = [];
  try {
    for (const ref of refs) {
      await remove(ref.id);
      done.push(ref);
    }
  } catch (error) {
    if (done.length > 0) await restore(done);
    throw error;
  }
}

const CHECKERS: Readonly<Record<AssistantCommandId, Checker>> = {
  "document.rename": async (input, { core }) => {
    const document = onlyOne(await documentsOf(core, input.ids));
    const name = needs(input.name, "error.commands.noName");
    return inApp("document.rename", async () => {
      const renamed = await core.renameDocument(document.id, name);
      return {
        done: { items: [documentItem(document, renamed)] },
        step: {
          undo: async () => {
            await core.renameDocument(document.id, document.name);
          },
          redo: async () => {
            await core.renameDocument(document.id, renamed.name);
          },
        },
        said: `Renamed the Document “${displayName(document)}” to “${displayName(renamed)}”.`,
      };
    });
  },

  "mind.rename": async (input, { core }) => {
    const mind = onlyOne(await mindsOf(core, input.ids));
    const name = needs(input.name, "error.commands.noName");
    return inApp("mind.rename", async () => {
      const renamed = await core.renameMind(mind.id, name);
      return {
        done: { items: [mindItem(mind, renamed)] },
        step: {
          undo: async () => {
            await core.renameMind(mind.id, mind.title);
          },
          redo: async () => {
            await core.renameMind(mind.id, renamed.title);
          },
        },
        said: `Renamed the Mind “${mindName(mind)}” to “${mindName(renamed)}”.`,
      };
    });
  },

  "document.moveTo": moveTo("document.moveTo"),
  "mind.moveTo": moveTo("mind.moveTo"),

  "document.tags": async (input, { core }) => {
    const name = needs(input.tag, "error.commands.noTag");
    const documents = await documentsOf(core, input.ids);
    return inApp("document.tags", async () => {
      const existing = (await core.listTags()).find((each) => same(each.name, name));
      const tag = existing ?? (await core.createTag({ name }));
      const made = existing === undefined;
      const changed = documents.filter((each) => !each.tags.some((one) => one.tagId === tag.id));
      const ids = changed.map((each) => each.id);
      const before = tagLinks(documents, ids, tag.id);
      if (ids.length > 0) await core.addTagToDocuments(ids, tag.id);
      const already = documents.length - changed.length;
      return {
        done: {
          items: changed.map((each) => documentItem(each)),
          into: { kind: "tag", id: tag.id, name: tag.name, made },
        },
        step: {
          undo: async () => {
            if (ids.length > 0) await core.revertDocumentTags(before);
            if (made) await core.deleteTag(tag.id);
          },
          redo: async () => {
            if (made) await core.restore([{ kind: "tag", id: tag.id }]);
            if (ids.length > 0) await core.addTagToDocuments(ids, tag.id);
          },
        },
        said: `Tagged ${plural(changed.length, "Document")} “${tag.name}”${made ? ", a new Tag" : ""}.${already > 0 ? ` ${plural(already, "Document")} had it already.` : ""}`,
      };
    });
  },

  "document.delete": async (input, { core }) => {
    const documents = await documentsOf(core, input.ids);
    const refs = documents.map((each) => ({ kind: "document" as const, id: each.id }));
    const remove = () => deleteAll(refs, (id) => core.deleteDocument(id), core.restore);
    return inApp("document.delete", async () => {
      await remove();
      return {
        done: { items: documents.map((each) => documentItem(each)) },
        step: { undo: () => core.restore(refs), redo: remove },
        said: `Moved ${plural(documents.length, "Document")} to Recently deleted, where the User can restore them for ${RECENTLY_DELETED_DAYS} days.`,
      };
    });
  },

  "mind.delete": async (input, { core }) => {
    const minds = await mindsOf(core, input.ids);
    const refs = minds.map((each) => ({ kind: "mind" as const, id: each.id }));
    const remove = () => deleteAll(refs, (id) => core.deleteMind(id), core.restore);
    return inApp("mind.delete", async () => {
      await remove();
      return {
        done: { items: minds.map((each) => mindItem(each)) },
        step: { undo: () => core.restore(refs), redo: remove },
        said: `Moved ${plural(minds.length, "Mind")} to Recently deleted, where the User can restore them for ${RECENTLY_DELETED_DAYS} days.`,
      };
    });
  },

  "mind.export": async (input, host) => {
    const mind = onlyOne(await mindsOf(host.core, input.ids));
    const given = input.folder === null ? null : await folderOnDisk(input.folder);
    const output = given === null ? null : await host.outputFolder();
    // The Output location, and any folder in it, is where the User chose new files go.
    const elsewhere = given !== null && (output === null || !isInside(output, given));
    const into = given !== null && given !== output ? given : null;
    const run = async (): Promise<Ran> => {
      const file = await host.exportMind(mind.id, input.format, into, elsewhere);
      return {
        done: {
          items: [mindItem(mind)],
          file: {
            path: file.path,
            name: file.name,
            folder: file.folder,
            inDefaultFolder: file.inDefaultFolder,
          },
        },
        step: null,
        said: `Saved “${mindName(mind)}” as “${file.name}” in ${elsewhere ? file.folder : `the Output location, ${file.folder}`}. A file can't be taken back with Undo.`,
      };
    };
    if (!elsewhere || given === null) {
      return {
        command: "mind.export",
        effects: [{ action: "write", scope: { kind: "output" } }],
        ask: null,
        run,
      };
    }
    const { fileName } = await host.core.previewMindExport(mind.id, { format: input.format });
    return {
      command: "mind.export",
      effects: [{ action: "write", scope: { kind: "folder", path: given } }],
      ask: {
        kind: "export",
        mind: { id: mind.id, name: mindName(mind) },
        format: input.format,
        folder: given,
        fileName,
      },
      run,
    };
  },

  "documents.add": async (input, { core }) => {
    const folder = await folderOnDisk(needs(input.folder, "error.commands.noFolderPath"));
    const files = await filesIn(folder, input.kinds);
    if (files.length === 0) {
      throw new InvalidInputError({ key: "error.commands.noFiles", values: { folder } });
    }
    return {
      command: "documents.add",
      effects: [{ action: "read", scope: { kind: "folder", path: folder } }, ...IN_APP],
      ask: { kind: "add", folder, files },
      async run() {
        const known = new Set((await core.listDocuments()).map((each) => each.id));
        const result = await core.addDocuments(files.map((name) => join(folder, name)));
        const added = result.documents.filter((each) => !known.has(each.id));
        const refs = added.map((each) => ({ kind: "document" as const, id: each.id }));
        const already = result.documents.length - added.length;
        return {
          done: { items: added.map((each) => documentItem(each)) },
          step:
            added.length === 0
              ? null
              : {
                  undo: () => deleteAll(refs, (id) => core.deleteDocument(id), core.restore),
                  redo: () => core.restore(refs),
                },
          said: `Added ${plural(added.length, "Document")} from ${folder}.${already > 0 ? ` ${plural(already, "was", "were")} in the Library already.` : ""}${result.skipped.length > 0 ? ` ${plural(result.skipped.length, "file")} couldn't be added.` : ""}`,
        };
      },
    };
  },
};

/** Checks a call: what it would do, and whether it asks first. Throws for one that can't run. */
export const checkCommand = (input: CommandInput, host: CommandsHost): Promise<CheckedCommand> =>
  CHECKERS[input.command](input, host);
