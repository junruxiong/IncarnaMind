/**
 * The Minds' Tools, for an Answer in the assistant (#202): what the User
 * types decides whether the assistant answers, changes a Mind or makes one,
 * and to change one, the model calls these. Every change arrives in its Mind
 * as Proposals, which the User accepts or rejects; a Checkpoint of the Mind
 * is saved first (#113), and nothing in the Mind counts as changed until
 * they accept.
 *
 * - `read_mind` reads a Mind the User names. The one in focus is in the
 *   Question context already.
 * - `edit_mind` changes text already in a Mind, as an instruction says: the
 *   selection, a section (named by its heading) or the whole Mind, rewritten
 *   as Edit with AI rewrites a selection (../../edits), on the Answer's model,
 *   its Citations kept.
 * - `add_to_mind` adds text the model wrote, with the Citations it recorded
 *   with `cite`, at the end of a section or after a paragraph, or at the end.
 * - `create_mind` makes a Mind titled as the model says, in the Folder of the
 *   Question's Search scope when that is one Folder, else in none, holding
 *   the text the model wrote, if any.
 *
 * A Mind is named by its name; left out, it is the one in focus. A call that
 * changed a Mind says so on its card (`MindChange`). What the model reads
 * back says what was done, or why nothing was, so its reply can say so.
 */
import { randomUUID } from "node:crypto";
import * as Y from "yjs";
import { setOrigins } from "../../../shared/blockOrigins";
import {
  pendingProposals,
  recordProposalRequest,
  resolveProposals,
} from "../../../shared/proposals";
import { countFor, countText, mostlyCjk } from "../../../shared/wordCount";
import type { NodeJSON } from "../../answers/blocks";
import {
  BLOCK_ID_ATTRIBUTE,
  type BlockOrigin,
  type EditWithAiResult,
  type Effect,
  type MindChange,
  type TextPlace,
  type ThirdPartySource,
  type WritingPlace,
} from "../../api";
import { plainText } from "../../mindText";
import { insertBlocks } from "../../noteAdding";
import { layout, textBlocks } from "../../proposals/write";
import type { PreparedChatModel } from "../../providers/chat";
import { MIND_TOOLS, type Tool, type ToolProviderInfo } from "..";
import { partialFields } from "../partialInput";

const PROVIDER: ToolProviderInfo = { kind: "minds", id: "minds", name: "Minds" };

/** About how much of a named Mind `read_mind` gives, in tokens; a window cuts it further. */
const READ_TOKENS = 12_000;

/** A Mind the Tools can read and change. */
export interface ChangeableMind {
  id: string;
  /** Its name as the User sees it: its title, else the name taken from its text. */
  name: string;
}

/** What the Minds' Tools need of the core. */
export interface MindsForTools {
  /** The Minds the User can read and change now, the most recently changed first. */
  list(): ChangeableMind[];
  /** A Mind's Yjs content, read without changing it. */
  read<T>(mindId: string, look: (blocks: Y.XmlFragment) => T): T;
  /** Changes a Mind's Yjs content in one transaction. */
  edit<T>(mindId: string, change: (blocks: Y.XmlFragment) => T): T;
  /**
   * The Mind as the model reads the one in focus, within about `tokens`
   * tokens, and where the third-party text in what it gives came from (#78).
   */
  text(
    mindId: string,
    name: string,
    tokens: number,
  ): { text: string; origins: readonly ThirdPartySource[] };
  /** Saves a Checkpoint of the Mind as it is, before the AI changes it (#113). */
  checkpoint(mindId: string): void;
  /** Makes a Mind, in the Folder `folderId` or in none. */
  create(title: string, folderId: string | null): ChangeableMind;
  /** The name of a Folder, or null when there is none by that id. */
  folderName(folderId: string): string | null;
  /** Says where a call is writing a Mind's new Blocks as they come (#202); `place` null once it has stopped. */
  writing(mindId: string, place: WritingPlace | null): void;
  /** Edit with AI over a stretch of a Mind (../../edits, `proposeInMind`). */
  proposeInMind(input: {
    mindId: string;
    from: TextPlace;
    to: TextPlace;
    instruction: string;
    prepared: PreparedChatModel;
    signal: AbortSignal;
    /** Runs just before the change's first Proposals are written. */
    beforeWriting(): void;
    /** More of its Proposals are in the Mind, as they come. */
    written(request: string, count: number): void;
  }): Promise<{
    result: EditWithAiResult;
    length: { before: string; after: string } | null;
    reason: string | null;
  }>;
}

export interface MindToolsOptions {
  minds: MindsForTools;
  /** The model the Answer is written with: edits are made on it too. */
  prepared: PreparedChatModel;
  /** The Mind the Question was asked from, and the Blocks the User selected in it; null: none. */
  focused: { mindId: string; blockIds: readonly string[] } | null;
  /** Where a new Mind goes: the Folder of the Question's Search scope when that is one Folder. */
  folderId: string | null;
  /** The Answer, and its Question, that Blocks it adds link back to. */
  origin: BlockOrigin;
  /**
   * Markdown as a Mind holds it: each Citation marker the Answer recorded a
   * Citation for, checked, and its figures as their shown text. Says how many
   * of its Citations weren't found and how many markers had no Citation.
   */
  blocksOf(markdown: string): { blocks: NodeJSON[]; notFound: number; unrecorded: number };
  /** A call changed a Mind: its card says so, as soon as it does, also while it writes. */
  changed(toolCallId: string, change: MindChange): void;
  /** The Answer's: once it aborts, calls still writing stop. */
  signal: AbortSignal;
  now(): Date;
}

/**
 * A name as the User or the model may write it: no quotes or "@" around it,
 * nor the Markdown mark a heading or list item has in <mind> ("## Risks"),
 * spaces made alike, any case.
 */
function nameKey(name: string): string {
  return name
    .trim()
    .replace(/^(?:@|#{1,6}\s+|[-*+]\s+|\d{1,3}[.)]\s+|>\s+)/, "")
    .replace(/^["'“‘「『《](.*)["'”’」』》]$/u, "$1")
    .replace(/\s+/g, " ")
    .trim()
    .toLocaleLowerCase();
}

const listed = (minds: readonly ChangeableMind[]) =>
  minds
    .slice(0, 10)
    .map((mind) => `“${mind.name}”`)
    .join(", ");

/**
 * The Mind a name means: the one so named, else the only one whose name
 * holds it; or what the model is told when there is none, or several.
 */
function findMind(minds: readonly ChangeableMind[], name: string): ChangeableMind | string {
  const key = nameKey(name);
  const exact = minds.find((mind) => nameKey(mind.name) === key);
  if (exact) return exact;
  const near = key === "" ? [] : minds.filter((mind) => nameKey(mind.name).includes(key));
  if (near.length === 1 && near[0]) return near[0];
  if (near.length > 1) {
    return `Several Minds have “${name}” in their name: ${listed(near)}. Ask the User which one, and change nothing yet.`;
  }
  return minds.length === 0
    ? `The User has no Mind called “${name}”, nor any other. Nothing was changed.`
    : `The User has no Mind called “${name}”. Their most recent Minds: ${listed(minds)}. If none is meant, say so and change nothing.`;
}

/** A Mind found, or the call fails, and the model reads why. */
function found(mind: ChangeableMind | string): ChangeableMind {
  if (typeof mind === "string") throw new Error(mind);
  return mind;
}

const text = (value: unknown, what: string): string => {
  if (typeof value !== "string" || value.trim() === "") throw new Error(`${what} must be text.`);
  return value.trim();
};

const optional = (value: unknown): string | null =>
  typeof value === "string" && value.trim() !== "" ? value.trim() : null;

const idOf = (element: Y.XmlElement): string => {
  const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
  return typeof id === "string" ? id : "";
};

const levelOf = (element: Y.XmlElement): number => {
  const level = Number(element.getAttribute("level"));
  return Number.isInteger(level) && level > 0 ? level : 1;
};

/** Where in `elements` the one `wanted` names is: by its whole text, else by its start; -1 for none. */
function indexNamed(elements: readonly Y.XmlElement[], wanted: string): number {
  const key = nameKey(wanted);
  const texts = elements.map((element) => nameKey(plainText(element)));
  const whole = texts.indexOf(key);
  return whole >= 0 || key === "" ? whole : texts.findIndex((text) => text.startsWith(key));
}

const isHeading = (element: Y.XmlElement | undefined) => element?.nodeName === "heading";

/**
 * The stretch of a Mind `edit_mind` changes, from one paragraph or heading
 * to another, whole: the section under the heading `part` (to the next
 * heading as high or higher), else the Blocks the User selected, else all of
 * it. A message when there is nothing there to change.
 */
function stretchOf(
  blocks: Y.XmlFragment,
  part: string | null,
  selected: readonly string[],
): { from: TextPlace; to: TextPlace } | string {
  const all = textBlocks(blocks).map((each) => each.element);
  let chosen = all;
  if (part !== null) {
    const headings = all.filter((element) => isHeading(element));
    const heading = headings[indexNamed(headings, part)];
    if (!heading) return `there is no heading “${part}” in it`;
    const start = all.indexOf(heading);
    const level = levelOf(heading);
    const end = all.findIndex(
      (element, index) =>
        index > start && element.nodeName === "heading" && levelOf(element) <= level,
    );
    chosen = all.slice(start, end < 0 ? all.length : end);
  } else if (selected.length > 0) {
    const ids = new Set(selected);
    const inSelection = (element: Y.XmlElement) => {
      for (let at: unknown = element; at instanceof Y.XmlElement; at = at.parent) {
        if (ids.has(idOf(at))) return true;
      }
      return false;
    };
    const picked = all.filter(inSelection);
    if (picked.length > 0) chosen = picked;
  }
  const first = chosen[0];
  const last = chosen.at(-1);
  if (!first || !last || chosen.every((element) => plainText(element).trim() === "")) {
    return "there is no text in it to change";
  }
  const size = layout(last).reduce((sum, child) => sum + child.size, 0);
  return { from: { blockId: idOf(first), offset: 0 }, to: { blockId: idOf(last), offset: size } };
}

/**
 * Where `add_to_mind` puts its Blocks: after the last top-level Block of the
 * section under the heading `after` names, or after the paragraph it names;
 * by that Block's id, or null for the end of the Mind. A message when it
 * names none.
 */
function placeAfter(
  blocks: Y.XmlFragment,
  after: string | null,
): { after: string | null } | string {
  if (after === null) return { after: null };
  const top = blocks.toArray().filter((each): each is Y.XmlElement => each instanceof Y.XmlElement);
  const headings = top.filter((element) => isHeading(element));
  const heading = headings[indexNamed(headings, after)];
  const texts = top.filter((element) => isHeading(element) || element.nodeName === "paragraph");
  const found = heading ?? texts[indexNamed(texts, after)];
  if (!found) {
    const names = headings.slice(0, 10).map((element) => `“${plainText(element)}”`);
    return `there is no heading or paragraph “${after}” in it${names.length > 0 ? `; its headings are ${names.join(", ")}` : ""}`;
  }
  if (!isHeading(found)) return { after: idOf(found) };
  const level = levelOf(found);
  let last = found;
  for (const element of top.slice(top.indexOf(found) + 1)) {
    if (isHeading(element) && levelOf(element) <= level) break;
    // A section's trailing empty lines stay after what is added.
    if (!(element.nodeName === "paragraph" && element.length === 0)) last = element;
  }
  return { after: idOf(last) };
}

/** How a text's length went, as the model is told: in words, or characters for Chinese and Japanese. */
function lengths(before: string, after: string): string {
  const was = countText(before);
  const now = countText(after);
  const unit = (count: ReturnType<typeof countText>) => (mostlyCjk(count) ? "characters" : "words");
  const from = countFor(was, "words");
  const to = countFor(now, "words");
  return unit(was) === unit(now)
    ? `from ${from} to ${to} ${unit(now)}`
    : `from ${from} ${unit(was)} to ${to} ${unit(now)}`;
}

const changes = (count: number) => (count === 1 ? "1 change" : `${count} changes`);

const WAITING =
  "They wait in the Mind for the User to accept or reject them; until then nothing in it has changed, and they can't be changed again.";

/** What `edit_mind`'s model reads of a result that proposed nothing, or why it failed. */
function notProposed(result: EditWithAiResult, name: string): string {
  switch (result.kind) {
    case "unchanged":
      return `No change was needed in “${name}”: the text already does what was asked. Nothing was proposed.`;
    case "has-proposals":
      return `“${name}” already has changes waiting there for the User. Nothing was proposed: they need to accept or reject those first.`;
    case "too-long":
      return `That is too much of “${name}” to change at once. Nothing was proposed: change one section at a time, naming its heading in part.`;
    case "changed":
      return `That text of “${name}” changed while it was being read. Nothing was proposed.`;
    case "local-mode":
      return "Local mode is on, and the model isn't on this computer. Nothing was proposed.";
    case "not-ready":
      return "The model can't be used now. Nothing was proposed.";
    case "failed":
      return `The change couldn't be made: ${result.error.message} Nothing was proposed.`;
    case "cancelled":
      return "The change was stopped. Nothing was proposed.";
    case "proposed":
      return "";
  }
}

/** The Minds' Tools for one Answer in the assistant (see the module's comment). */
export function mindTools(options: MindToolsOptions): Tool[] {
  const { minds, focused } = options;

  /** The Mind a call means: the one it names, else the one in focus; or why there is none. */
  const targetOf = (input: Record<string, unknown>): ChangeableMind | string => {
    const all = minds.list();
    const wanted = optional(input.mind);
    if (wanted !== null) return findMind(all, wanted);
    const inFocus = focused && all.find((mind) => mind.id === focused.mindId);
    return (
      inFocus ??
      "No Mind is open beside the assistant, and the User named none. Nothing was changed: say so, and that they can open a Mind, name one, or ask for a new Mind."
    );
  };

  /** What a call reads or writes: the Mind it means, or "" when it means none there is. */
  const effectOn =
    (action: "read" | "write") =>
    (input: Record<string, unknown>): Effect[] => {
      const mind = targetOf(input);
      return [{ action, scope: { kind: "mind", mindId: typeof mind === "string" ? "" : mind.id } }];
    };

  /**
   * Where a call's Blocks go in a Mind: Proposals of one request, linked back
   * to the Answer, each written after the last (`writeMore`), from the place
   * the call named; a Checkpoint first, unless the call made the Mind.
   */
  interface Writer {
    mind: ChangeableMind;
    /** The `after` the call named, for telling a different place. */
    named: string | null;
    request: string;
    /** The top-level Block the next ones go after; null: the Mind's end. */
    anchor: string | null;
    /** How many of the call's Blocks are in. */
    written: number;
    made: boolean;
  }

  const writerFor = (mind: ChangeableMind, after: string | null, made: boolean): Writer => {
    const place = minds.read(mind.id, (note) => placeAfter(note, after));
    if (typeof place === "string") {
      throw new Error(`Nothing was added: ${place}. Don't add it to another Mind instead.`);
    }
    return {
      mind,
      named: after,
      request: randomUUID(),
      anchor: place.after,
      written: 0,
      made,
    };
  };

  /** Writes those of `blocks` the writer hasn't yet, after the ones it has. */
  const writeMore = (writer: Writer, blocks: readonly NodeJSON[]): void => {
    const fresh = blocks.slice(writer.written);
    if (fresh.length === 0) return;
    const first = writer.written === 0;
    if (first && !writer.made) minds.checkpoint(writer.mind.id);
    minds.edit(writer.mind.id, (note) => {
      const anchor = note
        .toArray()
        .find(
          (each): each is Y.XmlElement =>
            each instanceof Y.XmlElement && writer.anchor !== null && idOf(each) === writer.anchor,
        );
      const ids = insertBlocks(note, fresh, { after: anchor ?? null, request: writer.request });
      writer.anchor = ids.at(-1) ?? writer.anchor;
      const doc = note.doc;
      if (!doc) return;
      setOrigins(doc, ids, options.origin);
      if (first) {
        recordProposalRequest(doc, writer.request, {
          source: "answer-added",
          providerId: options.prepared.provider.id,
          modelId: options.prepared.modelId,
          createdAt: options.now().toISOString(),
        });
      }
    });
    writer.written = blocks.length;
  };

  const changeOf = (writer: Writer, created: boolean): MindChange => ({
    mindId: writer.mind.id,
    created,
    request: writer.request,
    proposals: minds.read(
      writer.mind.id,
      (note) => pendingProposals(note).filter((each) => each.request === writer.request).length,
    ),
  });

  /** Takes back what a writer wrote: the call ended up meaning another Mind, or another place. */
  const withdraw = (writer: Writer): void => {
    if (writer.written === 0) return;
    minds.edit(writer.mind.id, (note) => {
      resolveProposals(note, (each) => each.request === writer.request, "reject");
    });
  };

  /**
   * Calls of `add_to_mind` and `create_mind` whose content is coming in as the
   * model writes it (`streamInput`), by id: each finished Block is written at
   * once, and "Writing…" stands where the next one goes. Null: the call can't
   * be written yet, or its input says nothing that streams.
   */
  const streams = new Map<string, { writer: Writer | null; created: boolean; breaks: number }>();
  /**
   * Calls already made: input the engine still hands over after the call ran
   * (it reads the stream behind the call) writes nothing more.
   */
  const ended = new Set<string>();

  options.signal.addEventListener("abort", () => {
    for (const stream of streams.values()) {
      if (stream.writer) minds.writing(stream.writer.mind.id, null);
    }
    streams.clear();
  });

  /**
   * More of a call's content: the Blocks it has finished (all but the one
   * being written) go in. `open` gives the writer once the fields before the
   * content say where; null while they don't yet.
   */
  const streamContent = (
    toolCallId: string,
    text: string,
    open: (fields: ReturnType<typeof partialFields>) => { writer: Writer; created: boolean } | null,
  ): void => {
    if (options.signal.aborted || ended.has(toolCallId)) return;
    const fields = partialFields(text);
    const content = fields.get("content");
    if (!content) return;
    let stream = streams.get(toolCallId);
    if (!stream) {
      stream = { writer: null, created: false, breaks: 0 };
      streams.set(toolCallId, stream);
    }
    // Only once a paragraph has ended is there a Block more to write.
    const breaks = content.value.split("\n\n").length - 1;
    if (breaks === stream.breaks) return;
    stream.breaks = breaks;
    if (!stream.writer) {
      const opened = open(fields);
      if (!opened) return;
      stream.writer = opened.writer;
      stream.created = opened.created;
    }
    const { writer } = stream;
    writeMore(writer, options.blocksOf(content.value).blocks.slice(0, -1));
    minds.writing(writer.mind.id, { after: writer.anchor });
    if (writer.written > 0) options.changed(toolCallId, changeOf(writer, stream.created));
  };

  /**
   * A call of `add_to_mind` or `create_mind` with all its input: the rest of
   * its Blocks, after any written as it came, into `mind` after `after`.
   */
  const finishContent = (
    toolCallId: string,
    mind: ChangeableMind,
    after: string | null,
    markdown: string,
    made: boolean,
  ): { change: MindChange; notFound: number; unrecorded: number } => {
    const stream = streams.get(toolCallId);
    streams.delete(toolCallId);
    ended.add(toolCallId);
    let writer = stream?.writer ?? null;
    if (writer && (writer.mind.id !== mind.id || writer.named !== after)) {
      withdraw(writer);
      minds.writing(writer.mind.id, null);
      writer = null;
    }
    writer ??= writerFor(mind, after, made);
    const { blocks, notFound, unrecorded } = options.blocksOf(markdown);
    writeMore(writer, blocks);
    minds.writing(writer.mind.id, null);
    const change = changeOf(writer, made);
    options.changed(toolCallId, change);
    return { change, notFound, unrecorded };
  };

  /** What the model reads of Blocks it added: how many, and what of their Citations didn't hold. */
  const added = (
    { change, notFound, unrecorded }: ReturnType<typeof finishContent>,
    where: string,
  ): string =>
    [
      `Proposed ${change.proposals === 1 ? "1 new Block" : `${change.proposals} new Blocks`} ${where}. ${WAITING}`,
      ...(notFound > 0
        ? [
            `${notFound === 1 ? "1 of their Citations" : `${notFound} of their Citations`} wasn't found in its source: say so.`,
          ]
        : []),
      ...(unrecorded > 0
        ? [
            `${unrecorded === 1 ? "1 marker" : `${unrecorded} markers`} had no Citation recorded with cite and ${unrecorded === 1 ? "was" : "were"} left out.`,
          ]
        : []),
    ].join(" ");

  const mindProperty = {
    type: "string",
    description:
      "The Mind's name, as the User calls it. Leave it out for the Mind the User is in, the one in <mind>.",
  };

  const read: Tool = {
    name: MIND_TOOLS.read,
    description:
      "Read one of the User's Minds (their notes) by its name, to see what it holds before changing it or adding to it. The Mind the User is in is already in <mind>: don't read it again.",
    inputSchema: {
      type: "object",
      properties: { mind: { ...mindProperty, description: "The Mind's name." } },
      required: ["mind"],
      additionalProperties: false,
    },
    provider: PROVIDER,
    providerTool: MIND_TOOLS.read,
    effects: effectOn("read"),
    untrustedResult: false,
    async call(input, { readThirdParty }) {
      const mind = found(findMind(minds.list(), text(input.mind, "mind")));
      const read = minds.text(mind.id, mind.name, READ_TOKENS);
      for (const source of read.origins) readThirdParty?.(source);
      return read.text;
    },
  };

  const edit: Tool = {
    name: MIND_TOOLS.edit,
    description:
      "Change text that is already in a Mind, as the User asks: shorten, rewrite, restructure, correct, translate or reword it. Don't write the new text: say what to change in instruction, and the text is rewritten for you, its Citations kept. It arrives in the Mind as changes the User accepts or rejects.",
    inputSchema: {
      type: "object",
      properties: {
        mind: mindProperty,
        part: {
          type: "string",
          description:
            "The heading of the section to change, as written in the Mind. Leave it out to change what the User selected, or else the whole Mind.",
        },
        instruction: {
          type: "string",
          description:
            'What to change, in the User\'s language, with anything from the conversation it needs, e.g. "Shorten to about 150 words, keeping every figure".',
        },
      },
      required: ["instruction"],
      additionalProperties: false,
    },
    provider: PROVIDER,
    providerTool: MIND_TOOLS.edit,
    effects: effectOn("write"),
    untrustedResult: false,
    async call(input, { toolCallId, signal }) {
      const instruction = text(input.instruction, "instruction");
      const mind = found(targetOf(input));
      const part = optional(input.part);
      // The selection counts only in the Mind it is in, and only when no section is named.
      const selected = focused?.mindId === mind.id && part === null ? focused.blockIds : [];
      const stretch = minds.read(mind.id, (blocks) => stretchOf(blocks, part, selected));
      if (typeof stretch === "string") {
        throw new Error(`Nothing was changed in “${mind.name}”: ${stretch}.`);
      }
      const { result, length, reason } = await minds.proposeInMind({
        mindId: mind.id,
        ...stretch,
        instruction,
        prepared: options.prepared,
        signal,
        beforeWriting: () => minds.checkpoint(mind.id),
        written: (request, proposals) =>
          options.changed(toolCallId, { mindId: mind.id, created: false, request, proposals }),
      });
      if (result.kind !== "proposed") {
        const why = notProposed(result, mind.name);
        if (result.kind === "unchanged") return why;
        throw new Error(why);
      }
      options.changed(toolCallId, {
        mindId: mind.id,
        created: false,
        request: result.request,
        proposals: result.proposals,
      });
      const went = length ? ` (${lengths(length.before, length.after)})` : "";
      return [
        `Proposed ${changes(result.proposals)} in “${mind.name}”${went}. ${WAITING}`,
        ...(reason ? [`Why, as the rewrite says: ${reason}`] : []),
        ...(result.droppedCitations > 0
          ? [
              `${result.droppedCitations === 1 ? "1 Citation" : `${result.droppedCitations} Citations`} would be removed with the words they belong to; the User sees that on the change: say so.`,
            ]
          : []),
        ...(result.outOfDate > 0
          ? [
              `${result.outOfDate === 1 ? "1 change is" : `${result.outOfDate} changes are`} already out of date: the User changed that text meanwhile.`,
            ]
          : []),
      ].join(" ");
    },
  };

  const contentProperty = {
    type: "string",
    description:
      "The text, in Markdown, cited as an Answer is, with markers like [^1] for Citations recorded with cite. A new section starts with its heading. Write it last, after the other fields.",
  };

  const add: Tool = {
    name: MIND_TOOLS.add,
    description:
      "Add new text to a Mind: a section, a paragraph or a list you write from the Documents or this conversation. Cite it as an Answer: call cite first and put its markers in content. It arrives in the Mind as changes the User accepts or rejects. To change text already there, use edit_mind.",
    inputSchema: {
      type: "object",
      properties: {
        mind: mindProperty,
        after: {
          type: "string",
          description:
            "The heading of the section to add it at the end of, or the first words of the paragraph to add it after, as written in the Mind. Leave it out to add it at the Mind's end.",
        },
        content: contentProperty,
      },
      required: ["content"],
      additionalProperties: false,
    },
    provider: PROVIDER,
    providerTool: MIND_TOOLS.add,
    effects: effectOn("write"),
    untrustedResult: false,
    streamInput(toolCallId, input) {
      streamContent(toolCallId, input, (fields) => {
        // Only once the Mind and the place it names are whole: a name cut short could mean another.
        const named = fields.get("mind");
        const place = fields.get("after");
        if ((named && !named.done) || (place && !place.done)) return null;
        const mind = targetOf({ mind: named?.value });
        if (typeof mind === "string") return null;
        return { writer: writerFor(mind, optional(place?.value), false), created: false };
      });
    },
    async call(input, { toolCallId }) {
      const markdown = text(input.content, "content");
      const mind = found(targetOf(input));
      const after = optional(input.after);
      const written = finishContent(toolCallId, mind, after, markdown, false);
      return added(written, `in “${mind.name}”, ${after ? `after “${after}”` : "at its end"}`);
    },
  };

  const create: Tool = {
    name: MIND_TOOLS.create,
    description:
      "Make a new Mind (a note) when the User asks for a new note, page or document, e.g. to summarise Documents or this conversation into one. Write its content as you would an Answer, with Citation markers from cite; leave content out for an empty Mind. It opens beside the User's Mind, its text as changes to accept or reject.",
    inputSchema: {
      type: "object",
      properties: {
        title: {
          type: "string",
          description: "A short title, from the User's words, in their language.",
        },
        content: contentProperty,
      },
      required: ["title"],
      additionalProperties: false,
    },
    provider: PROVIDER,
    providerTool: MIND_TOOLS.create,
    // The Mind it writes to isn't made yet.
    effects: () => [{ action: "write", scope: { kind: "mind", mindId: "" } }],
    untrustedResult: false,
    streamInput(toolCallId, input) {
      streamContent(toolCallId, input, (fields) => {
        const title = fields.get("title");
        if (!title?.done || title.value.trim() === "") return null;
        const mind = minds.create(title.value.trim(), options.folderId);
        return { writer: writerFor(mind, null, true), created: true };
      });
    },
    async call(input, { toolCallId }) {
      const title = text(input.title, "title");
      const markdown = optional(input.content);
      // Made already, as its content came in.
      const streamed = streams.get(toolCallId)?.writer?.mind;
      const mind = streamed ?? minds.create(title, options.folderId);
      const folder = options.folderId === null ? null : minds.folderName(options.folderId);
      const made = `Made the Mind “${mind.name}”${folder ? ` in the Folder “${folder}”` : ""}, open beside the User's Mind`;
      if (markdown === null) {
        options.changed(toolCallId, {
          mindId: mind.id,
          created: true,
          request: "",
          proposals: 0,
        });
        return `${made}. It is empty.`;
      }
      const written = finishContent(toolCallId, mind, null, markdown, true);
      return `${made}. ${added(written, "in it")}`;
    },
  };

  return [read, edit, add, create];
}
