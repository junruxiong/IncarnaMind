import { randomUUID } from "node:crypto";
import { getSchema, type JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { updateYFragment, yXmlFragmentToProseMirrorRootNode } from "@tiptap/y-tiptap";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import type { MindClient } from "./mindClient";

/** The schema of the renderer's editor: what a Mind can hold. */
export const mindSchema = getSchema(noteExtensions());

/**
 * The Mind's Blocks as the editor reads them from Yjs. Throws if anything in
 * them breaks the editor's schema.
 */
export function readMind(client: MindClient): ProseMirrorNode {
  const doc = yXmlFragmentToProseMirrorRootNode(client.blocks, mindSchema);
  if (doc.childCount > 0) doc.check();
  return doc;
}

/** Writes a whole Mind, as JSON, into its Yjs document the way the editor's binding does. */
export function writeMind(client: MindClient, blocks: JSONContent[]): void {
  const doc = mindSchema.nodeFromJSON({ type: "doc", content: blocks });
  client.doc.transact(() =>
    updateYFragment(client.doc, client.blocks, doc, { mapping: new Map(), isOMark: new Map() }),
  );
}

/** Changes the Mind's top-level Blocks, as JSON, and writes them back as the editor would. */
export function editMind(
  client: MindClient,
  change: (blocks: JSONContent[]) => JSONContent[],
): void {
  const current = client.blocks.length > 0 ? (readMind(client).toJSON().content ?? []) : [];
  writeMind(client, change(current));
}

const inline = (text: string): JSONContent[] => (text ? [{ type: "text", text }] : []);

/** A paragraph Note. `off` switches it out of Question context. */
export function note(text: string, { off = false } = {}): JSONContent {
  return {
    type: "paragraph",
    attrs: { id: randomUUID(), ...(off ? { includeInContext: false } : {}) },
    content: inline(text),
  };
}

export function heading(level: number, text: string): JSONContent {
  return { type: "heading", attrs: { id: randomUUID(), level }, content: inline(text) };
}

/** A Question Block, optionally with a model picked for it. */
export function question(
  text: string,
  model?: { providerId: string; modelId: string },
): JSONContent & { attrs: { id: string } } {
  return { type: "question", attrs: { id: randomUUID(), ...model }, content: inline(text) };
}

/** Each top-level Block's type, with an Answer's status. */
export function outline(client: MindClient): string[] {
  const types: string[] = [];
  readMind(client).forEach((block) => {
    types.push(block.type.name === "answer" ? `answer:${block.attrs.status}` : block.type.name);
  });
  return types;
}

/** The top-level Answer with this Block ID. */
export function answerIn(client: MindClient, answerId: string): ProseMirrorNode {
  let found: ProseMirrorNode | null = null;
  readMind(client).forEach((block) => {
    if (block.type.name === "answer" && block.attrs.id === answerId) found = block;
  });
  if (!found) throw new Error(`No Answer ${answerId} in the Mind.`);
  return found;
}

/** An Answer's text, its Blocks separated by blank lines. */
export function answerText(client: MindClient, answerId: string): string {
  const answer = answerIn(client, answerId);
  return answer.textBetween(0, answer.content.size, "\n\n");
}
