import { type Editor, Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import type * as Y from "yjs";
import type { BlockOrigin } from "../../../core/api";
import { allOrigins, observeOrigins } from "../../../shared/blockOrigins";
import { textBlockEdited } from "./citationNumbers";
import { BLOCK_ID_ATTRIBUTE } from "./noteSchema";

/**
 * Notes added from an Answer in the assistant (#201) keep a link back to it
 * (src/shared/blockOrigins.ts). Here the editor knows them, for the link's
 * mark in the margin and its card (`ConversationLinks`) and the Block menu
 * (`originOf`); just added, they are washed for a moment (`flashBlocks`,
 * DESIGN.md: "a brief wash on text just added"), so the eye finds them.
 */
export interface BlockOriginsOptions {
  doc: Y.Doc | null;
}

interface Origins {
  origins: ReadonlyMap<string, BlockOrigin>;
  flashing: ReadonlySet<string>;
  decorations: DecorationSet;
}

type Meta = { origins?: ReadonlyMap<string, BlockOrigin>; flashing?: ReadonlySet<string> };

const originsKey = new PluginKey<Origins>("blockOrigins");

/** How long a Block just added stays washed. */
const FLASH_MS = 1600;

/** A run of Blocks added together, from one Answer: its first Block marks it in the margin. */
export interface OriginRun {
  pos: number;
  blockId: string;
  origin: BlockOrigin;
}

const sameAnswer = (a: BlockOrigin, b: BlockOrigin) =>
  a.from.artifactId === b.from.artifactId &&
  a.from.anchor.kind === b.from.anchor.kind &&
  JSON.stringify(a.from.anchor) === JSON.stringify(b.from.anchor);

function decorate(
  doc: ProseMirrorNode,
  origins: ReadonlyMap<string, BlockOrigin>,
  flashing: ReadonlySet<string>,
): DecorationSet {
  if (origins.size === 0 && flashing.size === 0) return DecorationSet.empty;
  const decorations: Decoration[] = [];
  doc.forEach((node, pos) => {
    const id = node.attrs[BLOCK_ID_ATTRIBUTE];
    if (typeof id !== "string") return;
    const linked = origins.has(id);
    const flash = flashing.has(id);
    if (!linked && !flash) return;
    decorations.push(
      Decoration.node(pos, pos + node.nodeSize, {
        ...(linked && { "data-origin": "conversation" }),
        ...(flash && { class: "block-just-added" }),
      }),
    );
  });
  return DecorationSet.create(doc, decorations);
}

export const BlockOrigins = Extension.create<BlockOriginsOptions>({
  name: "blockOrigins",

  addOptions() {
    return { doc: null };
  },

  addProseMirrorPlugins() {
    const { doc } = this.options;
    const editor = this.editor;
    return [
      new Plugin<Origins>({
        key: originsKey,
        state: {
          init: (_config, state) => {
            const origins = doc ? allOrigins(doc) : new Map<string, BlockOrigin>();
            return {
              origins,
              flashing: new Set(),
              decorations: decorate(state.doc, origins, new Set()),
            };
          },
          apply: (tr, value) => {
            const meta = tr.getMeta(originsKey) as Meta | undefined;
            if (meta) {
              const origins = meta.origins ?? value.origins;
              const flashing = meta.flashing ?? value.flashing;
              return { origins, flashing, decorations: decorate(tr.doc, origins, flashing) };
            }
            if (!tr.docChanged || (value.origins.size === 0 && value.flashing.size === 0)) {
              return value;
            }
            // Typing in one text block just moves them. Anything else may have replaced a Block
            // (which drops its decoration) or brought one back, as an undo does: look again.
            if (textBlockEdited(tr) !== null) {
              return { ...value, decorations: value.decorations.map(tr.mapping, tr.doc) };
            }
            return { ...value, decorations: decorate(tr.doc, value.origins, value.flashing) };
          },
        },
        props: {
          decorations: (state) => originsKey.getState(state)?.decorations,
        },
        view: () => {
          const stop = doc
            ? observeOrigins(doc, () => {
                if (editor.isDestroyed) return;
                editor.view.dispatch(
                  editor.state.tr.setMeta(originsKey, { origins: allOrigins(doc) } satisfies Meta),
                );
              })
            : () => undefined;
          return { destroy: stop };
        },
      }),
    ];
  },
});

/** Where a Block came from, if it was added from an Answer. */
export function originOf(state: EditorState, blockId: unknown): BlockOrigin | null {
  if (typeof blockId !== "string") return null;
  return originsKey.getState(state)?.origins.get(blockId) ?? null;
}

/** Whether any Block in the Mind came from an Answer. */
export const hasOrigins = (state: EditorState): boolean =>
  (originsKey.getState(state)?.origins.size ?? 0) > 0;

/** The runs of Blocks added from one Answer each, top to bottom. */
export function originRuns(state: EditorState): OriginRun[] {
  const origins = originsKey.getState(state)?.origins;
  if (!origins || origins.size === 0) return [];
  const runs: OriginRun[] = [];
  let previous: BlockOrigin | null = null;
  state.doc.forEach((node, pos) => {
    const id = node.attrs[BLOCK_ID_ATTRIBUTE];
    const origin = typeof id === "string" ? origins.get(id) : undefined;
    if (origin && typeof id === "string" && !(previous && sameAnswer(previous, origin))) {
      runs.push({ pos, blockId: id, origin });
    }
    previous = origin ?? null;
  });
  return runs;
}

/** Washes Blocks just added for a moment. */
export function flashBlocks(editor: Editor, blockIds: readonly string[]): void {
  if (editor.isDestroyed) return;
  editor.view.dispatch(
    editor.state.tr.setMeta(originsKey, { flashing: new Set(blockIds) } satisfies Meta),
  );
  setTimeout(() => {
    if (editor.isDestroyed) return;
    editor.view.dispatch(
      editor.state.tr.setMeta(originsKey, { flashing: new Set() } satisfies Meta),
    );
  }, FLASH_MS);
}
