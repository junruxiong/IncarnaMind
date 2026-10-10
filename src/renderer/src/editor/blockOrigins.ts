import { type Editor, Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import type * as Y from "yjs";
import type { BlockOrigin } from "../../../core/api";
import { allOrigins, observeOrigins } from "../../../shared/blockOrigins";
import { BLOCK_ID_ATTRIBUTE } from "./noteSchema";

/**
 * Notes added from a chat (#201): each keeps a link back to its Answer
 * (src/shared/blockOrigins.ts). Here they get a tooltip naming the chat, and
 * the Block menu opens the chat (`originOf`); just added, they are washed for
 * a moment (`flashBlocks`), so the eye finds them.
 */
export interface BlockOriginsOptions {
  /** The Mind's Yjs document, which holds the origins. */
  doc: Y.Doc | null;
  /** The tooltip of a Block with this origin: "From the chat “…”". */
  describe(origin: BlockOrigin): string;
}

interface Origins {
  origins: ReadonlyMap<string, BlockOrigin>;
  /** Blocks just added, washed for a moment. */
  flashing: ReadonlySet<string>;
  decorations: DecorationSet;
}

type Meta = { origins?: ReadonlyMap<string, BlockOrigin>; flashing?: ReadonlySet<string> };

const originsKey = new PluginKey<Origins>("blockOrigins");

/** How long a Block just added stays washed (DESIGN.md D124: "a brief wash on text just added"). */
const FLASH_MS = 1600;

function decorate(
  doc: ProseMirrorNode,
  origins: ReadonlyMap<string, BlockOrigin>,
  flashing: ReadonlySet<string>,
  describe: (origin: BlockOrigin) => string,
): DecorationSet {
  if (origins.size === 0 && flashing.size === 0) return DecorationSet.empty;
  const decorations: Decoration[] = [];
  doc.forEach((node, pos) => {
    const id = node.attrs[BLOCK_ID_ATTRIBUTE];
    if (typeof id !== "string") return;
    const origin = origins.get(id);
    const flash = flashing.has(id);
    if (!origin && !flash) return;
    decorations.push(
      Decoration.node(pos, pos + node.nodeSize, {
        ...(origin && { title: describe(origin), "data-origin": "chat" }),
        ...(flash && { class: "block-just-added" }),
      }),
    );
  });
  return DecorationSet.create(doc, decorations);
}

export const BlockOrigins = Extension.create<BlockOriginsOptions>({
  name: "blockOrigins",

  addOptions() {
    return { doc: null, describe: () => "" };
  },

  addProseMirrorPlugins() {
    const { doc, describe } = this.options;
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
              decorations: decorate(state.doc, origins, new Set(), describe),
            };
          },
          apply: (tr, value) => {
            const meta = tr.getMeta(originsKey) as Meta | undefined;
            if (meta) {
              const origins = meta.origins ?? value.origins;
              const flashing = meta.flashing ?? value.flashing;
              return {
                origins,
                flashing,
                decorations: decorate(tr.doc, origins, flashing, describe),
              };
            }
            if (!tr.docChanged) return value;
            // A Block came or went (an undo may bring one back): look again. Typing just moves them.
            if (tr.doc.childCount !== tr.before.childCount) {
              return {
                ...value,
                decorations: decorate(tr.doc, value.origins, value.flashing, describe),
              };
            }
            return { ...value, decorations: value.decorations.map(tr.mapping, tr.doc) };
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

/** Where the Block with this id came from, if it was added from a chat. */
export function originOf(state: EditorState, blockId: unknown): BlockOrigin | null {
  if (typeof blockId !== "string") return null;
  return originsKey.getState(state)?.origins.get(blockId) ?? null;
}

/** Washes the Blocks with these ids for a moment: just added. */
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
