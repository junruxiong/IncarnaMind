/**
 * A window's local copy of a Mind's Yjs document, kept in step with the
 * core's: it starts from the core's full state, sends every local change to
 * the core, and applies the changes the core pushes. The editor holds one for
 * the Mind it shows; the assistant's card of changes (#202) holds one for each
 * Mind it accepts or rejects changes in.
 *
 * A Mind the core opens read-only (a newer version wrote it) is only a copy to
 * look at: nothing is ever sent from it.
 */
import * as Y from "yjs";
import type { MindAccess } from "../../core/api";
import { core } from "./core";

/** Marks changes that came from the core, so they aren't sent back to it. */
const FROM_CORE = Symbol("from the core");

export interface OpenedCopy {
  access: MindAccess;
  /** Null when the Mind can't be shown ("update-required"). */
  doc: Y.Doc | null;
}

/**
 * Opens a copy of the Mind: `opened` hears it once its state has arrived,
 * `failed` why it couldn't be opened. Returns what closes it.
 */
export function openMindCopy(
  mindId: string,
  opened: (copy: OpenedCopy) => void,
  failed: (error: unknown) => void,
): () => void {
  const local = new Y.Doc();
  let active = true;
  let editable = false;
  // Listen before opening, so nothing pushed in between is missed. Yjs merges in any order.
  const stopListening = core.on("mind.update", ({ mindId: changed, update }) => {
    if (changed === mindId) Y.applyUpdate(local, update, FROM_CORE);
  });
  const send = (update: Uint8Array, origin: unknown) => {
    if (origin !== FROM_CORE && editable) core.applyMindUpdate(mindId, update).catch(failed);
  };
  local.on("update", send);
  core.openMind(mindId).then(
    ({ state, access }) => {
      if (!active) return;
      if (access === "update-required") {
        opened({ access, doc: null });
        return;
      }
      editable = access === "edit";
      Y.applyUpdate(local, state, FROM_CORE);
      opened({ access, doc: local });
    },
    (error: unknown) => {
      if (active) failed(error);
    },
  );
  return () => {
    active = false;
    stopListening();
    local.off("update", send);
    core.closeMind(mindId).catch(() => undefined);
  };
}
