import { create } from "zustand";
import type { Document, LibraryGroup, LinkedFolder, Mind } from "../../core/api";

/**
 * The questions a command puts to the User before it acts: whether to delete a
 * Mind, a Document or a Folder, or unlink a Linked folder. A menu on any surface asks for one, and `RequestedDialogs` (mounted
 * once) shows it, so no row owns a dialog. When Undo replaces the
 * confirmations (#228), the commands act at once and this goes.
 */
interface Requests {
  deleteMind: Mind | null;
  deleteDocument: Document | null;
  deleteFolder: LibraryGroup | null;
  unlink: { linked: LinkedFolder; name: string } | null;
  /** The Folder whose name and description the Library opens to edit. */
  editFolder: string | null;
  /** The Tag Manage Tags opens on, to rename or to delete. */
  manageTag: { id: string; mode: "edit" | "delete" } | null;
  ask(request: Partial<Omit<Requests, "ask" | "clear">>): void;
  clear(key: keyof Omit<Requests, "ask" | "clear">): void;
}

export const useRequests = create<Requests>((set) => ({
  deleteMind: null,
  deleteDocument: null,
  deleteFolder: null,
  unlink: null,
  editFolder: null,
  manageTag: null,
  ask: (request) => set(request),
  clear: (key) => set({ [key]: null }),
}));
