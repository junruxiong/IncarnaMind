import type { MessageKey, MessageParams } from "../../shared/i18n";
import type { Undoable } from "./undo";

/**
 * The kinds of thing the User works with, the same for every gesture: menus, selection,
 * drag and ⌘K. A conversation is the assistant's, never a Mind (ADR-0017).
 */
export type ObjectKind =
  | "mind"
  | "conversation"
  | "document"
  | "folder"
  | "tag"
  | "linkedFolder"
  | "citation"
  | "answer"
  | "block"
  | "tab";

export interface ObjectRef {
  kind: ObjectKind;
  id: string;
}

export type Translate = (key: MessageKey, params?: MessageParams) => string;

export type { Undoable };
