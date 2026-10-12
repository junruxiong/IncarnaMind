import type { MessageKey, MessageParams } from "../../shared/i18n";

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
  | "chart"
  | "answer"
  | "block"
  | "tab";

export interface ObjectRef {
  kind: ObjectKind;
  id: string;
}

export type Translate = (key: MessageKey, params?: MessageParams) => string;

/** A change that can be taken back: ⌘Z and the undo line. */
export interface Undoable {
  /** What happened, said plainly: "Moved 3 Documents to Finance". */
  message: string;
  undo(): Promise<void>;
}
