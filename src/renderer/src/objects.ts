import type { ReactNode } from "react";
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

/** What dropping things on a target would do, said on the target while it can accept them. */
export interface DropEffect {
  kind: "move" | "add" | "attach" | "import";
  label: string;
}

export interface ObjectType {
  label(ref: ObjectRef, t: Translate): string;
  icon(ref: ObjectRef): ReactNode;
  rename?(ref: ObjectRef, name: string): Promise<Undoable>;
  /** To Recently deleted. */
  remove?(refs: readonly ObjectRef[]): Promise<Undoable>;
  accepts?(target: ObjectRef, dragged: readonly ObjectRef[]): DropEffect | null;
}
