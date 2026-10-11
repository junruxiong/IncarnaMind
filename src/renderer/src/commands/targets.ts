import type { Document } from "../../../core/api";
import type { ObjectKind, ObjectRef } from "../objects";
import { useAppStore } from "../store";
import type { CommandContext } from "./registry";

export const targetsOf = (ctx: CommandContext): readonly ObjectRef[] =>
  ctx.selection.length > 0 ? ctx.selection : ctx.focus ? [ctx.focus] : [];

export const idsOf = (ctx: CommandContext, kind: ObjectKind): string[] =>
  targetsOf(ctx).flatMap((ref) => (ref.kind === kind ? [ref.id] : []));

export function documentsOf(ctx: CommandContext): Document[] {
  const { documents } = useAppStore.getState();
  return idsOf(ctx, "document").flatMap((id) => {
    const document = documents.find((each) => each.id === id);
    return document ? [document] : [];
  });
}

/** Only while every target is a Document that passes `test`. */
export function everyDocument(ctx: CommandContext, test: (document: Document) => boolean): boolean {
  const count = targetsOf(ctx).length;
  const documents = documentsOf(ctx);
  return count > 0 && documents.length === count && documents.every(test);
}
