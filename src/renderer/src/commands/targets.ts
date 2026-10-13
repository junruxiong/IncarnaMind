import type { Document } from "../../../core/api";
import type { ObjectKind, ObjectRef } from "../objects";
import { useAppStore } from "../store";
import type { CommandContext } from "./registry";

export const targetsOf = (ctx: CommandContext): readonly ObjectRef[] =>
  ctx.selection.length > 0 ? ctx.selection : ctx.focus ? [ctx.focus] : [];

/** Only while one thing is selected or focused: what opens, renames or exports one. */
export const onlyOne = (ctx: CommandContext): boolean => targetsOf(ctx).length <= 1;

export const idsOf = (ctx: CommandContext, kind: ObjectKind): string[] =>
  targetsOf(ctx).flatMap((ref) => (ref.kind === kind ? [ref.id] : []));

export function documentsOf(ctx: CommandContext): Document[] {
  const ids = new Set(idsOf(ctx, "document"));
  return ids.size > 0 ? useAppStore.getState().documents.filter((each) => ids.has(each.id)) : [];
}

/** Only while every target is a Document that passes `test`. */
export function everyDocument(ctx: CommandContext, test: (document: Document) => boolean): boolean {
  const count = targetsOf(ctx).length;
  const documents = documentsOf(ctx);
  return count > 0 && documents.length === count && documents.every(test);
}
