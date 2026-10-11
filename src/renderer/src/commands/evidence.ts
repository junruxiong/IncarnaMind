import type { Document } from "../../../core/api";
import { countedByChoice, leftOutOfEvidence } from "../../../shared/evidence";
import { countAsEvidence } from "../evidence";
import { useAppStore } from "../store";
import type { Command, CommandContext } from "./registry";

/** The Documents a command acts on: the selection, else the focus. */
function documentsOf(ctx: CommandContext): Document[] {
  const refs = ctx.selection.length > 0 ? ctx.selection : ctx.focus ? [ctx.focus] : [];
  const { documents } = useAppStore.getState();
  return refs.flatMap((ref) => {
    const document = documents.find((each) => each.id === ref.id);
    return document ? [document] : [];
  });
}

/** Only while every Document it acts on passes. */
const everyOne = (ctx: CommandContext, test: (document: Document) => boolean) => {
  const count = ctx.selection.length > 0 ? ctx.selection.length : ctx.focus ? 1 : 0;
  const documents = documentsOf(ctx);
  return count > 0 && documents.length === count && documents.every(test);
};

/** "Count it" for the files IncarnaMind wrote (#80), and taking it back. */
export const evidenceCommands: readonly Command[] = [
  {
    id: "document.countAsEvidence",
    label: (ctx, t) => {
      const count = documentsOf(ctx).length;
      return count > 1 ? t("evidence.countMany", { count }) : t("evidence.count");
    },
    group: "main",
    appliesTo: ["document"],
    when: (ctx) => {
      const { settings, library } = useAppStore.getState();
      return everyOne(ctx, (document) => leftOutOfEvidence(document, settings, library));
    },
    reversible: true,
    run: (ctx) => countAsEvidence(documentsOf(ctx), true),
  },
  {
    id: "document.leaveOutOfEvidence",
    label: (ctx, t) => {
      const count = documentsOf(ctx).length;
      return count > 1 ? t("evidence.leaveOutMany", { count }) : t("evidence.leaveOut");
    },
    group: "main",
    appliesTo: ["document"],
    // Only where taking the choice back leaves it out: not while Settings or its Folder counts it.
    when: (ctx) => {
      const { settings, library } = useAppStore.getState();
      return everyOne(ctx, (document) => countedByChoice(document, settings, library));
    },
    reversible: true,
    run: (ctx) => countAsEvidence(documentsOf(ctx), false),
  },
];
