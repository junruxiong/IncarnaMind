import { countedByChoice, leftOutOfEvidence } from "../../../shared/evidence";
import { countAsEvidence } from "../evidence";
import { useAppStore } from "../store";
import type { Command } from "./registry";
import { documentsOf, everyDocument } from "./targets";

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
      return everyDocument(ctx, (document) => leftOutOfEvidence(document, settings, library));
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
      return everyDocument(ctx, (document) => countedByChoice(document, settings, library));
    },
    reversible: true,
    run: (ctx) => countAsEvidence(documentsOf(ctx), false),
  },
];
