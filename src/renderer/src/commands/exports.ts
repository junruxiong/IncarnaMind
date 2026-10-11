import { EXPORT_FORMATS } from "../../../core/api";
import { exportMind, exportPreferences, FORMAT_LABELS, useExportOptions } from "../exports";
import { currentTabOf } from "../layoutSlice";
import { useAppStore } from "../store";
import type { Command, CommandContext } from "./registry";

/**
 * The Mind a command exports: the tab it was run on (the group's ⋯ menu passes it), else the
 * Mind in the tab the User is in.
 */
function mindOf(ctx: CommandContext): string | null {
  const { layout } = useAppStore.getState();
  const tab =
    ctx.focus?.kind === "tab"
      ? layout?.groups.flatMap((group) => group.tabs).find((each) => each.id === ctx.focus?.id)
      : currentTabOf(layout);
  return tab?.item.kind === "mind" ? tab.item.mindId : null;
}

/**
 * Exporting a Mind (#100): ⇧⌘E exports again in the last format, as the tab strip's Export
 * does; each other format has its own entry, which becomes the last; and Export options… opens
 * the sheet. In ⌘K and the group's ⋯ menu.
 */
export const exportCommands: readonly Command[] = [
  {
    id: "mind.export",
    label: (_ctx, t) => t("export.as", { format: t(FORMAT_LABELS[exportPreferences().format]) }),
    keywords: ["export.keywords"],
    shortcut: { keys: "CmdOrCtrl+Shift+E" },
    group: "file",
    // A new file under a free name: nothing is replaced, and the file stays the User's to delete.
    reversible: true,
    when: (ctx) => mindOf(ctx) !== null,
    run: (ctx) => {
      const mindId = mindOf(ctx);
      if (mindId) return exportMind(mindId);
    },
  },
  ...EXPORT_FORMATS.map(
    (format): Command => ({
      id: `mind.exportAs.${format}`,
      label: (_ctx, t) => t("export.as", { format: t(FORMAT_LABELS[format]) }),
      keywords: ["export.keywords"],
      group: "file",
      reversible: true,
      // The last format is "mind.export", with its keys.
      when: (ctx) => mindOf(ctx) !== null && exportPreferences().format !== format,
      run: (ctx) => {
        const mindId = mindOf(ctx);
        if (mindId) return exportMind(mindId, format);
      },
    }),
  ),
  {
    id: "mind.exportOptions",
    label: "export.options",
    keywords: ["export.keywords"],
    group: "file",
    // Only opens the sheet; what it saves is the User's choice of options.
    reversible: true,
    when: (ctx) => mindOf(ctx) !== null,
    run: (ctx) => {
      const mindId = mindOf(ctx);
      if (mindId) useExportOptions.getState().open(mindId);
    },
  },
];
