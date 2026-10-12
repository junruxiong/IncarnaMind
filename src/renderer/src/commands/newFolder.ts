import type { MessageKey } from "../../../shared/i18n";
import { currentTabOf } from "../layoutSlice";
import { useAppStore } from "../store";
import { type Command, type CommandContext, fromMenu } from "./registry";
import { idsOf, targetsOf } from "./targets";

const store = () => useAppStore.getState();

interface Chosen {
  mindIds: string[];
  documentIds: string[];
}

const NOTHING: Chosen = { mindIds: [], documentIds: [] };
const countOf = ({ mindIds, documentIds }: Chosen) => mindIds.length + documentIds.length;

/** The Documents checked in the Library, while it is in front. */
function checkedInLibrary(): string[] {
  const { layout, selectedDocuments } = store();
  return currentTabOf(layout)?.item.kind === "library" ? [...selectedDocuments] : [];
}

/**
 * What "New Folder with these" takes: the Documents and Minds a menu names or
 * the selection holds. Until the shared selection (#225), the Library's checked
 * Documents are the selection, and one of them stands for all.
 */
function chosenBy(ctx: CommandContext): Chosen {
  const targets = targetsOf(ctx);
  const checked = checkedInLibrary();
  const [only] = targets;
  if (
    ctx.selection.length === 0 &&
    (targets.length === 0 || (only?.kind === "document" && checked.includes(only.id)))
  ) {
    return { mindIds: [], documentIds: checked };
  }
  if (!targets.every((target) => target.kind === "mind" || target.kind === "document")) {
    return NOTHING;
  }
  return { mindIds: idsOf(ctx, "mind"), documentIds: idsOf(ctx, "document") };
}

function labelFor(chosen: Chosen): MessageKey {
  const count = countOf(chosen);
  if (count === 0) return "folder.newWith";
  if (count === 1)
    return chosen.mindIds.length > 0 ? "folder.newWith.mind" : "folder.newWith.document";
  if (chosen.documentIds.length === count) return "folder.newWith.documents";
  return chosen.mindIds.length === count ? "folder.newWith.minds" : "folder.newWith.items";
}

/** New Folder (⇧⌘N) and New Folder with these (⌃⌘N on macOS), named in their row (see `FolderGroup`). */
export const newFolderCommands: readonly Command[] = [
  {
    ...fromMenu("folder.new"),
    group: "main",
    reversible: true,
    run: () => store().createFolder(),
  },
  {
    id: "folder.newWith",
    label: (ctx, t) => {
      const chosen = chosenBy(ctx);
      return t(labelFor(chosen), { count: countOf(chosen) });
    },
    shortcut: { keys: "Ctrl+Cmd+N", macOnly: true },
    group: "move",
    reversible: true,
    when: (ctx) => countOf(chosenBy(ctx)) > 0,
    run: (ctx) => store().createFolder(chosenBy(ctx)),
  },
];
