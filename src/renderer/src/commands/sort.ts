import type { SortScope } from "../sortChoices";
import { useSortMenu } from "../sortMenu";
import { useAppStore } from "../store";
import type { Command, CommandContext } from "./registry";
import { idsOf } from "./targets";

/** The place a Folder's or a Linked folder's sort is kept under: the Folder its contents are in. */
function scopeOf(ctx: CommandContext): { scope: SortScope; button: string } | undefined {
  const folder = idsOf(ctx, "folder")[0];
  if (folder) return { scope: `folder:${folder}`, button: `folder:${folder}` };
  const linkedId = idsOf(ctx, "linkedFolder")[0];
  const linked = useAppStore.getState().linkedFolders.find((each) => each.id === linkedId);
  return linked && { scope: `folder:${linked.folderId}`, button: `linkedFolder:${linked.id}` };
}

export const sortCommands: readonly Command[] = [
  {
    id: "folder.sort",
    label: "sort.menu",
    group: "main",
    appliesTo: ["folder", "linkedFolder"],
    when: (ctx) => scopeOf(ctx) !== undefined,
    reversible: true,
    run(ctx) {
      const at = scopeOf(ctx);
      if (!at) return;
      // After the menu has closed, beside the ⋯ it opened from.
      requestAnimationFrame(() =>
        useSortMenu.getState().show({
          scope: at.scope,
          withFolders: true,
          anchor: document.querySelector<HTMLButtonElement>(
            `[data-object-menu="${CSS.escape(at.button)}"]`,
          ),
        }),
      );
    },
  },
];
