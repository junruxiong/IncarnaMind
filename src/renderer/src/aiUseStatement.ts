import { core } from "./core";
import { translateNow as t } from "./i18n";
import { useAppStore } from "./store";
import { type Undoable, useUndo } from "./undo";

/**
 * Exports a Mind's AI use statement (#739): the core makes it as a new Mind,
 * which opens in a new tab beside the one in front, so the User can edit it. The
 * Mind the statement is of is left as it is. Undo sends the new Mind to Recently
 * deleted, as for any Mind made; redo brings it back and shows it.
 */
export async function exportAiUseStatement(mindId: string): Promise<Undoable | undefined> {
  try {
    const mind = await core.createAiUseStatement(mindId);
    const made: Undoable = {
      message: t("aiStatement.made", { name: mind.title }),
      name: t("aiStatement.command"),
      undo: async () => {
        await core.deleteMind(mind.id);
      },
      redo: async () => {
        await core.restore([{ kind: "mind", id: mind.id }]);
        useAppStore.getState().openMind(mind.id);
      },
    };
    useUndo.getState().push(made);
    await useAppStore.getState().showNewMind(mind);
    return made;
  } catch (error) {
    useAppStore.getState().reportError(error);
    return undefined;
  }
}
