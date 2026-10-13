import type { BlockOrigin } from "../../../core/api";
import { openConversation } from "../conversations";
import { core } from "../core";
import { translateNow as t } from "../i18n";
import { useAppStore } from "../store";

/**
 * Opens the conversation a Block was added from, in the assistant at its
 * Answer, flashed (#745). One in Recently deleted, or gone, is said to be.
 */
export async function openOrigin(origin: BlockOrigin): Promise<void> {
  if (origin.from.artifactKind !== "conversation") return;
  try {
    const target = await core.resolveReference(origin.from);
    if (target.status === "available") {
      await openConversation(origin.from.artifactId, origin.from.anchor.messageId);
      return;
    }
    const title = "name" in target && target.name ? target.name : t("conversation.untitled");
    useAppStore
      .getState()
      .reportError(
        new Error(target.status === "deleted" ? t("origin.deleted", { title }) : t("origin.gone")),
      );
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
}
