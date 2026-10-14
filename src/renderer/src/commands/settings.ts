import { settingsPageLabels, settingsPages } from "../settingsPages";
import { useAppStore } from "../store";
import type { Command } from "./registry";

/** Each Settings page but General, which is Settings… itself, opened from ⌘K. */
export const settingsPageCommands: readonly Command[] = settingsPages
  .filter((page) => page !== "general")
  .map((page) => ({
    id: `settings.${page}`,
    label: (_ctx, t) => t("command.settings.page", { page: t(settingsPageLabels[page]) }),
    keywords: ["settings.title"],
    group: "open",
    reversible: true,
    run: () => useAppStore.getState().openSettings(page),
  }));
