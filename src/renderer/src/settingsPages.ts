/** The pages of the Settings dialog, in the order its list shows them. */
export const settingsPages = [
  "general",
  "models",
  "search",
  "meetings",
  "connectors",
  "skills",
  "agents",
  "privacy",
] as const;

export type SettingsPage = (typeof settingsPages)[number];

/** Pages that were renamed or folded into another; a link to one opens where its settings went. */
const movedPages: Record<string, SettingsPage> = { tools: "privacy", notifications: "general" };

export const resolveSettingsPage = (page: unknown): SettingsPage =>
  isSettingsPage(page)
    ? page
    : typeof page === "string"
      ? (movedPages[page] ?? "general")
      : "general";

export const isSettingsPage = (page: unknown): page is SettingsPage =>
  settingsPages.some((each) => each === page);
