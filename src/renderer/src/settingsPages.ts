/** The pages of the Settings dialog, in the order its list shows them. */
export const settingsPages = [
  "general",
  "models",
  "search",
  "tools",
  "connectors",
  "skills",
  "privacy",
] as const;

export type SettingsPage = (typeof settingsPages)[number];

export const isSettingsPage = (page: unknown): page is SettingsPage =>
  settingsPages.some((each) => each === page);
