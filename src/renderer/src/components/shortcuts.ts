const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/** The new-tab shortcut as this system writes it, for tooltips and menus. */
export const NEW_MIND_SHORTCUT = isMac ? "⌘T" : "Ctrl+T";
