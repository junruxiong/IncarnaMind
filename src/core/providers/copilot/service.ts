/** Apart from the request adapter, so the renderer can import it. */
export const COPILOT_SERVICE = { id: "copilot", name: "GitHub Copilot" } as const;

/** GitHub's own error for an account without Copilot, and nothing else. */
export const NO_COPILOT_ACCESS = /\bno_copilot_access\b|\bno access to github copilot\b/i;
