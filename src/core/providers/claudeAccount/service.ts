/**
 * Where Questions asked with a Claude account go, for data-flow consent: to
 * Anthropic, under the User's Claude account. Its own service, apart from an
 * Anthropic API key's, as different terms apply. Kept apart from the request
 * adapter so the renderer can import it.
 */
export const CLAUDE_ACCOUNT_SERVICE = { id: "claude-account", name: "Claude" } as const;
