/**
 * Where Questions asked with the ChatGPT plan go, for data-flow consent: to
 * OpenAI, under the User's ChatGPT account. Its own service, apart from an
 * OpenAI API key's, as different terms apply. Kept apart from the endpoint
 * adapter so the renderer can import it.
 */
export const CHATGPT_SERVICE = { id: "chatgpt-plan", name: "ChatGPT" } as const;
