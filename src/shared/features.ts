/** Feature flags. Each is off until the feature can ship. */
export interface Features {
  /**
   * Sign in with ChatGPT and use the User's plan for chat. Waits until OpenAI
   * admits IncarnaMind to that program (ADR-0005); v1 doesn't depend on it.
   * Turned on, it shows only a disabled placeholder for now. The experimental
   * "ChatGPT plan (via Codex sign-in)" provider is separate: the User turns it
   * on in Settings, and the official sign-in would replace only its sign-in part.
   */
  signInWithChatGpt: boolean;
}

export const features: Readonly<Features> = {
  signInWithChatGpt: false,
};
