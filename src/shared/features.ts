/** Feature flags. Each is off until the feature can ship. */
export interface Features {
  /**
   * Sign in with ChatGPT and use the User's plan for chat. Waits until OpenAI
   * admits IncarnaMind to that program (ADR-0005); v1 doesn't depend on it.
   * Turned on, it shows only a disabled placeholder for now.
   */
  signInWithChatGpt: boolean;
}

export const features: Readonly<Features> = {
  signInWithChatGpt: false,
};
