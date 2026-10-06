import type { ChatReadiness, DataFlow, SecretProtection } from "./api";

/** Thrown when a caller of the core's public interface passes malformed input. */
export class InvalidInputError extends Error {
  override name = "InvalidInputError";
}

/** Thrown when a secret can't be stored safely on this device (see `getSecretStorage`). */
export class SecretStorageError extends Error {
  override name = "SecretStorageError";
  constructor(readonly protection: SecretProtection) {
    super(
      protection === "plain-text"
        ? "No keyring is running, so API keys would be stored in plain text. Start GNOME Keyring or KWallet, or accept plain-text storage in Settings."
        : "This system can't encrypt API keys, so they can't be saved.",
    );
  }
}

/** Thrown when the User declined a data flow: nothing was sent. */
export class ConsentDeclinedError extends Error {
  override name = "ConsentDeclinedError";
  constructor(readonly flow: DataFlow) {
    super(`You declined sending data to ${flow.service.name}, so nothing was sent.`);
  }
}

/** Thrown when Questions can't be answered yet; `readiness` says what to configure. */
export class ChatNotReadyError extends Error {
  override name = "ChatNotReadyError";
  constructor(readonly readiness: Extract<ChatReadiness, { ready: false }>) {
    super(`Chat isn't set up: ${readiness.reason}.`);
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
