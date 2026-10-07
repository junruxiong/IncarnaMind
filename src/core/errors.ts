import type {
  ChatReadiness,
  DataFlow,
  EmbeddingModelStatus,
  ProviderError,
  SecretProtection,
} from "./api";

/** Thrown when a caller of the core's public interface passes malformed input. */
export class InvalidInputError extends Error {
  override name = "InvalidInputError";
}

/** Thrown when a caller names something that doesn't exist or has been deleted. */
export class NotFoundError extends Error {
  override name = "NotFoundError";
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

/**
 * Thrown when automatic tagging's model can't be used yet, e.g. Jev is set up
 * but its key can't be read on this device. Documents wait until it can.
 */
export class TaggingNotReadyError extends Error {
  override name = "TaggingNotReadyError";
}

/**
 * Thrown by a vector search while the embedding model can't be used: the
 * built-in one isn't downloaded or can't start (`status`), or the provider the
 * User chose instead can't be used, e.g. its key is missing (`error`).
 */
export class EmbeddingModelNotReadyError extends Error {
  override name = "EmbeddingModelNotReadyError";
  constructor(
    /** The built-in model's state; null when another provider is in use. */
    readonly status: EmbeddingModelStatus | null,
    /** Why the chosen provider can't be used, when it isn't the built-in model. */
    readonly error: ProviderError | null = null,
  ) {
    super(
      status === null
        ? `The embedding provider can't be used${error ? ` (${error.kind}: ${error.message})` : ""}.`
        : status.error
          ? `The embedding model isn't ready (${status.error.kind}: ${status.error.message}).`
          : `The embedding model isn't ready (${status.state}).`,
    );
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
