import type { MessageKey } from "../../shared/i18n";

/**
 * What a failure says, split three ways: the core's own sentence for the User, plain words when the
 * sentence is the system's code, and the technical text for a "Details" disclosure.
 */
export interface ErrorDescription {
  /** The sentence the error carries, without Electron's wrapping, its class name or a stack. */
  message: string;
  /** Plain words, in the dictionary, for a failure of the system's (a full disk, a name too long). */
  reason: MessageKey | null;
  /**
   * The core method that failed, the error's class and its sentence, for the Details of a notice.
   * Never Electron's wrapping or a stack: they say nothing the User or a bug report can use.
   */
  detail: string;
}

/**
 * Electron rejects a call to the main process with the error written out:
 * "Error invoking remote method 'core:x': InvalidInputError: …".
 */
const IPC_WRAPPING = /^Error invoking remote method '([^']*)': ([\s\S]*)$/;
const CLASS_NAME = /^(\w*Error): ([\s\S]*)$/;
/** A line of a stack, which a message may carry when someone wrote the stack into it. */
const STACK_FROM = /\r?\n\s+at\s[\s\S]*$/;

/** The failures of the system a User can read plain words for, by their code. */
const SYSTEM_REASONS: readonly (readonly [RegExp, MessageKey])[] = [
  [/\b(EACCES|EPERM|EROFS)\b/, "error.system.notAllowed"],
  [/\b(ENOSPC|EDQUOT)\b/, "error.system.full"],
  [/\b(ENOENT|ENOTDIR)\b/, "error.system.gone"],
  [/\bENAMETOOLONG\b/, "error.system.nameTooLong"],
  [/\bEEXIST\b/, "error.system.exists"],
];

/**
 * Takes a failure apart. This is the one place the wrapping Electron puts around an error from the
 * core is taken off, so no notice or line shows it.
 */
export function describeError(error: unknown): ErrorDescription {
  const text = error instanceof Error ? error.message : String(error);
  const wrapped = IPC_WRAPPING.exec(text);
  const inner = wrapped ? (wrapped[2] ?? "") : text;
  const named = CLASS_NAME.exec(inner);
  const className = named ? named[1] : error instanceof Error ? error.name : null;
  const message = (named ? (named[2] ?? "") : inner).replace(STACK_FROM, "").trim();
  const reason = SYSTEM_REASONS.find(([code]) => code.test(message))?.[1] ?? null;
  const said = `${className || "Error"}: ${message}`;
  return { message, reason, detail: wrapped?.[1] ? `${wrapped[1]}\n${said}` : said };
}

/** The sentence of an error from the core, for a line that has no room for more. */
export function errorMessage(error: unknown): string {
  return describeError(error).message;
}
