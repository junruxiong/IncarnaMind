import type { DocumentFailureReason } from "../../api";

/** Why a file's text couldn't be extracted. */
export class ExtractionError extends Error {
  override name = "ExtractionError";
  constructor(
    readonly reason: DocumentFailureReason,
    message: string,
  ) {
    super(message);
  }
}
