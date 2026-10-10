import type { WebPageFailure, WebPageFailureReason } from "../../api";

/** Why a web page couldn't be saved or fetched again. */
export class WebPageError extends Error {
  override name = "WebPageError";
  constructor(
    readonly reason: WebPageFailureReason,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }

  get failure(): WebPageFailure {
    return { reason: this.reason, message: this.message };
  }
}
