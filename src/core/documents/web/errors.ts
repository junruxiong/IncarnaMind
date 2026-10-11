import type { WebPageFailure, WebPageFailureReason } from "../../api";

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
