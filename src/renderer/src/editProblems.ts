import type { ProviderError } from "../../core/api";
import type { EditProblem } from "./editWithAi";

/**
 * Why an Edit with AI failed, for its line: sending turned off under it says
 * so in the interface's words, with nothing to try again, as an Answer does
 * (#1206); any other failure offers Try again.
 */
export const failedEdit = (error: ProviderError, retry: () => void): EditProblem =>
  error.kind === "consent-revoked"
    ? { kind: "consent-revoked", error }
    : { kind: "failed", message: error.message, retry };

interface LineChoice {
  label: string;
  onClick(): void;
}

/** What the line for an Edit with AI that gave no changes says, wherever it shows. */
export interface EditProblemWords {
  text: string;
  role: "alert" | "status";
  /** The one thing that helps; none where closing the line is all there is. */
  action?: LineChoice;
  /** A sign-in route's second way back. */
  also?: LineChoice | undefined;
}
