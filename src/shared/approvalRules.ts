/**
 * The rules approvals decide by that the approval card needs too (see
 * src/core/approvals.ts, `decide`): which Effects run without asking, and when
 * a Tool-level "always" (always allow, always run) no longer covers a call.
 */
import type { Effect, ThirdPartyReason } from "../core/api";

/**
 * Whether an Effect runs without asking: reading the Documents, a Skill, or
 * a Connector's service; sending to a Connector's service, which the User's
 * consent to it covers (the call asks for that first).
 */
export const runsByDefault = ({ action, scope }: Effect): boolean =>
  scope.kind === "service"
    ? action === "read" || action === "network"
    : action === "read" && (scope.kind === "documents" || scope.kind === "skill");

/** Changing something, or sending data off this computer. */
const writesOrSends = ({ action }: Effect): boolean => action === "write" || action === "network";

/**
 * Whether a call asks because its Run had read third-party text, whatever the
 * User's policy (#78, ADR-0013): it doesn't run by default, and it writes or
 * sends data, so someone else's words could have steered it. A Tool-level
 * "always" was given for calls the User's own words lead to, and doesn't
 * cover it; the card offers none.
 */
export function asksAfterThirdParty(call: {
  effects: readonly Effect[];
  thirdParty?: readonly ThirdPartyReason[];
}): boolean {
  return (
    (call.thirdParty?.length ?? 0) > 0 &&
    !call.effects.every(runsByDefault) &&
    call.effects.some(writesOrSends)
  );
}
