/**
 * Whether a command the assistant runs asks the User first (#1110, ADR-0013): decided here
 * alone, from what its entry in `ASSISTANT_COMMANDS` says it does and what the check found, never
 * by the command's own code. It runs at once only when it changes a few things inside
 * IncarnaMind, which one Undo takes back, or saves a file in the Output location itself.
 * Everything else asks, with Allow or Don't allow and never "always":
 * - a file saved anywhere else, and a folder's files read;
 * - a change of more than `COMMAND_AT_ONCE` things;
 * - a delete once the Answer has read text the User didn't write here, which could have asked
 *   for it (Passages, a Connector's reply, a web page);
 * - a delete of the Mind the User is in.
 */

import { runsByDefault } from "../../../shared/approvalRules";
import { ASSISTANT_COMMANDS, isAppCommand } from "../../../shared/commands";
import { COMMAND_AT_ONCE, type CommandAsk, type CommandAskReason } from "../../api";
import type { CheckedCommand } from "./run";

/** How many of the things a change would make the approval names. */
const NAMES_SHOWN = 50;

export interface CommandContext {
  /** The Answer had read untrusted content before the call: Passages, a Connector's reply. */
  tainted: boolean;
  /** The Answer counts as having read third-party text: a web page, the document being checked. */
  readThirdParty: boolean;
  /** The Mind the User is in, if any. */
  focusedMindId: string | null;
}

function reasonFor(checked: CheckedCommand, context: CommandContext): CommandAskReason | null {
  const { deletes } = ASSISTANT_COMMANDS[checked.command];
  if (
    deletes &&
    checked.items.some((item) => item.kind === "mind" && item.id === context.focusedMindId)
  ) {
    return "focusedMind";
  }
  if (checked.items.length > COMMAND_AT_ONCE) return "many";
  if (deletes && (context.tainted || context.readThirdParty)) return "afterReading";
  return null;
}

/** What the User is asked before the call runs; null: it runs at once. */
export function askOf(checked: CheckedCommand, context: CommandContext): CommandAsk | null {
  const { command } = checked;
  const { does } = ASSISTANT_COMMANDS[command];
  if (checked.outside) return checked.outside;
  if (isAppCommand(command)) {
    const reason = reasonFor(checked, context);
    if (reason) {
      return {
        kind: "change",
        command,
        reason,
        count: checked.items.length,
        names: checked.items.slice(0, NAMES_SHOWN).map((item) => item.name),
        into: checked.into,
      };
    }
  }
  // Nothing outside to show, and only an in-app change or the Output location runs at once.
  if ((does === "app" || does === "export") && checked.effects.every(runsByDefault)) return null;
  throw new Error(`${checked.command} would do what nothing can ask the User about.`);
}
