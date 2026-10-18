/**
 * What the checks of an Answer's writes left to do (#1111), from its calls:
 * the lines under it, and the sentences their Check again and Check the rest
 * check. Only data.
 */
import type { AnswerToolCall, MindChange } from "../../core/api";
import type { Language } from "../../core/language";
import { formatResetTime } from "./cost";
import type { ShownClaim } from "./editor/claimChecks";
import type { Translate } from "./objects";

export type SelfCheck = NonNullable<MindChange["selfCheck"]> & { mindId: string };

/** Each write's check, in the order of the calls, with something left to check: a failure, or sentences past the most one write checks. */
export const selfChecksOf = (toolCalls: readonly AnswerToolCall[]): SelfCheck[] =>
  toolCalls.flatMap((call) => {
    const change = call.mindChange;
    const check = change?.selfCheck;
    return change && check && (check.failed > 0 || check.sent < check.cited)
      ? [{ mindId: change.mindId, ...check }]
      : [];
  });

/**
 * One line under the Answer: its words, more parts before its action, and the
 * action: check the sentences left now, or change the spending cap. A plan's
 * limit has none: nothing can be checked until it resets.
 */
export interface SelfCheckLine {
  kind: "failed" | "cost-cap" | "plan-limit" | "capped";
  text: string;
  details: string[];
  action: { label: string; does: "check" | "change-cap" } | null;
}

export function selfCheckLines(
  check: SelfCheck,
  t: Translate,
  language: Language,
  now = new Date(),
): SelfCheckLine[] {
  const { failed, stop } = check;
  const lines: SelfCheckLine[] = [];
  const one = failed === 1;
  if (failed > 0 && stop?.reason === "cost-cap") {
    lines.push({
      kind: "cost-cap",
      text: one ? t("selfCheck.costCap.one") : t("selfCheck.costCap", { count: failed }),
      details: [],
      action: { label: t("selfCheck.changeCap"), does: "change-cap" },
    });
  } else if (failed > 0 && stop?.reason === "plan-limit") {
    lines.push({
      kind: "plan-limit",
      text: one ? t("selfCheck.planLimit.one") : t("selfCheck.planLimit", { count: failed }),
      details: stop.resetsAt
        ? [t("selfCheck.resumesAt", { time: formatResetTime(stop.resetsAt, language, now) })]
        : [],
      action: null,
    });
  } else if (failed > 0) {
    lines.push({
      kind: "failed",
      text: one ? t("selfCheck.failed.one") : t("selfCheck.failed", { count: failed }),
      details: [],
      action: { label: t("claim.checkAgain"), does: "check" },
    });
  }
  if (check.sent < check.cited) {
    lines.push({
      kind: "capped",
      text: t("selfCheck.capped", { checked: check.checked, count: check.cited }),
      details: [],
      action: { label: t("selfCheck.checkRest"), does: "check" },
    });
  }
  return lines;
}

/** The sentences a write's check left not checked, as they are now: one edited since needs its own check again. */
export const leftToCheck = (
  claims: readonly ShownClaim[],
  request: string,
): { from: number; to: number }[] =>
  claims
    .filter((claim) => claim.check.request === request && claim.shown.state === "not-checked")
    .map((claim) => ({ from: claim.sentence.from, to: claim.sentence.to }));
