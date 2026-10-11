import type { ChatReadiness } from "../../../../core/api";
import type { MessageKey } from "../../../../shared/i18n";
import { signInAgain } from "../../chatGptSignIn";
import { useT } from "../../i18n";
import type { SettingsPage } from "../../settingsPages";
import { useAppStore } from "../../store";
import { ProblemLine } from "../ProblemLine";
import { readinessKey, serviceName } from "./shared";

/** Where to fix it: a declined data flow is allowed again on the Privacy page, the rest under Models. */
export const settingsPageFor = (
  readiness: Extract<ChatReadiness, { ready: false }>,
): SettingsPage => (readiness.reason === "consent-declined" ? "privacy" : "models");

/**
 * The one action that fixes `readiness`: for an ended ChatGPT sign-in,
 * signing in again right there; else opening the Settings page that fixes it,
 * labelled `label`. `before` runs first, e.g. to dismiss the problem.
 */
export function readinessAction(
  readiness: Extract<ChatReadiness, { ready: false }>,
  t: ReturnType<typeof useT>,
  openSettings: (page: SettingsPage) => void,
  label: MessageKey,
  before?: () => void,
): { label: string; onClick(): void } {
  if (readiness.reason === "sign-in-required") {
    return {
      label: t("chatgpt.signInAgain"),
      onClick: () => {
        before?.();
        void signInAgain(openSettings);
      },
    };
  }
  return {
    label: t(label),
    onClick: () => {
      before?.();
      openSettings(settingsPageFor(readiness));
    },
  };
}

/**
 * What to configure before Questions can be asked. `inLine`: ahead of an
 * action on the same line, so without the full stop that ends it otherwise.
 */
export function ReadinessExplanation({
  readiness,
  inLine = false,
}: {
  readiness: Extract<ChatReadiness, { ready: false }>;
  inLine?: boolean;
}) {
  const t = useT();
  const service = "provider" in readiness ? serviceName(readiness.provider, t) : "";
  const words = t(readinessKey(readiness.reason), { service });
  return <>{inLine ? words.replace(/[.。]$/, "") : words}</>;
}

/**
 * Shown in a Mind while Questions can't be asked, with a way to fix it.
 * Answers (#29) disable asking with the same explanation.
 */
export function ChatReadinessNotice() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const openSettings = useAppStore((state) => state.openSettings);
  if (!readiness || readiness.ready) return null;
  return (
    <ProblemLine
      role="status"
      testId="chat-readiness"
      className="mb-3"
      action={readinessAction(readiness, t, openSettings, "providers.readiness.setUp")}
    >
      <ReadinessExplanation readiness={readiness} inLine />
    </ProblemLine>
  );
}
