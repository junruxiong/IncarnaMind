import type { ChatReadiness } from "../../../../core/api";
import { signInAgain } from "../../chatGptSignIn";
import { useT } from "../../i18n";
import { turnOffLocalMode } from "../../localMode";
import { useModelSetup } from "../../modelSetup";
import { useRequests } from "../../requests";
import type { SettingsPage } from "../../settingsPages";
import { useAppStore } from "../../store";
import { ProblemLine } from "../ProblemLine";
import { readinessKey, serviceName } from "./shared";

/** Where to fix it: a declined data flow is allowed again on the Privacy page, the rest under Models. */
export const settingsPageFor = (
  readiness: Extract<ChatReadiness, { ready: false }>,
): SettingsPage => (readiness.reason === "consent-declined" ? "privacy" : "models");

/**
 * The one action that fixes `readiness`, labelled `label` (its Settings page
 * by default): for an ended ChatGPT sign-in, "Sign in again", right there;
 * else opening the Settings page that fixes it. `before` runs first, e.g. to
 * dismiss the problem.
 */
export function readinessAction(
  readiness: Extract<ChatReadiness, { ready: false }>,
  t: ReturnType<typeof useT>,
  openSettings: (page: SettingsPage) => void,
  label: string,
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
    label,
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
 * Why Questions can't be asked, as one line with the one thing that fixes it.
 * Local mode refusing a cloud model has two ways out, so it offers both: a
 * local model, or turning local mode off (which can be undone).
 * `refused`: a Question was just asked and not sent, rather than Questions
 * being off in general.
 */
export function ReadinessLine({
  readiness,
  testId,
  className,
  setUpLabel,
  refused = false,
}: {
  readiness: Extract<ChatReadiness, { ready: false }>;
  testId: string;
  className?: string;
  setUpLabel: string;
  refused?: boolean;
}) {
  const t = useT();
  const openSettings = useAppStore((state) => state.openSettings);
  if (readiness.reason === "local-mode") {
    return (
      <ProblemLine
        role="status"
        testId={testId}
        {...(className && { className })}
        action={{
          label: t("localMode.useLocal"),
          testId: "local-mode-use-local",
          onClick: () => useRequests.getState().ask({ chooseLocalModel: true }),
        }}
        also={{
          label: t("localMode.turnOff"),
          testId: "local-mode-turn-off",
          onClick: () =>
            turnOffLocalMode().catch((error: unknown) => useAppStore.getState().reportError(error)),
        }}
      >
        {refused ? (
          <>{t("localMode.refused", { service: serviceName(readiness.provider, t) })}</>
        ) : (
          <ReadinessExplanation readiness={readiness} inLine />
        )}
      </ProblemLine>
    );
  }
  return (
    <ProblemLine
      role="status"
      testId={testId}
      {...(className && { className })}
      action={readinessAction(readiness, t, openSettings, setUpLabel)}
    >
      <ReadinessExplanation readiness={readiness} inLine />
    </ProblemLine>
  );
}

/**
 * Shown in a Mind while Questions can't be asked, with a way to fix it.
 * Answers (#29) disable asking with the same explanation.
 */
export function ChatReadinessNotice() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  // The setup card, waiting with the Question, already says it.
  const settingUp = useModelSetup((state) => state.waiting !== null);
  if (!readiness || readiness.ready || settingUp) return null;
  return (
    <ReadinessLine
      readiness={readiness}
      testId="chat-readiness"
      className="mb-3"
      setUpLabel={t("providers.readiness.setUp")}
    />
  );
}
