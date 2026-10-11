import { useState } from "react";
import type { ChatGptAccount, ConnectionTestResult } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonClass,
  errorTextClass,
  ghostButtonClass,
  noticeClass,
  primaryButtonClass,
  rowTextClass,
  sectionTitleClass,
} from "../ui";
import {
  ChatGptSignInState,
  ChatGptStorageNotice,
  ContinueWithChatGptButton,
  ManageUsageLink,
  useChatGptPlan,
  useChatGptStorage,
  useContinueWithChatGpt,
} from "./ChatGptSignIn";
import { TestResult } from "./ProviderForm";

/**
 * Settings → Models → ChatGPT plan: "Continue with ChatGPT" (Sign in with
 * ChatGPT), then the account, whether Questions use the plan, a test, and
 * Sign out. The plan's model is chosen in Chat model above, the one place the
 * default model is chosen. Signing in here doesn't change the default model
 * unless there is none yet.
 */
export function ChatGptPlanSettings() {
  const t = useT();
  const [status, setStatus] = useChatGptPlan();
  const readiness = useAppStore((state) => state.chatReadiness);
  const chatModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const [busy, setBusy] = useState<"testing" | "saving" | "signing-out" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unconfirmed, setUnconfirmed] = useState(false);
  const sign = useContinueWithChatGpt({ keepDefault: true });
  const storage = useChatGptStorage();

  const inUse =
    readiness !== null && "provider" in readiness && readiness.provider.kind === "chatgpt";
  // The model Questions use with the plan: the one chosen in Chat model, else the plan's default.
  const planModel = (inUse && chatModel?.modelId) || (status?.models[0]?.id ?? "");
  const signedIn = status?.account.state === "signed-in";
  const waiting = sign.waiting || status?.signingIn === true;

  const run = async (action: "testing" | "saving" | "signing-out", task: () => Promise<void>) => {
    setBusy(action);
    setError(null);
    try {
      await task();
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy(null);
    }
  };

  const signIn = () => {
    setTest(null);
    setUnconfirmed(false);
    void sign.run();
  };

  const signOut = () =>
    run("signing-out", async () => {
      setTest(null);
      const result = await core.signOutOfChatGpt();
      setStatus(result.status);
      setUnconfirmed(!result.revoked);
    });

  const testConnection = () =>
    run("testing", async () => {
      setTest(null);
      setTest(await core.testChatConnection({ kind: "chatgpt", modelId: planModel }));
    });

  const use = () =>
    run("saving", async () => {
      await core.saveChatProvider({ kind: "chatgpt", modelId: planModel });
    });

  return (
    <section data-testid="chatgpt-settings" className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <h4 className={sectionTitleClass}>{t("chatgpt.title")}</h4>
        <p className={rowTextClass}>{t("chatgpt.description")}</p>
      </div>

      {status && (
        <p data-testid="chatgpt-account" className="text-ui text-ink-secondary">
          <AccountLine account={status.account} />
        </p>
      )}

      {waiting ? (
        <ChatGptSignInState sign={sign} />
      ) : !signedIn ? (
        <div className="flex flex-col items-start gap-2">
          <ChatGptStorageNotice storage={storage} />
          <ContinueWithChatGptButton
            onClick={signIn}
            disabled={status === null || !storage.canSave}
          />
          <ChatGptSignInState sign={sign} />
        </div>
      ) : (
        <>
          {inUse && (
            <p data-testid="chatgpt-in-use" className="text-[13px] leading-5 text-ink-secondary">
              {t("chatgpt.inUse")}
              <span aria-hidden="true"> · </span>
              <ManageUsageLink />
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            {!inUse && (
              <button
                type="button"
                data-testid="chatgpt-use"
                disabled={busy !== null || planModel === ""}
                onClick={() => void use()}
                className={primaryButtonClass}
              >
                {t("chatgpt.use")}
              </button>
            )}
            <button
              type="button"
              data-testid="chatgpt-test"
              disabled={busy !== null || planModel === ""}
              onClick={() => void testConnection()}
              className={buttonClass}
            >
              {busy === "testing" ? t("providers.form.testing") : t("providers.form.test")}
            </button>
            <button
              type="button"
              data-testid="chatgpt-sign-out"
              disabled={busy !== null}
              onClick={() => void signOut()}
              className={ghostButtonClass}
            >
              {t("chatgpt.signOut")}
            </button>
          </div>
        </>
      )}

      {unconfirmed && !signedIn && (
        <p role="note" data-testid="chatgpt-sign-out-unconfirmed" className={noticeClass}>
          {t("chatgpt.signedOut.unconfirmed")} <ManageUsageLink />
        </p>
      )}
      {test && <TestResult result={test} onRetry={() => void testConnection()} />}
      {error && (
        <p role="alert" data-testid="chatgpt-settings-error" className={errorTextClass}>
          {error}
        </p>
      )}
    </section>
  );
}

function AccountLine({ account }: { account: ChatGptAccount }) {
  const t = useT();
  if (account.state === "signed-out") return <>{t("chatgpt.account.signedOut")}</>;
  if (account.state === "expired") return <>{t("chatgpt.account.expired")}</>;
  return (
    <>{t("chatgpt.account.signedIn", { account: account.email ?? t("chatgpt.account.unknown") })}</>
  );
}
