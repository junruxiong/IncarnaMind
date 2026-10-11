import { useState } from "react";
import type { ChatGptAccount, ConnectionTestResult } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonClass,
  errorTextClass,
  fieldLabelClass,
  ghostButtonClass,
  inputClass,
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
  useContinueWithChatGpt,
} from "./ChatGptSignIn";
import { TestResult } from "./ProviderForm";

/**
 * Settings → Models → ChatGPT plan: "Continue with ChatGPT" (Sign in with
 * ChatGPT), then the plan's model for Questions, a test, and Sign out. Signing
 * in here doesn't change the default model unless there is none yet.
 */
export function ChatGptPlanSettings() {
  const t = useT();
  const [status, setStatus] = useChatGptPlan();
  const readiness = useAppStore((state) => state.chatReadiness);
  const chatModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState<"testing" | "saving" | "signing-out" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unconfirmed, setUnconfirmed] = useState(false);
  const sign = useContinueWithChatGpt({ keepDefault: true });

  const inUse =
    readiness !== null && "provider" in readiness && readiness.provider.kind === "chatgpt";
  const models = status?.models ?? [];
  const chosen = model || (inUse && chatModel ? chatModel.modelId : "") || (models[0]?.id ?? "");
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
      setTest(await core.testChatConnection({ kind: "chatgpt", modelId: chosen }));
    });

  const use = () =>
    run("saving", async () => {
      await core.saveChatProvider({ kind: "chatgpt", modelId: chosen });
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
          <ChatGptStorageNotice />
          <ContinueWithChatGptButton onClick={signIn} disabled={status === null} />
          <ChatGptSignInState sign={sign} />
        </div>
      ) : (
        <>
          <label className={fieldLabelClass}>
            {t("chatgpt.model")}
            <select
              data-testid="chatgpt-model"
              value={chosen}
              onChange={(event) => {
                setModel(event.target.value);
                setTest(null);
              }}
              className={inputClass}
            >
              {models.map((each) => (
                <option key={each.id} value={each.id}>
                  {each.name}
                </option>
              ))}
            </select>
          </label>
          {inUse && (
            <p data-testid="chatgpt-in-use" className="text-[13px] leading-5 text-ink-secondary">
              {t("chatgpt.inUse")}
              <span aria-hidden="true"> · </span>
              <ManageUsageLink />
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              data-testid="chatgpt-use"
              disabled={busy !== null || chosen === "" || (inUse && chosen === chatModel?.modelId)}
              onClick={() => void use()}
              className={primaryButtonClass}
            >
              {t("chatgpt.use")}
            </button>
            <button
              type="button"
              disabled={busy !== null || chosen === ""}
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
