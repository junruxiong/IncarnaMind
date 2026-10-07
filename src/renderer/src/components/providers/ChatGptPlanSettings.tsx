import { useEffect, useId, useState } from "react";
import type { ChatGptAccount, ChatGptPlanStatus, ConnectionTestResult } from "../../../../core/api";
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
  rowTitleClass,
  ruledListClass,
  ruledRowClass,
  sectionTitleClass,
  successTextClass,
} from "../ui";
import { TestResult } from "./ProviderForm";

/** The ChatGPT plan provider's status, kept up to date with the core's event. */
export function useChatGptPlan(): [ChatGptPlanStatus | null, (status: ChatGptPlanStatus) => void] {
  const [status, setStatus] = useState<ChatGptPlanStatus | null>(null);
  useEffect(() => {
    core.getChatGptPlan().then(setStatus, () => undefined);
    return core.on("chatGptPlan.changed", setStatus);
  }, []);
  return [status, setStatus];
}

/**
 * Settings → Experimental: the "ChatGPT plan (via Codex sign-in)" provider.
 * Off by default. Turning it on shows what it is (not an official OpenAI
 * integration, may be blocked, counts against the plan's limits) and the
 * sign-in; turning it off signs out and removes the provider.
 */
export function ChatGptPlanSettings() {
  const t = useT();
  const id = useId();
  const [status, setStatus] = useChatGptPlan();
  const readiness = useAppStore((state) => state.chatReadiness);
  const chatModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState<"switching" | "testing" | "saving" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const inUse =
    readiness !== null && "provider" in readiness && readiness.provider.kind === "chatgpt";
  const models = status?.models ?? [];
  const chosen = model || (inUse && chatModel ? chatModel.modelId : "") || (models[0]?.id ?? "");

  const run = async (action: "switching" | "testing" | "saving", task: () => Promise<void>) => {
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

  const toggle = (enabled: boolean) =>
    run("switching", async () => {
      setTest(null);
      setStatus(await core.setChatGptPlanEnabled(enabled));
    });

  const signIn = async () => {
    setError(null);
    setTest(null);
    try {
      const result = await core.signInToChatGpt();
      if (result.ok) setStatus(result.status);
      // Cancelling is the User's own doing: nothing to explain.
      else if (result.error.kind !== "cancelled") {
        setError(t(`codex.signIn.error.${result.error.kind}`));
      }
    } catch (failure) {
      setError(errorMessage(failure));
    }
  };

  const signOut = () =>
    run("switching", async () => {
      setTest(null);
      setStatus(await core.signOutOfChatGpt());
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
    <section data-testid="experimental-settings" className="flex flex-col">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("codex.experimental.title")}</h4>
      <div className={ruledListClass}>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={id} className={rowTitleClass}>
              {t("codex.provider.name")}
            </label>
            <span id={`${id}-description`} className={rowTextClass}>
              {t("codex.switch.description")}
            </span>
          </div>
          <input
            id={id}
            type="checkbox"
            role="switch"
            aria-checked={status?.enabled ?? false}
            data-testid="codex-switch"
            className="switch"
            checked={status?.enabled ?? false}
            disabled={status === null || busy === "switching"}
            aria-describedby={`${id}-description`}
            onChange={(event) => void toggle(event.target.checked)}
          />
        </div>
      </div>

      {status?.enabled && (
        <div className="mt-3 flex flex-col gap-3">
          <div role="note" data-testid="codex-warning" className={noticeClass}>
            <p className="font-semibold text-ink">{t("codex.warning.title")}</p>
            <p className="mt-1">{t("codex.warning.body")}</p>
          </div>

          <p data-testid="codex-account" className="text-ui text-ink-secondary">
            <AccountLine account={status.account} />
          </p>

          {status.signingIn ? (
            <div className="flex items-center gap-2 text-ui text-ink-secondary">
              <span className="flex-1">{t("codex.signingIn")}</span>
              <button
                type="button"
                data-testid="codex-cancel"
                onClick={() => void core.cancelChatGptSignIn()}
                className={buttonClass}
              >
                {t("codex.cancel")}
              </button>
            </div>
          ) : status.account.state !== "signed-in" ? (
            <div>
              <button
                type="button"
                data-testid="codex-sign-in"
                onClick={() => void signIn()}
                className={primaryButtonClass}
              >
                {status.account.state === "expired" ? t("codex.signInAgain") : t("codex.signIn")}
              </button>
            </div>
          ) : (
            <>
              <label className={fieldLabelClass}>
                {t("codex.model")}
                <select
                  data-testid="codex-model"
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
              {inUse && <p className={successTextClass}>{t("codex.inUse")}</p>}
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  data-testid="codex-use"
                  disabled={
                    busy !== null || chosen === "" || (inUse && chosen === chatModel?.modelId)
                  }
                  onClick={() => void use()}
                  className={primaryButtonClass}
                >
                  {t("codex.use")}
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
                  data-testid="codex-sign-out"
                  disabled={busy !== null}
                  onClick={() => void signOut()}
                  className={ghostButtonClass}
                >
                  {t("codex.signOut")}
                </button>
              </div>
            </>
          )}

          {test && <TestResult result={test} />}
          {error && (
            <p role="alert" data-testid="codex-error" className={errorTextClass}>
              {error}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function AccountLine({ account }: { account: ChatGptAccount }) {
  const t = useT();
  if (account.state === "signed-out") return <>{t("codex.account.signedOut")}</>;
  if (account.state === "expired") return <>{t("codex.account.expired")}</>;
  const plan = account.plan ? account.plan.charAt(0).toUpperCase() + account.plan.slice(1) : null;
  return (
    <>
      {t("codex.account.signedIn", { account: account.email ?? t("codex.account.unknown") })}
      {plan && <span className="ml-2 text-ink-meta">{t("codex.account.plan", { plan })}</span>}
    </>
  );
}
