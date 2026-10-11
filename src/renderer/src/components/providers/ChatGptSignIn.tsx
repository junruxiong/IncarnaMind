import { useEffect, useState } from "react";
import type { ChatGptPlanStatus, SecretStorageStatus } from "../../../../core/api";
import { continueWithChatGpt, type SignedIn, useSignIn, useWelcome } from "../../chatGptSignIn";
import { CHATGPT_USAGE_URL } from "../../chatGptUsage";
import { core } from "../../core";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonStyle,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  primaryButtonClass,
} from "../ui";
import { useModal } from "../useModal";
import { ChatGptMark } from "./ChatGptMark";
import { SecretStorageNotice } from "./ProviderForm";

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
 * "Continue with ChatGPT", in the black button with the ChatGPT logo that
 * OpenAI's guidelines ask for (https://developers.openai.com/siwc/ui-ux-guidelines).
 */
export function ContinueWithChatGptButton({
  onClick,
  disabled = false,
  size = "md",
  testId = "chatgpt-continue",
}: {
  onClick(): void;
  disabled?: boolean;
  size?: "md" | "lg";
  testId?: string;
}) {
  const t = useT();
  return (
    <button
      type="button"
      data-testid={testId}
      disabled={disabled}
      onClick={onClick}
      className={buttonStyle("primary", size)}
    >
      <ChatGptMark className="size-4" />
      {t("chatgpt.continue")}
    </button>
  );
}

/**
 * Where no keyring keeps secrets safe (Linux without one), the sign-in can't
 * be kept until the User accepts storing it in plain text, as for API keys.
 */
export function ChatGptStorageNotice() {
  const [status, setStatus] = useState<SecretStorageStatus | null>(null);
  useEffect(() => {
    core.getSecretStorage().then(setStatus, () => undefined);
  }, []);
  if (!status || status.canSave) return null;
  const accept = () =>
    core
      .acceptPlainTextSecretStorage()
      .then(setStatus, (error: unknown) => useAppStore.getState().reportError(error));
  return <SecretStorageNotice status={status} onAccept={() => void accept()} />;
}

/** "Manage usage ↗": ChatGPT's own page, where the User sees and limits what apps use of the plan. */
export function ManageUsageLink({ className = "" }: { className?: string }) {
  const t = useT();
  return (
    <a
      href={CHATGPT_USAGE_URL}
      target="_blank"
      rel="noreferrer"
      data-testid="chatgpt-manage-usage"
      className={`font-semibold text-accent underline-offset-2 hover:underline ${className}`}
    >
      {t("chatgpt.manageUsage")} ↗
    </a>
  );
}

/**
 * A "Continue with ChatGPT" action's state: whether it's waiting for the
 * browser, and what went wrong, in words. `onSignedIn` gets the provider once
 * the User is signed in, whichever way they got there (Use another account too).
 */
export function useContinueWithChatGpt(
  options: { keepDefault?: boolean; onSignedIn?(signedIn: SignedIn): void } = {},
) {
  const t = useT();
  const { waiting, failure } = useSignIn();
  const run = async (input: { anotherAccount?: boolean } = {}): Promise<void> => {
    const result = await continueWithChatGpt({
      ...input,
      ...(options.keepDefault && { keepDefault: true }),
    });
    if (result && !("kind" in result)) options.onSignedIn?.(result);
  };
  return {
    run,
    waiting,
    failure,
    /** The failure as one line, in the User's language. */
    failureText: failure ? t(`chatgpt.signIn.error.${failure.kind}`) : null,
    cancel: () => void core.cancelChatGptSignIn(),
  };
}

/**
 * What a "Continue with ChatGPT" shows while it waits or after it failed: the
 * browser hint with Cancel, or the failure in plain words with its one action
 * (`retry`: "Sign in again", where no Continue with ChatGPT button is beside
 * it), and OpenAI's own words behind "Details".
 */
export function ChatGptSignInState({
  sign,
  retry = false,
}: {
  sign: ReturnType<typeof useContinueWithChatGpt>;
  retry?: boolean;
}) {
  const t = useT();
  if (sign.waiting) {
    return (
      <p className="flex flex-wrap items-center gap-x-2 text-[13px] leading-5 text-ink-secondary">
        <span>{t("chatgpt.signingIn")}</span>
        <button
          type="button"
          data-testid="chatgpt-cancel"
          onClick={sign.cancel}
          className="font-semibold text-accent underline-offset-2 hover:underline"
        >
          {t("chatgpt.cancel")}
        </button>
      </p>
    );
  }
  const { failure, failureText } = sign;
  if (!failure || !failureText) return null;
  const action =
    failure.kind === "other-account"
      ? { testId: "chatgpt-another-account", label: t("chatgpt.anotherAccount"), another: true }
      : retry && failure.kind !== "secret-storage"
        ? { testId: "chatgpt-retry", label: t("chatgpt.signInAgain"), another: false }
        : null;
  return (
    <div role="alert" data-testid="chatgpt-error" className="text-[13px] leading-5 break-words">
      <p className="text-danger">
        {failureText}
        {action && (
          <>
            <span aria-hidden="true"> · </span>
            <button
              type="button"
              data-testid={action.testId}
              onClick={() => void sign.run(action.another ? { anotherAccount: true } : {})}
              className="font-semibold text-accent underline-offset-2 hover:underline"
            >
              {action.label}
            </button>
          </>
        )}
      </p>
      <details className="text-ink-meta">
        <summary className="cursor-default">{t("chatgpt.details")}</summary>
        <p data-testid="chatgpt-error-details">{failure.message}</p>
      </details>
    </div>
  );
}

/**
 * "You're using your ChatGPT plan": shown once, after the first sign-in on
 * this device, as OpenAI's guidelines ask, with "Got it".
 */
export function ChatGptWelcomeDialog() {
  const t = useT();
  const open = useWelcome((state) => state.open);
  const dialog = useModal(open);
  const close = () => useWelcome.setState({ open: false });
  return (
    <dialog
      ref={dialog}
      data-testid="chatgpt-welcome"
      aria-labelledby="chatgpt-welcome-title"
      onCancel={close}
      className={`${dialogClass} w-[26rem]`}
    >
      {open && (
        <div className={dialogBodyClass}>
          <ChatGptMark className="size-8 text-ink" />
          <h2 id="chatgpt-welcome-title" className={dialogTitleClass}>
            {t("chatgpt.welcome.title")}
          </h2>
          <p className={dialogTextClass}>
            {t("chatgpt.welcome.body")} <ManageUsageLink />
          </p>
          <div className={dialogActionsClass}>
            <button
              type="button"
              data-testid="chatgpt-welcome-ok"
              onClick={close}
              className={primaryButtonClass}
            >
              {t("chatgpt.welcome.ok")}
            </button>
          </div>
        </div>
      )}
    </dialog>
  );
}
