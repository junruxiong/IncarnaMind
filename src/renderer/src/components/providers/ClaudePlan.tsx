import { useEffect, useId, useRef, useState } from "react";
import type { ClaudeAccount, ClaudeSignInErrorKind } from "../../../../core/api";
import type { MessageKey } from "../../../../shared/i18n";
import {
  cancelSignInWithClaude,
  forgetClaudeSignInFailure,
  signInWithClaude,
  signOutOfClaude,
  useClaudeAccount,
  useClaudeSignIn,
} from "../../claudeSignIn";
import { core } from "../../core";
import { useGap, useT } from "../../i18n";
import { useAppStore } from "../../store";
import { ChevronDownLineIcon, ChevronRightLineIcon } from "../lineIcons";
import { ProblemMark } from "../ProblemLine";
import { buttonStyle, rowButtonsClass, rowTextClass, rowTitleClass, ruledRowClass } from "../ui";
import { ChatGptStorageNotice, useChatGptStorage } from "./ChatGptSignIn";
import { PasteSignInAddress, type PasteWords } from "./PasteSignInAddress";

const PASTE: PasteWords = {
  testId: "claude-paste",
  question: "claudePlan.paste.question",
  open: "claudePlan.paste.open",
  problems: {
    "not-sign-in-address": "claudePlan.paste.problem.not-sign-in-address",
    "other-sign-in": "claudePlan.paste.problem.other-sign-in",
    finished: "chatgpt.paste.problem.finished",
  },
};

const FAILURES: Record<ClaudeSignInErrorKind, MessageKey> = {
  "sign-in-cancelled": "claudeAccount.problem.signInCancelled",
  refused: "claudeAccount.problem.refused",
  "unexpected-response": "claudeAccount.problem.unexpectedResponse",
  "secret-storage": "chatgpt.signIn.error.secret-storage",
  unavailable: "claudePlan.failed",
  failed: "claudePlan.failed",
};

const linkClass = "font-semibold text-accent underline-offset-2 hover:underline";

/** Whether this build offers the Claude account route (ADR-0019): never a row where it doesn't. */
function useOffered(): boolean {
  return useClaudeAccount()?.available === true;
}

/**
 * "Use my Claude plan", the first-run row under Other ways (#872). Picking it opens it in place
 * to say, before anything is sent, what using the plan means; then it waits for the browser.
 */
export function ClaudePlanWay({ onUseKey }: { onUseKey(): void }) {
  const t = useT();
  const offered = useOffered();
  const waiting = useClaudeSignIn((state) => state.waiting);
  const [open, setOpen] = useState(false);
  const panel = useId();
  if (!offered) return null;
  const expanded = open || waiting;
  const Chevron = expanded ? ChevronDownLineIcon : ChevronRightLineIcon;
  return (
    <div data-testid="way-claude" className="flex flex-col">
      <button
        type="button"
        data-testid="way-claude-open"
        aria-expanded={expanded}
        aria-controls={panel}
        onClick={() => {
          if (!waiting) setOpen(!open);
        }}
        className="flex items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-hover"
      >
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-ui font-semibold text-ink">{t("claudePlan.way.title")}</span>
          <span className="text-[13px] leading-5 text-ink-secondary">
            {t("claudePlan.way.line")}
          </span>
        </span>
        <Chevron aria-hidden="true" className="size-3.5 shrink-0 text-ink-meta" />
      </button>
      {expanded && (
        <div id={panel} className="flex flex-col gap-2 px-2 pt-1 pb-2">
          <ClaudePlanSignIn titled onClose={() => setOpen(false)} onUseKey={onUseKey} />
        </div>
      )}
    </div>
  );
}

/**
 * Settings › Models › "Already pay for a plan?": the Claude row. Signed out, Sign in with Claude
 * opens the same words in place as first run; signed in, the account and Sign out.
 */
export function ClaudePlanRow() {
  const t = useT();
  const status = useClaudeAccount();
  const waiting = useClaudeSignIn((state) => state.waiting);
  const [open, setOpen] = useState(false);
  const row = useRef<HTMLLIElement>(null);
  const offered = status?.available === true;
  const openedAt = useAppStore((state) => state.settingsAt === "claude-plan");
  const signedIn = status?.account.state === "signed-in";

  useEffect(() => {
    if (!offered || !openedAt) return;
    useAppStore.setState({ settingsAt: null });
    if (!signedIn && !useClaudeSignIn.getState().waiting) setOpen(true);
    row.current?.scrollIntoView({ block: "center" });
  }, [offered, openedAt, signedIn]);

  if (!status || !offered) return null;
  const { account } = status;
  const steps = !signedIn && (open || waiting);
  return (
    <li ref={row} data-testid="plans-claude-account" className={ruledRowClass}>
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className={rowTitleClass}>{t("plans.claude.title")}</span>
        {!steps && <AccountLines account={account} />}
        {steps && (
          <div className="mt-2 flex flex-col gap-2">
            <ClaudePlanSignIn keepDefault onClose={() => setOpen(false)} />
          </div>
        )}
      </div>
      <div className={rowButtonsClass}>
        {signedIn ? (
          <button
            type="button"
            data-testid="claude-plan-sign-out"
            onClick={() => void signOutOfClaude()}
            className={buttonStyle("secondary", "sm")}
          >
            {t("claudePlan.signOut")}
          </button>
        ) : (
          !steps && (
            <button
              type="button"
              data-testid="plans-claude-sign-in"
              onClick={() =>
                account.state === "expired"
                  ? void signInWithClaude({ keepDefault: true })
                  : setOpen(true)
              }
              className={buttonStyle("secondary", "sm")}
            >
              {t(
                account.state === "expired"
                  ? "signInRoute.wayBack.signInAgain"
                  : "claudePlan.signIn",
              )}
            </button>
          )
        )}
      </div>
    </li>
  );
}

function AccountLines({ account }: { account: ClaudeAccount }) {
  const t = useT();
  const gap = useGap();
  if (account.state === "signed-in") {
    const name = account.email ?? account.organization ?? t("claudePlan.account.unknown");
    return (
      <>
        <span data-testid="claude-plan-account" className={rowTextClass}>
          {t("claudePlan.signedIn", { account: name })}
        </span>
        <span data-testid="claude-plan-usage" className={rowTextClass}>
          {t("claudePlan.usage")}
        </span>
      </>
    );
  }
  if (account.state === "expired") {
    return (
      <span data-testid="claude-plan-account" className={rowTextClass}>
        {t("claudeAccount.problem.signInExpired")}
      </span>
    );
  }
  const line = t("claudePlan.settings.line");
  const usage = t("claudePlan.usage");
  return (
    <span data-testid="claude-plan-account" className={rowTextClass}>
      {line}
      {gap(line, usage)}
      {usage}
    </span>
  );
}

/**
 * What using the plan means, said before anything is sent, with the usage limits first (#872);
 * then the wait for the browser, or how it failed. `onClose`: Cancel, back to the row.
 * `onUseKey`: where an API key can be pasted instead, if the place has one.
 */
function ClaudePlanSignIn({
  titled = false,
  keepDefault = false,
  onClose,
  onUseKey,
}: {
  titled?: boolean;
  keepDefault?: boolean;
  onClose(): void;
  onUseKey?(): void;
}) {
  const t = useT();
  const waiting = useClaudeSignIn((state) => state.waiting);
  const failure = useClaudeSignIn((state) => state.failure);
  const storage = useChatGptStorage();
  const cancel = useRef<HTMLButtonElement>(null);

  // Opened, the focus is on the safe choice (ADR-0019).
  useEffect(() => {
    if (!useClaudeSignIn.getState().waiting) cancel.current?.focus();
  }, []);

  if (waiting) {
    return (
      <div data-testid="claude-plan-waiting" className="flex flex-col gap-2">
        <p role="status" className="flex items-start gap-2 text-ui font-semibold text-ink">
          <span aria-hidden="true" className="mt-[7px] size-1.5 shrink-0 rounded-full bg-accent" />
          {t("claudePlan.waiting")}
        </p>
        <p data-testid="claude-plan-usage" className="text-[13px] leading-5 text-ink-secondary">
          {t("claudePlan.usage")}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            data-testid="claude-plan-reopen"
            onClick={() =>
              void core
                .reopenClaudeSignIn()
                .catch((error: unknown) => useAppStore.getState().reportError(error))
            }
            className={buttonStyle("secondary", "sm")}
          >
            {t("claudePlan.reopen")}
          </button>
          <button
            type="button"
            data-testid="claude-plan-cancel"
            onClick={() => {
              cancelSignInWithClaude();
              onClose();
            }}
            className={buttonStyle("ghost", "sm")}
          >
            {t("claudePlan.cancel")}
          </button>
        </div>
        <PasteSignInAddress
          paste={(address) => core.pasteClaudeSignInAddress(address)}
          words={PASTE}
          startOpen
        />
      </div>
    );
  }

  return (
    <div data-testid="claude-plan-notice" className="flex flex-col gap-2">
      {titled && <p className="text-ui font-semibold text-ink">{t("claudePlan.notice.title")}</p>}
      <ul className="flex list-disc flex-col gap-1 pl-4 text-[13px] leading-5 text-ink-secondary">
        <li data-testid="claude-plan-usage" className="font-semibold text-ink">
          {t("claudePlan.usage")}
        </li>
        <li>{t("claudePlan.sends")}</li>
        <li>{t("claudePlan.off")}</li>
      </ul>
      <ChatGptStorageNotice storage={storage} />
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          data-testid="claude-plan-sign-in"
          disabled={!storage.canSave}
          onClick={() =>
            void signInWithClaude({ keepDefault }).then((signedIn) => {
              if (signedIn) onClose();
            })
          }
          className={buttonStyle("primary", "sm")}
        >
          {t("claudePlan.signIn")}
        </button>
        <button
          ref={cancel}
          type="button"
          data-testid="claude-plan-cancel"
          onClick={() => {
            forgetClaudeSignInFailure();
            onClose();
          }}
          className={buttonStyle("ghost", "sm")}
        >
          {t("claudePlan.cancel")}
        </button>
      </div>
      {failure && (
        <div
          role="alert"
          data-testid="claude-plan-error"
          className="text-[13px] leading-5 break-words"
        >
          <p className="text-ink">
            <ProblemMark />
            {t(FAILURES[failure.kind])}
            {onUseKey && failure.kind !== "sign-in-cancelled" && (
              <>
                <span aria-hidden="true"> · </span>
                <button
                  type="button"
                  data-testid="claude-plan-use-key"
                  onClick={onUseKey}
                  className={linkClass}
                >
                  {t("signInRoute.wayBack.pasteApiKey")}
                </button>
              </>
            )}
          </p>
          <details className="text-ink-meta">
            <summary className="cursor-default">{t("claudePlan.details")}</summary>
            <p data-testid="claude-plan-error-details">{failure.message}</p>
          </details>
        </div>
      )}
    </div>
  );
}
