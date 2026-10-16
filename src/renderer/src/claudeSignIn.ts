/**
 * Signing in with a Claude account from anywhere in the window: the first-run row, Settings ›
 * Models, a problem line. The core runs one sign-in at a time, so its state (the account, waiting,
 * or how it failed) is one for the whole window, and every place that offers it says the same.
 */
import { useEffect } from "react";
import { create } from "zustand";
import type { ClaudeAccountStatus, ClaudeSignInErrorKind } from "../../core/api";
import { catalogProvider } from "../../core/providers/catalog/providers";
import { core } from "./core";
import { errorMessage } from "./errors";
import { translateNow } from "./i18n";
import { useAppStore } from "./store";

export interface ClaudeSignInFailure {
  kind: ClaudeSignInErrorKind;
  message: string;
}

interface ClaudeSignInState {
  status: ClaudeAccountStatus | null;
  waiting: boolean;
  failure: ClaudeSignInFailure | null;
}

export const useClaudeSignIn = create<ClaudeSignInState>(() => ({
  status: null,
  waiting: false,
  failure: null,
}));

core.on("claudeAccount.changed", (status) => useClaudeSignIn.setState({ status }));

let asked = false;

/** The account's status, read once something shows it. */
export function useClaudeAccount(): ClaudeAccountStatus | null {
  useEffect(() => {
    if (asked) return;
    asked = true;
    core.getClaudeAccount().then(
      (status) => useClaudeSignIn.setState({ status }),
      (error: unknown) => {
        asked = false;
        console.error("The Claude account's status couldn't be read.", error);
      },
    );
  }, []);
  return useClaudeSignIn((state) => state.status);
}

// A newer sign-in replaces one still waiting, which then ends as cancelled and must say nothing.
let latest = 0;
let cancelled = false;

async function chooseClaudePlan(keepDefault: boolean): Promise<void> {
  const providers = await core.listChatProviders();
  const existing = providers.find((provider) => provider.kind === "claude-account");
  const chatModel = useAppStore.getState().settings?.user.chatModel ?? null;
  if (existing && (keepDefault || chatModel?.providerId === existing.id)) return;
  const modelId = catalogProvider("anthropic")?.roles.answers;
  if (!modelId) throw new Error("The catalog names no Answers model for Anthropic.");
  await core.saveChatProvider({ kind: "claude-account", modelId });
  if (keepDefault && chatModel) {
    await useAppStore.getState().updateSettings({ user: { chatModel } });
  }
}

/**
 * Signs in through Anthropic's page, then saves the route as a provider, which allows its flows
 * (ADR-0022); `keepDefault` leaves the default model as it was. True once signed in. A failure
 * is kept for the row to say; the User's own Cancel says nothing.
 */
export async function signInWithClaude(options: { keepDefault?: boolean } = {}): Promise<boolean> {
  latest += 1;
  const run = latest;
  cancelled = false;
  useClaudeSignIn.setState({ waiting: true, failure: null });
  let failure: ClaudeSignInFailure | null = null;
  try {
    const current = await core.getClaudeAccount();
    // Signed in already (e.g. its provider was removed since): no need for the browser again.
    const result =
      current.account.state === "signed-in"
        ? { ok: true as const, status: current }
        : await core.signInToClaude();
    if (!result.ok) {
      failure = result.error;
      return false;
    }
    useClaudeSignIn.setState({ status: result.status });
    await chooseClaudePlan(options.keepDefault === true);
    return true;
  } catch (error) {
    failure = { kind: "failed", message: errorMessage(error, translateNow) };
    return false;
  } finally {
    if (run === latest) {
      const quiet = failure?.kind === "sign-in-cancelled" && cancelled;
      useClaudeSignIn.setState({ waiting: false, failure: quiet ? null : failure });
    }
  }
}

export function cancelSignInWithClaude(): void {
  cancelled = true;
  core.cancelClaudeSignIn().catch((error: unknown) => useAppStore.getState().reportError(error));
}

export const forgetClaudeSignInFailure = (): void => useClaudeSignIn.setState({ failure: null });

/**
 * "Sign in again" from a problem line: Settings › Models opens at the Claude row, which waits
 * there for the code Anthropic's page shows. True once signed in, so the Answer can be asked again.
 */
export function signInToClaudeAgain(): Promise<boolean> {
  const signingIn = signInWithClaude({ keepDefault: true });
  useAppStore.getState().openSettings("models", "claude-plan");
  return signingIn;
}

export async function signOutOfClaude(): Promise<void> {
  try {
    useClaudeSignIn.setState({ status: await core.signOutOfClaude(), failure: null });
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
}
