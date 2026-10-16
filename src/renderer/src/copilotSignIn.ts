/**
 * Signing in with GitHub for the Copilot plan, from first run's Other ways or Settings › Models:
 * one sign-in at a time, so its state (the status, how the last one failed, whether its notice is
 * open) is one for the whole window, and both places say the same.
 */
import { useEffect } from "react";
import { create } from "zustand";
import type { CopilotSignInErrorKind, CopilotStatus } from "../../core/api";
import { copilotWay } from "../../core/providers/copilot/service";
import { core } from "./core";
import { errorMessage } from "./errors";
import { translateNow } from "./i18n";
import { useAppStore } from "./store";

/** "no-models": signed in, but the plan listed no model for this account. */
export type CopilotFailureKind = CopilotSignInErrorKind | "no-models";

export interface CopilotFailure {
  kind: CopilotFailureKind;
  message: string;
}

interface CopilotState {
  status: CopilotStatus | null;
  failure: CopilotFailure | null;
  open: boolean;
}

export const useCopilot = create<CopilotState>()(() => ({
  status: null,
  failure: null,
  open: false,
}));

let loaded = false;
core.on("copilot.changed", (status) => useCopilot.setState({ status }));

export function useCopilotStatus(): CopilotStatus | null {
  useEffect(() => {
    if (loaded) return;
    loaded = true;
    core.getCopilot().then(
      (status) => useCopilot.setState({ status }),
      (error: unknown) => {
        loaded = false;
        console.error("The Copilot plan's status couldn't be read.", error);
      },
    );
  }, []);
  return useCopilot((state) => state.status);
}

export const openCopilotNotice = (open: boolean) =>
  useCopilot.setState(open ? { open, failure: null } : { open });

/**
 * Signs in with GitHub and, where the User has no Copilot provider yet, saves one with the plan's
 * first model, which is choosing the plan for Questions (ADR-0022). `keepDefault` leaves the
 * default model for new Minds as it was. True once signed in; a cancelled sign-in says nothing.
 */
export async function signInWithCopilot(options: { keepDefault?: boolean } = {}): Promise<boolean> {
  useCopilot.setState({ failure: null });
  try {
    const status = useCopilot.getState().status ?? (await core.getCopilot());
    const way = copilotWay(status.profiles);
    const result = await core.signInToCopilot(way ? { profile: way.id } : {});
    if (!result.ok) {
      if (result.error.kind !== "sign-in-cancelled") useCopilot.setState({ failure: result.error });
      return false;
    }
    useCopilot.setState({ status: result.status, open: false });
    return await choosePlan(result.status, options.keepDefault === true);
  } catch (error) {
    useCopilot.setState({
      failure: { kind: "failed", message: errorMessage(error, translateNow) },
    });
    return false;
  }
}

async function choosePlan(status: CopilotStatus, keepDefault: boolean): Promise<boolean> {
  const providers = await core.listChatProviders();
  if (providers.some((provider) => provider.kind === "copilot")) return true;
  const modelId = status.models[0]?.id;
  if (!modelId) {
    useCopilot.setState({
      failure: { kind: "no-models", message: "GitHub Copilot listed no models for this account." },
    });
    return false;
  }
  const chatModel = useAppStore.getState().settings?.user.chatModel ?? null;
  await core.saveChatProvider({ kind: "copilot", modelId });
  // Saving makes the plan the default for new Minds; from Settings, a default the User had stays.
  if (keepDefault && chatModel)
    await useAppStore.getState().updateSettings({ user: { chatModel } });
  return true;
}

export const cancelCopilotSignIn = () =>
  core.cancelCopilotSignIn().catch((error: unknown) => useAppStore.getState().reportError(error));

/**
 * "Sign in again" from a problem line: Settings › Models opens at the Copilot plan with its
 * notice, where the code will show. True once signed in there, false if Settings closes first.
 */
export function signInToCopilotAgain(): Promise<boolean> {
  openCopilotNotice(true);
  useAppStore.getState().openSettings("models", "copilot");
  return new Promise((resolve) => {
    const settle = (signedIn: boolean) => {
      stopCopilot();
      stopSettings();
      resolve(signedIn);
    };
    const stopCopilot = useCopilot.subscribe((state) => {
      if (state.status?.account.state === "signed-in" && !state.status.signingIn) settle(true);
    });
    const stopSettings = useAppStore.subscribe((state) => {
      if (!state.settingsOpen) settle(false);
    });
  });
}
