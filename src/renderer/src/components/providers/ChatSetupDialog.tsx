import { useEffect, useState } from "react";
import type { ChatProvider } from "../../../../core/api";
import { core } from "../../core";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { buttonStyle, dialogClass } from "../ui";
import { useModal } from "../useModal";
import { type ChatProviderChoice, ChatProviderChoices, OllamaAction } from "./ChatProviderChoices";
import { useOllama } from "./OllamaCard";

/**
 * First-run setup: shown while no chat provider is configured, until the User
 * sets one up or chooses "set up later". Notes and Documents work either way.
 * Where Answers come from is one choice among rows (none preselected): local
 * models with Ollama, an API key, or the experimental ChatGPT plan.
 */
export function ChatSetupDialog() {
  const readiness = useAppStore((state) => state.chatReadiness);
  const dismissed = useAppStore((state) => state.settings?.device.chatSetupDismissed);
  const open =
    readiness?.ready === false && readiness.reason === "no-provider" && dismissed === false;
  // The dialog itself takes focus, so no choice looks picked: none is preselected.
  const dialog = useModal(open, { focusDialog: true });

  return (
    <dialog
      ref={dialog}
      tabIndex={-1}
      data-testid="chat-setup"
      aria-labelledby="chat-setup-title"
      // Esc means "set up later".
      onCancel={(event) => {
        event.preventDefault();
        void useAppStore.getState().updateSettings({ device: { chatSetupDismissed: true } });
      }}
      className={`${dialogClass} w-[560px] outline-none`}
    >
      {/* Readiness turns ready once a provider is saved, which closes this dialog. */}
      {open && <ChatSetup />}
    </dialog>
  );
}

function ChatSetup() {
  const t = useT();
  const updateSettings = useAppStore((state) => state.updateSettings);
  const openSettings = useAppStore((state) => state.openSettings);
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const [choice, setChoice] = useState<ChatProviderChoice | null>(null);
  const ollama = useOllama(() => undefined);

  useEffect(() => {
    core.listChatProviders().then(setProviders, () => undefined);
  }, []);

  const later = () => void updateSettings({ device: { chatSetupDismissed: true } });

  return (
    <div className="flex flex-col gap-6 px-8 pt-8 pb-6">
      <div className="flex flex-col gap-2">
        <h2
          id="chat-setup-title"
          className="font-serif text-[26px] leading-[34px] font-semibold tracking-[-0.005em] text-ink [font-variation-settings:'opsz'_36]"
        >
          {t("providers.setup.title")}
        </h2>
        <p className="text-[15px] leading-[22px] text-ink-secondary">{t("providers.setup.body")}</p>
      </div>

      <ChatProviderChoices
        choice={choice}
        onChoose={setChoice}
        ollama={ollama}
        providers={providers}
        onSaved={() => undefined}
        includeChatGpt
      />

      {/* Kept in view while the dialog scrolls, e.g. with the API key form open. */}
      <div className="sticky bottom-0 -mx-8 -mb-6 flex flex-wrap items-center gap-2 bg-sheet px-8 pt-3 pb-6">
        <button
          type="button"
          data-testid="chat-setup-later"
          onClick={later}
          className={`-ml-4 ${buttonStyle("ghost", "lg")}`}
        >
          {t("providers.setup.later")}
        </button>
        <div className="ml-auto flex items-center gap-2">
          {/* With Ollama running, local models are one click away, though nothing is chosen. */}
          {(choice === "ollama" || (choice === null && ollama.status?.running)) && (
            <OllamaAction ollama={ollama} large />
          )}
          {choice === "chatgpt" && (
            <button
              type="button"
              data-testid="chat-setup-chatgpt"
              onClick={() => {
                later();
                openSettings("chat-model");
              }}
              className={buttonStyle("primary", "lg")}
            >
              {t("providers.choice.continue")}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
