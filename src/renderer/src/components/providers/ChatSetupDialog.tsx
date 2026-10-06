import { useEffect, useState } from "react";
import type { ChatProvider } from "../../../../core/api";
import { core } from "../../core";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { useModal } from "../useModal";
import { OllamaCard } from "./OllamaCard";
import { ProviderForm } from "./ProviderForm";
import { buttonClass } from "./shared";

/**
 * First-run setup: shown while no chat provider is configured, until the User
 * sets one up or chooses "set up later". Notes and Documents work either way.
 */
export function ChatSetupDialog() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const dismissed = useAppStore((state) => state.settings?.device.chatSetupDismissed);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const open =
    readiness?.ready === false && readiness.reason === "no-provider" && dismissed === false;
  const dialog = useModal(open);
  const [providers, setProviders] = useState<ChatProvider[]>([]);

  useEffect(() => {
    if (open) core.listChatProviders().then(setProviders, () => undefined);
  }, [open]);

  const later = () => void updateSettings({ device: { chatSetupDismissed: true } });

  return (
    <dialog
      ref={dialog}
      data-testid="chat-setup"
      aria-labelledby="chat-setup-title"
      // Esc means "set up later".
      onCancel={(event) => {
        event.preventDefault();
        later();
      }}
      className="m-auto max-h-[90vh] w-[34rem] max-w-[calc(100vw-2rem)] overflow-y-auto rounded-[9px] bg-white p-5 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      {open && (
        <>
          <h2 id="chat-setup-title" className="text-lg font-semibold">
            {t("providers.setup.title")}
          </h2>
          <p className="mt-1 text-sm text-gray-600">{t("providers.setup.body")}</p>

          <div className="mt-4">
            {/* Readiness turns ready once saved, which closes this dialog. */}
            <OllamaCard onSaved={() => undefined} />
          </div>

          <h3 className="mt-4 mb-2 text-sm font-medium">{t("providers.setup.orApiKey")}</h3>
          <ProviderForm providers={providers} onSaved={() => undefined} />

          <div className="mt-5 flex justify-end border-t border-gray-100 pt-3">
            <button
              type="button"
              data-testid="chat-setup-later"
              onClick={later}
              className={buttonClass}
            >
              {t("providers.setup.later")}
            </button>
          </div>
        </>
      )}
    </dialog>
  );
}
