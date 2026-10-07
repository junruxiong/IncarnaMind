import { type FormEvent, useCallback, useEffect, useId, useState } from "react";
import type { ChatProvider } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { useChatGptPlan } from "./ChatGptPlanSettings";
import { ReadinessExplanation } from "./ChatReadinessNotice";
import { OllamaCard } from "./OllamaCard";
import { ProviderForm } from "./ProviderForm";
import { buttonClass, inputClass, providerLabel } from "./shared";

/** Settings → Chat model: the provider in use, the default model, and changing either. */
export function ChatModelSettings() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const chatModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const [changing, setChanging] = useState(false);
  const [model, setModel] = useState(chatModel?.modelId ?? "");
  const [chatGptPlan] = useChatGptPlan();
  const modelFieldId = useId();

  const refresh = useCallback(() => {
    core.listChatProviders().then(setProviders, () => undefined);
  }, []);

  // Readiness changes whenever a provider is saved or removed.
  useEffect(() => {
    if (readiness) refresh();
  }, [readiness, refresh]);

  useEffect(() => setModel(chatModel?.modelId ?? ""), [chatModel?.modelId]);

  const current = chatModel
    ? providers.find((each) => each.id === chatModel.providerId)
    : undefined;

  const saveModel = (event: FormEvent) => {
    event.preventDefault();
    if (!current || model.trim() === "") return;
    void updateSettings({ user: { chatModel: { providerId: current.id, modelId: model.trim() } } });
  };

  const remove = async () => {
    if (!current) return;
    try {
      await core.deleteChatProvider(current.id);
    } catch (failure) {
      useAppStore.setState({ actionError: errorMessage(failure) });
    }
  };

  const done = () => {
    setChanging(false);
    refresh();
  };

  return (
    <section data-testid="chat-model-settings">
      <h3 className="mb-1 text-sm font-medium">{t("providers.settings.title")}</h3>
      {readiness && !readiness.ready && readiness.reason !== "no-provider" && (
        <p className="mb-2 rounded-[9px] bg-amber-50 p-2 text-sm text-amber-900">
          <ReadinessExplanation readiness={readiness} />
        </p>
      )}

      {current ? (
        <div className="flex flex-col gap-2">
          <p className="text-sm">
            <span className="text-gray-600">{t("providers.settings.provider")}</span>
            {providerLabel(current, t)}
          </p>
          <form onSubmit={saveModel} className="flex items-end gap-2">
            <div className="flex-1 text-sm text-gray-600">
              <label htmlFor={modelFieldId}>{t("providers.settings.defaultModel")}</label>
              {current.kind === "chatgpt" ? (
                // The ChatGPT plan takes only the models its endpoint accepts.
                <select
                  id={modelFieldId}
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  className={inputClass}
                >
                  {chatGptPlan?.models.map((each) => (
                    <option key={each.id} value={each.id}>
                      {each.name}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  id={modelFieldId}
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  spellCheck={false}
                  className={inputClass}
                />
              )}
            </div>
            <button
              type="submit"
              disabled={model.trim() === "" || model.trim() === chatModel?.modelId}
              className={buttonClass}
            >
              {t("providers.settings.saveModel")}
            </button>
          </form>
          {!changing && (
            <div className="flex gap-2">
              <button type="button" onClick={() => setChanging(true)} className={buttonClass}>
                {t("providers.settings.change")}
              </button>
              <button type="button" onClick={() => void remove()} className={buttonClass}>
                {t("providers.settings.remove")}
              </button>
            </div>
          )}
        </div>
      ) : (
        <p className="text-sm text-gray-600">{t("providers.settings.none")}</p>
      )}

      {(changing || !current) && (
        <div className="mt-3 flex flex-col gap-3">
          <OllamaCard onSaved={done} />
          <ProviderForm providers={providers} onSaved={done} />
          {current && (
            <div>
              <button type="button" onClick={() => setChanging(false)} className={buttonClass}>
                {t("providers.settings.cancel")}
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
