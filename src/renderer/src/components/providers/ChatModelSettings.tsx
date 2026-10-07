import { type FormEvent, useCallback, useEffect, useId, useState } from "react";
import type { ChatProvider } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonClass,
  fieldLabelClass,
  inputClass,
  noticeClass,
  pageIntroClass,
  rowButtonsClass,
  rowTitleClass,
  ruledListClass,
  ruledRow,
  ruledRowClass,
} from "../ui";
import { useChatGptPlan } from "./ChatGptPlanSettings";
import { type ChatProviderChoice, ChatProviderChoices } from "./ChatProviderChoices";
import { ReadinessExplanation } from "./ChatReadinessNotice";
import { useOllama } from "./OllamaCard";
import { providerLabel } from "./shared";

/**
 * Settings → Chat model: the provider in use and the default model, as rows
 * split by rules, and changing the provider with the same choices as first
 * run (local models with Ollama, or an API key).
 */
export function ChatModelSettings() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const chatModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const [changing, setChanging] = useState(false);
  const [choice, setChoice] = useState<ChatProviderChoice | null>(null);
  const [model, setModel] = useState(chatModel?.modelId ?? "");
  const [chatGptPlan] = useChatGptPlan();
  const modelFieldId = useId();

  const refresh = useCallback(() => {
    core.listChatProviders().then(setProviders, () => undefined);
  }, []);

  const done = () => {
    setChanging(false);
    setChoice(null);
    refresh();
  };
  const ollama = useOllama(done);

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

  return (
    <section data-testid="chat-model-settings" className="flex flex-col gap-3">
      {readiness && !readiness.ready && readiness.reason !== "no-provider" && (
        <p role="note" className={noticeClass}>
          <ReadinessExplanation readiness={readiness} />
        </p>
      )}

      {current ? (
        <div className={ruledListClass}>
          <div className={ruledRowClass}>
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className={rowTitleClass}>{t("providers.settings.provider")}</span>
              <span data-testid="chat-provider-current" className="text-ui text-ink-secondary">
                {providerLabel(current, t)}
              </span>
            </div>
            {!changing && (
              <div className={rowButtonsClass}>
                <button type="button" onClick={() => setChanging(true)} className={buttonClass}>
                  {t("providers.settings.change")}
                </button>
                <button type="button" onClick={() => void remove()} className={buttonClass}>
                  {t("providers.settings.remove")}
                </button>
              </div>
            )}
          </div>
          <form onSubmit={saveModel} className={ruledRow("end")}>
            <div className="min-w-0">
              <label htmlFor={modelFieldId} className={fieldLabelClass}>
                {t("providers.settings.defaultModel")}
              </label>
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
        </div>
      ) : (
        <p className={pageIntroClass}>{t("providers.settings.none")}</p>
      )}

      {(changing || !current) && (
        <div className="flex flex-col items-start gap-3">
          <div className="w-full">
            <ChatProviderChoices
              choice={choice}
              onChoose={setChoice}
              ollama={ollama}
              providers={providers}
              onSaved={done}
              ollamaActionInRow
            />
          </div>
          {current && (
            <button
              type="button"
              onClick={() => {
                setChanging(false);
                setChoice(null);
              }}
              className={buttonClass}
            >
              {t("providers.settings.cancel")}
            </button>
          )}
        </div>
      )}
    </section>
  );
}
