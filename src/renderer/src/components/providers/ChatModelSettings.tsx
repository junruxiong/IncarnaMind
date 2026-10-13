import { type FormEvent, useCallback, useEffect, useId, useState } from "react";
import type { ChatGptPlanModel, ChatProvider } from "../../../../core/api";
import { core } from "../../core";
import { formatSpent } from "../../cost";
import { useLanguage, useT } from "../../i18n";
import { useAppStore } from "../../store";
import { useCostSummary } from "../SpendingCapSettings";
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
import { useChatGptPlan } from "./ChatGptSignIn";
import { type ChatProviderChoice, ChatProviderChoices } from "./ChatProviderChoices";
import { ReadinessExplanation } from "./ChatReadinessNotice";
import { useOllama } from "./OllamaCard";
import { providerLabel } from "./shared";

/**
 * Settings → Models → Chat model: the provider in use and the default model, as rows
 * split by rules, and changing the provider with the same choices as first
 * run (local models with Ollama, or an API key). With the ChatGPT plan, the
 * default model is a choice of the plan's models.
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
      useAppStore.getState().reportError(failure);
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
              <ProviderCostLine providerId={current.id} />
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
                // The ChatGPT plan takes only the models its endpoint accepts. This is the one place
                // its model is chosen: the plan's own section has the account, usage and sign-out.
                <select
                  id={modelFieldId}
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  className={inputClass}
                >
                  {planModelOptions(chatGptPlan?.models ?? [], chatModel?.modelId).map((each) => (
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
              withMoreProviders
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

/**
 * What a provider billed per token cost this month, in its own currency
 * ("This month: $3.20"), once it has been used. A plan's line is in its own
 * section, and a model on this computer costs nothing.
 */
function ProviderCostLine({ providerId }: { providerId: string }) {
  const t = useT();
  const language = useLanguage();
  const summary = useCostSummary("models");
  const used = summary?.providers.find(
    (each) => each.providerId === providerId && each.billing === "per-token",
  );
  if (!used) return null;
  const spent = formatSpent(used.spent, t, language);
  const tokens = new Intl.NumberFormat(language).format(used.unpricedTokens);
  return (
    <span data-testid="chat-provider-cost" className="text-[13px] leading-5 text-ink-meta">
      {spent ? t("cost.provider.thisMonth", { spent }) : t("cost.provider.unpriced", { tokens })}
    </span>
  );
}

/**
 * The ChatGPT plan's models to choose from, with the one in use kept even when
 * the account no longer lists it, so the select never shows another model than
 * the one Questions use.
 */
function planModelOptions(
  models: readonly ChatGptPlanModel[],
  inUse: string | undefined,
): readonly ChatGptPlanModel[] {
  if (!inUse || models.some((each) => each.id === inUse)) return models;
  return [{ id: inUse, name: inUse }, ...models];
}
