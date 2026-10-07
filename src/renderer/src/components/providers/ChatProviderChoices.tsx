import { type ReactNode, useId } from "react";
import type { ChatProvider } from "../../../../core/api";
import { useT } from "../../i18n";
import {
  buttonStyle,
  choiceListClass,
  choiceRadioClass,
  choiceRowClass,
  choiceTextClass,
  choiceTitleClass,
} from "../ui";
import { type Ollama, OllamaProgress, ollamaDescription } from "./OllamaCard";
import { ProviderForm } from "./ProviderForm";

/** Where Answers come from: local models, an API key, or (experimental) a ChatGPT plan. */
export type ChatProviderChoice = "ollama" | "api-key" | "chatgpt";

interface ChatProviderChoicesProps {
  /** Nothing is chosen at first: no provider is preselected. */
  choice: ChatProviderChoice | null;
  onChoose(choice: ChatProviderChoice): void;
  ollama: Ollama;
  /** Saved providers, to tell the User when a key is already stored. */
  providers: readonly ChatProvider[];
  onSaved(provider: ChatProvider): void;
  /** First run offers the ChatGPT plan, which is set up in Settings. */
  includeChatGpt?: boolean;
  /** Settings puts Ollama's button in its row; first run puts it at the dialog's foot. */
  ollamaActionInRow?: boolean;
}

/**
 * The ways Answers can come from, as rows split by rules in one box (not
 * cards). The chosen row is washed and opens below its words: the API key's
 * form, or Ollama's download progress.
 */
export function ChatProviderChoices(props: ChatProviderChoicesProps) {
  const { choice, onChoose, ollama, providers, onSaved } = props;
  const { includeChatGpt = false, ollamaActionInRow = false } = props;
  const t = useT();
  const name = useId();

  const radio = (value: ChatProviderChoice) => (
    <input
      id={`${name}-${value}`}
      type="radio"
      name={name}
      value={value}
      data-testid={`chat-choice-${value}`}
      checked={choice === value}
      onChange={() => onChoose(value)}
      className={choiceRadioClass}
    />
  );

  return (
    <div
      role="radiogroup"
      aria-label={t("providers.choice.label")}
      data-testid="chat-provider-choices"
      className={choiceListClass}
    >
      <Choice
        inputId={`${name}-ollama`}
        chosen={choice === "ollama"}
        details={
          (ollamaActionInRow || ollama.pulling || ollama.error) && (
            <div className="flex flex-col items-start gap-3">
              <OllamaProgress ollama={ollama} />
              {ollamaActionInRow && <OllamaAction ollama={ollama} />}
            </div>
          )
        }
      >
        {radio("ollama")}
        <span className="flex min-w-0 flex-col gap-1">
          <span className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
            <span className={choiceTitleClass}>{t("providers.ollama.title")}</span>
            <span className="text-label font-semibold text-success">
              {t("providers.choice.local")}
            </span>
          </span>
          <span data-testid="ollama-status" className={choiceTextClass}>
            {ollamaDescription(ollama, t)}
          </span>
        </span>
      </Choice>

      <Choice
        inputId={`${name}-api-key`}
        chosen={choice === "api-key"}
        details={<ProviderForm providers={providers} onSaved={onSaved} />}
      >
        {radio("api-key")}
        <span className="flex min-w-0 flex-col gap-1">
          <span className={choiceTitleClass}>{t("providers.choice.apiKey")}</span>
          <span className={choiceTextClass}>{t("providers.choice.apiKey.body")}</span>
        </span>
      </Choice>

      {includeChatGpt && (
        <Choice inputId={`${name}-chatgpt`} chosen={choice === "chatgpt"}>
          {radio("chatgpt")}
          <span className="flex min-w-0 flex-col gap-1">
            <span className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <span className={choiceTitleClass}>{t("providers.kind.chatgpt")}</span>
              <span className="text-label font-semibold text-ink-meta">
                {t("providers.choice.experimental")}
              </span>
            </span>
            <span className={choiceTextClass}>{t("providers.choice.chatgpt.body")}</span>
          </span>
        </Choice>
      )}
    </div>
  );
}

/** One choice: its row, and, while chosen, what it opens, lined up with its words. */
function Choice({
  inputId,
  chosen,
  details,
  children,
}: {
  /** The choice's radio, which `children` holds. */
  inputId: string;
  chosen: boolean;
  details?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={chosen ? "bg-wash" : ""}>
      <label htmlFor={inputId} className={choiceRowClass}>
        {children}
      </label>
      {chosen && details && <div className="pr-4 pb-4 pl-11">{details}</div>}
    </div>
  );
}

/** Ollama's button: use it (pulling the recommended model), or look for it again. */
export function OllamaAction({ ollama, large = false }: { ollama: Ollama; large?: boolean }) {
  const t = useT();
  const { status, pulling } = ollama;
  const size = large ? "lg" : "md";
  if (status?.running) {
    return (
      <button
        type="button"
        data-testid="ollama-use"
        disabled={pulling}
        onClick={() => void ollama.use()}
        className={buttonStyle("primary", size)}
      >
        {t("providers.ollama.use", { model: status.recommendedModel })}
      </button>
    );
  }
  return (
    <button
      type="button"
      disabled={status === null}
      onClick={() => void ollama.detect()}
      className={buttonStyle("secondary", size)}
    >
      {t("providers.ollama.checkAgain")}
    </button>
  );
}
