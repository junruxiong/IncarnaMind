import { type ReactNode, useId } from "react";
import type { ChatProvider, ChatProviderKind } from "../../../../core/api";
import type { Language } from "../../../../core/language";
import { keyProviderGroups, kindOfCatalogId } from "../../../../core/providers/catalog/providers";
import { CHATGPT_SERVICE } from "../../../../core/providers/chatgpt/service";
import { useLanguage, useT } from "../../i18n";
import type { KeyRefusal } from "../../keyRefusals";
import { OnThisComputer } from "../OnThisComputer";
import {
  buttonStyle,
  choiceListClass,
  choiceRadioClass,
  choiceRowClass,
  choiceTextClass,
  choiceTitleClass,
} from "../ui";
import {
  ChatGptSignInState,
  ChatGptStorageNotice,
  ContinueWithChatGptButton,
  useChatGptStorage,
  useContinueWithChatGpt,
} from "./ChatGptSignIn";
import { KeepLocalChoice } from "./KeepLocalChoice";
import { LocalServersAlso, localChoice, useLocalServers } from "./LocalServerChoices";
import { type Ollama, OllamaProgress, ollamaDescription } from "./OllamaCard";
import { ProviderForm, ProviderSendsLine } from "./ProviderForm";
import { keyProviderName } from "./shared";

const SHORT_NAMES: Partial<
  Record<ChatProviderKind, "providers.kind.short.qwen" | "providers.kind.short.deepseek">
> = {
  qwen: "providers.kind.short.qwen",
  deepseek: "providers.kind.short.deepseek",
};

/** Where Answers come from: local models, an API key, or the User's ChatGPT plan. */
export type ChatProviderChoice = "ollama" | "api-key" | "chatgpt" | ReturnType<typeof localChoice>;

interface ChatProviderChoicesProps {
  /** Nothing is chosen at first: no provider is preselected. */
  choice: ChatProviderChoice | null;
  onChoose(choice: ChatProviderChoice): void;
  ollama: Ollama;
  /** Saved providers, to tell the User when a key is already stored. */
  providers: readonly ChatProvider[];
  /** A provider was saved: by its key, or by signing in with ChatGPT. */
  onSaved(provider: ChatProvider): void;
  /** Settings puts Ollama's button in its row; first run puts it at the dialog's foot. */
  ollamaActionInRow?: boolean;
  /** First run offers the ChatGPT plan here; Settings has a section of its own for it. */
  includeChatGpt?: boolean;
  /** Settings also lists the best-effort providers, under "More providers"; first run lists only first-class ones. */
  withMoreProviders?: boolean;
  /** A key refused by its provider, whose field the API key's form opens on (#765). */
  refused?: KeyRefusal | null;
}

/**
 * The ways Answers can come from, as rows split by rules in one box (not
 * cards). The chosen row is washed and opens below its words: the API key's
 * form, or Ollama's download progress.
 */
export function ChatProviderChoices(props: ChatProviderChoicesProps) {
  const { choice, onChoose, ollama, providers, onSaved } = props;
  const { ollamaActionInRow = false, includeChatGpt = false, withMoreProviders = false } = props;
  const { refused = null } = props;
  const t = useT();
  const language = useLanguage();
  const name = useId();
  const local = useLocalServers(withMoreProviders);

  const apiKeyBody = withMoreProviders
    ? t("providers.choice.apiKey.body")
    : t("providers.choice.apiKey.firstRun.body", {
        providers: providerList(keyProviderGroups().main, t, language),
      });

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
          (ollamaActionInRow || ollama.pulling || ollama.error || ollama.status?.running) && (
            <div className="flex flex-col items-start gap-3">
              <OllamaProgress ollama={ollama} />
              {ollamaActionInRow && <OllamaAction ollama={ollama} />}
              {ollama.status?.running && (
                <KeepLocalChoice keepLocal={ollama.keepLocal} disabled={ollama.pulling} />
              )}
            </div>
          )
        }
      >
        {radio("ollama")}
        <span className="flex min-w-0 flex-col gap-1">
          <span className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
            <span className={choiceTitleClass}>{t("providers.ollama.title")}</span>
            <OnThisComputer className="text-label font-semibold">
              {t("providers.choice.local")}
            </OnThisComputer>
          </span>
          <span data-testid="ollama-status" className={choiceTextClass}>
            {ollamaDescription(ollama, t)}
          </span>
          <LocalServersAlso
            closed={local.closed}
            onOpen={(catalogId) => {
              local.open(catalogId);
              onChoose(localChoice(catalogId));
            }}
          />
        </span>
      </Choice>

      {local.shown.map((catalogId) => (
        <Choice
          key={catalogId}
          inputId={`${name}-${localChoice(catalogId)}`}
          chosen={choice === localChoice(catalogId)}
          details={
            <ProviderForm
              only={catalogId}
              providers={providers}
              onSaved={onSaved}
              withMoreProviders
            />
          }
        >
          {radio(localChoice(catalogId))}
          <span className="flex min-w-0 flex-col gap-1">
            <span className={choiceTitleClass}>{keyProviderName(catalogId, t, language)}</span>
            <span data-testid={`${localChoice(catalogId)}-status`} className={choiceTextClass}>
              {local.found[catalogId]?.running
                ? t("providers.local.rowRunning")
                : t("providers.local.rowNotRunning")}
            </span>
          </span>
        </Choice>
      ))}

      <Choice
        inputId={`${name}-api-key`}
        chosen={choice === "api-key"}
        details={
          <ProviderForm
            providers={providers}
            onSaved={onSaved}
            withMoreProviders={withMoreProviders}
            refused={refused}
          />
        }
      >
        {radio("api-key")}
        <span className="flex min-w-0 flex-col gap-1">
          <span className={choiceTitleClass}>{t("providers.choice.apiKey")}</span>
          <span className={choiceTextClass}>{apiKeyBody}</span>
        </span>
      </Choice>

      {includeChatGpt && (
        <Choice
          inputId={`${name}-chatgpt`}
          chosen={choice === "chatgpt"}
          details={<ChatGptChoice onSaved={onSaved} />}
        >
          {radio("chatgpt")}
          <span className="flex min-w-0 flex-col gap-1">
            <span className={choiceTitleClass}>{t("chatgpt.title")}</span>
            <span className={choiceTextClass}>{t("providers.choice.chatgpt.body")}</span>
          </span>
        </Choice>
      )}
    </div>
  );
}

/** The ChatGPT plan's row, chosen: "Continue with ChatGPT", which saves the plan once signed in. */
function ChatGptChoice({ onSaved }: { onSaved(provider: ChatProvider): void }) {
  const sign = useContinueWithChatGpt({ onSignedIn: ({ provider }) => onSaved(provider) });
  const storage = useChatGptStorage();
  return (
    <div className="flex flex-col items-start gap-2">
      <ChatGptStorageNotice storage={storage} />
      {!sign.waiting && (
        <>
          <ProviderSendsLine provider={CHATGPT_SERVICE.name} />
          <ContinueWithChatGptButton onClick={() => void sign.run()} disabled={!storage.canSave} />
        </>
      )}
      <ChatGptSignInState sign={sign} />
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

const OLLAMA_DOWNLOAD_URL = "https://ollama.com/download";

/** Ollama's button: use it (pulling the recommended model), or get it and look for it again. */
export function OllamaAction({ ollama, large = false }: { ollama: Ollama; large?: boolean }) {
  const t = useT();
  const { status, pulling } = ollama;
  const size = large ? "lg" : "md";
  if (status?.running) {
    return (
      <button
        type="button"
        data-testid="ollama-use"
        disabled={pulling || !ollama.keepLocal.asked}
        onClick={() => void ollama.use()}
        className={buttonStyle("primary", size)}
      >
        {t("providers.ollama.use", { model: status.recommendedModel })}
      </button>
    );
  }
  return (
    <span className="flex flex-wrap items-center gap-2">
      {status !== null && (
        // The main process opens it in the User's browser.
        <button
          type="button"
          data-testid="ollama-get"
          onClick={() => window.open(OLLAMA_DOWNLOAD_URL, "_blank", "noreferrer")}
          className={buttonStyle("primary", size)}
        >
          {t("providers.ollama.get")} ↗
        </button>
      )}
      <button
        type="button"
        data-testid="ollama-check-again"
        disabled={status === null}
        onClick={() => void ollama.detect()}
        className={buttonStyle("secondary", size)}
      >
        {t("providers.ollama.checkAgain")}
      </button>
    </span>
  );
}

/** The providers by their short names, in the language's list: "Anthropic, OpenAI or Google". */
function providerList(
  ids: readonly string[],
  t: ReturnType<typeof useT>,
  language: Language,
): string {
  const names = ids.map((id) => {
    const short = SHORT_NAMES[kindOfCatalogId(id)];
    return short ? t(short) : keyProviderName(id, t, language);
  });
  const last = names.pop();
  if (last === undefined) return "";
  if (names.length === 0) return last;
  return t("providers.list.last", { rest: names.join(t("providers.list.separator")), last });
}
