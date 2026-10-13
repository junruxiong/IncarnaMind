import { useEffect, useState } from "react";
import type { AgentClient, ChatProvider } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useLanguage, useT } from "../../i18n";
import { giveUpWaiting } from "../../modelSetup";
import { useAppStore } from "../../store";
import { buttonStyle, hintClass } from "../ui";
import { ChatGptMark } from "./ChatGptMark";
import {
  ChatGptSignInState,
  ChatGptStorageNotice,
  useChatGptStorage,
  useContinueWithChatGpt,
} from "./ChatGptSignIn";
import { OllamaAction } from "./ChatProviderChoices";
import {
  OllamaProgress,
  type Ollama as OllamaState,
  ollamaDescription,
  useOllama,
} from "./OllamaCard";
import { ProviderForm } from "./ProviderForm";

const CLAUDE_DOWNLOAD = "https://claude.ai/download";

type View = "deepseek" | "key" | "local" | "chatgpt" | null;

/**
 * The first own Question with no model set (#103, D198): the assistant's composer becomes this
 * card while the Question waits above it. One recommended way leads (the ChatGPT plan; in Chinese,
 * a DeepSeek key), the others are folded under "Other ways", and Set up later is last. Choosing a
 * service is the consent for it (#718), so no dialog is added here.
 */
export function ModelSetupCard() {
  const t = useT();
  const chinese = useLanguage() === "zh-CN";
  const [view, setView] = useState<View>(null);
  const [others, setOthers] = useState(false);
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const ollama = useOllama(() => undefined);
  useEffect(() => {
    core.listChatProviders().then(setProviders, () => undefined);
  }, []);
  const open = (next: Exclude<View, null>) => setView(view === next ? null : next);

  return (
    <section
      data-testid="model-setup"
      aria-labelledby="model-setup-title"
      className="flex max-h-[70vh] flex-col gap-3 overflow-y-auto px-4 py-3"
    >
      <div className="flex flex-col gap-1">
        <h2 id="model-setup-title" className="text-[15px] leading-6 font-semibold text-ink">
          {t("setup.title")}
        </h2>
        <p className="text-[13px] leading-5 text-ink-secondary">{t("setup.body")}</p>
      </div>

      {chinese ? (
        <>
          <button
            type="button"
            data-testid="setup-primary"
            aria-expanded={view === "deepseek"}
            onClick={() => open("deepseek")}
            className={buttonStyle("primary", "lg")}
          >
            {t("setup.deepseek.use")}
          </button>
          <p className={hintClass}>{t("setup.deepseek.line")}</p>
          {view === "deepseek" && (
            <ProviderForm providers={providers} onSaved={() => undefined} initialKind="deepseek" />
          )}
        </>
      ) : (
        <ChatGptWay primary />
      )}

      <details
        open={others}
        onToggle={(event) => setOthers(event.currentTarget.open)}
        className="flex flex-col gap-2"
      >
        <summary
          data-testid="setup-other"
          className="cursor-default text-[13px] leading-5 font-semibold text-ink-secondary"
        >
          {t("setup.other")}
        </summary>
        <div className="mt-2 flex flex-col gap-2">
          {chinese ? (
            <>
              <Way
                testId="way-key"
                title={t("setup.way.key")}
                line={t("setup.way.key.line")}
                onClick={() => open("key")}
              />
              <Way
                testId="way-local"
                title={t("setup.way.local")}
                line={t("setup.way.local.line")}
                onClick={() => open("local")}
              />
              <Way
                testId="way-chatgpt"
                title={t("setup.way.chatgpt")}
                line={t("setup.way.chatgpt.line")}
                onClick={() => open("chatgpt")}
              />
            </>
          ) : (
            <>
              <Way
                testId="way-key"
                title={t("setup.way.key")}
                line={t("setup.way.key.line")}
                onClick={() => open("key")}
              />
              <Way
                testId="way-local"
                title={t("setup.way.local")}
                line={t("setup.way.local.line")}
                onClick={() => open("local")}
              />
            </>
          )}
          {view === "key" && <ProviderForm providers={providers} onSaved={() => undefined} />}
          {view === "local" && <LocalWay ollama={ollama} />}
          {view === "chatgpt" && <ChatGptWay />}
          <ClaudeWay />
        </div>
      </details>

      <div>
        <button
          type="button"
          data-testid="setup-later"
          onClick={giveUpWaiting}
          className={`-ml-3 ${buttonStyle("ghost", "md")}`}
        >
          {t("providers.setup.later")}
        </button>
      </div>
    </section>
  );
}

function Way({
  testId,
  title,
  line,
  onClick,
}: {
  testId: string;
  title: string;
  line: string;
  onClick(): void;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      className="flex flex-col items-start gap-0.5 rounded-md px-2 py-1.5 text-left hover:bg-hover"
    >
      <span className="text-ui font-semibold text-ink">{title}</span>
      <span className="text-[13px] leading-5 text-ink-secondary">{line}</span>
    </button>
  );
}

/** Signing in with ChatGPT: the recommended button in English, a way under Other ways in Chinese. */
function ChatGptWay({ primary = false }: { primary?: boolean }) {
  const t = useT();
  const sign = useContinueWithChatGpt({ onSignedIn: () => undefined });
  const storage = useChatGptStorage();
  return (
    <div className="flex flex-col items-start gap-2">
      <ChatGptStorageNotice storage={storage} />
      {!sign.waiting && (
        <button
          type="button"
          data-testid={primary ? "setup-primary" : "setup-chatgpt"}
          disabled={!storage.canSave}
          onClick={() => void sign.run()}
          className={buttonStyle("primary", primary ? "lg" : "md")}
        >
          <ChatGptMark className="size-4" />
          {primary ? t("setup.chatgpt.use") : t("chatgpt.continue")}
        </button>
      )}
      {primary && <p className={hintClass}>{t("setup.chatgpt.line")}</p>}
      <ChatGptSignInState sign={sign} />
    </div>
  );
}

function LocalWay({ ollama }: { ollama: OllamaState }) {
  const t = useT();
  return (
    <div className="flex flex-col items-start gap-2 pl-2">
      <p data-testid="ollama-status" className="text-[13px] leading-5 text-ink-secondary">
        {ollamaDescription(ollama, t)}
      </p>
      <OllamaProgress ollama={ollama} />
      <OllamaAction ollama={ollama} />
    </div>
  );
}

/** "Already pay for Claude?": Anthropic doesn't let other apps use the plan, so the Claude app connects instead. */
function ClaudeWay() {
  const t = useT();
  const reportError = useAppStore((state) => state.reportError);
  const [client, setClient] = useState<AgentClient | null | undefined>(undefined);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const folders = useAppStore((state) => state.library?.groups ?? []);
  useEffect(() => {
    core.listAgentClients().then(
      (all) => setClient(all.find((each) => each.id === "claude-desktop") ?? null),
      () => setClient(null),
    );
  }, []);
  const connect = async () => {
    setBusy(true);
    setProblem(null);
    try {
      await core.setAgentsEnabled(true);
      const { result } = await core.setUpAgent({
        client: "claude-desktop",
        folderIds: folders.map((folder) => folder.id),
      });
      if (result.kind === "copy")
        await navigator.clipboard.writeText(result.text).catch(reportError);
      setDone(true);
    } catch (error) {
      setProblem(errorMessage(error, t));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div data-testid="setup-claude" className="flex flex-col items-start gap-1 px-2 py-1.5">
      <span className="text-ui font-semibold text-ink">{t("setup.claude.title")}</span>
      <span className="text-[13px] leading-5 text-ink-secondary">{t("setup.claude.line")}</span>
      {client?.installed === false ? (
        <>
          <span className={hintClass}>{t("setup.claude.missing")}</span>
          <a
            href={CLAUDE_DOWNLOAD}
            target="_blank"
            rel="noreferrer"
            className="text-[13px] font-semibold text-accent underline-offset-2 hover:underline"
          >
            {t("setup.claude.download")} ↗
          </a>
        </>
      ) : (
        <button
          type="button"
          data-testid="setup-claude-connect"
          disabled={busy || done || folders.length === 0}
          onClick={() => void connect()}
          className={buttonStyle("secondary", "sm")}
        >
          {t("setup.claude.connect")}
        </button>
      )}
      {problem && (
        <p role="alert" className="text-[13px] leading-5 text-danger">
          {problem}
        </p>
      )}
    </div>
  );
}
