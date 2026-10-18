import { type ReactNode, useEffect, useId, useRef, useState } from "react";
import type { ChatModelGroup, ChatProvider } from "../../../../core/api";
import { useAnswers } from "../../answers";
import { continueSetup } from "../../conversations";
import { core } from "../../core";
import { useLanguage, useT } from "../../i18n";
import { useAppStore } from "../../store";
import { CheckLineIcon } from "../lineIcons";
import { ROUTE_LABELS, TranscriptionRouteSelect } from "../meetings/TranscriptionRouteSelect";
import { buttonStyle } from "../ui";
import { CHATGPT_RECIPIENT } from "./ChatGptSignIn";
import { jobSelectClass, ROWS, RoleSelect, useJobSummary, useModelRoles } from "./ModelDefaults";
import { providerName } from "./ProviderPage";
import { SendsLine } from "./SendsLine";

/** The company a provider's questions go to, or null where they stay on this computer. */
function recipientOf(provider: ChatProvider, name: string): string | null {
  if (provider.kind === "ollama") return null;
  if (provider.kind === "chatgpt") return CHATGPT_RECIPIENT;
  if (provider.kind === "claude-account") return "Anthropic";
  if (provider.kind === "copilot") return "GitHub";
  return name;
}

/**
 * Once a key or sign-in works, the setup card says which model each job uses, with Change in
 * place, and waits for Continue, which is never held back. The Question that waited is asked then.
 */
export function SetupReady() {
  const t = useT();
  const language = useLanguage();
  const id = useId();
  const user = useAppStore((state) => state.settings?.user);
  const groups = useAnswers((state) => state.models) ?? [];
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const roles = useModelRoles(groups);
  const next = useRef<HTMLButtonElement>(null);

  // The menus were read before this provider existed.
  useEffect(() => {
    useAnswers.setState({ models: null });
    useAnswers.getState().loadModels();
    core
      .listChatProviders()
      .then(setProviders, (error: unknown) =>
        console.error("The saved chat providers couldn't be listed for the setup card.", error),
      );
    next.current?.focus();
  }, []);

  const chosen = user?.chatModel;
  const group = groups.find((each) => each.provider.id === chosen?.providerId);
  const name = group ? providerName(group.provider, t, language) : null;
  const recipient = group && name ? recipientOf(group.provider, name) : null;
  // Only when the Answers model can't read a scan itself, which a model's own record says.
  const readsImages = chosen ? group?.about[chosen.modelId]?.readsImages : undefined;
  const rows = ROWS.filter((row) => row.role !== "imageModel" || readsImages === false);

  return (
    <section
      data-testid="model-setup"
      data-ready=""
      aria-labelledby={`${id}-title`}
      className="setup-card flex min-h-0 max-h-[85vh] flex-col gap-2.5 overflow-y-auto px-4"
    >
      <h2
        id={`${id}-title`}
        data-setup-title
        data-testid="setup-ready-title"
        className="flex items-center gap-2 text-[15px] leading-6 font-semibold text-ink"
      >
        <CheckLineIcon className="size-4 shrink-0 text-success" />
        {name ? t("setup.ready", { name }) : t("setup.title")}
      </h2>
      <section data-testid="setup-jobs" className="flex flex-col">
        <h3 className="text-[13px] leading-5 font-semibold text-ink-secondary">
          {t("setup.jobs.title")}
        </h3>
        {rows.map((row) => (
          <Job
            key={row.role}
            id={`${id}-${row.role}`}
            testId={row.role}
            label={t(row.label)}
            tip={t(row.tip)}
            summary={<RoleSummary role={row.role} groups={groups} roles={roles} />}
            editor={(done) => (
              <RoleSelect
                id={`${id}-${row.role}`}
                role={row.role}
                groups={groups}
                providers={providers}
                roles={roles}
                className={jobSelectClass}
                autoFocus
                onChosen={done}
                onBlur={done}
                onEscape={done}
              />
            )}
          />
        ))}
        <Job
          id={`${id}-meetings`}
          testId="meetings"
          label={t("models.defaults.meetings")}
          tip={t("models.defaults.meetings.tip")}
          summary={<RouteSummary />}
          editor={(done) => (
            <TranscriptionRouteSelect
              id={`${id}-meetings`}
              className={jobSelectClass}
              autoFocus
              onChosen={done}
              onBlur={done}
              onEscape={done}
            />
          )}
        />
      </section>
      {recipient && <SendsLine provider={recipient} testId="setup-ready-sends" />}
      <div data-setup-foot>
        <button
          ref={next}
          type="button"
          data-testid="setup-continue"
          onClick={continueSetup}
          className={buttonStyle("primary", "md")}
        >
          {t("setup.continue")}
        </button>
      </div>
    </section>
  );
}

function RoleSummary({
  role,
  groups,
  roles,
}: {
  role: (typeof ROWS)[number]["role"];
  groups: readonly ChatModelGroup[];
  roles: ReturnType<typeof useModelRoles>;
}) {
  return <>{useJobSummary(role, groups, roles)}</>;
}

function RouteSummary() {
  const t = useT();
  const route = useAppStore((state) => state.settings?.device.transcriptionRoute) ?? "local";
  return <>{t(ROUTE_LABELS[route])}</>;
}

/** One job: its name, what it uses, and Change, which swaps the words for the choice, where they are. */
function Job({
  id,
  testId,
  label,
  tip,
  summary,
  editor,
}: {
  id: string;
  testId: string;
  label: string;
  tip: string;
  summary: ReactNode;
  editor(done: () => void): ReactNode;
}) {
  const t = useT();
  const [editing, setEditing] = useState(false);
  const change = useRef<HTMLButtonElement>(null);
  const done = () => {
    setEditing(false);
    // Back on the words, the focus on the button that opened the choice.
    requestAnimationFrame(() => change.current?.focus());
  };
  return (
    <div
      data-testid={`setup-job-${testId}`}
      className="flex min-h-9 flex-wrap items-center gap-x-4 gap-y-1 py-1"
    >
      <span
        title={tip}
        id={`${id}-label`}
        className="w-28 shrink-0 text-[13px] leading-5 font-semibold text-ink"
      >
        {label}
      </span>
      <span id={`${id}-tip`} className="sr-only">
        {tip}
      </span>
      {editing ? (
        editor(done)
      ) : (
        <>
          <span
            data-testid={`setup-job-${testId}-value`}
            className="min-w-0 flex-1 truncate text-[13px] leading-5 text-ink-secondary"
          >
            {summary}
          </span>
          <button
            ref={change}
            type="button"
            data-testid={`setup-job-${testId}-change`}
            aria-label={`${t("setup.jobs.change")}: ${label}`}
            onClick={() => setEditing(true)}
            className="shrink-0 text-[13px] font-semibold text-accent underline-offset-2 hover:underline"
          >
            {t("setup.jobs.change")}
          </button>
        </>
      )}
    </div>
  );
}
