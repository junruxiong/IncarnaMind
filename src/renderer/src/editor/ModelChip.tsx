import { isMacOS } from "@tiptap/core";
import { Children, type ReactNode, useEffect, useState } from "react";
import type * as Y from "yjs";
import type { ChatGptPlanStatus, ChatModelChoice, ChatProvider } from "../../../core/api";
import { isCloudModelName } from "../../../core/providers/ollamaCloud";
import type { MessageKey } from "../../../shared/i18n";
import { mindModelOf, observeMindModel, setMindModel } from "../../../shared/mindModel";
import { useAnswers } from "../answers";
import { CheckLineIcon } from "../components/lineIcons";
import { OnThisComputer } from "../components/OnThisComputer";
import { ChatGptMark } from "../components/providers/ChatGptMark";
import {
  ChatGptSignInState,
  ManageUsageLink,
  useContinueWithChatGpt,
} from "../components/providers/ChatGptSignIn";
import { usePopoverMenu } from "../components/usePopoverMenu";
import { useComposer } from "../composer";
import { core } from "../core";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ChevronDownSmallIcon } from "./icons";

/** Where a model runs, as the menu groups them (DESIGN.md, Composer › Model). */
type ModelGroup = "keys" | "plan" | "local" | "cloud";

const GROUPS: readonly ModelGroup[] = ["keys", "plan", "local", "cloud"];

/**
 * `cloud`: the model runs on ollama.com though its provider, Ollama, is on this
 * computer. The provider's own service says so only for the default model, so
 * the model list's word comes first.
 */
const groupOf = (provider: ChatProvider, cloud: boolean): ModelGroup =>
  provider.kind === "chatgpt"
    ? "plan"
    : cloud
      ? "cloud"
      : provider.service === null
        ? "local"
        : "keys";

const GROUP_NAMES: Record<ModelGroup, MessageKey> = {
  keys: "composer.model.group.keys",
  plan: "composer.model.group.plan",
  local: "composer.model.group.local",
  cloud: "composer.model.group.cloud",
};

/** What each group's models are like, on their second line. */
const HINTS: Record<ModelGroup, MessageKey> = {
  keys: "composer.model.hint.keys",
  plan: "composer.model.hint.plan",
  local: isMacOS() ? "composer.model.hint.localMac" : "composer.model.hint.local",
  cloud: "composer.model.hint.cloud",
};

/** How a model is reached, after its name in the chip's tooltip. */
const VIA: Record<ModelGroup, MessageKey> = {
  keys: "composer.model.via.keys",
  plan: "composer.model.via.plan",
  local: isMacOS() ? "composer.model.via.localMac" : "composer.model.via.local",
  cloud: "composer.model.via.cloud",
};

const same = (a: ChatModelChoice | null, b: ChatModelChoice | null) =>
  a !== null && b !== null && a.providerId === b.providerId && a.modelId === b.modelId;

/**
 * The model chosen for the Mind, kept up to date as it changes here or elsewhere;
 * with no Mind (`doc` null), the one chosen for the composer until the app quits.
 */
export function useMindModel(doc: Y.Doc | null): ChatModelChoice | null {
  const [choice, setChoice] = useState(() => (doc ? mindModelOf(doc) : null));
  const withoutMind = useComposer((state) => state.modelWithoutMind);
  useEffect(() => {
    if (!doc) return;
    setChoice(mindModelOf(doc));
    return observeMindModel(doc, () => setChoice(mindModelOf(doc)));
  }, [doc]);
  return doc ? choice : withoutMind;
}

function chooseModel(doc: Y.Doc | null, choice: ChatModelChoice): void {
  if (doc) setMindModel(doc, choice);
  else useComposer.setState({ modelWithoutMind: choice });
}

/**
 * The composer's quiet model chip, before the Search scope: the model the
 * Mind's next Questions are asked with, and a menu to choose another, grouped
 * by where it runs (your keys, the ChatGPT plan, this computer), each with a
 * one-line hint. A choice is remembered for the Mind (`setMindModel`); a Mind
 * with none uses the default for new Minds, which the menu names at its foot.
 * Until the User is signed in with ChatGPT, the plan's group offers "Continue
 * with ChatGPT"; while its model is chosen, it says the plan is in use, with
 * "Manage usage". Not shown before any chat model is set up.
 */
export function ModelChip({ doc, onChosen }: { doc: Y.Doc | null; onChosen(): void }) {
  const t = useT();
  const chosen = useMindModel(doc);
  const defaultModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const readiness = useAppStore((state) => state.chatReadiness);
  const localMode = useAppStore((state) => state.embedding?.localOnly === true);
  const openSettings = useAppStore((state) => state.openSettings);
  const groups = useAnswers((state) => state.models);
  const loadModels = useAnswers((state) => state.loadModels);
  // Over the composer, not on it.
  const menu = usePopoverMenu({ around: (chip) => chip.closest<HTMLElement>(".composer") });
  const [plan, setPlan] = useState<ChatGptPlanStatus | null>(null);
  const sign = useContinueWithChatGpt({
    keepDefault: true,
    onSignedIn: ({ provider, modelId }) => {
      loadModels();
      chooseModel(doc, { providerId: provider.id, modelId });
      menu.close();
      onChosen();
    },
  });

  // The menu lists what each provider offers: asked for when it is about to open.
  useEffect(() => {
    if (!menu.open) return;
    loadModels();
    core.getChatGptPlan().then(setPlan, () => undefined);
  }, [menu.open, loadModels]);
  useEffect(() => core.on("chatGptPlan.changed", setPlan), []);

  const current = chosen ?? defaultModel;
  if (!current) return null;
  const providerOf = (providerId: string): ChatProvider | null =>
    groups?.find((group) => group.provider.id === providerId)?.provider ??
    (readiness && "provider" in readiness && readiness.provider.id === providerId
      ? readiness.provider
      : null);
  const provider = providerOf(current.providerId);
  const isCloud = (providerId: string, modelId: string) => {
    const group = groups?.find((each) => each.provider.id === providerId);
    // A model the provider doesn't list (a Mind's own choice) is cloud by its name, or as the provider says of its default.
    return group?.models.includes(modelId)
      ? group.cloud.includes(modelId)
      : isCloudModelName(modelId) || (provider?.kind === "ollama" && provider.service !== null);
  };
  const via = provider ? t(VIA[groupOf(provider, isCloud(provider.id, current.modelId))]) : null;
  const title = via ? `${current.modelId} · ${via}` : current.modelId;
  const label = via
    ? t("composer.model.label", { model: current.modelId, via })
    : t("composer.model.labelPlain", { model: current.modelId });

  const choose = (choice: ChatModelChoice) => {
    chooseModel(doc, choice);
    menu.close();
    onChosen();
  };

  // Each provider's models; the one in use is listed even if its provider didn't list it.
  const offered = (groups ?? []).map((group) => ({
    provider: group.provider,
    cloud:
      group.provider.id === current.providerId &&
      !group.models.includes(current.modelId) &&
      isCloud(group.provider.id, current.modelId)
        ? [current.modelId, ...group.cloud]
        : group.cloud,
    models:
      group.provider.id === current.providerId && !group.models.includes(current.modelId)
        ? [current.modelId, ...group.models]
        : group.models,
  }));

  return (
    <span className="composer-model-anchor">
      <button
        type="button"
        {...menu.buttonProps}
        data-testid="composer-model"
        aria-label={label}
        title={title}
        onPointerEnter={loadModels}
        onFocus={loadModels}
        className="composer-chip composer-model"
      >
        <span className="truncate">{current.modelId}</span>
        <ChevronDownSmallIcon className="composer-model-chevron" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("composer.model.menu")}
        data-testid="composer-model-menu"
        className="model-menu"
      >
        {menu.open && (
          <>
            <div className="model-menu-list">
              <p className="model-menu-intro">{t("composer.model.intro")}</p>
              {GROUPS.map((group) => {
                // A provider can sit in two groups: Ollama's local models and its cloud ones.
                const inGroup = offered
                  .map((each) => ({
                    ...each,
                    models: each.models.filter(
                      (modelId) => groupOf(each.provider, each.cloud.includes(modelId)) === group,
                    ),
                  }))
                  .filter((each) => each.models.length > 0);
                if (group === "plan") {
                  return (
                    <PlanGroup
                      key={group}
                      signedIn={plan?.account.state === "signed-in"}
                      inUse={provider?.kind === "chatgpt"}
                      sign={sign}
                    >
                      {inGroup.flatMap(({ provider: each, models }) =>
                        models.map((modelId) => (
                          <ModelItem
                            key={`${each.id}\n${modelId}`}
                            provider={each}
                            modelId={modelId}
                            hint={t(HINTS[group])}
                            checked={same({ providerId: each.id, modelId }, current)}
                            unavailable={localMode}
                            onChoose={choose}
                          />
                        )),
                      )}
                    </PlanGroup>
                  );
                }
                if (inGroup.length === 0) return null;
                return (
                  // biome-ignore lint/a11y/useSemanticElements: a group of a menu's items, which a fieldset can't hold.
                  <div key={group} role="group" aria-label={t(GROUP_NAMES[group])}>
                    <p aria-hidden="true" className="model-menu-heading">
                      {t(GROUP_NAMES[group])}
                    </p>
                    {inGroup.flatMap(({ provider: each, models }) =>
                      models.map((modelId) => (
                        <ModelItem
                          key={`${each.id}\n${modelId}`}
                          provider={each}
                          modelId={modelId}
                          hint={t(HINTS[group])}
                          onThisComputer={group === "local"}
                          checked={same({ providerId: each.id, modelId }, current)}
                          // Local mode: only a model on this computer can be asked.
                          unavailable={localMode && group !== "local"}
                          onChoose={choose}
                        />
                      )),
                    )}
                  </div>
                );
              })}
              {groups === null || groups.length === 0 ? (
                <p className="model-menu-intro">{t("composer.model.loading")}</p>
              ) : null}
            </div>
            <div className="model-menu-foot">
              <span>
                {defaultModel
                  ? t("composer.model.default", { model: defaultModel.modelId })
                  : t("composer.model.defaultUnset")}
              </span>
              <span aria-hidden="true"> · </span>
              <button
                type="button"
                role="menuitem"
                data-testid="composer-model-settings"
                onClick={() => {
                  menu.close();
                  openSettings("models");
                }}
                className="model-menu-link"
              >
                {t("composer.model.change")}
              </button>
            </div>
          </>
        )}
      </div>
    </span>
  );
}

/**
 * One model in the menu: its id, a hint, and who runs it. `unavailable`: listed
 * but not chosen, with the reason as its hint (a cloud model in local mode).
 */
function ModelItem({
  provider,
  modelId,
  hint,
  onThisComputer = false,
  checked,
  unavailable,
  onChoose,
}: {
  provider: ChatProvider;
  modelId: string;
  hint: string;
  /** The hint says it stays on this computer: it takes that mark. */
  onThisComputer?: boolean;
  checked: boolean;
  unavailable: boolean;
  onChoose(choice: ChatModelChoice): void;
}) {
  const t = useT();
  return (
    <button
      type="button"
      role="menuitemradio"
      aria-checked={checked}
      aria-disabled={unavailable || undefined}
      data-testid="composer-model-option"
      data-model={modelId}
      data-provider={provider.id}
      data-unavailable={unavailable || undefined}
      onClick={() => {
        if (!unavailable) onChoose({ providerId: provider.id, modelId });
      }}
      className="model-menu-item"
    >
      <CheckLineIcon
        className="model-menu-check"
        style={{ visibility: checked ? "visible" : "hidden" }}
      />
      <span className="model-menu-name">
        <span className="truncate">{modelId}</span>
        <span className="model-menu-hint">
          {unavailable ? (
            t("composer.model.localModeOff")
          ) : onThisComputer ? (
            <OnThisComputer>{hint}</OnThisComputer>
          ) : (
            hint
          )}
        </span>
      </span>
      <span className="model-menu-provider">{providerName(provider, t)}</span>
    </button>
  );
}

/**
 * The ChatGPT plan's group: its models once signed in, with "Using ChatGPT
 * plan · Manage usage" while one is chosen, as OpenAI's guidelines ask; else
 * "Continue with ChatGPT", which signs in and chooses the plan for this Mind.
 */
function PlanGroup({
  signedIn,
  inUse,
  sign,
  children,
}: {
  signedIn: boolean;
  inUse: boolean;
  sign: ReturnType<typeof useContinueWithChatGpt>;
  children: ReactNode;
}) {
  const t = useT();
  const items = Children.count(children);
  if (signedIn && items === 0) return null;
  return (
    // biome-ignore lint/a11y/useSemanticElements: a group of a menu's items, which a fieldset can't hold.
    <div role="group" aria-label={t(GROUP_NAMES.plan)} data-testid="composer-model-plan">
      <p aria-hidden="true" className="model-menu-heading">
        {t(GROUP_NAMES.plan)}
      </p>
      {signedIn ? (
        <>
          {children}
          {inUse && (
            <p data-testid="composer-model-plan-in-use" className="model-menu-intro">
              {t("chatgpt.inUse")}
              <span aria-hidden="true"> · </span>
              <ManageUsageLink />
            </p>
          )}
        </>
      ) : sign.waiting || sign.failureText ? (
        <div className="model-menu-intro">
          <ChatGptSignInState sign={sign} retry />
        </div>
      ) : (
        <button
          type="button"
          role="menuitem"
          data-testid="composer-model-chatgpt"
          onClick={() => void sign.run()}
          className="model-menu-item"
        >
          <ChatGptMark className="model-menu-check" />
          <span className="model-menu-name">
            <span className="truncate">{t("chatgpt.continue")}</span>
            <span className="model-menu-hint">{t("chatgpt.menu.hint")}</span>
          </span>
        </button>
      )}
    </div>
  );
}

/** Who runs a model, short: "Anthropic", "OpenAI", "ChatGPT", "Ollama", or a server's host. */
function providerName(provider: ChatProvider, t: ReturnType<typeof useT>): string {
  if (provider.kind === "chatgpt") return "ChatGPT";
  if (provider.kind === "openai-compatible") {
    return provider.service?.name ?? t("providers.kind.openai-compatible");
  }
  return t(`providers.kind.${provider.kind}`);
}
