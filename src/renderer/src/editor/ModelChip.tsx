import { isMacOS } from "@tiptap/core";
import { useEffect, useState } from "react";
import type * as Y from "yjs";
import type { ChatModelChoice, ChatProvider } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { mindModelOf, observeMindModel, setMindModel } from "../../../shared/mindModel";
import { useAnswers } from "../answers";
import { CheckLineIcon } from "../components/lineIcons";
import { usePopoverMenu } from "../components/usePopoverMenu";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ChevronDownSmallIcon } from "./icons";

/** Where a model runs, as the menu groups them (DESIGN.md, Composer › Model). */
type ModelGroup = "keys" | "plan" | "local";

const GROUPS: readonly ModelGroup[] = ["keys", "plan", "local"];

const groupOf = (provider: ChatProvider): ModelGroup =>
  provider.kind === "chatgpt" ? "plan" : provider.service === null ? "local" : "keys";

const GROUP_NAMES: Record<ModelGroup, MessageKey> = {
  keys: "composer.model.group.keys",
  plan: "composer.model.group.plan",
  local: "composer.model.group.local",
};

/** What each group's models are like, on their second line. */
const HINTS: Record<ModelGroup, MessageKey> = {
  keys: "composer.model.hint.keys",
  plan: "composer.model.hint.plan",
  local: isMacOS() ? "composer.model.hint.localMac" : "composer.model.hint.local",
};

/** How a model is reached, after its name in the chip's tooltip. */
const VIA: Record<ModelGroup, MessageKey> = {
  keys: "composer.model.via.keys",
  plan: "composer.model.via.plan",
  local: isMacOS() ? "composer.model.via.localMac" : "composer.model.via.local",
};

const same = (a: ChatModelChoice | null, b: ChatModelChoice | null) =>
  a !== null && b !== null && a.providerId === b.providerId && a.modelId === b.modelId;

/** The model chosen for the Mind, kept up to date as it changes here or elsewhere. */
function useMindModel(doc: Y.Doc): ChatModelChoice | null {
  const [choice, setChoice] = useState(() => mindModelOf(doc));
  useEffect(() => {
    setChoice(mindModelOf(doc));
    return observeMindModel(doc, () => setChoice(mindModelOf(doc)));
  }, [doc]);
  return choice;
}

/**
 * The composer's quiet model chip, before the Search scope: the model the
 * Mind's next Questions are asked with, and a menu to choose another, grouped
 * by where it runs (your keys, the ChatGPT plan, this computer), each with a
 * one-line hint. A choice is remembered for the Mind (`setMindModel`); a Mind
 * with none uses the default for new Minds, which the menu names at its foot.
 * Not shown before any chat model is set up.
 */
export function ModelChip({ doc, onChosen }: { doc: Y.Doc; onChosen(): void }) {
  const t = useT();
  const chosen = useMindModel(doc);
  const defaultModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const readiness = useAppStore((state) => state.chatReadiness);
  const openSettings = useAppStore((state) => state.openSettings);
  const groups = useAnswers((state) => state.models);
  const loadModels = useAnswers((state) => state.loadModels);
  // Over the composer, not on it.
  const menu = usePopoverMenu({ around: (chip) => chip.closest<HTMLElement>(".composer") });

  // The menu lists what each provider offers: asked for when it is about to open.
  useEffect(() => {
    if (menu.open) loadModels();
  }, [menu.open, loadModels]);

  const current = chosen ?? defaultModel;
  if (!current) return null;
  const providerOf = (providerId: string): ChatProvider | null =>
    groups?.find((group) => group.provider.id === providerId)?.provider ??
    (readiness && "provider" in readiness && readiness.provider.id === providerId
      ? readiness.provider
      : null);
  const provider = providerOf(current.providerId);
  const via = provider ? t(VIA[groupOf(provider)]) : null;
  const title = via ? `${current.modelId} · ${via}` : current.modelId;
  const label = via
    ? t("composer.model.label", { model: current.modelId, via })
    : t("composer.model.labelPlain", { model: current.modelId });

  const choose = (choice: ChatModelChoice) => {
    setMindModel(doc, choice);
    menu.close();
    onChosen();
  };

  // Each provider's models; the one in use is listed even if its provider didn't list it.
  const offered = (groups ?? []).map((group) => ({
    provider: group.provider,
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
                const inGroup = offered.filter(
                  (each) => groupOf(each.provider) === group && each.models.length > 0,
                );
                if (inGroup.length === 0) return null;
                return (
                  // biome-ignore lint/a11y/useSemanticElements: a group of a menu's items, which a fieldset can't hold.
                  <div key={group} role="group" aria-label={t(GROUP_NAMES[group])}>
                    <p aria-hidden="true" className="model-menu-heading">
                      {t(GROUP_NAMES[group])}
                      {group === "plan" && (
                        <span className="model-menu-heading-note">
                          {" · "}
                          {t("composer.model.group.planNote")}
                        </span>
                      )}
                    </p>
                    {inGroup.flatMap(({ provider: each, models }) =>
                      models.map((modelId) => {
                        const choice = { providerId: each.id, modelId };
                        const checked = same(choice, current);
                        return (
                          <button
                            key={`${each.id}\n${modelId}`}
                            type="button"
                            role="menuitemradio"
                            aria-checked={checked}
                            data-testid="composer-model-option"
                            data-model={modelId}
                            data-provider={each.id}
                            onClick={() => choose(choice)}
                            className="model-menu-item"
                          >
                            <CheckLineIcon
                              className="model-menu-check"
                              style={{ visibility: checked ? "visible" : "hidden" }}
                            />
                            <span className="model-menu-name">
                              <span className="truncate">{modelId}</span>
                              <span className="model-menu-hint">{t(HINTS[group])}</span>
                            </span>
                            <span className="model-menu-provider">{providerName(each, t)}</span>
                          </button>
                        );
                      }),
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

/** Who runs a model, short: "Anthropic", "OpenAI", "ChatGPT", "Ollama", or a server's host. */
function providerName(provider: ChatProvider, t: ReturnType<typeof useT>): string {
  if (provider.kind === "chatgpt") return "ChatGPT";
  if (provider.kind === "openai-compatible") {
    return provider.service?.name ?? t("providers.kind.openai-compatible");
  }
  return t(`providers.kind.${provider.kind}`);
}
