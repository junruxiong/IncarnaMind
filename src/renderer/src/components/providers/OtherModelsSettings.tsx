import { useCallback, useEffect, useId, useState } from "react";
import type { ChatModelGroup, OtherModel } from "../../../../core/api";
import type { MessageKey } from "../../../../shared/i18n";
import { modelName } from "../../../../shared/modelName";
import { useAnswers } from "../../answers";
import { core } from "../../core";
import { ChevronDownSmallIcon, ChevronRightSmallIcon } from "../../editor/icons";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { rowStatusClass, rowTitleClass, ruledListClass } from "../ui";
import { providerLabel } from "./shared";

type Why = OtherModel["why"];

const ROW = "flex min-h-[34px] items-center gap-2.5 py-1";

const WHYS: readonly Why[] = ["unknown", "older", "cannot-answer"];

const WHY_TITLES: Record<Why, MessageKey> = {
  unknown: "models.other.group.unknown",
  older: "models.other.group.older",
  "cannot-answer": "models.other.group.cannotAnswer",
};

const REASONS: Record<NonNullable<OtherModel["reason"]>, MessageKey> = {
  completion: "models.other.reason.completion",
  audio: "models.other.reason.audio",
  "deep-research": "models.other.reason.deepResearch",
  "computer-use": "models.other.reason.computerUse",
};

/** What a provider lists beyond its menu, then the ids the User added that it doesn't list; by id, so a tick moves nothing. */
const rowsOf = (group: ChatModelGroup): OtherModel[] => [
  ...group.other,
  ...group.added
    .filter((id) => !group.other.some((each) => each.id === id))
    .map((id): OtherModel => ({ id, why: "unknown" })),
];

/**
 * Settings → Models → Other models: a disclosure, closed at first, of what each
 * provider lists that its model menu leaves out, grouped by why. Ticking a row
 * puts it in the menu; one that can't answer Questions here has no box. Not
 * shown while no provider has any.
 */
export function OtherModelsSettings() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const [groups, setGroups] = useState<ChatModelGroup[]>([]);
  const [open, setOpen] = useState(false);
  const panelId = useId();

  const refresh = useCallback(() => {
    core.listChatModels().then(setGroups, () => undefined);
  }, []);
  useEffect(() => {
    if (readiness) refresh();
  }, [readiness, refresh]);

  const toggle = async (providerId: string, modelId: string, inMenu: boolean) => {
    try {
      await core.setChatModelInMenu({ providerId, modelId, inMenu });
    } catch (failure) {
      useAppStore.getState().reportError(failure);
    }
    useAnswers.setState({ models: null });
    refresh();
  };

  const shown = groups.filter((group) => rowsOf(group).length > 0);
  if (shown.length === 0) return null;
  const count = shown.reduce((sum, group) => sum + rowsOf(group).length, 0);
  const Chevron = open ? ChevronDownSmallIcon : ChevronRightSmallIcon;
  return (
    <section data-testid="other-models-settings">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        data-testid="other-models-toggle"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2.5 py-3.5 text-left"
      >
        <Chevron aria-hidden="true" className="size-4 shrink-0 text-ink-meta" />
        <span className="flex min-w-0 flex-col">
          <span className={rowTitleClass}>
            {t("models.other.title")}
            <span className="font-normal text-ink-meta"> · {count}</span>
          </span>
          <span className={rowStatusClass}>
            {shown.length === 1 && shown[0]
              ? t("models.other.sub", { provider: providerLabel(shown[0].provider, t) })
              : t("models.other.subMany")}
          </span>
        </span>
      </button>
      {open && (
        <div id={panelId} className="ml-[26px] flex flex-col gap-3.5">
          <p className={`-mb-1 ${rowStatusClass}`}>{t("models.other.tick")}</p>
          {shown.map((group) => (
            <ProviderRows
              key={group.provider.id}
              group={group}
              named={shown.length > 1}
              onToggle={toggle}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function ProviderRows({
  group,
  named,
  onToggle,
}: {
  group: ChatModelGroup;
  named: boolean;
  onToggle(providerId: string, modelId: string, inMenu: boolean): void;
}) {
  const t = useT();
  const rows = rowsOf(group);
  return (
    <div className="flex flex-col gap-3">
      {named && <p className={rowTitleClass}>{providerLabel(group.provider, t)}</p>}
      {WHYS.map((why) => {
        const inGroup = rows.filter((row) => row.why === why);
        if (inGroup.length === 0) return null;
        return (
          // biome-ignore lint/a11y/useSemanticElements: a titled list of rows, which a fieldset would box.
          <div key={why} role="group" aria-label={t(WHY_TITLES[why])}>
            <p className="mb-1 text-[12px] leading-4 font-semibold text-ink-meta">
              {t(WHY_TITLES[why])}
            </p>
            <div className={ruledListClass}>
              {inGroup.map((row) => (
                <ModelRow
                  key={row.id}
                  row={row}
                  name={modelName(row.id, {
                    kind: group.provider.kind,
                    endpoint: group.provider.endpoint,
                    names: group.names,
                  })}
                  checked={group.added.includes(row.id)}
                  onToggle={(inMenu) => onToggle(group.provider.id, row.id, inMenu)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ModelRow({
  row,
  name,
  checked,
  onToggle,
}: {
  row: OtherModel;
  name: string;
  checked: boolean;
  onToggle(inMenu: boolean): void;
}) {
  const t = useT();
  if (row.why === "cannot-answer") {
    return (
      <div
        data-testid="other-model"
        data-model={row.id}
        data-unavailable
        className={`${ROW} pl-[26px]`}
      >
        <span
          title={name === row.id ? undefined : row.id}
          className="min-w-0 break-all text-ui text-ink-secondary"
        >
          {name}
        </span>
        <span className="ml-auto shrink-0 text-[13px] leading-5 text-ink-meta">
          {row.reason ? t(REASONS[row.reason]) : null}
        </span>
      </div>
    );
  }
  return (
    <label
      data-testid="other-model"
      data-model={row.id}
      className={`${ROW} cursor-pointer hover:bg-hover`}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onToggle(event.target.checked)}
        className="size-4 shrink-0 accent-ink"
      />
      <span
        title={name === row.id ? undefined : row.id}
        className="min-w-0 flex-1 break-all text-ui text-ink"
      >
        {name}
      </span>
      {row.why === "unknown" && (
        <span className="ml-auto shrink-0 text-[13px] leading-5 text-ink-meta">
          {t("models.other.mayNotWork")}
        </span>
      )}
    </label>
  );
}
