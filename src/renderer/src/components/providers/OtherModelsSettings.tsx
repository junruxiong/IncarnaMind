import { useCallback, useEffect, useState } from "react";
import type { ChatModelGroup } from "../../../../core/api";
import { useAnswers } from "../../answers";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { rowStatusClass, rowTextClass, ruledListClass, ruledRow, sectionTitleClass } from "../ui";
import { providerLabel } from "./shared";

/**
 * Settings → Models → Other models: what each provider lists that its model
 * menu leaves out until the User adds it, with "In the menu" to add one (and to
 * take it out again). Not shown while no provider has any.
 */
export function OtherModelsSettings() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const [groups, setGroups] = useState<ChatModelGroup[]>([]);

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
      useAppStore.setState({ actionError: errorMessage(failure) });
    }
    useAnswers.setState({ models: null });
    refresh();
  };

  const shown = groups.filter((group) => group.other.length + group.added.length > 0);
  if (shown.length === 0) return null;
  return (
    <section data-testid="other-models-settings">
      <h4 className={`mb-0.5 ${sectionTitleClass}`}>{t("models.other.title")}</h4>
      <p className="mb-2 max-w-[560px] text-[13px] leading-5 text-ink-meta">
        {t("models.other.note")}
      </p>
      <div className="flex flex-col gap-4">
        {shown.map((group) => (
          <div key={group.provider.id}>
            {groups.length > 1 && (
              <p className={`mb-1 ${rowStatusClass}`}>{providerLabel(group.provider, t)}</p>
            )}
            <div className={ruledListClass}>
              {[...group.added, ...group.other].map((modelId) => {
                const inMenu = group.added.includes(modelId);
                return (
                  <label
                    key={modelId}
                    data-testid="other-model"
                    data-model={modelId}
                    className={`${ruledRow("center")} cursor-pointer`}
                  >
                    <span className={`min-w-0 break-all ${rowTextClass}`}>{modelId}</span>
                    <span className="flex shrink-0 items-center gap-2 text-[13px] leading-5 text-ink-secondary">
                      <input
                        type="checkbox"
                        checked={inMenu}
                        onChange={(event) =>
                          void toggle(group.provider.id, modelId, event.target.checked)
                        }
                        className="size-4 accent-ink"
                      />
                      {t("models.other.inMenu")}
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
