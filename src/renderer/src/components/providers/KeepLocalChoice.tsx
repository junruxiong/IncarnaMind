import { useEffect, useId, useState } from "react";
import { core } from "../../core";
import { useT } from "../../i18n";
import { checkBoxClass, checkHintClass, checkRowClass } from "../ui";

/** What Local mode would pause: the cloud services set up, and the connected agents. */
interface CloudSetUp {
  services: string[];
  agents: number;
}

async function readCloudSetUp(): Promise<CloudSetUp> {
  const [providers, embedding, rerank, agents] = await Promise.all([
    core.listChatProviders(),
    core.getEmbeddingSettings(),
    core.getRerankSettings(),
    core.getAgents(),
  ]);
  const names = [
    ...providers.map((provider) => provider.service?.name),
    embedding.provider.service?.name,
    rerank.enabled ? rerank.service?.name : undefined,
  ].filter((name): name is string => name !== undefined);
  return {
    services: [...new Set(names)],
    agents: agents.enabled ? agents.pairings.length : 0,
  };
}

/**
 * "Also keep everything on this computer", asked under the button that
 * chooses a local model (#545). Ticked at first only when nothing in the cloud
 * is set up, or Local mode is already on; the User's own choice then wins.
 */
export function useKeepLocal(active: boolean) {
  const [setUp, setSetUp] = useState<CloudSetUp | null>(null);
  const [localOnly, setLocalOnly] = useState(false);
  const [chosen, setChosen] = useState<boolean | null>(null);
  const [asked, setAsked] = useState(false);

  useEffect(() => {
    if (!active) return;
    let current = true;
    Promise.all([readCloudSetUp(), core.getEmbeddingSettings()]).then(
      ([cloud, embedding]) => {
        if (!current) return;
        setSetUp(cloud);
        setLocalOnly(embedding.localOnly);
        setAsked(true);
      },
      // Without knowing what is set up, the box isn't shown and Local mode stays as it was.
      () => current && setAsked(true),
    );
    return () => {
      current = false;
    };
  }, [active]);

  const nothing = setUp !== null && setUp.services.length === 0 && setUp.agents === 0;
  return {
    setUp,
    asked,
    nothing,
    checked: chosen ?? (nothing || localOnly),
    choose: setChosen,
  };
}

export type KeepLocal = ReturnType<typeof useKeepLocal>;

export function KeepLocalChoice({
  keepLocal,
  disabled,
}: {
  keepLocal: KeepLocal;
  disabled: boolean;
}) {
  const t = useT();
  const id = useId();
  const { setUp } = keepLocal;
  if (setUp === null) return null;

  const parts = [
    ...setUp.services,
    ...(setUp.agents > 0
      ? [
          setUp.agents === 1
            ? t("localMode.keep.agents.one")
            : t("localMode.keep.agents", { count: setUp.agents }),
        ]
      : []),
  ];
  const last = parts.pop();
  const services =
    last === undefined
      ? ""
      : parts.length === 0
        ? last
        : t("localMode.keep.list.last", { rest: parts.join(t("providers.list.separator")), last });

  return (
    <div data-testid="keep-local" className="flex flex-col gap-1">
      <label htmlFor={id} className={checkRowClass}>
        <input
          id={id}
          type="checkbox"
          className={checkBoxClass}
          data-testid="keep-local-box"
          aria-describedby={`${id}-line`}
          checked={keepLocal.checked}
          disabled={disabled}
          onChange={(event) => keepLocal.choose(event.target.checked)}
        />
        {t("localMode.keep.label")}
      </label>
      <p id={`${id}-line`} data-testid="keep-local-line" className={checkHintClass}>
        {keepLocal.nothing ? t("localMode.keep.nothing") : t("localMode.keep.pauses", { services })}
      </p>
    </div>
  );
}
