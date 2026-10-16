import { Fragment, useEffect, useState } from "react";
import type { LocalServerStatus } from "../../../../core/api";
import { LOCAL_PRESETS } from "../../../../core/providers/catalog/providers";
import { core } from "../../core";
import { useLanguage, useT } from "../../i18n";
import { choiceTextClass } from "../ui";
import { keyProviderName } from "./shared";

/** The choice of a local server's row, by its catalog id. */
export const localChoice = (catalogId: string) => `local-${catalogId}` as const;

/**
 * The servers on this computer beside Ollama: each is looked for at its default address when
 * the card opens, and a running one has its row at once. The others are opened by their names.
 */
export function useLocalServers(enabled: boolean) {
  const [found, setFound] = useState<Record<string, LocalServerStatus>>({});
  const [opened, setOpened] = useState<string[]>([]);

  useEffect(() => {
    if (!enabled) return;
    for (const catalogId of LOCAL_PRESETS) {
      core.detectLocalServer({ catalogId }).then(
        (status) => setFound((now) => ({ ...now, [catalogId]: status })),
        () => undefined,
      );
    }
  }, [enabled]);

  const open = (catalogId: string) => setOpened((now) => [...new Set([...now, catalogId])]);
  const shown = enabled
    ? LOCAL_PRESETS.filter((id) => opened.includes(id) || found[id]?.running)
    : [];
  return {
    found,
    shown,
    closed: enabled ? LOCAL_PRESETS.filter((id) => !shown.includes(id)) : [],
    open,
  };
}

/** "Also works with LM Studio · Unsloth Studio", the names the ones to open. */
export function LocalServersAlso({
  closed,
  onOpen,
}: {
  closed: readonly string[];
  onOpen(catalogId: string): void;
}) {
  const t = useT();
  const language = useLanguage();
  if (closed.length === 0) return null;
  return (
    <span data-testid="local-servers-also" className={choiceTextClass}>
      {t("providers.local.also")}{" "}
      {closed.map((catalogId, index) => (
        <Fragment key={catalogId}>
          {index > 0 && " · "}
          <button
            type="button"
            data-testid={`local-open-${catalogId}`}
            onClick={() => onOpen(catalogId)}
            className="underline"
          >
            {keyProviderName(catalogId, t, language)}
          </button>
        </Fragment>
      ))}
    </span>
  );
}
