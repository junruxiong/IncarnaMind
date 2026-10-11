import { useId, useState } from "react";
import type { SourceProviderInfo } from "../../../core/api";
import { useFindSources } from "../findSources";
import { useT } from "../i18n";
import { ProblemLine } from "./ProblemLine";
import { buttonStyle, inputClass } from "./ui";

/**
 * A provider's optional key of the User's own (openFDA's): a line under the
 * search that opens to a field, kept in the keychain by the core and never
 * shown again. Saving an empty field removes it.
 */
export function SourceKeyField({ provider }: { provider: SourceProviderInfo }) {
  const t = useT();
  const hintId = useId();
  const [open, setOpen] = useState(false);
  const [key, setKey] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const save = async (value: string | null) => {
    const failed = await useFindSources.getState().setKey(provider.id, value);
    setProblem(failed);
    if (!failed) {
      setKey("");
      setOpen(false);
    }
  };
  const name = { provider: provider.name };
  return (
    <div className="mt-2 text-[13px] leading-5 text-ink-secondary">
      <button
        type="button"
        aria-expanded={open}
        data-testid="find-sources-key-open"
        onClick={() => setOpen(!open)}
        className="underline decoration-rule-strong underline-offset-2 hover:text-ink"
      >
        {t("sources.key.open", name)}
      </button>
      {provider.key === "set" && (
        <span data-testid="find-sources-key-set" className="ml-2 text-ink-meta">
          {t("sources.key.set")}
        </span>
      )}
      {open && (
        <form
          className="mt-1.5 flex flex-wrap items-center gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (key.trim() !== "") void save(key);
          }}
        >
          <input
            type="password"
            autoComplete="off"
            spellCheck={false}
            data-testid="find-sources-key"
            aria-label={t("sources.key.title", name)}
            aria-describedby={hintId}
            placeholder={t("sources.key.placeholder")}
            value={key}
            onChange={(event) => setKey(event.target.value)}
            className={`${inputClass} mt-0 min-w-48 flex-1`}
          />
          <button
            type="submit"
            data-testid="find-sources-key-save"
            disabled={key.trim() === ""}
            className={buttonStyle("primary")}
          >
            {t("sources.key.save")}
          </button>
          {provider.key === "set" && (
            <button
              type="button"
              data-testid="find-sources-key-remove"
              onClick={() => void save(null)}
              className={buttonStyle("ghost")}
            >
              {t("sources.key.remove")}
            </button>
          )}
          <p id={hintId} className="basis-full text-ink-meta">
            {t("sources.key.hint", name)}
          </p>
        </form>
      )}
      {problem && (
        <ProblemLine className="mt-1" testId="find-sources-key-problem" title={problem}>
          {t("sources.key.failed")}
        </ProblemLine>
      )}
    </div>
  );
}
