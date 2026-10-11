import { useEffect, useId, useState } from "react";
import type { OutputSettings } from "../../../core/api";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { folderName } from "../linkedFolders";
import { chooseOutputLocation } from "../outputFiles";
import { PathText } from "./PathText";
import { ProblemLine } from "./ProblemLine";
import {
  buttonClass,
  rowTextClass,
  ruledListClass,
  ruledRow,
  sectionNoteClass,
  sectionTitleClass,
} from "./ui";

/**
 * Settings → General → Output location: where new files go, changed with the
 * system's folder dialog; and, once IncarnaMind has asked about any, the
 * Linked folders it may or may not add files to, each with its switch.
 */
export function OutputLocationSettings() {
  const t = useT();
  const titleId = useId();
  const [settings, setSettings] = useState<OutputSettings | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    core.getOutputSettings().then(
      (loaded) => {
        if (current) setSettings(loaded);
      },
      (failure: unknown) => {
        if (current) setProblem(errorMessage(failure));
      },
    );
    const stop = core.on("outputSettings.changed", setSettings);
    return () => {
      current = false;
      stop();
    };
  }, []);

  const change = async (action: () => Promise<OutputSettings | null>) => {
    setProblem(null);
    try {
      const updated = await action();
      if (updated) setSettings(updated);
    } catch (failure) {
      setProblem(errorMessage(failure));
    }
  };

  if (!settings) return null;
  const { location, linkedFolders } = settings;

  return (
    <section data-testid="output-location-settings" className="flex flex-col gap-7">
      <div>
        <h4 id={titleId} className={sectionTitleClass}>
          {t("outputLocation.title")}
        </h4>
        <p className={sectionNoteClass}>{t("outputLocation.body")}</p>
        <div className="flex min-w-0 items-center gap-2">
          {location ? (
            <PathText
              path={location}
              testId="output-location-path"
              className="flex-1 rounded-md border border-rule bg-frame px-3 py-1.5 font-mono text-[13px] leading-5 text-ink-secondary"
            />
          ) : (
            <span
              data-testid="output-location-path"
              className="min-w-0 flex-1 text-[13px] leading-5 text-ink-meta"
            >
              {t("outputLocation.notChosen")}
            </span>
          )}
          <button
            type="button"
            data-testid="output-location-change"
            aria-describedby={titleId}
            onClick={() => void change(chooseOutputLocation)}
            className={buttonClass}
          >
            {t(location ? "outputLocation.change" : "outputLocation.choose")}
          </button>
        </div>
        {problem && (
          <ProblemLine testId="output-location-problem" className="mt-2">
            {t("outputLocation.failed", { reason: problem })}
          </ProblemLine>
        )}
      </div>
      {linkedFolders.length > 0 && (
        <div data-testid="linked-folder-writing">
          <h4 className={sectionTitleClass}>{t("outputLocation.linked.title")}</h4>
          <p className={sectionNoteClass}>{t("outputLocation.linked.body")}</p>
          <ul className={ruledListClass}>
            {linkedFolders.map((each) => (
              <li
                key={each.linkedFolderId}
                data-testid="linked-folder-writing-row"
                data-allowed={each.allowed}
                className={ruledRow("center", true)}
              >
                <div className="flex min-w-0 flex-col">
                  <span className="truncate text-ui text-ink">{folderName(each.path)}</span>
                  <PathText path={each.path} className={rowTextClass} />
                </div>
                <input
                  type="checkbox"
                  role="switch"
                  aria-checked={each.allowed}
                  aria-label={t("outputLocation.linked.allow", { name: folderName(each.path) })}
                  data-testid="linked-folder-writing-switch"
                  checked={each.allowed}
                  onChange={(event) => {
                    const allowed = event.target.checked;
                    void change(() => core.setLinkedFolderWriting(each.linkedFolderId, allowed));
                  }}
                  className="switch"
                />
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
