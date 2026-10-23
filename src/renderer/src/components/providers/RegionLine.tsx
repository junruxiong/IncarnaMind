import { useEffect, useRef, useState } from "react";
import type { Endpoint } from "../../../../core/providers/catalog";
import { useLanguage, useT } from "../../i18n";
import { MetaLine } from "../MetaLine";
import { LineAction } from "../ProblemLine";
import { fieldClass } from "../ui";

/**
 * The region a saved provider's key was found in (#1572), as a quiet line: "Using the
 * International endpoint · Change". Change shows the regions in place; another opens that
 * region's own page, which takes a key made there.
 */
export function RegionLine({
  regions,
  current,
  onRegion,
}: {
  regions: readonly Endpoint[];
  current: Endpoint;
  onRegion(endpoint: string): void;
}) {
  const t = useT();
  const words = useLanguage() === "zh-CN" ? "zh" : "en";
  const [choosing, setChoosing] = useState(false);
  const select = useRef<HTMLSelectElement>(null);
  const change = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (choosing) select.current?.focus();
  }, [choosing]);
  const done = () => {
    setChoosing(false);
    requestAnimationFrame(() => change.current?.focus());
  };

  if (choosing) {
    return (
      <label className="flex items-center gap-2 text-[13px] leading-5 text-ink-secondary">
        {t("providers.form.region")}
        <select
          ref={select}
          data-testid="provider-page-region"
          value={current.id}
          onChange={(event) => onRegion(event.target.value)}
          onBlur={() => setChoosing(false)}
          onKeyDown={(event) => {
            if (event.key !== "Escape") return;
            event.preventDefault();
            event.stopPropagation();
            done();
          }}
          className={`${fieldClass} h-7 w-auto py-0`}
        >
          {regions.map((each) => (
            <option key={each.id} value={each.id}>
              {each.label[words]}
            </option>
          ))}
        </select>
      </label>
    );
  }
  return (
    <MetaLine
      data-testid="provider-page-region-line"
      className="text-[13px] leading-5 text-ink-secondary"
      parts={[
        t("providers.settings.region", { region: current.label[words] }),
        <LineAction
          key="change"
          ref={change}
          action={{
            label: t("providers.settings.changeRegion"),
            testId: "provider-page-region-change",
            onClick: () => setChoosing(true),
          }}
        />,
      ]}
    />
  );
}
