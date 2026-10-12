import { type KeyboardEvent, useEffect, useId, useState } from "react";
import type { CostSummary, Currency, MonthlyCap } from "../../../core/api";
import { core } from "../core";
import { formatMoney, formatSpent } from "../cost";
import { useLanguage, useT } from "../i18n";
import type { SettingsPage } from "../settingsPages";
import { useAppStore } from "../store";
import {
  fieldClass,
  hintClass,
  rowTextClass,
  rowTitleClass,
  ruledListClass,
  ruledRowClass,
  sectionTitleClass,
} from "./ui";

/** A first cap, when the User turns it on: about what a light month on a strong model costs. */
const FIRST_CAP: Record<Currency, number> = { USD: 20, CNY: 150 };

/**
 * This month's cost so far, read again each time Settings shows `page` and
 * whenever the cap changes. Null until read.
 */
export function useCostSummary(page: SettingsPage): CostSummary | null {
  const shown = useAppStore((state) => state.settingsOpen && state.settingsPage === page);
  const cap = useAppStore((state) => state.settings?.user.monthlyCap);
  const reportError = useAppStore((state) => state.reportError);
  const [summary, setSummary] = useState<CostSummary | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: a changed cap is when to read it again, not what it is read from.
  useEffect(() => {
    if (shown) core.getCostSummary().then(setSummary, reportError);
  }, [shown, cap, reportError]);
  return summary;
}

/**
 * Settings → General → Monthly spending cap (#83): what models billed per
 * token cost this month, and one optional cap, off by default. Its amount is
 * edited in place: Enter or leaving the field saves, Esc puts it back.
 */
export function SpendingCapSettings() {
  const t = useT();
  const language = useLanguage();
  const cap = useAppStore((state) => state.settings?.user.monthlyCap);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const summary = useCostSummary("general");
  const id = useId();
  const [draft, setDraft] = useState("");
  useEffect(() => {
    if (cap) setDraft(String(cap.amount));
  }, [cap]);
  if (cap === undefined) return null;

  const save = (next: MonthlyCap | null) => void updateSettings({ user: { monthlyCap: next } });
  const currency: Currency = cap?.currency ?? (language === "zh-CN" ? "CNY" : "USD");
  const typed = Number(draft);
  const valid = draft.trim() !== "" && Number.isFinite(typed) && typed >= 0;

  const commit = () => {
    if (!cap) return;
    if (!valid) setDraft(String(cap.amount));
    else if (typed !== cap.amount) save({ amount: typed, currency: cap.currency });
  };
  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") commit();
    if (event.key === "Escape" && cap && draft !== String(cap.amount)) {
      // Esc puts a changed amount back, and doesn't close Settings as well.
      event.preventDefault();
      setDraft(String(cap.amount));
    }
  };

  const spent = summary && formatSpent(summary.spent, t, language);
  const unpriced = summary?.providers.reduce((sum, each) => sum + each.unpricedTokens, 0) ?? 0;
  const spentLine = !summary
    ? null
    : cap && summary.spentTowardCap
      ? t("cost.cap.spentOf", {
          spent: formatMoney(summary.spentTowardCap, language),
          cap: formatMoney(cap, language),
        })
      : spent
        ? t("cost.cap.spent", { spent })
        : t("cost.cap.nothing");

  return (
    <section data-testid="spending-cap-settings">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("cost.cap.title")}</h4>
      <div className={ruledListClass}>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={id} className={rowTitleClass}>
              {t("cost.cap.switch")}
            </label>
            <p id={`${id}-description`} className={rowTextClass}>
              {t("cost.cap.body")}
            </p>
            {spentLine && (
              <p data-testid="spending-cap-spent" className={hintClass}>
                {spentLine}
              </p>
            )}
            {unpriced > 0 && (
              <p data-testid="spending-cap-unpriced" className={hintClass}>
                {t("cost.cap.unpriced", {
                  tokens: new Intl.NumberFormat(language).format(unpriced),
                })}
              </p>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-3">
            {cap && (
              <div className="flex items-center gap-1">
                <select
                  aria-label={t("cost.cap.currency")}
                  data-testid="spending-cap-currency"
                  value={cap.currency}
                  onChange={(event) =>
                    save({
                      amount: valid ? typed : cap.amount,
                      currency: event.target.value === "CNY" ? "CNY" : "USD",
                    })
                  }
                  className={`${fieldClass} w-16`}
                >
                  <option value="USD">$</option>
                  <option value="CNY">¥</option>
                </select>
                <input
                  type="number"
                  inputMode="decimal"
                  min="0"
                  step="any"
                  aria-label={t("cost.cap.amount")}
                  data-testid="spending-cap-amount"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onBlur={commit}
                  onKeyDown={onKeyDown}
                  className={`${fieldClass} w-24`}
                />
              </div>
            )}
            <input
              id={id}
              type="checkbox"
              role="switch"
              aria-checked={cap !== null}
              aria-describedby={`${id}-description`}
              data-testid="spending-cap-switch"
              checked={cap !== null}
              onChange={(event) =>
                save(
                  event.target.checked
                    ? { amount: valid && typed > 0 ? typed : FIRST_CAP[currency], currency }
                    : null,
                )
              }
              className="switch"
            />
          </div>
        </div>
      </div>
    </section>
  );
}
