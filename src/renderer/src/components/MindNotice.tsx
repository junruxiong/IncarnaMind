import { useT } from "../i18n";

/**
 * Said at the top of a Mind this version can't edit: "update" for one of a
 * kind a newer version made (it isn't shown), "read-only" for one a newer
 * version wrote the content of (it is shown, and never written to).
 */
export function MindNotice({ kind }: { kind: "update" | "read-only" }) {
  const t = useT();
  const key = kind === "update" ? "mind.update" : "mind.readOnly";
  return (
    <aside
      role="status"
      data-testid={`mind-notice-${kind}`}
      // Reaching 12px into the margins, so its text keeps the Mind's text edge.
      className="-mx-3 mb-6 flex items-start gap-2.5 rounded-lg bg-accent-wash p-3 font-sans text-ui text-ink-strong"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        className="mt-0.5 size-4 shrink-0 text-accent-strong"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5" />
        <path d="M12 8v.01" />
      </svg>
      <p className="min-w-0 flex-1">
        <strong className="font-semibold text-ink">{t(`${key}.title`)}</strong> {t(`${key}.body`)}
      </p>
    </aside>
  );
}
