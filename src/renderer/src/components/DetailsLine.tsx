import { type ReactNode, useId, useState } from "react";
import { useT } from "../i18n";

/**
 * One short line with "Details" after it, which opens the rest in place (DESIGN.md › Words on
 * screen). For what leaves the computer: the line says where it goes, Details lists what.
 */
export function DetailsLine({
  children,
  details,
  testId,
  className = "",
}: {
  children: ReactNode;
  details: ReactNode;
  testId: string;
  className?: string;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const panel = useId();
  return (
    <div data-testid={testId} className={className}>
      <p>
        {children}
        <span aria-hidden="true"> · </span>
        <button
          type="button"
          data-testid={`${testId}-details`}
          aria-expanded={open}
          aria-controls={panel}
          onClick={() => setOpen(!open)}
          className="rounded-sm font-semibold text-accent underline-offset-2 hover:underline"
        >
          {t("details.label")}
        </button>
      </p>
      {open && (
        <p id={panel} data-testid={`${testId}-list`} className="mt-0.5">
          {details}
        </p>
      )}
    </div>
  );
}
