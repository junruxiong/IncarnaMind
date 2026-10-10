import type { ReactNode } from "react";

/** The one action a problem offers. */
export interface ProblemAction {
  label: string;
  onClick(): void;
  testId?: string;
  /** Not while it is already running. */
  disabled?: boolean;
}

/**
 * A problem, or the result of something the User did, where it happened: one
 * line in the meta tone with at most one action, "Couldn't save: the folder
 * is read-only · Choose another" (DESIGN.md, Problems). Never a modal, and
 * never a toast that goes away by itself: it stays until the cause is gone.
 *
 * `alert` is read out at once, for something that went wrong; `status` waits
 * for a pause, for a result such as "Saved … · Show in Finder".
 */
export function ProblemLine({
  children,
  action,
  role = "alert",
  testId,
  title,
  className = "",
}: {
  children: ReactNode;
  action?: ProblemAction | undefined;
  role?: "alert" | "status";
  testId?: string;
  /** More to read on hover, such as the provider's own words. */
  title?: string | undefined;
  className?: string;
}) {
  return (
    <p
      role={role}
      data-testid={testId}
      title={title}
      className={`font-sans text-[13px] leading-5 break-words text-ink-meta ${className}`}
    >
      <span>{children}</span>
      {action && (
        <>
          <span aria-hidden="true"> · </span>
          <button
            type="button"
            data-testid={action.testId}
            disabled={action.disabled}
            onClick={action.onClick}
            className="rounded-sm font-semibold text-accent underline-offset-2 hover:underline disabled:opacity-50 disabled:hover:no-underline"
          >
            {action.label}
          </button>
        </>
      )}
    </p>
  );
}
