import type { ReactNode } from "react";
import { MetaLine } from "./MetaLine";

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
  also,
  role = "alert",
  testId,
  title,
  className = "",
}: {
  children: ReactNode;
  action?: ProblemAction | undefined;
  /** A second choice, after the first, where the problem has two ways out (local mode refusing a Question). */
  also?: ProblemAction | undefined;
  role?: "alert" | "status";
  testId?: string;
  /** More to read on hover, such as the provider's own words. */
  title?: string | undefined;
  className?: string;
}) {
  const buttons = [action, also].filter(
    (each): each is ProblemAction => each !== undefined && each.label !== "",
  );
  return (
    <div
      role={role}
      data-testid={testId}
      title={title}
      className={`font-sans text-[13px] leading-5 break-words text-ink-meta ${className}`}
    >
      <MetaLine
        as="div"
        parts={[
          children,
          ...buttons.map((each) => (
            <button
              key={each.label}
              type="button"
              data-testid={each.testId}
              disabled={each.disabled}
              onClick={each.onClick}
              className="rounded-sm font-semibold text-accent underline-offset-2 hover:underline disabled:opacity-50 disabled:hover:no-underline"
            >
              {each.label}
            </button>
          )),
        ]}
      />
    </div>
  );
}
