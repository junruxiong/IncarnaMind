import type { JSX, SVGProps } from "react";
import type { CitationCheck } from "../../../core/api";

/**
 * The Mind editor's own glyphs, drawn on a 24px grid with round caps as in
 * the approved mockups (DESIGN.md). Each Citation check state has its own
 * shape (tick, exclamation mark, dash), so colour is never the only signal.
 */
type IconProps = SVGProps<SVGSVGElement>;

function Stroke({ weight, children, ...props }: IconProps & { weight: number }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={weight}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

/** A filled play triangle: ask the Question. */
export const AskPlayIcon = (props: IconProps) => (
  <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
    <path
      d="M8 5.2v13.6a1 1 0 0 0 1.52.85l11-6.8a1 1 0 0 0 0-1.7l-11-6.8A1 1 0 0 0 8 5.2z"
      fill="currentColor"
    />
  </svg>
);

/** The marks of the Citation checks, small and bold: tick, "!", dash, and a dot while checking. */
const MARKS: Record<CitationCheck, (props: IconProps) => JSX.Element> = {
  found: (props) => (
    <Stroke weight={3} {...props}>
      <path d="m5 12.5 4.5 4.5L19 7.5" />
    </Stroke>
  ),
  "not-found": (props) => (
    <Stroke weight={3} {...props}>
      <path d="M12 6v8" />
      <path d="M12 18.5v.01" />
    </Stroke>
  ),
  "cant-check": (props) => (
    <Stroke weight={3} {...props}>
      <path d="M7 12h10" />
    </Stroke>
  ),
  checking: (props) => (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <circle cx="12" cy="12" r="4" />
    </svg>
  ),
};

/** The state of a Citation check as a small mark, for the margin and narrow markers. */
export function CheckMarkIcon({ check, ...props }: IconProps & { check: CitationCheck }) {
  const Mark = MARKS[check];
  return <Mark {...props} />;
}

/** The same states in a circle, for the state line of the Citation card. */
export function CheckStateIcon({ check, ...props }: IconProps & { check: CitationCheck }) {
  return (
    <Stroke weight={2} {...props}>
      <circle cx="12" cy="12" r="9" />
      {check === "found" && <path d="m8.5 12 2.5 2.5 4.5-5" />}
      {check === "not-found" && (
        <>
          <path d="M12 7.5v5" />
          <path d="M12 16.2v.01" />
        </>
      )}
      {check === "cant-check" && <path d="M8 12h8" />}
      {check === "checking" && <path d="M12 7.5V12l3 2" />}
    </Stroke>
  );
}

export const ChevronDownSmallIcon = (props: IconProps) => (
  <Stroke weight={2} {...props}>
    <path d="m6 9 6 6 6-6" />
  </Stroke>
);

export const ChevronRightSmallIcon = (props: IconProps) => (
  <Stroke weight={2} {...props}>
    <path d="m9 6 6 6-6 6" />
  </Stroke>
);

/** A cross: take something out (a chip of the Search scope, a Skill). */
export const RemoveIcon = (props: IconProps) => (
  <Stroke weight={2.5} {...props}>
    <path d="M6 6l12 12M18 6 6 18" />
  </Stroke>
);

/** A circular arrow: write the Answer again. */
export const RegenerateSmallIcon = (props: IconProps) => (
  <Stroke weight={2} {...props}>
    <path d="M20 11a8 8 0 1 0-2.3 5.7" />
    <path d="M20 4v7h-7" />
  </Stroke>
);
