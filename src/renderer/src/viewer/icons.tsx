import type { SVGProps } from "react";

/** The viewer's own icons, drawn on a 24px grid like the design's mockups. */
type IconProps = SVGProps<SVGSVGElement>;

function Stroke({ strokeWidth = 1.75, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    />
  );
}

/** A box with a narrow column on its left: the outline panel. */
export const OutlineToggleIcon = (props: IconProps) => (
  <Stroke {...props}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M9 4v16" />
  </Stroke>
);

export const PreviousPageIcon = (props: IconProps) => (
  <Stroke strokeWidth={2} {...props}>
    <path d="m15 6-6 6 6 6" />
  </Stroke>
);

export const NextPageIcon = (props: IconProps) => (
  <Stroke strokeWidth={2} {...props}>
    <path d="m9 6 6 6-6 6" />
  </Stroke>
);

export const ZoomOutIcon = (props: IconProps) => (
  <Stroke strokeWidth={2} {...props}>
    <path d="M5 12h14" />
  </Stroke>
);

export const ZoomInIcon = (props: IconProps) => (
  <Stroke strokeWidth={2} {...props}>
    <path d="M12 5v14M5 12h14" />
  </Stroke>
);

/** Two bars with arrows between them: fit the page to the width. */
export const FitWidthIcon = (props: IconProps) => (
  <Stroke {...props}>
    <path d="M4 5v14M20 5v14M8 12h8M10.5 9 8 12l2.5 3M13.5 9 16 12l-2.5 3" />
  </Stroke>
);

/** An arrow leaving a box: open in another app. */
export const OpenExternallyIcon = (props: IconProps) => (
  <Stroke {...props}>
    <path d="M14 4h6v6" />
    <path d="M20 4 10 14" />
    <path d="M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5" />
  </Stroke>
);

export const CloseViewerIcon = (props: IconProps) => (
  <Stroke strokeWidth={2} {...props}>
    <path d="M6 6l12 12M18 6 6 18" />
  </Stroke>
);

/** Points right; turned a quarter to point down when its entry is expanded. */
export const ExpandIcon = (props: IconProps) => (
  <Stroke strokeWidth={2} {...props}>
    <path d="m9 6 6 6-6 6" />
  </Stroke>
);

/** The Citation check's tick: the quote was found. */
export const TickIcon = (props: IconProps) => (
  <Stroke strokeWidth={3} {...props}>
    <path d="m5 12.5 4.5 4.5L19 7.5" />
  </Stroke>
);

/** The Citation check's exclamation mark: the quote wasn't found. */
export const ExclamationIcon = (props: IconProps) => (
  <Stroke strokeWidth={3} {...props}>
    <path d="M12 6v8" />
    <path d="M12 18.5v.01" />
  </Stroke>
);
