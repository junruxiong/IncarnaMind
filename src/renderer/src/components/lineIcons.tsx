import type { ReactNode, SVGProps } from "react";
import type { DocumentKind } from "../../../core/api";

type IconProps = SVGProps<SVGSVGElement>;

/**
 * The chrome's icons (sidebar, pane headers, dialogs), drawn as in the
 * mockups: 1.75px strokes in the text's colour on a 24px grid, shown at 16px.
 */
function LineIcon({ children, strokeWidth = 1.75, ...props }: IconProps & { children: ReactNode }) {
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
    >
      {children}
    </svg>
  );
}

export function PlusLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M12 5v14M5 12h14" />
    </LineIcon>
  );
}

/** A Mind: a notebook. */
export function MindLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M7 3h10a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z" />
      <path d="M9 8h6M9 12h6M9 16h3" />
    </LineIcon>
  );
}

const FOLDER = "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z";

export function FolderLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d={FOLDER} />
    </LineIcon>
  );
}

export function FolderPlusLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d={FOLDER} />
      <path d="M12 10.5v5M9.5 13h5" />
    </LineIcon>
  );
}

/** Other Documents, the files added on their own: two pages, one behind the other. */
export function LooseDocumentsLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M8 7V5a2 2 0 0 1 2-2h5l4 4v10a2 2 0 0 1-2 2h-1" />
      <path d="M7 7h5l4 4v8a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z" />
    </LineIcon>
  );
}

/** A Document: a page with a folded corner; text and Markdown ones have lines on it. */
export function DocumentLineIcon({ kind, ...props }: IconProps & { kind: DocumentKind }) {
  return (
    <LineIcon {...props}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      {kind !== "pdf" && <path d="M9 13h6M9 17h4" />}
    </LineIcon>
  );
}

export function ChevronDownLineIcon(props: IconProps) {
  return (
    <LineIcon strokeWidth={2} {...props}>
      <path d="m6 9 6 6 6-6" />
    </LineIcon>
  );
}

export function ChevronRightLineIcon(props: IconProps) {
  return (
    <LineIcon strokeWidth={2} {...props}>
      <path d="m9 6 6 6-6 6" />
    </LineIcon>
  );
}

/** Settings: three sliders. */
export function SettingsLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1" />
      <circle cx="15" cy="6" r="2" />
      <circle cx="9" cy="12" r="2" />
      <circle cx="17" cy="18" r="2" />
    </LineIcon>
  );
}

/** Export: down onto a line. */
export function ExportLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M12 4v11" />
      <path d="m7 10 5 5 5-5" />
      <path d="M5 20h14" />
    </LineIcon>
  );
}

export function CloseLineIcon(props: IconProps) {
  return (
    <LineIcon strokeWidth={2} {...props}>
      <path d="M6 6l12 12M18 6 6 18" />
    </LineIcon>
  );
}

export function TagLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0L3 13V5a2 2 0 0 1 2-2h8l7.6 7.6a2 2 0 0 1 0 2.8z" />
      <circle cx="7.5" cy="7.5" r="1.25" />
    </LineIcon>
  );
}

export function MoreLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <circle cx="5" cy="12" r="1" fill="currentColor" />
      <circle cx="12" cy="12" r="1" fill="currentColor" />
      <circle cx="19" cy="12" r="1" fill="currentColor" />
    </LineIcon>
  );
}

export function TrashLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M4 7h16M10 11v6M14 11v6" />
      <path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" />
      <path d="M9 7V4h6v3" />
    </LineIcon>
  );
}

/** Duplicate: one page over another. */
export function DuplicateLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <rect x="8" y="8" width="12" height="12" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </LineIcon>
  );
}

export function PencilLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16z" />
      <path d="M13.5 6.5l4 4" />
    </LineIcon>
  );
}

/** Merging one thing into another: two paths joining into one. */
export function MergeLineIcon(props: IconProps) {
  return (
    <LineIcon {...props}>
      <path d="M6 4v3.5l6 6 6-6V4M12 13.5V20" />
    </LineIcon>
  );
}

/** Applied automatically: a small four-pointed spark, drawn at 10–12px beside a Tag's name. */
export function SparkLineIcon(props: IconProps) {
  return (
    <LineIcon strokeWidth={2} {...props}>
      <path d="M12 3c.6 4.6 3.4 7.4 8 8-4.6.6-7.4 3.4-8 8-.6-4.6-3.4-7.4-8-8 4.6-.6 7.4-3.4 8-8z" />
    </LineIcon>
  );
}

export function CheckLineIcon(props: IconProps) {
  return (
    <LineIcon strokeWidth={2.25} {...props}>
      <path d="m5 12.5 4.5 4.5L19 7.5" />
    </LineIcon>
  );
}

/** The app's mark in the sidebar header: a serif "I" on a 16px ink square. */
export function AppMark() {
  return (
    <span
      aria-hidden="true"
      className="flex size-4 shrink-0 items-center justify-center rounded-sm bg-ink font-serif text-[12px] leading-none font-bold text-white"
    >
      I
    </span>
  );
}
