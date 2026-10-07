import type { SVGProps } from "react";
import type { DocumentKind } from "../../../core/api";

type IconProps = SVGProps<SVGSVGElement>;

export function LogoIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" {...props}>
      <defs>
        <linearGradient id="incarnamind-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0.1" stopColor="#6366f1" />
          <stop offset="0.45" stopColor="#0ea5e9" />
          <stop offset="0.9" stopColor="#10b981" />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="15" fill="url(#incarnamind-logo)" />
      <path
        d="M10.5 21.5v-11M16 21.5v-6.5M21.5 21.5v-9"
        stroke="#fff"
        strokeWidth="2.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** A notebook, in the old app's flat colour style. */
export function MindIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <rect x="2.5" y="1.5" width="11" height="13" rx="2" fill="#bfdbfe" stroke="#3b82f6" />
      <path d="M5.5 5h5M5.5 8h5M5.5 11h3" stroke="#3b82f6" strokeLinecap="round" />
    </svg>
  );
}

/** A page with a folded corner: red for PDFs, indigo for text and Markdown. */
export function DocumentIcon({ kind, ...props }: IconProps & { kind: DocumentKind }) {
  const [fill, stroke] = kind === "pdf" ? ["#fee2e2", "#ef4444"] : ["#e0e7ff", "#6366f1"];
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <path
        d="M4.5 1.5h5l3 3v9a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1z"
        fill={fill}
        stroke={stroke}
        strokeLinejoin="round"
      />
      <path d="M9.5 1.5v3h3" fill="none" stroke={stroke} strokeLinejoin="round" />
      <path d="M6 8.5h4M6 11h4" stroke={stroke} strokeLinecap="round" />
    </svg>
  );
}

export function PencilIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path
        d="M10.5 2.5l3 3L6 13H3v-3z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function PlusIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M2.5 4h11M6.5 4V2.5h3V4M4 4l.7 9.1a1 1 0 0 0 1 .9h4.6a1 1 0 0 0 1-.9L12 4M6.8 6.5v5M9.2 6.5v5" />
    </svg>
  );
}

/** An arrow out of a tray: exporting a Mind to a file. */
export function ExportIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M8 10V2.5M5 5.5l3-3 3 3M3 9.5v3a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-3" />
    </svg>
  );
}

/** Six dots: the handle a Block is dragged by, like the old editor's "holder". */
export function GripIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" {...props}>
      <circle cx="5.5" cy="3.5" r="1.25" />
      <circle cx="10.5" cy="3.5" r="1.25" />
      <circle cx="5.5" cy="8" r="1.25" />
      <circle cx="10.5" cy="8" r="1.25" />
      <circle cx="5.5" cy="12.5" r="1.25" />
      <circle cx="10.5" cy="12.5" r="1.25" />
    </svg>
  );
}

export function MinusIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path d="M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function Chevron({ d, ...props }: IconProps & { d: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path
        d={d}
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export const ChevronUpIcon = (props: IconProps) => <Chevron d="M4 10l4-4 4 4" {...props} />;

export const ChevronDownIcon = (props: IconProps) => <Chevron d="M4 6l4 4 4-4" {...props} />;

/** Two arrows pointing out to the sides: fit to width. */
export function FitWidthIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M2 3v10M14 3v10M5 8h6M6.5 6L5 8l1.5 2M9.5 6L11 8l-1.5 2" />
    </svg>
  );
}

export function SettingsIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M4 7h9M17 7h3M4 17h3M11 17h9" />
      <circle cx="15" cy="7" r="2" />
      <circle cx="9" cy="17" r="2" />
    </svg>
  );
}

/** GitHub's mark, from Primer Octicons (MIT). */
export function GitHubIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" />
    </svg>
  );
}

/** A folder, in the same flat colour style as the Mind and Document icons. */
export function FolderIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <path
        d="M1.5 4a1 1 0 0 1 1-1h3.4l1.5 1.5h6.1a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1z"
        fill="#fef3c7"
        stroke="#f59e0b"
        strokeLinejoin="round"
      />
      <path d="M1.5 6.5h13" stroke="#f59e0b" />
    </svg>
  );
}

/** A stack of pages: every Document, whatever its Folder. */
export function AllDocumentsIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <rect x="4.5" y="1.5" width="9" height="11" rx="1" fill="#f3f4f6" stroke="#9ca3af" />
      <rect x="2.5" y="3.5" width="9" height="11" rx="1" fill="#e0e7ff" stroke="#6366f1" />
      <path d="M5 7.5h4M5 10h4" stroke="#6366f1" strokeLinecap="round" />
    </svg>
  );
}

export function FolderPlusIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M14.5 8V5.5a1 1 0 0 0-1-1H7.4L5.9 3H2.5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h5.5" />
      <path d="M12.5 10v4M10.5 12h4" />
    </svg>
  );
}

/** A folder with an arrow into it: "Move to…". */
export function MoveToFolderIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M1.5 4a1 1 0 0 1 1-1h3.4l1.5 1.5h6.1a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1z" />
      <path d="M5.5 9h5M8.5 7l2 2-2 2" />
    </svg>
  );
}

/** A label tag with a hole: Tags. */
export function TagIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M2.5 3.5a1 1 0 0 1 1-1h3.6l6.2 6.2a1 1 0 0 1 0 1.4l-3.2 3.2a1 1 0 0 1-1.4 0L2.5 7.1z" />
      <circle cx="5.5" cy="5.5" r="1" />
    </svg>
  );
}

/** Points right; rotate it 90° to point down. */
export function ChevronIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path
        d="M6 4l4 4-4 4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function CheckIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path
        d="M3.5 8.5l3 3 6-7"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** A triangle pointing right: ask a Question, like the old editor's run button. */
export function AskIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M5.5 3.6v8.8a.6.6 0 0 0 .9.5l6.6-4.4a.6.6 0 0 0 0-1L6.4 3.1a.6.6 0 0 0-.9.5z" />
    </svg>
  );
}

/** A rounded square: stop writing. */
export function StopIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" {...props}>
      <rect x="4" y="4" width="8" height="8" rx="1.5" />
    </svg>
  );
}

/** A circular arrow: write it again. */
export function RegenerateIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M13 8a5 5 0 1 1-1.5-3.6M13 2.5v2.4h-2.4" />
    </svg>
  );
}

/** A speech bubble with a question mark: a Question, in menus. */
export function QuestionIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M8 13.5c3.3 0 6-2.3 6-5.2S11.3 3 8 3 2 5.4 2 8.3c0 1.3.5 2.4 1.4 3.3L3 14l2.6-1a6.8 6.8 0 0 0 2.4.5z" />
      <path d="M6.6 6.9a1.5 1.5 0 1 1 2 1.4c-.4.2-.6.5-.6.9M8 10.6v.1" />
    </svg>
  );
}

/** A magnifying glass: a search of the Documents. */
export function SearchIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
      {...props}
    >
      <circle cx="7" cy="7" r="4.25" />
      <path d="M10.2 10.2l3.3 3.3" />
    </svg>
  );
}

/** A plug: a Connector, and the calls an Answer makes through one. */
export function PlugIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M6 2.5v3M10 2.5v3" />
      <path d="M4.25 5.5h7.5v2a3.75 3.75 0 0 1-7.5 0z" />
      <path d="M8 11.25v2.25" />
    </svg>
  );
}

/** A circle with a tick: the quote was found on the page. */
export function QuoteFoundIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden="true" {...props}>
      <circle cx="8" cy="8" r="6.25" strokeWidth="1.5" />
      <path
        d="M5.3 8.2l1.9 1.9 3.6-4"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** A circle with a cross: the quote wasn't found on the page. */
export function QuoteNotFoundIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden="true" {...props}>
      <circle cx="8" cy="8" r="6.25" strokeWidth="1.5" />
      <path d="M5.9 5.9l4.2 4.2M10.1 5.9l-4.2 4.2" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** A dashed circle with a dash: the quote can't be checked. */
export function CantCheckIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden="true" {...props}>
      <circle cx="8" cy="8" r="6.25" strokeWidth="1.5" strokeDasharray="2.6 1.8" />
      <path d="M5.5 8h5" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** A clock: the quote is checked once the Answer is finished. */
export function CheckingIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden="true" {...props}>
      <circle cx="8" cy="8" r="6.25" strokeWidth="1.5" />
      <path d="M8 4.8V8l2.2 1.4" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** A Skill: a four-pointed spark, for packaged know-how. */
export function SkillIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M8 1.75c.45 2.9 1.35 3.8 4.25 4.25C9.35 6.45 8.45 7.35 8 10.25 7.55 7.35 6.65 6.45 3.75 6 6.65 5.55 7.55 4.65 8 1.75z" />
      <path d="M12.25 10.5c.2 1.15.6 1.55 1.75 1.75-1.15.2-1.55.6-1.75 1.75-.2-1.15-.6-1.55-1.75-1.75 1.15-.2 1.55-.6 1.75-1.75z" />
    </svg>
  );
}
