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

export function TrashIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" {...props}>
      <path
        d="M2.5 4.5h11M6.5 4.5v-2h3v2M4 4.5l.7 9h6.6l.7-9"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
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
