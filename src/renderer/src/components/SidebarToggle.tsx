import { createContext, useContext, useEffect, useRef } from "react";
import { files } from "../core";
import { useT } from "../i18n";
import { SidebarLineIcon } from "./lineIcons";
import { iconButtonClass } from "./ui";

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/** The shortcut as this system writes it. */
export const SIDEBAR_SHORTCUT = isMac ? "⌘\\" : "Ctrl+\\";

/** What the window tells the sidebar's toggle buttons (see `Workspace` in App.tsx). */
export interface SidebarControl {
  /** The sidebar is out of the way (the User's choice, or a window too narrow). */
  hidden: boolean;
  /** Hidden, but slid out over the content for now. */
  peeking: boolean;
  toggle(): void;
  /** True once if the button the User used has moved to another place, so it takes the focus again. */
  takeRefocus(): boolean;
}

export const SidebarControlContext = createContext<SidebarControl | null>(null);

/**
 * The sidebar button (DESIGN.md, title bar). On macOS it sits in the
 * sidebar's header beside the traffic lights, and in the card's tab band, in
 * the same place, while the sidebar is hidden; on Windows and Linux it is at
 * the start of the tab strip. Only one of the two is ever drawn.
 */
export function SidebarToggle({ place }: { place: "header" | "band" }) {
  const t = useT();
  const control = useContext(SidebarControlContext);
  const button = useRef<HTMLButtonElement>(null);
  const takeRefocus = control?.takeRefocus;
  const inHeader =
    files.titleBar === "inset" && control !== null && (!control.hidden || control.peeking);
  const here = place === "header" ? inHeader : !inHeader;

  // The button the User used was replaced by its twin in the other place: focus stays on a button.
  useEffect(() => {
    if (here && takeRefocus?.()) button.current?.focus();
  }, [here, takeRefocus]);

  if (!control || !here) return null;
  const label = control.hidden ? t("sidebar.show") : t("sidebar.hide");
  return (
    <button
      ref={button}
      type="button"
      data-testid="sidebar-toggle"
      aria-label={label}
      aria-expanded={!control.hidden}
      aria-controls="sidebar"
      title={`${label} (${SIDEBAR_SHORTCUT})`}
      onClick={control.toggle}
      className={`${iconButtonClass} sidebar-toggle text-ink-secondary`}
    >
      <SidebarLineIcon className="size-4" />
    </button>
  );
}
