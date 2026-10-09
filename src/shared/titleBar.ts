/**
 * The window's title bar. The app draws the top band itself (the sidebar's
 * header, the Mind tabs, the Library's header and the viewer's toolbar), and
 * the system draws only its window buttons in it, as Notion, Linear, Slack
 * and VS Code do:
 * - "inset" (macOS): the traffic lights, in the sidebar's header;
 * - "overlay" (Windows): minimise, maximise and close, over the band's right end;
 * - "native" (Linux): the system's own title bar above the band, as window
 *   managers differ too much to draw it in the page.
 *
 * The window's options are in src/main/titleBar.ts; the room the page keeps
 * for the buttons, and the band's dragging, in styles.css ("The title bar").
 */
export type TitleBar = "inset" | "overlay" | "native";

export function titleBarOf(platform: string): TitleBar {
  if (platform === "darwin") return "inset";
  if (platform === "win32") return "overlay";
  return "native";
}
