import { useNotices } from "./notices";

/** Puts text on the clipboard and says so, for a few seconds. */
export async function copyText(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
  useNotices.getState().confirm("info.copied");
}
