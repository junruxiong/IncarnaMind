import { create } from "zustand";

/** How long "Copied" shows before it goes by itself. */
export const COPIED_SHOWN_MS = 3000;

export const useCopied = create<{ shown: boolean }>()(() => ({ shown: false }));

let timer: ReturnType<typeof setTimeout> | undefined;

/** Puts text on the clipboard and says so, for a few seconds. */
export async function copyText(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
  useCopied.setState({ shown: true });
  clearTimeout(timer);
  timer = setTimeout(() => useCopied.setState({ shown: false }), COPIED_SHOWN_MS);
}
