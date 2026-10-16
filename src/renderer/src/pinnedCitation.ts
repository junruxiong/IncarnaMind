let pinned: { mindId: string; at: number } | null = null;

/** The Citation whose card was open when the Document came to the front of the window, by its place in the Mind. */
export const pinCitation = (mindId: string, at: number): void => {
  pinned = { mindId, at };
};

export const unpinCitation = (): void => {
  pinned = null;
};

/** Whether the Mind's Citation at `at` had its card open when the User left; true once, as the Mind comes back. */
export function takePinnedCitation(mindId: string, at: number): boolean {
  const same = pinned?.mindId === mindId && pinned.at === at;
  if (same) pinned = null;
  return same;
}
