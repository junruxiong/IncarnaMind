/**
 * A Citation's card opens when its marker, or its mark in the margin, is
 * hovered or focused. The mark lives outside the marker's node view, so it
 * tells the marker with this event, which the node view listens to.
 */
const PEEK = "citation-peek";

export interface Peek {
  /** Who asks: the card stays open while anyone does. */
  source: string;
  on: boolean;
}

export const peekCitation = (marker: HTMLElement, source: string, on: boolean) =>
  marker.dispatchEvent(new CustomEvent<Peek>(PEEK, { detail: { source, on } }));

/** Listens on a marker for those asking to show or hide its card. */
export function onPeek(marker: HTMLElement, listener: (peek: Peek) => void): () => void {
  const handle = (event: Event) => {
    if (event instanceof CustomEvent) listener(event.detail as Peek);
  };
  marker.addEventListener(PEEK, handle);
  return () => marker.removeEventListener(PEEK, handle);
}
