import type { ReactNode } from "react";

/** Stands in for an element in a translated message, which is split around it. */
export const SLOT = "";

/** A translated message with an element where its `{name}` placeholder was, filled with `SLOT`. */
export function Slotted({ message, slot }: { message: string; slot: ReactNode }) {
  const [before, after] = message.split(SLOT);
  return (
    <>
      {before}
      {slot}
      {after}
    </>
  );
}
