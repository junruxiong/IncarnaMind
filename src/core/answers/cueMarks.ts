import { clockTime } from "../../shared/clock";
import { parseCueTarget } from "../../shared/units";
import type { PageText } from "../documents/passages";

/**
 * A Transcript's Passage with each cue's time before it ("[14:26] …"), so
 * the model can say when something was said (#136). Each cue is a line of its
 * own, found by its start, in order. Any other Passage is returned as it is.
 */
export function withCueMarks(text: string, units: readonly PageText[]): string {
  let marked = "";
  let from = 0;
  for (const unit of units) {
    if (unit.kind !== "cues") continue;
    for (const anchor of unit.anchors ?? []) {
      const cue = parseCueTarget(anchor.target);
      const head = unit.text.slice(anchor.start, anchor.end).trim().slice(0, 200);
      if (!cue || !head) continue;
      let at = -1;
      for (let line = from; line >= 0 && line < text.length; ) {
        const rest = text.slice(line, line + head.length);
        if (rest.length > 0 && head.startsWith(rest)) {
          at = line;
          break;
        }
        const next = text.indexOf("\n", line);
        line = next < 0 ? -1 : next + 1;
      }
      if (at < 0) continue;
      marked += `${text.slice(from, at)}[${clockTime(cue.startMs)}] `;
      from = at;
    }
  }
  return marked + text.slice(from);
}
