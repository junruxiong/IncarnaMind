import type { ReactNode } from "react";
import type { StyledRun } from "../../../core/api";
import { isWebLink } from "./styledText";

/**
 * One run of text a citation style formatted, as React elements: italics, bold, small
 * capitals, underlining, raised and lowered text, and a web address as a link that opens in
 * the browser. The text is only ever text (ADR-0016): nothing in it is read as markup.
 */
export function StyledRunView({ run }: { run: StyledRun }): ReactNode {
  let node: ReactNode = run.text;
  if (run.script === "sup") node = <sup>{node}</sup>;
  if (run.script === "sub") node = <sub>{node}</sub>;
  if (run.underline) node = <u>{node}</u>;
  if (run.smallCaps) node = <span className="[font-variant:small-caps]">{node}</span>;
  if (run.bold) node = <b>{node}</b>;
  if (run.italic) node = <i>{node}</i>;
  if (run.link && isWebLink(run.link)) {
    node = (
      <a href={run.link} target="_blank" rel="noreferrer" className="references-link">
        {node}
      </a>
    );
  }
  return node;
}

/** Runs, one after another. */
export function StyledRuns({ runs }: { runs: readonly StyledRun[] }) {
  return runs.map((run, index) => (
    // biome-ignore lint/suspicious/noArrayIndexKey: runs have no identity but their place.
    <StyledRunView key={index} run={run} />
  ));
}
