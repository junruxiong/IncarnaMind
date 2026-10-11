/**
 * An export's Citations in the Mind's citation style (#289): Citations side by side are one
 * cluster, as the editor shows them; each cluster is formatted, in the text or as a note, and
 * the Bibliography is added at the end. Numbered, IncarnaMind's own style, keeps its footnotes
 * and is never passed through here.
 */
import type { CitationStyle } from "../../shared/citationStyles";
import type { CitationFormatRequest, CitedWork, FormattedCitations, StyledRun } from "../api";
import { type Block, type Inline, trimTrailingSpace } from "../mindText";

/** Every run of inline content in the Blocks, in order, mapped by `map`. */
function mapInlines(blocks: readonly Block[], map: (content: Inline[]) => Inline[]): Block[] {
  const visit = (block: Block): Block => {
    switch (block.kind) {
      case "paragraph":
      case "heading":
      case "question":
        return { ...block, content: map(block.content) };
      case "list":
        return { ...block, items: block.items.map((item) => item.map(visit)) };
      case "quote":
        return { ...block, content: block.content.map(visit) };
      case "table":
        return {
          ...block,
          rows: block.rows.map((row) =>
            row.map((cell) => ({ ...cell, content: cell.content.map(visit) })),
          ),
        };
      default:
        return block;
    }
  };
  return blocks.map(visit);
}

/** Where a cluster starts in its content, how far it reaches, and its Citations. */
interface Cluster {
  start: number;
  /** One past its last Citation. */
  end: number;
  citations: Extract<Inline, { kind: "citation" }>[];
}

const isBlank = (inline: Inline | undefined) =>
  inline?.kind === "text" && inline.text.trim() === "" && !inline.text.includes("\n");

/** The clusters in one run of content: Citations with nothing but spaces between them. */
function clustersIn(content: readonly Inline[]): Cluster[] {
  const found: Cluster[] = [];
  let index = 0;
  while (index < content.length) {
    const first = content[index];
    if (first?.kind !== "citation") {
      index++;
      continue;
    }
    const cluster: Cluster = { start: index, end: index + 1, citations: [first] };
    for (;;) {
      const next = content[cluster.end];
      if (next?.kind === "citation") {
        cluster.citations.push(next);
        cluster.end++;
      } else if (isBlank(next) && content[cluster.end + 1]?.kind === "citation") {
        cluster.end++;
      } else {
        break;
      }
    }
    found.push(cluster);
    index = cluster.end;
  }
  return found;
}

const sameFormat = (a: StyledRun, b: StyledRun) =>
  a.italic === b.italic &&
  a.bold === b.bold &&
  a.smallCaps === b.smallCaps &&
  a.underline === b.underline &&
  a.script === b.script &&
  a.link === b.link;

/** A cluster's runs without the work each belongs to, neighbours formatted alike joined. */
function runsOf(runs: FormattedCitations["clusters"][number]["runs"]): StyledRun[] {
  const joined: StyledRun[] = [];
  for (const { work: _, ...run } of runs) {
    const last = joined.at(-1);
    if (last && sameFormat(last, run))
      joined[joined.length - 1] = { ...last, text: last.text + run.text };
    else joined.push(run);
  }
  return joined;
}

/**
 * The Blocks with their Citations in `style`, and its Bibliography at the end: `format` is the
 * core's `formatCitations`. Blocks with no Citation come back as they were.
 */
export async function styleCitations(
  blocks: readonly Block[],
  style: CitationStyle,
  format: (request: CitationFormatRequest) => Promise<FormattedCitations>,
): Promise<Block[]> {
  const works: CitedWork[][] = [];
  mapInlines(blocks, (content) => {
    for (const cluster of clustersIn(content)) {
      works.push(cluster.citations.map((citation) => citation.work));
    }
    return content;
  });
  if (works.length === 0) return [...blocks];

  const formatted = await format({
    style: style.id,
    clusters: works.map((each) => ({ works: each })),
  });
  let next = 0;
  const styled = mapInlines(blocks, (content) => {
    const clusters = clustersIn(content);
    if (clusters.length === 0) return content;
    const out: Inline[] = [];
    let at = 0;
    for (const cluster of clusters) {
      out.push(...content.slice(at, cluster.start));
      const runs = runsOf(formatted.clusters[next]?.runs ?? []);
      next++;
      const note = formatted.form === "note";
      // A note's number, or a raised number, goes right after the word, as in print.
      if (note || runs[0]?.script === "sup") trimTrailingSpace(out);
      out.push({
        kind: "cite",
        runs,
        note,
        unverified: cluster.citations.some((citation) => citation.footnote.unverified),
      });
      at = cluster.end;
    }
    out.push(...content.slice(at));
    return out;
  });
  if (formatted.bibliography.length === 0 || style.heading === null) return styled;
  return [
    ...styled,
    {
      kind: "bibliography",
      heading: style.heading,
      items: formatted.bibliography.map((entry) => ({ label: entry.label, runs: entry.runs })),
      hangingIndent: formatted.hangingIndent,
    },
  ];
}
