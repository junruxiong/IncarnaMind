import type { WordBorder, WordCellLook, WordLook, WordTableLook } from "../../../core/api";
import { border, pt } from "./wordLookCss";
import { WORD_LOOK_DOM_ATTRIBUTE } from "./wordSchema";

const SIDES = ["top", "right", "bottom", "left"] as const;
const WORD_TABLE = `table[${WORD_LOOK_DOM_ATTRIBUTE}]`;
/** An inch, for a column the file gives no width, such as one the User adds. */
const UNSIZED_COLUMN = 72;
/** Word's cell margins where neither the table nor its style sets them. */
const CELL_MARGIN = 5.4;
/** Half Word's widest border. */
const BORDER_ROOM = 3;

const attribute = (key: string) => `[${WORD_LOOK_DOM_ATTRIBUTE}="${key}"]`;

const rule = (selector: string, declarations: string[]) =>
  declarations.length > 0 ? `${selector} { ${declarations.join("; ")}; }` : "";

// "hidden" wins over the table's line on that edge, as a cell's "none" does in Word.
const line = (value: WordBorder) => (value.style === "nil" ? "hidden" : border(value));

const padding = (margins: WordTableLook["margins"]) =>
  SIDES.flatMap((side) => {
    const value = margins?.[side];
    return value === undefined ? [] : [`padding-${side}: ${pt(value)}`];
  });

function width(table: WordTableLook, textWidth: number | null): string {
  if (table.width?.unit === "percent") {
    return textWidth ? pt((textWidth * table.width.value) / 100) : `${table.width.value}%`;
  }
  const columns = table.columns?.reduce((sum, each) => sum + each, 0);
  if (columns) return pt(columns);
  return table.width ? pt(table.width.value) : "auto";
}

function placement(table: WordTableLook): string[] {
  if (table.align === "center") return ["margin-left: auto", "margin-right: auto"];
  if (table.align === "right") return ["margin-left: auto"];
  return table.indent && table.indent > 0 ? [`margin-left: ${pt(table.indent)}`] : [];
}

function tableRules(scope: string, key: string, table: WordTableLook, textWidth: number | null) {
  const selector = `${scope} table${attribute(key)}`;
  const cells = (rows: string, cell: string) =>
    `${scope} :where(table${attribute(key)} > tbody > ${rows} > ${cell})`;
  const outer = table.borders ?? {};
  const across = table.borders?.insideH;
  const down = table.borders?.insideV;
  return [
    rule(selector, [
      `width: ${width(table, textWidth)}`,
      ...placement(table),
      ...SIDES.flatMap((side) => {
        const value = outer[side];
        return value ? [`border-${side}: ${border(value)}`] : [];
      }),
    ]),
    rule(cells("tr", "*"), padding(table.margins)),
    across ? rule(cells("tr + tr", "*"), [`border-top: ${border(across)}`]) : "",
    across ? rule(cells("tr:not(:last-child)", "*"), [`border-bottom: ${border(across)}`]) : "",
    down ? rule(cells("tr", "* + *"), [`border-left: ${border(down)}`]) : "",
    down ? rule(cells("tr", ":not(:last-child)"), [`border-right: ${border(down)}`]) : "",
    ...(table.columns ?? []).map(
      (column, index) =>
        `${selector} > colgroup > col:nth-child(${index + 1}) { width: ${pt(column)}; }`,
    ),
  ];
}

function cellRule(scope: string, key: string, cell: WordCellLook): string {
  return rule(
    `${scope} :is(td, th)${attribute(key)}`,
    [
      cell.shade ? `background-color: #${cell.shade}` : "",
      cell.vertical === "center" ? "vertical-align: middle" : "",
      cell.vertical === "bottom" ? "vertical-align: bottom" : "",
      cell.noWrap ? "white-space: nowrap" : "",
      ...padding(cell.margins),
      ...SIDES.flatMap((side) => {
        const value = cell.borders?.[side];
        return value ? [`border-${side}: ${line(value)}`] : [];
      }),
    ].filter(Boolean),
  );
}

/** Only the file's lines, collapsed as in Word: insideH and insideV on the cells between, a cell's own sides over them. */
export function wordTableCss(scope: string, look: WordLook): string[] {
  if (Object.keys(look.tables).length === 0) return [];
  return [
    // Room for the outer lines' halves outside the table, which the scrolling frame would cut.
    `${scope} :where(.tableWrapper:has(> ${WORD_TABLE})) { margin: 0 calc(-1 * ${pt(BORDER_ROOM)}); padding: 0 ${pt(BORDER_ROOM)} 1px; }`,
    `${scope} :where(${WORD_TABLE}) { width: auto; border: 0; border-radius: 0; border-collapse: collapse; table-layout: fixed; }`,
    `${scope} :where(${WORD_TABLE} > colgroup > col) { width: ${pt(UNSIZED_COLUMN)}; }`,
    `${scope} :where(${WORD_TABLE} > tbody > tr > :is(td, th)) { min-width: 0; padding: 0 ${pt(CELL_MARGIN)}; border: 0; border-radius: 0; vertical-align: top; background: none; font: inherit; color: inherit; text-align: inherit; }`,
    `${scope} :where(${WORD_TABLE} > tbody > tr > * > * + *) { margin-top: 0; }`,
    ...Object.entries(look.tables).flatMap(([key, table]) =>
      tableRules(scope, key, table, look.width),
    ),
    ...Object.entries(look.cells).map(([key, cell]) => cellRule(scope, key, cell)),
  ].filter(Boolean);
}
