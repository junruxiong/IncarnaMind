import type { ChartKind } from "../../../shared/charts";

/**
 * What a chart on screen can do, for its right-click menu and its commands
 * (#436): only the chart's own view knows whether it shows its table and
 * can open its title to rename it in place.
 */
export interface ChartHandle {
  kind(): ChartKind;
  /** Whether it can be drawn as `kind`: a scatter chart needs two columns of numbers. */
  fits(kind: ChartKind): boolean;
  setKind(kind: ChartKind): void;
  /** Shows its values as a table. */
  asTable(): boolean;
  toggleTable(): void;
  /** Its table changed since it was made: it can draw the table as it is now. */
  canUpdate(): boolean;
  update(): void;
  rename(): void;
  remove(): void;
}

/** The charts on screen, by their Block's id. */
export const chartHandles = new Map<string, ChartHandle>();
