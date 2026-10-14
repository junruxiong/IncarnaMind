import type { ChartKind } from "../../../shared/charts";

/** Only a chart's own view knows whether it shows its table, and can open its title to rename it. */
export interface ChartHandle {
  kind(): ChartKind;
  fits(kind: ChartKind): boolean;
  setKind(kind: ChartKind): void;
  asTable(): boolean;
  toggleTable(): void;
  canUpdate(): boolean;
  update(): void;
  rename(): void;
  remove(): void;
}

export const chartHandles = new Map<string, ChartHandle>();
