/**
 * Pass rates of the everyday requests, asked a few times each since a real
 * model varies: per request and model, held to the request's floor, and per
 * language and model.
 */

export interface RequestRun {
  id: string;
  language: "en" | "zh-CN";
  model: string;
  passed: boolean;
}

export interface RowRate {
  id: string;
  language: "en" | "zh-CN";
  model: string;
  passed: number;
  runs: number;
  floor: number;
  under: boolean;
}

/** One row per request and model, in the order first run; `floors` by request id. */
export function rowRates(
  runs: readonly RequestRun[],
  floors: ReadonlyMap<string, number>,
): RowRate[] {
  const rows = new Map<string, RowRate>();
  for (const run of runs) {
    const key = `${run.id}\n${run.model}`;
    const row = rows.get(key) ?? {
      id: run.id,
      language: run.language,
      model: run.model,
      passed: 0,
      runs: 0,
      floor: floors.get(run.id) ?? 1,
      under: false,
    };
    row.runs++;
    if (run.passed) row.passed++;
    row.under = row.passed / row.runs < row.floor;
    rows.set(key, row);
  }
  return [...rows.values()];
}

/** How many runs passed, per model and language. */
export function languageRates(
  runs: readonly RequestRun[],
): { model: string; language: "en" | "zh-CN"; passed: number; runs: number }[] {
  const groups = new Map<
    string,
    { model: string; language: "en" | "zh-CN"; passed: number; runs: number }
  >();
  for (const run of runs) {
    const key = `${run.model}\n${run.language}`;
    const group = groups.get(key) ?? {
      model: run.model,
      language: run.language,
      passed: 0,
      runs: 0,
    };
    group.runs++;
    if (run.passed) group.passed++;
    groups.set(key, group);
  }
  return [...groups.values()];
}
