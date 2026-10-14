const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;

/**
 * How idle Checkpoints thin out as they age: every one for a day, then one an
 * hour until a week old, one a day until 90 days old, then one a week. In each
 * hour, day or week (counted in UTC from the epoch) the newest stays: the Mind
 * as it was when that stretch ended.
 */
const STEPS: readonly { olderThan: number; every: number }[] = [
  { olderThan: 90 * DAY, every: 7 * DAY },
  { olderThan: 7 * DAY, every: DAY },
  { olderThan: DAY, every: HOUR },
];

/** The idle Checkpoints to thin out now, of those given: their ids. */
export function idleToThin(
  idle: readonly { id: string; createdAt: string }[],
  now: Date,
): string[] {
  const newestFirst = [...idle].sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
  const kept = new Set<string>();
  const thin: string[] = [];
  for (const checkpoint of newestFirst) {
    const at = Date.parse(checkpoint.createdAt);
    const step = STEPS.find((each) => now.getTime() - at >= each.olderThan);
    if (!step) continue;
    const stretch = `${step.every}:${Math.floor(at / step.every)}`;
    if (kept.has(stretch)) thin.push(checkpoint.id);
    else kept.add(stretch);
  }
  return thin;
}
