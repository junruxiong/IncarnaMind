/** "Nice" steps for a chart's value axis from `min` to `max`: round numbers, as Office picks them. */
export function niceScale(
  min: number,
  max: number,
  steps = 5,
): { min: number; max: number; step: number } {
  if (max === min) {
    if (max === 0) return { min: 0, max: 1, step: 0.2 };
    return max > 0 ? niceScale(0, max, steps) : niceScale(min, 0, steps);
  }
  const raw = (max - min) / steps;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((each) => each * magnitude >= raw) ?? 10) * magnitude;
  return { min: Math.floor(min / step) * step, max: Math.ceil(max / step) * step, step };
}
