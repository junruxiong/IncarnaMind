/**
 * The grouping check's timings (R1, R2, O5) at a small size: the check runs
 * them at 5,000 Documents and 100,000 Passages; here they only have to work.
 */
import { describe, expect, test } from "vitest";
import {
  MAIN_THREAD_BAR_MS,
  median,
  syntheticVectors,
  timeKMeans,
  timeMeans,
} from "../../eval/grouping/lib/timing";
import { dot, topicCount } from "../../src/core/topics/grouping";

describe("the grouping check's timings", () => {
  test("generated vectors are unit vectors, the same for a seed", () => {
    const vectors = syntheticVectors(50, 8, 5, 3);
    for (const vector of vectors) expect(dot(vector, vector)).toBeCloseTo(1, 5);
    expect(syntheticVectors(50, 8, 5, 3)).toEqual(vectors);
  });

  test("k-means is timed as the grouping worker runs it: k from the count, 3 runs", () => {
    const timing = timeKMeans(300, 16);
    expect(timing).toMatchObject({ documents: 300, dimensions: 16, k: topicCount(300) });
    expect(timing.runs).toHaveLength(3);
    expect(timing.seconds).toBeGreaterThan(0);
  });

  test("the means are timed cold (loading the index) and warm, on a generated database", async () => {
    const timing = await timeMeans({
      passages: 2_000,
      documents: 40,
      dimensions: 16,
      textLength: 200,
      repeats: 2,
    });
    expect(timing).toMatchObject({ passages: 2_000, documents: 40, barMs: MAIN_THREAD_BAR_MS });
    expect(timing.coldMs).toHaveLength(2);
    expect(timing.warmMs).toHaveLength(2);
    expect(timing.databaseMb).toBeGreaterThan(0);
    expect(timing.passes).toBe(median(timing.coldMs) <= MAIN_THREAD_BAR_MS);
  });

  test("median", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
  });
});
