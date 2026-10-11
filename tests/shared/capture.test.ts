import { describe, expect, test } from "vitest";
import { toPcm16 } from "../../src/shared/capture";

describe("The capture window's samples", () => {
  test("become 16-bit PCM, full scale both ways, clipped beyond it", () => {
    const pcm = toPcm16(new Float32Array([0, 0.5, -0.5, 1, -1, 1.5, -2]));
    expect([...pcm]).toEqual([0, 16384, -16384, 32767, -32768, 32767, -32768]);
  });
});
