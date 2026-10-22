import { describe, expect, test } from "vitest";
import type { DocumentKind } from "../../src/core";
import { createReadingMeter } from "../../src/core/documents/readingSpeed";

/**
 * A meter on a clock only the test moves, keeping its device value as JSON, as SQLite does.
 * `read` reads a Document whose job starts on a warm worker and takes `ms`.
 */
function startMeter(kept?: string) {
  let time = 0;
  let embedding = "";
  let stored = kept;
  let writes = 0;
  let next = 0;
  const meter = createReadingMeter({
    clock: () => time,
    embedding: () => embedding,
    read: () => (stored === undefined ? undefined : JSON.parse(stored)),
    write: (speed) => {
      stored = JSON.stringify(speed);
      writes++;
    },
    reportError: (error) => {
      throw error;
    },
  });
  return {
    meter,
    kept: () => stored,
    writes: () => writes,
    advance(ms: number) {
      time += ms;
    },
    useEmbedding(model: string) {
      embedding = model;
    },
    read(kind: DocumentKind, bytes: number, ms: number) {
      const id = `document-${next++}`;
      meter.started(id, true);
      time += ms;
      meter.processed(id, kind, bytes, true);
    },
  };
}

const files = (count: number, kind: DocumentKind, size: number) =>
  Array.from({ length: count }, () => ({ kind, size }));

/** Small Markdown, text and CSV files, of 800 to 1,400 bytes. */
const smallFiles = (count: number) =>
  Array.from({ length: count }, (_, index) => ({
    kind: (["markdown", "text", "csv"] as const)[index % 3] ?? "text",
    size: 800 + (index % 7) * 100,
  }));

describe("How long reading takes on this computer (#1506)", () => {
  test("nothing is said until files of a kind have been read here, then it is what they took", () => {
    const { meter, read } = startMeter();
    expect(meter.seconds(smallFiles(2000))).toBeNull();

    // The issue's folder: 2,000 small files read in 25 s, between 9 and 16 ms each.
    for (const [index, file] of smallFiles(2000).entries()) {
      read(file.kind, file.size, 9 + (index % 8));
    }

    const again = meter.seconds(smallFiles(2000));
    expect(again).toBeGreaterThan(25 / 2);
    expect(again).toBeLessThan(25 * 2);
    // Fewer files, less time: twenty take a fraction of a second.
    expect(meter.seconds(smallFiles(20))).toBeLessThan(1);
    // A kind never read here: no time, rather than a guess.
    expect(meter.seconds([...smallFiles(20), { kind: "pdf", size: 2_000_000 }])).toBeNull();
  });

  test("a kind's time follows its files' sizes: so much per file, and so much per megabyte", () => {
    const { meter, read } = startMeter();
    // PDFs here take 40 ms each, and 2 s a megabyte.
    for (const size of [30_000, 120_000, 500_000, 2_000_000, 6_000_000, 15_000_000]) {
      read("pdf", size, 40 + (2 * size) / 1000);
    }

    expect(meter.seconds(files(500, "pdf", 100_000))).toBeCloseTo(500 * (0.04 + 0.2));
    expect(meter.seconds(files(3, "pdf", 40_000_000))).toBeCloseTo(3 * (0.04 + 80));
  });

  test("files all of one size can't tell what a file costs from what a megabyte does: each costs what it took", () => {
    const { meter, read } = startMeter();
    for (let index = 0; index < 3; index++) read("text", 4_000, 10);

    expect(meter.seconds(files(10, "text", 4_000))).toBeCloseTo(0.1);
    expect(meter.seconds(files(10, "text", 400_000))).toBeCloseTo(0.1);
  });

  test("a job on a worker new to its kind doesn't count: it starts the worker, or loads PDF.js", () => {
    const { meter, advance } = startMeter();
    meter.started("first", false);
    advance(400);
    meter.processed("first", "pdf", 50_000, true);
    expect(meter.seconds(files(1, "pdf", 50_000))).toBeNull();

    // Started again on a new worker (the last one crashed): only the new start counts.
    meter.started("again", true);
    advance(30);
    meter.started("again", false);
    advance(400);
    meter.processed("again", "pdf", 50_000, true);
    expect(meter.seconds(files(1, "pdf", 50_000))).toBeNull();
  });

  test("embedding is part of reading: a Document counts once embedded, without its wait for its turn", () => {
    const { meter, advance, useEmbedding } = startMeter();
    useEmbedding("multilingual-e5-small");
    meter.started("notes", true);
    advance(10);
    meter.processed("notes", "markdown", 2_000, false);
    expect(meter.seconds(files(1, "markdown", 2_000))).toBeNull();

    advance(60_000);
    meter.embedded("notes", 90);

    expect(meter.seconds(files(1, "markdown", 2_000))).toBeCloseTo(0.1);
  });

  test("turning embeddings on or off, or another model, starts again: reading takes another time", () => {
    const { meter, read, useEmbedding } = startMeter();
    read("markdown", 2_000, 10);
    expect(meter.seconds(files(1, "markdown", 2_000))).toBeCloseTo(0.01);

    useEmbedding("multilingual-e5-small");
    expect(meter.seconds(files(1, "markdown", 2_000))).toBeNull();
    read("markdown", 2_000, 100);
    expect(meter.seconds(files(1, "markdown", 2_000))).toBeCloseTo(0.1);

    useEmbedding("");
    expect(meter.seconds(files(1, "markdown", 2_000))).toBeNull();
  });

  test("it is kept on this computer, so a restart still knows; what it can't read counts as nothing read", () => {
    const first = startMeter();
    first.read("csv", 5_000, 20);
    // Kept at most every 5 s while Documents are read (each is a database write), and on closing.
    for (let index = 0; index < 100; index++) first.read("csv", 5_000, 20);
    expect(first.writes()).toBe(1);
    first.advance(5_000);
    first.read("csv", 5_000, 20);
    expect(first.writes()).toBe(2);
    first.read("csv", 5_000, 20);
    first.meter.close();
    expect(first.writes()).toBe(3);

    const restarted = startMeter(first.kept());
    expect(restarted.meter.seconds(files(3, "csv", 5_000))).toBeCloseTo(0.06);

    const unreadable = startMeter(
      JSON.stringify({ embedding: "", kinds: { csv: { count: "many" } } }),
    );
    expect(unreadable.meter.seconds(files(3, "csv", 5_000))).toBeNull();
  });
});
