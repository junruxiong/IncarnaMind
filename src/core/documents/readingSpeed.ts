/**
 * How fast this computer reads Documents (#1506), so "Link this folder?" can say how long
 * reading a folder takes here, and says nothing until files of its kinds have been read here.
 *
 * For each kind, a straight line through how long each Document took to read against its size:
 * its processing job, and its embedding when embeddings are on (not its wait for its turn).
 * Only a job on a worker that has run its kind before counts (see `ProcessorHandlers.onStart`),
 * and only a file read as text: scanned pages or pictures take as long as they are many, which
 * no size tells. The line is this computer's, kept as a device value (docs/agents/portability.md,
 * rule 5), and starts again when the embedding model changes, since embedding is part of reading.
 */
import type { DocumentKind } from "../api";
import { isRecord } from "../errors";

/** The device value it is kept in. */
export const READING_SPEED = "readingSpeed";

const MEGABYTE = 1_000_000;

/** Measurements are kept at most this often while Documents are read: each is a database write. */
const KEEP_EVERY_MS = 5_000;

/**
 * One kind's Documents read here: how many, their mean size in megabytes and mean time in
 * seconds, and the sums of the squares of their sizes' distances from the mean and of the
 * products of both distances (Welford's), from which the line is fitted.
 */
interface KindReadings {
  count: number;
  megabytes: number;
  seconds: number;
  sizeSquares: number;
  products: number;
}

export interface ReadingSpeed {
  /** The embedding model the Documents were read with; "" while embeddings are off. */
  embedding: string;
  kinds: Partial<Record<string, KindReadings>>;
}

const FIELDS = ["count", "megabytes", "seconds", "sizeSquares", "products"] as const;

const isKindReadings = (value: unknown): value is KindReadings =>
  isRecord(value) &&
  FIELDS.every((field) => Number.isFinite(value[field])) &&
  typeof value.count === "number" &&
  value.count >= 1;

/** What is kept, if it was read with this embedding model; nothing read otherwise. */
function speedFor(stored: unknown, embedding: string): ReadingSpeed {
  const kinds: ReadingSpeed["kinds"] = {};
  if (isRecord(stored) && stored.embedding === embedding && isRecord(stored.kinds)) {
    for (const [kind, readings] of Object.entries(stored.kinds)) {
      if (isKindReadings(readings)) kinds[kind] = readings;
    }
  }
  return { embedding, kinds };
}

function withReading(
  speed: ReadingSpeed,
  kind: DocumentKind,
  bytes: number,
  seconds: number,
): ReadingSpeed {
  const before = speed.kinds[kind] ?? {
    count: 0,
    megabytes: 0,
    seconds: 0,
    sizeSquares: 0,
    products: 0,
  };
  const count = before.count + 1;
  const size = bytes / MEGABYTE;
  const apart = size - before.megabytes;
  const megabytes = before.megabytes + apart / count;
  const mean = before.seconds + (seconds - before.seconds) / count;
  const readings: KindReadings = {
    count,
    megabytes,
    seconds: mean,
    sizeSquares: before.sizeSquares + apart * (size - megabytes),
    products: before.products + apart * (seconds - mean),
  };
  return { ...speed, kinds: { ...speed.kinds, [kind]: readings } };
}

/**
 * Seconds per file and per megabyte: the least-squares line, through the means, held to costing
 * no less than nothing per megabyte and per file. Files all of one size can't tell the two
 * apart: they cost their mean per file.
 */
function line({ megabytes, seconds, sizeSquares, products }: KindReadings) {
  const slope = sizeSquares > 0 ? Math.max(0, products / sizeSquares) : 0;
  const perMegabyte = megabytes > 0 ? Math.min(slope, seconds / megabytes) : 0;
  return { perFile: seconds - perMegabyte * megabytes, perMegabyte };
}

export interface ReadingMeterOptions {
  /** Milliseconds, from a clock that only moves forward. */
  clock(): number;
  /** The embedding model Documents are embedded with now; "" while embeddings are off. */
  embedding(): string;
  /** The device value, as kept, and keeping it. */
  read(): unknown;
  write(speed: ReadingSpeed): void;
  reportError(error: unknown): void;
}

export function createReadingMeter(options: ReadingMeterOptions) {
  /** When each Document's processing job started, for those whose time counts. */
  const started = new Map<string, number>();
  /** Documents processed, waiting to be embedded, with how long processing took. */
  const embedding = new Map<string, { kind: DocumentKind; bytes: number; ms: number }>();
  /** What is measured for the embedding model in use; `unkept` until written to the device value. */
  let measured: ReadingSpeed | null = null;
  let unkept = false;
  let keptAt = Number.NEGATIVE_INFINITY;

  const keep = () => {
    if (!unkept || !measured) return;
    unkept = false;
    keptAt = options.clock();
    try {
      options.write(measured);
    } catch (error) {
      options.reportError(error);
    }
  };

  const speed = (): ReadingSpeed => {
    const model = options.embedding();
    if (measured?.embedding !== model) {
      keep();
      measured = speedFor(options.read(), model);
    }
    return measured;
  };

  const add = (kind: DocumentKind, bytes: number, ms: number) => {
    measured = withReading(speed(), kind, bytes, ms / 1000);
    unkept = true;
    if (options.clock() - keptAt >= KEEP_EVERY_MS) keep();
  };

  return {
    /** A Document's processing job starts; `warm` as `ProcessorHandlers.onStart` gives it. */
    started(documentId: string, warm: boolean): void {
      if (warm) started.set(documentId, options.clock());
      else started.delete(documentId);
    },

    /**
     * Its job read its file's text, `bytes` of a `kind`. `ready`: it is read; otherwise it is
     * read once embedded, which takes its share of the time too.
     */
    processed(documentId: string, kind: DocumentKind, bytes: number, ready: boolean): void {
      const begun = started.get(documentId);
      started.delete(documentId);
      embedding.delete(documentId);
      if (begun === undefined) return;
      const ms = options.clock() - begun;
      if (ready) add(kind, bytes, ms);
      else embedding.set(documentId, { kind, bytes, ms });
    },

    /** It was embedded, in `ms`: it is read. */
    embedded(documentId: string, ms: number): void {
      const processed = embedding.get(documentId);
      embedding.delete(documentId);
      if (processed) add(processed.kind, processed.bytes, processed.ms + ms);
    },

    /**
     * How long reading these files takes here, in seconds. Null while a kind among them
     * hasn't been read here with the embedding model in use.
     */
    seconds(files: Iterable<{ kind: DocumentKind; size: number }>): number | null {
      const { kinds } = speed();
      let total = 0;
      for (const { kind, size } of files) {
        const readings = kinds[kind];
        if (!readings) return null;
        const { perFile, perMegabyte } = line(readings);
        total += perFile + (perMegabyte * size) / MEGABYTE;
      }
      return total;
    },

    /** Keeps what was measured since it was last kept. */
    close: keep,
  };
}
