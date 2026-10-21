/**
 * Diarization error rate, as NIST's md-eval scores it: over the reference's speech time, the
 * share missed, falsely found, or given to the wrong speaker, with overlapping speech counted
 * for each speaker in it, after mapping hypothesis speakers to reference ones so that they
 * share the most time. Scored in 10 ms frames; a collar leaves out the time around each
 * reference boundary.
 */

export interface Turn {
  startMs: number;
  endMs: number;
  speaker: string;
}

export interface DiarizationError {
  /** (missed + false alarm + confusion) / reference speech. */
  der: number;
  missedMs: number;
  falseAlarmMs: number;
  confusionMs: number;
  /** Reference speech, overlap counted once per speaker. */
  speechMs: number;
  referenceSpeakers: number;
  hypothesisSpeakers: number;
}

const FRAME_MS = 10;

/** Reads RTTM lines: `SPEAKER <file> 1 <start s> <duration s> <NA> <NA> <speaker> …`. */
export function readRttm(text: string): Turn[] {
  return text
    .split("\n")
    .map((line) => line.trim().split(/\s+/))
    .filter((fields) => fields[0] === "SPEAKER" && fields.length >= 8)
    .map((fields) => {
      const startMs = Number(fields[3]) * 1000;
      return {
        startMs,
        endMs: startMs + Number(fields[4]) * 1000,
        speaker: fields[7] as string,
      };
    });
}

/** Who speaks in each frame, as speaker indexes. */
function framesOf(turns: readonly Turn[], speakers: readonly string[], frames: number): number[][] {
  const index = new Map(speakers.map((speaker, at) => [speaker, at]));
  const out: number[][] = Array.from({ length: frames }, () => []);
  for (const turn of turns) {
    const who = index.get(turn.speaker) as number;
    const from = Math.max(0, Math.round(turn.startMs / FRAME_MS));
    const to = Math.min(frames, Math.round(turn.endMs / FRAME_MS));
    for (let frame = from; frame < to; frame++) {
      const here = out[frame] as number[];
      if (!here.includes(who)) here.push(who);
    }
  }
  return out;
}

/** The assignment of rows to columns with the most total `gain` (Hungarian method); -1 for none. */
export function bestAssignment(gain: readonly (readonly number[])[]): number[] {
  const rows = gain.length;
  const columns = gain[0]?.length ?? 0;
  const n = Math.max(rows, columns);
  if (n === 0) return [];
  const most = Math.max(0, ...gain.flat());
  const cost = (row: number, column: number) =>
    row < rows && column < columns ? most - (gain[row]?.[column] ?? 0) : most;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const match = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let row = 1; row <= n; row++) {
    match[0] = row;
    let column0 = 0;
    const least = new Array<number>(n + 1).fill(Number.POSITIVE_INFINITY);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[column0] = true;
      const row0 = match[column0] as number;
      let delta = Number.POSITIVE_INFINITY;
      let column1 = 0;
      for (let column = 1; column <= n; column++) {
        if (used[column]) continue;
        const reduced = cost(row0 - 1, column - 1) - (u[row0] as number) - (v[column] as number);
        if (reduced < (least[column] as number)) {
          least[column] = reduced;
          way[column] = column0;
        }
        if ((least[column] as number) < delta) {
          delta = least[column] as number;
          column1 = column;
        }
      }
      for (let column = 0; column <= n; column++) {
        if (used[column]) {
          u[match[column] as number] = (u[match[column] as number] as number) + delta;
          v[column] = (v[column] as number) - delta;
        } else least[column] = (least[column] as number) - delta;
      }
      column0 = column1;
    } while (match[column0] !== 0);
    do {
      const column1 = way[column0] as number;
      match[column0] = match[column1] as number;
      column0 = column1;
    } while (column0 !== 0);
  }
  const assigned = new Array<number>(rows).fill(-1);
  for (let column = 1; column <= n; column++) {
    const row = (match[column] as number) - 1;
    if (row < rows && column - 1 < columns) assigned[row] = column - 1;
  }
  return assigned;
}

export function diarizationError(
  reference: readonly Turn[],
  hypothesis: readonly Turn[],
  { collarMs = 0 }: { collarMs?: number } = {},
): DiarizationError {
  const end = Math.max(
    0,
    ...reference.map((turn) => turn.endMs),
    ...hypothesis.map((turn) => turn.endMs),
  );
  const frames = Math.ceil(end / FRAME_MS);
  const referenceSpeakers = [...new Set(reference.map((turn) => turn.speaker))];
  const hypothesisSpeakers = [...new Set(hypothesis.map((turn) => turn.speaker))];
  const ref = framesOf(reference, referenceSpeakers, frames);
  const hyp = framesOf(hypothesis, hypothesisSpeakers, frames);
  const scored = new Array<boolean>(frames).fill(true);
  const collar = Math.round(collarMs / FRAME_MS);
  if (collar > 0) {
    for (const turn of reference) {
      for (const edge of [turn.startMs, turn.endMs]) {
        const at = Math.round(edge / FRAME_MS);
        for (let frame = Math.max(0, at - collar); frame < Math.min(frames, at + collar); frame++) {
          scored[frame] = false;
        }
      }
    }
  }
  const shared = referenceSpeakers.map(() => new Array<number>(hypothesisSpeakers.length).fill(0));
  for (let frame = 0; frame < frames; frame++) {
    if (!scored[frame]) continue;
    for (const r of ref[frame] as number[]) {
      for (const h of hyp[frame] as number[])
        (shared[r] as number[])[h] = ((shared[r] as number[])[h] as number) + 1;
    }
  }
  const mapped = bestAssignment(shared);
  let missed = 0;
  let falseAlarm = 0;
  let confusion = 0;
  let speech = 0;
  for (let frame = 0; frame < frames; frame++) {
    if (!scored[frame]) continue;
    const inRef = ref[frame] as number[];
    const inHyp = hyp[frame] as number[];
    const correct = inRef.filter((r) => inHyp.includes(mapped[r] as number)).length;
    speech += inRef.length;
    missed += Math.max(0, inRef.length - inHyp.length);
    falseAlarm += Math.max(0, inHyp.length - inRef.length);
    confusion += Math.min(inRef.length, inHyp.length) - correct;
  }
  return {
    der: speech === 0 ? 0 : (missed + falseAlarm + confusion) / speech,
    missedMs: missed * FRAME_MS,
    falseAlarmMs: falseAlarm * FRAME_MS,
    confusionMs: confusion * FRAME_MS,
    speechMs: speech * FRAME_MS,
    referenceSpeakers: referenceSpeakers.length,
    hypothesisSpeakers: hypothesisSpeakers.length,
  };
}
