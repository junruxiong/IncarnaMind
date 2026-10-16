/**
 * The text recognition model in a process of its own: a picture that hangs it
 * is given up after the time limit, the process is stopped, and the next
 * picture is read by a new one.
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import {
  createChannelTextRecognizer,
  type TextRecognizerChannel,
  type TextRecognizerRequest,
  type TextRecognizerResponse,
} from "../../src/core/documents/scan/channel";

/** Processes that answer every load, and read a picture file unless it is told to hang. */
function processes(hangs: (page: number) => boolean) {
  const started: { stopped: boolean }[] = [];
  const start = (): TextRecognizerChannel => {
    const process = { stopped: false };
    started.push(process);
    let respond: (response: TextRecognizerResponse) => void = () => undefined;
    return {
      send(request: TextRecognizerRequest) {
        if (request.type === "load") queueMicrotask(() => respond({ id: request.id }));
        else if (request.type === "readFile" && !hangs(request.page)) {
          queueMicrotask(() =>
            respond({ id: request.id, lines: [], size: { width: 100, height: 50 } }),
          );
        }
      },
      onResponse: (listener) => {
        respond = listener;
      },
      onExit: () => undefined,
      stop: () => {
        process.stopped = true;
      },
    };
  };
  return { start, started };
}

const FILES = { detector: "detector.onnx", recognizer: "recognizer.onnx" };

describe("the text recognition process", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("a picture it doesn't read in time stops it, and the next picture is read by a new one", async () => {
    vi.useFakeTimers();
    const { start, started } = processes((page) => page === 1);
    const recognizer = createChannelTextRecognizer(start);

    await recognizer.load(FILES);
    const hanging = recognizer.readFile(new Uint8Array([1]), 1);
    const given = expect(hanging).rejects.toThrow("wasn't read in");
    await vi.advanceTimersByTimeAsync(60_000);
    await given;
    expect(started).toEqual([{ stopped: true }]);

    await recognizer.load(FILES);
    expect(await recognizer.readFile(new Uint8Array([2]), 2)).toEqual({
      lines: [],
      width: 100,
      height: 50,
    });
    expect(started).toEqual([{ stopped: true }, { stopped: false }]);
  });
});
