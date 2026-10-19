import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { createSettling, SHOWN_AFTER_MS } from "../../src/renderer/src/settling";

describe("What has been there long enough to show", () => {
  let heard: string[][];
  const settling = () => createSettling((settled) => heard.push([...settled].sort()));
  beforeEach(() => {
    vi.useFakeTimers();
    heard = [];
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("a wait that ends sooner than the limit is never shown", () => {
    const wait = settling();
    wait.update(["processing"]);
    vi.advanceTimersByTime(SHOWN_AFTER_MS - 1);
    wait.update([]);
    vi.advanceTimersByTime(SHOWN_AFTER_MS * 2);
    expect(heard).toEqual([]);
  });

  test("one that lasts is shown from the limit on, and goes at once when it ends", () => {
    const wait = settling();
    wait.update(["processing"]);
    vi.advanceTimersByTime(SHOWN_AFTER_MS - 1);
    expect(heard).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(heard).toEqual([["processing"]]);
    wait.update([]);
    expect(heard).toEqual([["processing"], []]);
  });

  test("a change in what else is there doesn't start it again", () => {
    const wait = settling();
    wait.update(["processing"]);
    vi.advanceTimersByTime(200);
    wait.update(["processing", "tagging"]);
    vi.advanceTimersByTime(100);
    expect(heard).toEqual([["processing"]]);
    vi.advanceTimersByTime(200);
    expect(heard).toEqual([["processing"], ["processing", "tagging"]]);
  });

  test("a wait that comes back starts from nothing", () => {
    const wait = settling();
    wait.update(["processing"]);
    vi.advanceTimersByTime(SHOWN_AFTER_MS);
    wait.update([]);
    wait.update(["processing"]);
    expect(heard).toEqual([["processing"], []]);
    vi.advanceTimersByTime(SHOWN_AFTER_MS - 1);
    expect(heard).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(heard).toEqual([["processing"], [], ["processing"]]);
  });

  test("stopped, nothing more is heard", () => {
    const wait = settling();
    wait.update(["processing"]);
    wait.stop();
    vi.advanceTimersByTime(SHOWN_AFTER_MS * 2);
    expect(heard).toEqual([]);
  });
});
