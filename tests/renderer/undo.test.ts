import { beforeEach, describe, expect, test } from "vitest";
import { type Undoable, UndoRefused, useUndo } from "../../src/renderer/src/undo";

/*
 * The session's undo stack (docs/designs/interaction-system.md, C.6): ⌘Z takes
 * the last change back, ⇧⌘Z does it again, a new change ends what can be
 * redone, and an undo the core refuses leaves the stack as it should be.
 */

/** A change that records what was done to it, in `log`. */
function change(name: string, log: string[], { redo = true } = {}): Undoable {
  return {
    message: `Changed ${name}`,
    name,
    undo: async () => {
      log.push(`undo ${name}`);
    },
    ...(redo && {
      redo: async () => {
        log.push(`redo ${name}`);
      },
    }),
  };
}

const undo = () => useUndo.getState().undo();
const redo = () => useUndo.getState().redo();
const push = (each: Undoable) => useUndo.getState().push(each);
const names = (changes: readonly Undoable[]) => changes.map((each) => each.name);

beforeEach(() => {
  useUndo.setState({ done: [], undone: [], line: null });
});

describe("the undo stack", () => {
  test("a change pushed shows its line, and ⌘Z takes it back", async () => {
    const log: string[] = [];
    const moved = change("Move", log);
    push(moved);
    expect(useUndo.getState().line?.change).toBe(moved);
    await undo();
    expect(log).toEqual(["undo Move"]);
    expect(useUndo.getState().line).toBeNull();
    expect(names(useUndo.getState().done)).toEqual([]);
    expect(names(useUndo.getState().undone)).toEqual(["Move"]);
  });

  test("several changes undo in reverse order and redo in order", async () => {
    const log: string[] = [];
    for (const name of ["A", "B", "C"]) push(change(name, log));
    await undo();
    await undo();
    await undo();
    await redo();
    await redo();
    expect(log).toEqual(["undo C", "undo B", "undo A", "redo A", "redo B"]);
    expect(names(useUndo.getState().done)).toEqual(["A", "B"]);
    expect(names(useUndo.getState().undone)).toEqual(["C"]);
    // Done again, a change shows its line again.
    expect(useUndo.getState().line?.change?.name).toBe("B");
  });

  test("a held ⌘Z steps back change by change, each undo after the one before", async () => {
    const log: string[] = [];
    let finish: () => void = () => undefined;
    const slow: Undoable = {
      message: "Slow",
      name: "Slow",
      undo: () =>
        new Promise<void>((resolve) => {
          finish = () => {
            log.push("undo Slow");
            resolve();
          };
        }),
    };
    push(change("A", log));
    push(slow);
    const first = undo();
    const second = undo();
    for (let i = 0; i < 10; i++) await Promise.resolve();
    // The second waits for the first.
    expect(log).toEqual([]);
    finish();
    await Promise.all([first, second]);
    expect(log).toEqual(["undo Slow", "undo A"]);
  });

  test("a new change ends what can be redone", async () => {
    const log: string[] = [];
    push(change("A", log));
    push(change("B", log));
    await undo();
    push(change("C", log));
    expect(names(useUndo.getState().undone)).toEqual([]);
    await redo();
    expect(log).toEqual(["undo B"]);
    expect(names(useUndo.getState().done)).toEqual(["A", "C"]);
  });

  test("the same change offered twice, by its action and by its command, is one step", () => {
    const moved = change("Move", []);
    push(moved);
    push(moved);
    expect(useUndo.getState().done).toEqual([moved]);
  });

  test("a change that can't be redone, once taken back, ends what can be redone", async () => {
    const log: string[] = [];
    push(change("A", log, { redo: false }));
    push(change("B", log));
    await undo();
    expect(names(useUndo.getState().undone)).toEqual(["B"]);
    await undo();
    expect(names(useUndo.getState().undone)).toEqual([]);
    await redo();
    expect(log).toEqual(["undo B", "undo A"]);
  });

  test("an undo the core refuses shows one problem line, and leaves the change done", async () => {
    const log: string[] = [];
    push(change("A", log));
    push({
      message: "Renamed",
      name: "Rename",
      undo: () =>
        Promise.reject(
          new Error("Error invoking remote method 'core:x': NotFoundError: It was deleted."),
        ),
    });
    await undo();
    const { line, done } = useUndo.getState();
    expect(line?.problem).toEqual({ step: "undo", reason: "It was deleted.", fix: null });
    // It can never be taken back: the next ⌘Z reaches the change before it.
    expect(names(done)).toEqual(["A"]);
    await undo();
    expect(log).toEqual(["undo A"]);
  });

  test("an undo refused with a fix stays on the stack; the fix lets it happen", async () => {
    const log: string[] = [];
    let folderBack = false;
    push({
      message: "Moved “Shortlist” to Research",
      name: "Move",
      undo: async () => {
        if (!folderBack)
          throw new UndoRefused("“Finance” is deleted", {
            label: "Restore Finance",
            run: async () => {
              folderBack = true;
            },
          });
        log.push("undo Move");
      },
      redo: async () => undefined,
    });
    await undo();
    const problem = useUndo.getState().line?.problem;
    expect(problem?.reason).toBe("“Finance” is deleted");
    expect(problem?.fix?.label).toBe("Restore Finance");
    expect(names(useUndo.getState().done)).toEqual(["Move"]);
    await problem?.fix?.run();
    expect(log).toEqual(["undo Move"]);
    expect(names(useUndo.getState().done)).toEqual([]);
    expect(names(useUndo.getState().undone)).toEqual(["Move"]);
  });

  test("a refused redo stays undone with its fix, or leaves without one", async () => {
    const failing: Undoable = {
      message: "Deleted",
      name: "Delete",
      undo: async () => undefined,
      redo: () => Promise.reject(new Error("It isn't there.")),
    };
    push(failing);
    await undo();
    await redo();
    expect(useUndo.getState().line?.problem).toEqual({
      step: "redo",
      reason: "It isn't there.",
      fix: null,
    });
    expect(useUndo.getState().undone).toEqual([]);
    expect(useUndo.getState().done).toEqual([]);
  });

  test("a change withdrawn leaves the stack and its line", () => {
    const log: string[] = [];
    const accepted = change("Accept", log);
    push(change("A", log));
    push(accepted);
    useUndo.getState().withdraw(accepted);
    expect(names(useUndo.getState().done)).toEqual(["A"]);
    expect(useUndo.getState().line).toBeNull();
  });
});
