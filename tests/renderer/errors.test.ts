import { describe, expect, test } from "vitest";
import { describeError, errorMessage } from "../../src/renderer/src/errors";

/*
 * What a failed call to the core says (#157). Electron rejects it with the error written out,
 * "Error invoking remote method 'core:x': InvalidInputError: …"; no line the User reads shows that
 * wrapping or a stack, and the technical text waits behind "Details".
 */

const ipc = (method: string, name: string, message: string) =>
  new Error(`Error invoking remote method '${method}': ${name}: ${message}`);

describe("an error from the core", () => {
  test("says the core's own sentence, without Electron's wrapping or the class's name", () => {
    const failure = ipc(
      "core:renameDocument",
      "InvalidInputError",
      "A Document's name can't be empty.",
    );
    expect(errorMessage(failure)).toBe("A Document's name can't be empty.");
    expect(describeError(failure).reason).toBeNull();
  });

  test("keeps the method and the class for the Details, not the wrapping", () => {
    const { detail } = describeError(
      ipc("core:renameDocument", "InvalidInputError", "A Document's name can't be empty."),
    );
    expect(detail).toBe(
      "core:renameDocument\nInvalidInputError: A Document's name can't be empty.",
    );
    expect(detail).not.toContain("Error invoking remote method");
  });

  test("is as readable in Chinese", () => {
    const failure = ipc("core:moveToFolder", "NotFoundError", "这个文件夹已被删除。");
    expect(errorMessage(failure)).toBe("这个文件夹已被删除。");
  });

  test("never shows a stack, even one written into the message", () => {
    const failure = ipc(
      "core:addDocuments",
      "Error",
      "Couldn't read it.\n    at read (/app/core.js:10:5)\n    at async run (/app/core.js:20:3)",
    );
    const { message, detail } = describeError(failure);
    expect(message).toBe("Couldn't read it.");
    expect(detail).not.toMatch(/\n\s+at /);
  });

  test("keeps a message that runs over several lines", () => {
    const failure = ipc("core:x", "InvalidInputError", "First line.\nSecond line.");
    expect(errorMessage(failure)).toBe("First line.\nSecond line.");
  });
});

describe("an error that did not cross the bridge", () => {
  test("loses its class's name in its message, and keeps it for the Details", () => {
    const { message, detail } = describeError(new TypeError("Cannot read properties of undefined"));
    expect(message).toBe("Cannot read properties of undefined");
    expect(detail).toBe("TypeError: Cannot read properties of undefined");
  });

  test("is made of what it is given when it isn't an Error", () => {
    expect(errorMessage("The disk went away")).toBe("The disk went away");
    expect(describeError(42).message).toBe("42");
  });

  test("reads a class's name from its text, as Electron writes it", () => {
    expect(describeError(new Error("NotFoundError: It was deleted.")).message).toBe(
      "It was deleted.",
    );
  });
});

describe("a failure of the system's", () => {
  test.each([
    ["EACCES: permission denied, rename '/a/b.pdf' -> '/a/c.pdf'", "error.system.notAllowed"],
    ["EROFS: read-only file system, open '/a/b.pdf'", "error.system.notAllowed"],
    ["ENOSPC: no space left on device, write", "error.system.full"],
    ["ENOENT: no such file or directory, open '/a/b.pdf'", "error.system.gone"],
    ["ENAMETOOLONG: name too long, rename '/a/b.pdf' -> '/a/c'", "error.system.nameTooLong"],
    ["EEXIST: file already exists, mkdir '/a/b'", "error.system.exists"],
  ])("%s gets plain words", (text, reason) => {
    const description = describeError(ipc("core:renameDocument", "Error", text));
    expect(description.reason).toBe(reason);
    // The system's own text, with the path in it, stays for the Details.
    expect(description.detail).toContain(text);
  });

  test("a sentence of the core's that names no code keeps its own words", () => {
    expect(describeError(new Error("Choose the folder that holds them now.")).reason).toBeNull();
  });
});
