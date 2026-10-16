import { expect, test } from "vitest";
import { useTaggingExplained } from "../../src/renderer/src/taggingExplained";

const explained = () => useTaggingExplained.getState();

test("each Folder is told what is sent once, for the whole run it is told in", () => {
  expect(explained().says(["contracts"])).toBe(true);
  explained().shown(["contracts"]);
  expect(explained().says(["contracts"])).toBe(true);
  explained().ended();
  expect(explained().says(["contracts"])).toBe(false);
  expect(explained().says(["contracts", null])).toBe(true);
});
