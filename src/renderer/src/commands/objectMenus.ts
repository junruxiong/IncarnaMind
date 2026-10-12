import { registerMenu } from "./menus";

// Later features add their items with registerMenu: Move to… (#229, #212), Add to note (#201), History (#300), Tag colour and merge (#205).

/** Export as… the last format (⇧⌘E), each other format, then the options sheet (#100). */
const EXPORTS = [
  "mind.export",
  "mind.exportAs.docx",
  "mind.exportAs.pdf",
  "mind.exportAs.html",
  "mind.exportAs.markdown",
  "mind.exportOptions",
] as const;

registerMenu("mind", [
  "mind.open",
  "mind.openSide",
  ...EXPORTS,
  "mind.citationStyle",
  "skill.fromMind",
  "mind.moveTo",
  "mind.rename",
  "mind.delete",
]);

registerMenu("document", [
  "document.open",
  "document.openTab",
  "document.ask",
  "document.startMind",
  "document.tags",
  "document.retry",
  "document.fetchAgain",
  "document.countAsEvidence",
  "document.leaveOutOfEvidence",
  "document.rename",
  "document.openExternally",
  "document.showInFolder",
  "file.copyPath",
  "document.openLivePage",
  "document.copyAddress",
  "file.info",
  "document.delete",
]);

registerMenu("folder", [
  "folder.open",
  "folder.newMind",
  "folder.ask",
  "folder.organize",
  "folder.editDescription",
  "folder.rename",
  "folder.delete",
]);

registerMenu("tag", ["tag.open", "tag.manage", "tag.rename", "tag.delete"]);

registerMenu("linkedFolder", [
  "linkedFolder.pause",
  "linkedFolder.layout",
  "linkedFolder.download",
  "linkedFolder.reveal",
  "file.copyPath",
  "file.info",
  "linkedFolder.locate",
  "linkedFolder.unlink",
]);

registerMenu("deleted", ["deleted.restore", "deleted.deletePermanently"]);

// A tab's menu is its Mind's, then the tab's own.
registerMenu("tab", [...EXPORTS, "mind.moveTo", "mind.delete"], 0);
registerMenu(
  "tab",
  [
    "tab.split",
    "tab.keep",
    "tab.pin",
    "tab.moveLeft",
    "tab.moveRight",
    "tab.rename",
    "tab.close",
    "tab.closeOthers",
    "tab.closeToTheRight",
  ],
  1,
);

registerMenu("answer", ["answer.copy", "answer.regenerate", "block.delete"]);

registerMenu("citation", ["citation.open", "citation.remove"]);

registerMenu("chart", [
  "chart.table",
  "chart.kind.bar",
  "chart.kind.line",
  "chart.kind.pie",
  "chart.kind.scatter",
  "chart.update",
  "chart.rename",
  "block.delete",
]);
