import { registerMenu } from "./menus";

// Later features add their items with registerMenu: Move to… (#229, #212), Copy path and Get info (#249),
// Add to note (#201), History (#300), Tag colour and merge (#205).

registerMenu("mind", [
  "mind.open",
  "mind.openSide",
  "mind.outline",
  "mind.export",
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
  "document.openLivePage",
  "document.copyAddress",
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
  "linkedFolder.locate",
  "linkedFolder.unlink",
]);

// A tab's menu is its Mind's, then the tab's own.
registerMenu("tab", ["mind.outline", "mind.export", "mind.moveTo", "mind.delete"], 0);
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
