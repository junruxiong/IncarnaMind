import { registerMenu } from "./menus";

// Later features add their items with registerMenu: Move to… (#229, #212), Copy path and Get info (#249),
// Add to note (#201), History (#300), Tag colour and merge (#205).

registerMenu("mind", [
  "mind.open",
  "mind.openSide",
  "mind.export",
  "mind.moveTo",
  "mind.rename",
  "mind.delete",
]);

registerMenu("document", [
  "document.open",
  "document.openTab",
  "document.ask",
  "document.startMind",
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

registerMenu("folder", ["folder.open", "folder.newMind", "folder.rename", "folder.delete"]);

registerMenu("tag", ["tag.open", "tag.manage"]);

registerMenu("linkedFolder", [
  "linkedFolder.pause",
  "linkedFolder.layout",
  "linkedFolder.download",
  "linkedFolder.reveal",
  "linkedFolder.unlink",
]);

registerMenu("tab", [
  "tab.split",
  "tab.keep",
  "tab.pin",
  "tab.moveLeft",
  "tab.moveRight",
  "tab.rename",
  "tab.close",
  "tab.closeOthers",
  "tab.closeToTheRight",
]);

registerMenu("answer", ["answer.copy", "answer.regenerate", "block.delete"]);

registerMenu("citation", ["citation.open", "citation.remove"]);
