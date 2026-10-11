import { files } from "../core";
import { useAppStore } from "../store";
import { useWebPages } from "../webPages";
import type { Command, CommandContext } from "./registry";
import { documentsOf, everyDocument } from "./targets";

const webPagesOf = (ctx: CommandContext) => documentsOf(ctx).filter((document) => document.web);

/** Only while every Document it acts on is a web page. */
const onWebPages = (ctx: CommandContext) => everyDocument(ctx, (document) => !!document.web);

const reportError = (error: unknown) => useAppStore.getState().reportError(error);

export const webPageCommands: readonly Command[] = [
  {
    id: "webPage.add",
    label: "sidebar.plus.addWebPage",
    group: "main",
    reversible: true,
    run: () => useWebPages.getState().openDialog(),
  },
  {
    id: "document.fetchAgain",
    label: "webPages.fetchAgain",
    group: "main",
    appliesTo: ["document"],
    when: onWebPages,
    // Older versions stay for the Citations that quote them (ADR-0010).
    reversible: true,
    run: async (ctx) => {
      await Promise.all(webPagesOf(ctx).map((each) => useWebPages.getState().fetchAgain(each)));
    },
  },
  {
    id: "document.openLivePage",
    label: "webPages.openLive",
    group: "file",
    appliesTo: ["document"],
    when: onWebPages,
    reversible: true,
    run: (ctx) => {
      for (const each of webPagesOf(ctx)) files.openDocumentExternally(each.id).catch(reportError);
    },
  },
  {
    id: "document.copyAddress",
    label: "webPages.copyAddress",
    group: "file",
    appliesTo: ["document"],
    when: onWebPages,
    reversible: true,
    run: (ctx) => {
      const addresses = webPagesOf(ctx).flatMap((each) => (each.web ? [each.web.address] : []));
      navigator.clipboard.writeText(addresses.join("\n")).catch(reportError);
    },
  },
];
