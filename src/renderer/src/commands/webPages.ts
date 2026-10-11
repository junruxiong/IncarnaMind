import type { Document } from "../../../core/api";
import { files } from "../core";
import { useAppStore } from "../store";
import { useWebPages } from "../webPages";
import type { Command, CommandContext } from "./registry";

/** The web page Documents a command acts on: the selection, else the focus. */
function webPagesOf(ctx: CommandContext): Document[] {
  const refs = ctx.selection.length > 0 ? ctx.selection : ctx.focus ? [ctx.focus] : [];
  const { documents } = useAppStore.getState();
  return refs.flatMap((ref) => {
    const document = documents.find((each) => each.id === ref.id);
    return document?.web ? [document] : [];
  });
}

/** Only while every Document it acts on is a web page. */
const onWebPages = (ctx: CommandContext) => {
  const count = ctx.selection.length > 0 ? ctx.selection.length : ctx.focus ? 1 : 0;
  return count > 0 && webPagesOf(ctx).length === count;
};

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
