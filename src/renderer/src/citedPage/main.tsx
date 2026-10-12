/**
 * The hidden window that draws a cited page to save it (#133): the Document
 * viewer's view for its kind, at the cited Units, with no quote washed and
 * no mark. Once the view has drawn the cited area, the window is made tall
 * enough to hold it, and says where it is, with the text drawn in it.
 */
import { Suspense } from "react";
import { createRoot } from "react-dom/client";
import { CITED_PAGE_BRIDGE_KEY, type CitedPageBridge } from "../../../shared/citedPage";
import { core } from "../core";
import "../fonts.css";
import "../styles.css";
import "../viewer/viewer.css";
import { useAppStore, type ViewerTarget } from "../store";
import { type CitedArea, type CitedAreaCapture, CitedAreaContext } from "../viewer/citedArea";
import { DocumentView } from "../viewer/DocumentView";
import "./citedPage.css";
import { boundsOf, firstOf, imagesIn, textIn } from "./measure";

declare global {
  interface Window {
    readonly [CITED_PAGE_BRIDGE_KEY]: CitedPageBridge;
  }
}

const bridge = window[CITED_PAGE_BRIDGE_KEY];

/** Room kept beside the cited area, in CSS pixels; none above or below, where the next lines are. */
const MARGIN = 4;
/** How many times the area is measured, each after it is drawn again, before giving up. */
const ROUNDS = 20;

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** The next frame drawn, or a moment, whichever comes first. */
const frame = () =>
  new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
    setTimeout(resolve, 100);
  });

/** Measures the cited area once it is drawn and the window holds it, and says where it is. */
async function measure(area: () => CitedArea | null): Promise<void> {
  for (let round = 0; round < ROUNDS; round++) {
    await document.fonts.ready;
    const drawn = area();
    if (drawn) await Promise.all(imagesIn(drawn).map((image) => image.decode().catch(() => {})));
    await frame();
    await frame();
    const current = area();
    const box = current && boundsOf(current);
    if (!current || !box) continue;
    if (box.top < 0) {
      firstOf(current)?.scrollIntoView({ block: "start" });
      continue;
    }
    const wanted = Math.ceil(box.bottom);
    if (wanted > innerHeight) {
      const height = await bridge.resize(wanted);
      if (height > innerHeight) {
        for (let wait = 0; wait < 30 && innerHeight < height; wait++) await frame();
        continue;
      }
    }
    const x = Math.max(0, Math.floor(box.left - MARGIN));
    const y = Math.max(0, Math.floor(box.top));
    const rect = {
      x,
      y,
      width: Math.min(innerWidth, Math.ceil(box.right + MARGIN)) - x,
      height: Math.min(innerHeight, Math.ceil(box.bottom)) - y,
    };
    bridge.drawn({ rect, text: textIn(current, rect) });
    return;
  }
  bridge.failed("The cited area wasn't drawn.");
}

async function start(): Promise<void> {
  const [request, settings] = await Promise.all([bridge.request(), core.getSettings()]);
  // In the interface language, as the viewer draws it: a comment's author, a picture's placeholder.
  useAppStore.setState({ settings });
  let area: (() => CitedArea | null) | null = null;
  const capture: CitedAreaCapture = {
    area: request.area,
    drawn(given) {
      const first = area === null;
      area = given;
      if (first) {
        measure(() => area?.() ?? null).catch((error: unknown) => bridge.failed(messageOf(error)));
      }
    },
    failed(reason) {
      if (area === null) bridge.failed(reason);
    },
  };
  const target: ViewerTarget = {
    documentId: request.document.id,
    pageFrom: request.pageFrom,
    pageTo: request.pageTo,
    request: 1,
  };
  const container = document.getElementById("root");
  if (!container) throw new Error("The page has no #root element.");
  createRoot(container).render(
    <CitedAreaContext.Provider value={capture}>
      <main className="cited-page viewer-pane flex h-screen min-h-0 min-w-0 flex-col overflow-hidden bg-sheet">
        <Suspense fallback={null}>
          <DocumentView document={request.document} target={target} />
        </Suspense>
      </main>
    </CitedAreaContext.Provider>,
  );
}

start().catch((error: unknown) => bridge.failed(messageOf(error)));
