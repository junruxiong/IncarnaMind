import { createCanvas, loadImage } from "@napi-rs/canvas";
import type { PagePicture } from "../../adapters";
import { greys } from "./pages";

// A chart's small print reads better a little larger, and a large photo needs no more than this.
const SMALLEST_SIDE = 1000;
const LARGEST_SIDE = 2000;

/** A saved web page's picture, drawn on white in shades of grey, for reading. */
export async function figurePicture(bytes: Uint8Array): Promise<PagePicture> {
  const image = await loadImage(Buffer.from(bytes));
  const longer = Math.max(image.width, image.height, 1);
  const scale = Math.min(LARGEST_SIDE, Math.max(SMALLEST_SIDE, longer)) / longer;
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const canvas = createCanvas(width, height);
  const context = canvas.getContext("2d");
  // A transparent picture's text is dark on nothing: on white, as a page shows it.
  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);
  return { width, height, pixels: greys(context.getImageData(0, 0, width, height).data) };
}
