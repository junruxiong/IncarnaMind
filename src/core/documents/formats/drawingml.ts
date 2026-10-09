/**
 * DrawingML, the drawing language of Office packages, as the slide renderer
 * needs it (ADR-0011): a theme's colours, fonts and style lists, colours with
 * their transforms (lumMod, tint, alpha…), fills and lines, all resolved to
 * CSS values. Pure, with no DOM: the viewer draws what this reads.
 *
 * Lengths in a package are EMUs; the renderer works in CSS pixels at 96 dpi,
 * so one pixel is 9,525 EMUs and one point is 4/3 of a pixel.
 */
import { child, elements, local, parseXml, type XmlElement } from "./xml";

export const EMU_PER_PX = 9525;
export const PX_PER_PT = 4 / 3;

/** EMUs, as written in an attribute, in CSS pixels; `fallback` when absent or not a number. */
export function emu(value: string | undefined, fallback = 0): number {
  if (value === undefined) return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number / EMU_PER_PX : fallback;
}

/** A number attribute, or `fallback`. */
export function numberAttr(
  element: XmlElement | undefined,
  name: string,
  fallback: number,
): number {
  const value = element?.attrs[name];
  if (value === undefined) return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/** A boolean attribute ("1", "true", "on"), or undefined when absent. */
export function boolAttr(element: XmlElement | undefined, name: string): boolean | undefined {
  const value = element?.attrs[name];
  if (value === undefined) return undefined;
  return value === "1" || value === "true" || value === "on";
}

export interface ThemeFonts {
  latin: string;
  ea: string;
  cs: string;
}

/** What a slide's drawing takes from its master's theme. */
export interface Theme {
  /** The scheme's twelve colours (dk1, lt1, dk2, lt2, accent1–6, hlink, folHlink), as RGB. */
  colours: Record<string, Rgb>;
  major: ThemeFonts;
  minor: ThemeFonts;
  /** The format scheme's fill, line and background fill styles, for style references. */
  fills: XmlElement[];
  lines: XmlElement[];
  backgrounds: XmlElement[];
  /** Each effect style's effect list (a:effectLst), for effect references. */
  effects: (XmlElement | undefined)[];
}

/** Office's own theme, for a package without one. */
const OFFICE_COLOURS: Record<string, string> = {
  dk1: "000000",
  lt1: "FFFFFF",
  dk2: "44546A",
  lt2: "E7E6E6",
  accent1: "4472C4",
  accent2: "ED7D31",
  accent3: "A5A5A5",
  accent4: "FFC000",
  accent5: "5B9BD5",
  accent6: "70AD47",
  hlink: "0563C1",
  folHlink: "954F72",
};

export function readTheme(xml: string | undefined): Theme {
  const theme: Theme = {
    colours: Object.fromEntries(
      Object.entries(OFFICE_COLOURS).map(([name, hex]) => [name, hexToRgb(hex)]),
    ),
    major: { latin: "Calibri Light", ea: "", cs: "" },
    minor: { latin: "Calibri", ea: "", cs: "" },
    fills: [],
    lines: [],
    backgrounds: [],
    effects: [],
  };
  if (!xml) return theme;
  const root = parseXml(xml);
  const elementsOf = child(root, "a:themeElements");
  const scheme = elementsOf && child(elementsOf, "a:clrScheme");
  for (const slot of scheme ? elements(scheme) : []) {
    const colour = elements(slot)[0];
    const rgb = colour && baseColour(colour, null);
    if (rgb) theme.colours[local(slot.name)] = rgb;
  }
  const fontScheme = elementsOf && child(elementsOf, "a:fontScheme");
  const fonts = (name: string): ThemeFonts | undefined => {
    const set = fontScheme && child(fontScheme, name);
    if (!set) return undefined;
    return {
      latin: child(set, "a:latin")?.attrs.typeface ?? "",
      ea: child(set, "a:ea")?.attrs.typeface ?? "",
      cs: child(set, "a:cs")?.attrs.typeface ?? "",
    };
  };
  theme.major = fonts("a:majorFont") ?? theme.major;
  theme.minor = fonts("a:minorFont") ?? theme.minor;
  const format = elementsOf && child(elementsOf, "a:fmtScheme");
  const list = (name: string) => {
    const found = format && child(format, name);
    return found ? elements(found) : [];
  };
  theme.fills = list("a:fillStyleLst");
  theme.lines = list("a:lnStyleLst");
  theme.backgrounds = list("a:bgFillStyleLst");
  theme.effects = list("a:effectStyleLst").map((style) => child(style, "a:effectLst"));
  return theme;
}

/** How a master names its colours: bg1 → lt1, tx1 → dk1 and so on (a dark theme swaps them). */
export type ColourMap = Record<string, string>;

export const DEFAULT_COLOUR_MAP: ColourMap = {
  bg1: "lt1",
  tx1: "dk1",
  bg2: "lt2",
  tx2: "dk2",
};

/** A colour map from a `p:clrMap` or an `a:overrideClrMapping`, over `base`. */
export function readColourMap(element: XmlElement | undefined, base: ColourMap): ColourMap {
  if (!element) return base;
  return { ...base, ...element.attrs };
}

/** What colours resolve against: the theme, the colour map, and a style reference's colour. */
export interface ColourContext {
  theme: Theme;
  map: ColourMap;
  /** The colour a style reference gives `phClr`. */
  placeholder?: Rgba | null;
}

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

export interface Rgba extends Rgb {
  /** 0–1. */
  a: number;
}

function hexToRgb(hex: string): Rgb {
  const value = Number.parseInt(hex.padStart(6, "0").slice(0, 6), 16);
  if (!Number.isFinite(value)) return { r: 0, g: 0, b: 0 };
  return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
}

const PRESET: Record<string, string> = {
  black: "000000",
  white: "FFFFFF",
  red: "FF0000",
  green: "008000",
  lime: "00FF00",
  blue: "0000FF",
  yellow: "FFFF00",
  cyan: "00FFFF",
  aqua: "00FFFF",
  magenta: "FF00FF",
  fuchsia: "FF00FF",
  gray: "808080",
  grey: "808080",
  silver: "C0C0C0",
  maroon: "800000",
  navy: "000080",
  olive: "808000",
  purple: "800080",
  teal: "008080",
  orange: "FFA500",
  darkBlue: "00008B",
  darkRed: "8B0000",
  darkGreen: "006400",
  darkGray: "A9A9A9",
  lightGray: "D3D3D3",
};

const COLOUR_ELEMENTS = new Set([
  "srgbClr",
  "schemeClr",
  "sysClr",
  "prstClr",
  "scrgbClr",
  "hslClr",
]);

/** The colour element inside a fill or colour container, e.g. the a:srgbClr of an a:solidFill. */
export function colourElement(container: XmlElement | undefined): XmlElement | undefined {
  return container
    ? elements(container).find((each) => COLOUR_ELEMENTS.has(local(each.name)))
    : undefined;
}

const linear = (channel: number) => {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};
const gamma = (value: number) => {
  const c = Math.min(1, Math.max(0, value));
  return Math.round(255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055));
};

function toHsl({ r, g, b }: Rgb): [number, number, number] {
  const rr = r / 255;
  const gg = g / 255;
  const bb = b / 255;
  const max = Math.max(rr, gg, bb);
  const min = Math.min(rr, gg, bb);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h =
    max === rr
      ? (gg - bb) / d + (gg < bb ? 6 : 0)
      : max === gg
        ? (bb - rr) / d + 2
        : (rr - gg) / d + 4;
  return [h * 60, s, l];
}

function fromHsl(h: number, s: number, l: number): Rgb {
  const hue = ((h % 360) + 360) % 360;
  const sat = Math.min(1, Math.max(0, s));
  const lum = Math.min(1, Math.max(0, l));
  const c = (1 - Math.abs(2 * lum - 1)) * sat;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = lum - c / 2;
  const [r, g, b] =
    hue < 60
      ? [c, x, 0]
      : hue < 120
        ? [x, c, 0]
        : hue < 180
          ? [0, c, x]
          : hue < 240
            ? [0, x, c]
            : hue < 300
              ? [x, 0, c]
              : [c, 0, x];
  return {
    r: Math.round((r + m) * 255),
    g: Math.round((g + m) * 255),
    b: Math.round((b + m) * 255),
  };
}

/** A colour element's colour before its transforms; null for a scheme colour with no context. */
function baseColour(element: XmlElement, context: ColourContext | null): Rgba | null {
  const val = element.attrs.val ?? "";
  switch (local(element.name)) {
    case "srgbClr":
      return { ...hexToRgb(val), a: 1 };
    case "sysClr":
      return {
        ...hexToRgb(element.attrs.lastClr ?? (val === "window" ? "FFFFFF" : "000000")),
        a: 1,
      };
    case "prstClr":
      return { ...hexToRgb(PRESET[val] ?? "000000"), a: 1 };
    case "scrgbClr":
      return {
        r: gamma(numberAttr(element, "r", 0) / 100000),
        g: gamma(numberAttr(element, "g", 0) / 100000),
        b: gamma(numberAttr(element, "b", 0) / 100000),
        a: 1,
      };
    case "hslClr":
      return {
        ...fromHsl(
          numberAttr(element, "hue", 0) / 60000,
          numberAttr(element, "sat", 0) / 100000,
          numberAttr(element, "lum", 0) / 100000,
        ),
        a: 1,
      };
    case "schemeClr": {
      if (!context) return null;
      if (val === "phClr") return context.placeholder ? { ...context.placeholder } : null;
      const name = context.map[val] ?? val;
      const rgb = context.theme.colours[name] ?? context.theme.colours[val];
      return rgb ? { ...rgb, a: 1 } : null;
    }
  }
  return null;
}

/** A colour element resolved, its transforms applied in order. */
export function resolveColour(
  element: XmlElement | undefined,
  context: ColourContext,
): Rgba | null {
  if (!element) return null;
  const base = baseColour(element, context);
  if (!base) return null;
  let colour: Rgba = base;
  for (const transform of elements(element)) {
    const value = numberAttr(transform, "val", 0) / 100000;
    const name = local(transform.name);
    switch (name) {
      case "alpha":
        colour = { ...colour, a: value };
        break;
      case "alphaMod":
        colour = { ...colour, a: colour.a * value };
        break;
      case "alphaOff":
        colour = { ...colour, a: Math.min(1, Math.max(0, colour.a + value)) };
        break;
      case "tint": {
        // Toward white, in linear light: a 40% tint is 40% of the colour and 60% white.
        const mix = (c: number) => gamma(linear(c) * value + (1 - value));
        colour = { r: mix(colour.r), g: mix(colour.g), b: mix(colour.b), a: colour.a };
        break;
      }
      case "shade": {
        const mix = (c: number) => gamma(linear(c) * value);
        colour = { r: mix(colour.r), g: mix(colour.g), b: mix(colour.b), a: colour.a };
        break;
      }
      case "lumMod":
      case "lumOff":
      case "satMod":
      case "satOff":
      case "hueMod":
      case "hueOff": {
        let [h, s, l] = toHsl(colour);
        if (name === "lumMod") l *= value;
        else if (name === "lumOff") l += value;
        else if (name === "satMod") s *= value;
        else if (name === "satOff") s += value;
        else if (name === "hueMod") h *= value;
        else h += numberAttr(transform, "val", 0) / 60000;
        colour = { ...fromHsl(h, s, l), a: colour.a };
        break;
      }
      case "comp": {
        const [h, s, l] = toHsl(colour);
        colour = { ...fromHsl(h + 180, s, l), a: colour.a };
        break;
      }
      case "inv":
        colour = { r: 255 - colour.r, g: 255 - colour.g, b: 255 - colour.b, a: colour.a };
        break;
      case "gray": {
        const grey = Math.round(0.2126 * colour.r + 0.7152 * colour.g + 0.0722 * colour.b);
        colour = { r: grey, g: grey, b: grey, a: colour.a };
        break;
      }
    }
  }
  return colour;
}

/** A colour as CSS. */
export function css(colour: Rgba): string {
  const hex = (value: number) => Math.round(value).toString(16).padStart(2, "0");
  const rgb = `#${hex(colour.r)}${hex(colour.g)}${hex(colour.b)}`;
  return colour.a >= 0.999
    ? rgb
    : `rgb(${colour.r} ${colour.g} ${colour.b} / ${Math.round(colour.a * 1000) / 1000})`;
}

/** The colour inside a container (an a:solidFill, a fontRef…), as CSS. */
export function containerColour(
  container: XmlElement | undefined,
  context: ColourContext,
): string | null {
  const colour = resolveColour(colourElement(container), context);
  return colour ? css(colour) : null;
}

/** How a shape, background or cell is filled, in CSS terms. */
export type Fill =
  | { kind: "none" }
  | { kind: "solid"; colour: string }
  | {
      kind: "gradient";
      /** Degrees clockwise from left-to-right, as DrawingML measures them. */
      angle: number;
      radial: boolean;
      stops: { at: number; colour: string }[];
    }
  | { kind: "image"; part: string };

const FILL_ELEMENTS = new Set([
  "noFill",
  "solidFill",
  "gradFill",
  "blipFill",
  "pattFill",
  "grpFill",
]);

/** The fill element of a properties element (spPr, bgPr, tcPr…), if it has one. */
export function fillElement(container: XmlElement | undefined): XmlElement | undefined {
  return container
    ? elements(container).find((each) => FILL_ELEMENTS.has(local(each.name)))
    : undefined;
}

/**
 * A fill element as a `Fill`. `part` turns a picture fill's relationship id
 * into the package part it names; a fill this can't draw (a pattern) is drawn
 * as a blend of its two colours. Undefined for a group fill (`grpFill`),
 * which the caller takes from the group.
 */
export function readFill(
  element: XmlElement | undefined,
  context: ColourContext,
  part?: (id: string) => string | undefined,
): Fill | undefined {
  if (!element) return undefined;
  switch (local(element.name)) {
    case "noFill":
      return { kind: "none" };
    case "solidFill": {
      const colour = containerColour(element, context);
      return colour ? { kind: "solid", colour } : { kind: "none" };
    }
    case "gradFill": {
      const list = child(element, "a:gsLst");
      const stops = (list ? elements(list) : [])
        .map((stop) => {
          const colour = resolveColour(colourElement(stop), context);
          return colour ? { at: numberAttr(stop, "pos", 0) / 100000, colour: css(colour) } : null;
        })
        .filter((stop): stop is { at: number; colour: string } => stop !== null)
        .sort((a, b) => a.at - b.at);
      if (stops.length === 0) return { kind: "none" };
      const lin = child(element, "a:lin");
      const path = child(element, "a:path");
      return {
        kind: "gradient",
        angle: numberAttr(lin, "ang", 0) / 60000,
        radial: path !== undefined,
        stops,
      };
    }
    case "blipFill": {
      const blip = child(element, "a:blip");
      const target = blip?.attrs["r:embed"] && part?.(blip.attrs["r:embed"]);
      return target ? { kind: "image", part: target } : { kind: "none" };
    }
    case "pattFill": {
      const fg = resolveColour(colourElement(child(element, "a:fgClr")), context);
      const bg = resolveColour(colourElement(child(element, "a:bgClr")), context);
      if (!fg && !bg) return { kind: "none" };
      if (!fg || !bg) return { kind: "solid", colour: css((fg ?? bg) as Rgba) };
      const mix = (a: number, b: number) => Math.round((a + b) / 2);
      return {
        kind: "solid",
        colour: css({
          r: mix(fg.r, bg.r),
          g: mix(fg.g, bg.g),
          b: mix(fg.b, bg.b),
          a: mix(fg.a, bg.a),
        }),
      };
    }
  }
  return undefined;
}

/** A line (a shape's outline, a cell's border): CSS pixels, a colour, a dash and its ends. */
export interface Line {
  width: number;
  colour: string;
  /** SVG dash array in multiples of the width, e.g. [4, 3]; null for solid. */
  dash: number[] | null;
  /** Arrowheads at the start and the end, for connectors. */
  head: string | null;
  tail: string | null;
}

const DASHES: Record<string, number[]> = {
  dot: [1, 1],
  sysDot: [1, 1],
  dash: [4, 3],
  sysDash: [3, 1],
  lgDash: [8, 3],
  dashDot: [4, 3, 1, 3],
  sysDashDot: [3, 1, 1, 1],
  lgDashDot: [8, 3, 1, 3],
  lgDashDotDot: [8, 3, 1, 3, 1, 3],
  sysDashDotDot: [3, 1, 1, 1, 1, 1],
};

/**
 * An a:ln as a `Line`, over `base` (a style reference's line). Null when the
 * line is off (`a:noFill`), undefined when neither says anything.
 */
export function readLine(
  ln: XmlElement | undefined,
  context: ColourContext,
  base?: Line | null,
): Line | null | undefined {
  if (!ln) return base;
  const fill = fillElement(ln);
  let colour: string | null | undefined;
  if (fill) {
    const read = readFill(fill, context);
    colour =
      read?.kind === "solid"
        ? read.colour
        : read?.kind === "gradient"
          ? (read.stops[0]?.colour ?? null)
          : null;
    if (!colour) return null;
  }
  const resolvedColour = colour ?? base?.colour;
  if (!resolvedColour) return base === null ? null : undefined;
  const dashName = child(ln, "a:prstDash")?.attrs.val;
  const end = (name: string) => {
    const type = child(ln, name)?.attrs.type;
    return type && type !== "none" ? type : null;
  };
  return {
    width: ln.attrs.w !== undefined ? Math.max(emu(ln.attrs.w), 0.5) : (base?.width ?? 1),
    colour: resolvedColour,
    dash: dashName !== undefined ? (DASHES[dashName] ?? null) : (base?.dash ?? null),
    head: child(ln, "a:headEnd") ? end("a:headEnd") : (base?.head ?? null),
    tail: child(ln, "a:tailEnd") ? end("a:tailEnd") : (base?.tail ?? null),
  };
}

/** An outer shadow (a:outerShdw): its offset and blur, CSS pixels, and colour. */
export interface Shadow {
  x: number;
  y: number;
  blur: number;
  colour: string;
}

/** An effect list's outer shadow; null when it has none (other effects aren't drawn). */
export function readShadow(
  effectLst: XmlElement | undefined,
  context: ColourContext,
): Shadow | null {
  const shadow = effectLst && child(effectLst, "a:outerShdw");
  if (!shadow) return null;
  const colour = resolveColour(colourElement(shadow), context);
  if (!colour || colour.a <= 0) return null;
  const distance = emu(shadow.attrs.dist);
  const direction = (numberAttr(shadow, "dir", 0) / 60000) * (Math.PI / 180);
  return {
    x: Math.round(distance * Math.cos(direction) * 100) / 100,
    y: Math.round(distance * Math.sin(direction) * 100) / 100,
    blur: emu(shadow.attrs.blurRad),
    colour: css(colour),
  };
}
