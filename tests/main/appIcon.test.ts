import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { section } from "../helpers/electronBuilder";

const REPO_ROOT = join(__dirname, "..", "..");
const ICON = join(REPO_ROOT, "build", "icon");
const builderConfig = () => readFileSync(join(REPO_ROOT, "electron-builder.yml"), "utf8");

function icnsTypes(icns: Buffer): string[] {
  const types: string[] = [];
  for (let at = 8; at < icns.length; at += icns.readUInt32BE(at + 4)) {
    types.push(icns.toString("latin1", at, at + 4));
  }
  return types;
}

/** An .ico entry's width is one byte, so 0 stands for 256. */
function icoSizes(ico: Buffer): number[] {
  return Array.from(
    { length: ico.readUInt16LE(4) },
    (_, index) => ico.readUInt8(6 + 16 * index) || 256,
  );
}

const pngWidth = (png: Buffer) => png.readUInt32BE(16);

describe("The app icon", () => {
  test("macOS shows the Icon Composer icon in Default and Dark, with an .icns of every size as the fallback", () => {
    const mac = section(builderConfig(), "mac");
    expect(mac).toMatch(/^ {2}icon: build\/icon\/icon\.icns$/m);
    expect(mac).toMatch(/^ {4}CFBundleIconName: AppIcon$/m);
    expect(mac).toMatch(/^ {4}- from: build\/icon\/Assets\.car\n {6}to: Assets\.car$/m);
    expect(existsSync(join(ICON, "Assets.car"))).toBe(true);

    const composer: unknown = JSON.parse(
      readFileSync(join(ICON, "AppIcon.icon", "icon.json"), "utf8"),
    );
    expect(composer).toMatchObject({
      "fill-specializations": expect.arrayContaining([
        expect.objectContaining({ appearance: "dark" }),
      ]),
      groups: [
        {
          layers: [
            {
              "image-name-specializations": [
                { value: "glyph-light.svg" },
                { appearance: "dark", value: "glyph-dark.svg" },
              ],
            },
          ],
        },
      ],
    });
    for (const glyph of ["glyph-light.svg", "glyph-dark.svg"]) {
      expect(existsSync(join(ICON, "AppIcon.icon", "Assets", glyph))).toBe(true);
    }

    // 16, 32, 128, 256 and 512 pt, each at 1x and 2x.
    expect(icnsTypes(readFileSync(join(ICON, "icon.icns")))).toEqual(
      expect.arrayContaining([
        "ic04",
        "ic05",
        "ic07",
        "ic08",
        "ic09",
        "ic10",
        "ic11",
        "ic12",
        "ic13",
        "ic14",
      ]),
    );
  });

  test("Windows has the icon at 16, 24, 32, 48, 64 and 256 px", () => {
    expect(section(builderConfig(), "win")).toMatch(/^ {2}icon: build\/icon\/icon\.ico$/m);
    expect(icoSizes(readFileSync(join(ICON, "icon.ico")))).toEqual([16, 24, 32, 48, 64, 256]);
  });

  test("Linux has a PNG of each size from 16 to 1024 px, each as big as its name says", () => {
    expect(section(builderConfig(), "linux")).toMatch(/^ {2}icon: build\/icon\/png$/m);
    const files = readdirSync(join(ICON, "png"));
    const sizes = files.map((file) => Number(/^(\d+)x\1\.png$/.exec(file)?.[1]));
    expect(sizes.sort((a, b) => a - b)).toEqual([16, 24, 32, 48, 64, 128, 256, 512, 1024]);
    for (const file of files) {
      expect(pngWidth(readFileSync(join(ICON, "png", file)))).toBe(Number.parseInt(file, 10));
    }
  });
});
