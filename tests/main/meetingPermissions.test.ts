import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { section } from "../helpers/electronBuilder";

const REPO_ROOT = join(__dirname, "..", "..");
const read = (path: string) => readFile(join(REPO_ROOT, path), "utf8");

describe("The macOS build may record a meeting (ADR-0015)", () => {
  test("it says why it asks for the microphone and the computer's sound, in English and Chinese", async () => {
    const mac = section(await read("electron-builder.yml"), "mac");
    expect(mac).toMatch(
      /^ {4}NSMicrophoneUsageDescription: "When you record a meeting, IncarnaMind records your microphone\. .+"$/m,
    );
    expect(mac).toMatch(
      /^ {4}NSAudioCaptureUsageDescription: "When you record a meeting, IncarnaMind records the computer's sound: .+"$/m,
    );
    expect(mac).toMatch(/^ {4}- from: build\/zh_CN\.lproj\n {6}to: zh_CN\.lproj$/m);

    const chinese = await read("build/zh_CN.lproj/InfoPlist.strings");
    expect(chinese).toMatch(/^"NSMicrophoneUsageDescription" = "录制会议时，[^"]*麦克风[^"]*";$/m);
    expect(chinese).toMatch(
      /^"NSAudioCaptureUsageDescription" = "录制会议时，[^"]*电脑声音[^"]*";$/m,
    );
  });

  test("a hardened-runtime build may open the microphone, and keeps electron-builder's entitlements", async () => {
    const mac = section(await read("electron-builder.yml"), "mac");
    expect(mac).toMatch(/^ {2}entitlements: build\/entitlements\.mac\.plist$/m);
    expect(mac).toMatch(/^ {2}entitlementsInherit: build\/entitlements\.mac\.plist$/m);

    const entitlements = await read("build/entitlements.mac.plist");
    for (const key of [
      "com.apple.security.device.audio-input",
      "com.apple.security.cs.allow-jit",
      "com.apple.security.cs.allow-unsigned-executable-memory",
      "com.apple.security.cs.disable-library-validation",
    ]) {
      expect(entitlements).toMatch(
        new RegExp(`<key>${key.replaceAll(".", "\\.")}</key>\\s*<true/>`),
      );
    }
  });
});
