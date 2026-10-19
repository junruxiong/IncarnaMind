import { describe, expect, test } from "vitest";
import { translate } from "../../src/shared/i18n";
import {
  isTheirs,
  numberedSpeaker,
  SPEAKER_ME,
  SPEAKER_THEM,
  SPEAKER_UNKNOWN,
  speakerKey,
  speakerName,
} from "../../src/shared/speakers";

describe("The words for who said a live line (#941)", () => {
  test.each([
    [SPEAKER_ME, "Me", "我"],
    [SPEAKER_THEM, "Them", "对方"],
    [SPEAKER_UNKNOWN, "Speaker ?", "发言人 ?"],
  ])("%s is worded in English and Chinese", (speaker, english, chinese) => {
    const key = speakerKey(speaker);
    expect(key).not.toBeNull();
    if (!key) return;
    expect(translate("en", key)).toBe(english);
    expect(translate("zh-CN", key)).toBe(chinese);
  });

  test("a person's name is shown as it is", () => {
    expect(speakerKey("Dana")).toBeNull();
    expect(speakerKey("Speaker 1")).toBeNull();
  });

  test("someone a cloud service told apart is numbered, worded in the interface's language, and is the other side", () => {
    const speaker = numberedSpeaker(2);
    expect(speakerName(speaker, (key, params) => translate("en", key, params))).toBe("Speaker 2");
    expect(speakerName(speaker, (key, params) => translate("zh-CN", key, params))).toBe("发言人 2");
    expect([speaker, SPEAKER_THEM].every(isTheirs)).toBe(true);
    expect([SPEAKER_ME, SPEAKER_UNKNOWN, "Dana", null].some(isTheirs)).toBe(false);
  });
});
