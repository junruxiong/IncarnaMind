import { describe, expect, test } from "vitest";
import { translate } from "../../src/shared/i18n";
import {
  isTheirs,
  roomVoice,
  SPEAKER_ME,
  SPEAKER_THEM,
  SPEAKER_UNKNOWN,
  speakerKey,
  speakerName,
  theirVoice,
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

  test("voices a cloud service told apart are numbered and worded in the interface's language; only the other side's count as theirs", () => {
    const en = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
      translate("en", key, params);
    const zh = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
      translate("zh-CN", key, params);
    expect(speakerName(theirVoice(2), en)).toBe("Them 2");
    expect(speakerName(theirVoice(2), zh)).toBe("对方 2");
    expect(speakerName(roomVoice(2), en)).toBe("Speaker 2");
    expect(speakerName(roomVoice(2), zh)).toBe("发言人 2");
    expect([theirVoice(2), SPEAKER_THEM].every(isTheirs)).toBe(true);
    // In a room anyone may be the User: a room's voices are never the other side.
    expect([roomVoice(2), SPEAKER_ME, SPEAKER_UNKNOWN, "Dana", null].some(isTheirs)).toBe(false);
  });
});
