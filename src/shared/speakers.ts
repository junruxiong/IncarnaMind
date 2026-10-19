import type { MessageKey, MessageParams } from "./i18n";

/**
 * Who a live line of a Meeting's Transcript is from, before speakers are told
 * apart (#941): the microphone, the computer's sound, or a room with no
 * computer sound. The Transcript's file keeps these words as they are, in a
 * voice span (`<v Me>`); the window words them in the interface's language.
 */
export const SPEAKER_ME = "Me";
export const SPEAKER_THEM = "Them";
export const SPEAKER_UNKNOWN = "Speaker ?";

/**
 * Someone a cloud service told apart as they spoke (#141), numbered in the order heard in the
 * Meeting: in a call, one of the other side ("Them 2"); in a room, anyone ("Speaker 2"), who may
 * be the User.
 */
export const theirVoice = (n: number) => `Them ${n}`;
export const roomVoice = (n: number) => `Speaker ${n}`;
const THEIRS = /^Them (\d+)$/;
const ROOM = /^Speaker (\d+)$/;

/** Another person's line: the computer's sound in a call, as one voice or told apart. */
export const isTheirs = (speaker: string | null) =>
  speaker !== null && (speaker === SPEAKER_THEM || THEIRS.test(speaker));

const KEYS: Readonly<Record<string, MessageKey>> = {
  [SPEAKER_ME]: "transcript.speaker.me",
  [SPEAKER_THEM]: "transcript.speaker.them",
  [SPEAKER_UNKNOWN]: "transcript.speaker.unknown",
};

export const speakerKey = (speaker: string): MessageKey | null => KEYS[speaker] ?? null;

export function speakerName(
  speaker: string,
  t: (key: MessageKey, params?: MessageParams) => string,
): string {
  const key = speakerKey(speaker);
  if (key) return t(key);
  const theirs = THEIRS.exec(speaker);
  if (theirs) return t("transcript.speaker.themNumbered", { n: theirs[1] as string });
  const room = ROOM.exec(speaker);
  return room ? t("transcript.speaker.numbered", { n: room[1] as string }) : speaker;
}
