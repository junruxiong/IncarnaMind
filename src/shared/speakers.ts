import type { MessageKey } from "./i18n";

/**
 * Who a live line of a Meeting's Transcript is from, before speakers are told
 * apart (#941): the microphone, the computer's sound, or a room with no
 * computer sound. The Transcript's file keeps these words as they are, in a
 * voice span (`<v Me>`); the window words them in the interface's language.
 */
export const SPEAKER_ME = "Me";
export const SPEAKER_THEM = "Them";
export const SPEAKER_UNKNOWN = "Speaker ?";

const KEYS: Readonly<Record<string, MessageKey>> = {
  [SPEAKER_ME]: "transcript.speaker.me",
  [SPEAKER_THEM]: "transcript.speaker.them",
  [SPEAKER_UNKNOWN]: "transcript.speaker.unknown",
};

/** The words for one of the three labels in the interface's language; null for any other speaker, who is named as is. */
export const speakerKey = (speaker: string): MessageKey | null => KEYS[speaker] ?? null;
