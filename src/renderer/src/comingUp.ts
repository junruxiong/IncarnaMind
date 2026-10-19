export interface CalendarMeeting {
  id: string;
  title: string;
  /** ISO 8601, UTC. */
  start: string;
  attendees: readonly string[];
}

const NONE: readonly CalendarMeeting[] = [];

// Calendars are connected in #190, which fills this in; until then nothing is coming up.
export const useComingUp = (): readonly CalendarMeeting[] => NONE;
