/**
 * One-click help during a call, in a window: the button pressed until
 * its Answer shows, so its outline is drawn at once; why it couldn't be asked;
 * and the buttons' shortcuts as this system writes them.
 */
import { useEffect } from "react";
import { create } from "zustand";
import type { ChatReadiness, Meeting, MeetingHelpKind, Settings } from "../../core/api";
import { helpKeysOf, isGeneralHelpKind } from "../../shared/meetingHelp";
import { keysHere } from "./commands/keymap";
import { openConversation, useConversations } from "./conversations";
import { core, files } from "./core";
import { errorMessage } from "./errors";
import { translateNow } from "./i18n";
import { isLive, useMeetings } from "./meetings";
import { useAppStore } from "./store";

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

interface HelpState {
  /** The button pressed, by Meeting, until its Answer is the Meeting's `help`. */
  pending: Readonly<Record<string, MeetingHelpKind>>;
  /** Why the last press couldn't be asked, by Meeting: no model ready, or a problem in words. */
  blocked: Readonly<
    Record<string, { readiness: Extract<ChatReadiness, { ready: false }> } | { message: string }>
  >;
  /** Whether the floating panel shows now. */
  panelShown: boolean;
}

export const useMeetingHelp = create<HelpState>()(() => ({
  pending: {},
  blocked: {},
  panelShown: false,
}));

function settle(meetingId: string, blocked?: HelpState["blocked"][string]): void {
  useMeetingHelp.setState((state) => {
    const { [meetingId]: _pending, ...pending } = state.pending;
    const { [meetingId]: _blocked, ...rest } = state.blocked;
    return { pending, blocked: blocked ? { ...rest, [meetingId]: blocked } : rest };
  });
}

// A press on a Meeting that has since stopped leaves no outline behind.
useMeetings.subscribe((state) => {
  const { pending } = useMeetingHelp.getState();
  for (const meetingId of Object.keys(pending)) {
    const meeting = state.meetings.find((each) => each.id === meetingId);
    if (!isLive(meeting)) settle(meetingId);
  }
});

/**
 * Asks for a button's help (`AskMeetingHelpInput`); resolves to the conversation and Answer, or
 * null when it couldn't be asked, which `blocked` then says.
 */
export async function askHelp(
  meeting: Meeting,
  kind: MeetingHelpKind,
  conversationId?: string | null,
): Promise<{ conversationId: string; answerId: string } | null> {
  useMeetingHelp.setState((state) => ({
    pending: { ...state.pending, [meeting.id]: kind },
    blocked: Object.fromEntries(
      Object.entries(state.blocked).filter(([meetingId]) => meetingId !== meeting.id),
    ),
  }));
  try {
    const result = await core.askMeetingHelp({
      meetingId: meeting.id,
      kind,
      ...(conversationId !== undefined && { conversationId }),
    });
    if (!result.asked) {
      settle(
        meeting.id,
        result.reason === "not-ready"
          ? { readiness: result.readiness }
          : { message: translateNow("meetings.help.failed", { message: result.skill }) },
      );
      return null;
    }
    settle(meeting.id);
    return { conversationId: result.conversation.id, answerId: result.answerId };
  } catch (error) {
    settle(meeting.id, { message: errorMessage(error, translateNow) });
    return null;
  }
}

/** The Meeting recording or paused now, if there is one. */
export const liveMeetingNow = (): Meeting | undefined =>
  useMeetings.getState().meetings.find((meeting) => isLive(meeting));

/**
 * Asks in the assistant, as its buttons and ⌘K do: in the conversation it shows, which opens
 * there if it was a new chat.
 */
export async function askHelpInAssistant(kind: MeetingHelpKind): Promise<void> {
  const meeting = liveMeetingNow();
  if (!meeting) return;
  const asked = await askHelp(meeting, kind, useConversations.getState().openId);
  if (asked && useConversations.getState().openId !== asked.conversationId) {
    await openConversation(asked.conversationId);
  }
}

/** A button's shortcut on this device, as this system writes it; null for none. */
export function helpKeysLabel(kind: MeetingHelpKind, settings: Settings | null): string | null {
  if (!isGeneralHelpKind(kind) || !settings) return null;
  const keys = helpKeysOf(kind, settings.device.meetingHelpKeys, isMac);
  return keys ? keysHere(keys) : null;
}

/**
 * In the app's window: a help shortcut pressed with no floating panel asks in the assistant;
 * the panel's Set up opens where a model is chosen; and whether the panel shows.
 */
export function useMeetingHelpInWindow(): void {
  useEffect(() => {
    const stops = [
      files.onMeetingHelp((kind) => void askHelpInAssistant(kind)),
      files.onSetUpModel(() => useAppStore.getState().openSettings("models")),
      files.onMeetingPanel((panelShown) => useMeetingHelp.setState({ panelShown })),
    ];
    return () => {
      for (const stop of stops) stop();
    };
  }, []);
}
