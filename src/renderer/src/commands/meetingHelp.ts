import {
  GENERAL_HELP,
  helpKindsFor,
  helpLabel,
  MEETING_TEMPLATES,
  type MeetingHelpKind,
  type MeetingHelpTemplate,
  TEMPLATE_HELP,
  templateLabel,
} from "../../../shared/meetingHelp";
import { core, files } from "../core";
import { askHelpInAssistant, helpKeysLabel, liveMeetingNow, useMeetingHelp } from "../meetingHelp";
import { canRecordMeetings } from "../meetings";
import { useAppStore } from "../store";
import type { Command } from "./registry";

/*
 * Help during a call, from ⌘K: each button, asked in the assistant; showing or hiding
 * the floating panel; and the meeting's template, which adds buttons for its kind.
 */

const meetingsOn = () => canRecordMeetings(useAppStore.getState().settings);

const whileLive = () =>
  liveMeetingNow() ? true : ({ reason: "meetings.command.noneLive" } as const);

const helpCommand = (kind: MeetingHelpKind, general: boolean): Command => ({
  id: `meeting.help.${kind}`,
  label: helpLabel(kind),
  keywords: ["meetings.help.label"],
  group: "main",
  // A Question in the assistant's conversation, which clearing the conversation takes away.
  reversible: true,
  when: () =>
    meetingsOn() &&
    (general || helpKindsFor(liveMeetingNow()?.helpTemplate ?? null).includes(kind)),
  enabled: whileLive,
  globalKeys: () => helpKeysLabel(kind, useAppStore.getState().settings),
  run: () => askHelpInAssistant(kind),
});

const templateCommand = (template: MeetingHelpTemplate | null): Command => ({
  id: `meeting.template-${template ?? "none"}`,
  label: (_ctx, t) =>
    t("meetings.template.command", {
      name: t(template ? templateLabel(template) : "meetings.template.none"),
    }),
  keywords: ["meetings.template.chip"],
  group: "main",
  // Choosing another template takes it back.
  reversible: true,
  when: meetingsOn,
  enabled: whileLive,
  async run() {
    const meeting = liveMeetingNow();
    if (meeting) await core.setMeetingTemplate({ meetingId: meeting.id, template });
  },
});

export const meetingHelpCommands: readonly Command[] = [
  ...GENERAL_HELP.map((kind) => helpCommand(kind, true)),
  ...MEETING_TEMPLATES.flatMap((template) =>
    TEMPLATE_HELP[template].map((kind) => helpCommand(kind, false)),
  ),
  {
    id: "meeting.panel",
    label: (_ctx, t) =>
      t(useMeetingHelp.getState().panelShown ? "meetings.panel.hide" : "meetings.panel.show"),
    group: "main",
    // Pressed again, it undoes itself.
    reversible: true,
    when: meetingsOn,
    enabled: whileLive,
    run: () => files.showMeetingPanel(!useMeetingHelp.getState().panelShown),
  },
  templateCommand(null),
  ...MEETING_TEMPLATES.map(templateCommand),
];
