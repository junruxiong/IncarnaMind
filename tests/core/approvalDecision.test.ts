/**
 * One approval decision (#62; docs/designs/agent-extensibility.md §4.3): a
 * call runs or asks the User first by its Effects and the User's policy for
 * it. The golden table is every case v1 decides, with v1's outcome: a
 * Connector's Tool, marked read-only or not, with no policy, "always" or
 * "ask"; and a Skill script, with no policy or "always". Each call's Effects
 * are the ones its Tool declares. Each case is decided in a Run that had read
 * untrusted content (tainted, #64) and in one that hadn't, with the same
 * outcome: for v1's subjects taint is only recorded (decision D3, §4.6).
 */
import { describe, expect, test } from "vitest";
import type { ApprovalPolicyValue, ApprovalSubject, Effect } from "../../src/core";
import { createApprovals } from "../../src/core/approvals";
import { connectorToolEffects } from "../../src/core/connectors";
import { createEventHub } from "../../src/core/events";
import { createScriptRunner } from "../../src/core/skills/scripts";
import { scriptEffects } from "../../src/core/skills/tools";
import { migrate, openDatabase } from "../../src/core/storage";

const CONNECTOR = { id: "connector-1", name: "Tides" };
const SKILL = { skillId: "skill-1", skillDir: "/data/skills/skill-1" };

/** Approvals over an empty database, with one Connector and one Skill. */
function approvalsWith(policy: ApprovalPolicyValue | null, subject: ApprovalSubject) {
  const db = openDatabase(":memory:");
  migrate(db);
  const approvals = createApprovals({
    db,
    events: createEventHub(),
    now: () => new Date().toISOString(),
    owners: {
      connectors: () => new Map([[CONNECTOR.id, CONNECTOR.name]]),
      skills: () => new Map([[SKILL.skillId, "toolbox"]]),
    },
  });
  if (policy) approvals.set({ subject, policy, riskAccepted: true });
  return approvals;
}

const tool = (name: string): ApprovalSubject => ({
  kind: "tool",
  connectorId: CONNECTOR.id,
  tool: name,
});
const script: ApprovalSubject = { kind: "skill-script", skillId: SKILL.skillId };

/** What a Connector Tool's call can do, as its Tool declares it. */
const connectorCall = (readOnly: boolean) =>
  connectorToolEffects({ id: `connector:${CONNECTOR.id}`, name: CONNECTOR.name }, readOnly);

/** What a Skill script's run can do on v1's Executor (sandbox level "none"), as `run_skill_script` declares it. */
const scriptRun = (): Effect[] => {
  const runner = createScriptRunner({
    executor: { level: "none", run: () => Promise.reject(new Error("Nothing runs here.")) },
  });
  return scriptEffects(SKILL, runner.access(SKILL.skillDir));
};

/**
 * v1's outcomes. Before Effects, a Connector's Tool asked when the User chose
 * "ask", or else unless it was read-only or always allowed
 * (`toolNeedsApproval`); a Skill script asked unless its Skill's scripts
 * always run (`scriptNeedsApproval`).
 */
const GOLDEN: {
  call: string;
  subject: ApprovalSubject;
  effects: () => Effect[];
  policy: ApprovalPolicyValue | null;
  outcome: "run" | "ask";
}[] = [
  {
    call: "a Connector Tool marked read-only",
    subject: tool("lookup_tide"),
    effects: () => connectorCall(true),
    policy: null,
    outcome: "run",
  },
  {
    call: "a Connector Tool marked read-only",
    subject: tool("lookup_tide"),
    effects: () => connectorCall(true),
    policy: "always",
    outcome: "run",
  },
  {
    call: "a Connector Tool marked read-only",
    subject: tool("lookup_tide"),
    effects: () => connectorCall(true),
    policy: "ask",
    outcome: "ask",
  },
  {
    call: "a Connector Tool not marked read-only",
    subject: tool("book_boat"),
    effects: () => connectorCall(false),
    policy: null,
    outcome: "ask",
  },
  {
    call: "a Connector Tool not marked read-only",
    subject: tool("book_boat"),
    effects: () => connectorCall(false),
    policy: "always",
    outcome: "run",
  },
  {
    call: "a Connector Tool not marked read-only",
    subject: tool("book_boat"),
    effects: () => connectorCall(false),
    policy: "ask",
    outcome: "ask",
  },
  {
    call: "a Skill script",
    subject: script,
    effects: scriptRun,
    policy: null,
    outcome: "ask",
  },
  {
    call: "a Skill script",
    subject: script,
    effects: scriptRun,
    policy: "always",
    outcome: "run",
  },
];

/** The golden table with its taint column: each case, in a Run that hadn't read untrusted content and in one that had. */
const GOLDEN_WITH_TAINT = GOLDEN.flatMap((row) =>
  [false, true].map((tainted) => ({ ...row, tainted })),
);

describe("One approval decision gives v1's outcomes", () => {
  test.each(GOLDEN_WITH_TAINT)(
    "$call, policy $policy, tainted $tainted: $outcome",
    ({ subject, effects, policy, tainted, outcome }) => {
      const approvals = approvalsWith(policy, subject);

      expect(approvals.decide({ subject, effects: effects(), tainted })).toBe(outcome);
    },
  );

  test("a policy covers only its own subject: another Tool of the Connector keeps its default", () => {
    const approvals = approvalsWith("always", tool("book_boat"));

    expect(
      approvals.decide({
        subject: tool("cancel_boat"),
        effects: connectorCall(false),
        tainted: false,
      }),
    ).toBe("ask");
  });
});
