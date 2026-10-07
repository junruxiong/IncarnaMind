/**
 * Approvals (#38): a Tool call that could change something asks the User
 * before it runs.
 *
 * - Read-only marks are hints. A Connector says which of its Tools only read
 *   (MCP's `readOnlyHint`); IncarnaMind can't check that. Such a Tool runs
 *   without asking unless the User switched it to "ask"; every other Tool
 *   asks first, unless the User always allows it.
 * - Policies replace the default for one subject: a Connector's Tool, or (#41)
 *   a Skill's scripts. They are rows in SQLite, sync-ready (ADR-0003): UUIDs,
 *   timestamps, soft deletes, and no secrets.
 * - Requests: an Answer's call that asks raises "approval.requested" and waits
 *   for `respond`, racing the Answer's abort signal, so stopping the Answer
 *   while it waits withdraws the request at once ("approval.resolved", deny).
 *   Closing the core denies whatever is still waiting.
 */
import { randomUUID } from "node:crypto";
import type {
  ApprovalDecision,
  ApprovalPolicy,
  ApprovalPolicyValue,
  ApprovalRequest,
  ApprovalSubject,
  ApprovalSubjectKind,
} from "./api";
import { InvalidInputError, isRecord, NotFoundError } from "./errors";
import type { createEventHub } from "./events";
import type { Database } from "./storage";

/** The names of what policies are about, as they are now; live ones only. */
export interface ApprovalOwners {
  /** Live Connectors' names, by id. */
  connectors(): ReadonlyMap<string, string>;
  /** Live Skills' names, by id. */
  skills(): ReadonlyMap<string, string>;
}

export interface ApprovalsOptions {
  db: Database;
  events: ReturnType<typeof createEventHub>;
  now: () => string;
  owners: ApprovalOwners;
}

/** A Tool call that asks, as an Answer hands it over. */
export type ToolCallToApprove = Omit<ApprovalRequest, "requestId" | "subject">;

export type Approvals = ReturnType<typeof createApprovals>;

interface PolicyRow {
  id: string;
  subject_kind: string;
  subject_id: string;
  policy: string;
  created_at: string;
  updated_at: string;
}

interface Pending {
  request: ApprovalRequest;
  /** Ends the wait with the User's decision. */
  settle(decision: ApprovalDecision): void;
}

const decisions: readonly ApprovalDecision[] = ["allow-once", "always-allow", "deny"];
const COLUMNS = "id, subject_kind, subject_id, policy, created_at, updated_at";

/** How a subject is stored: its kind, and one text id. */
function keyOf(subject: ApprovalSubject): { kind: ApprovalSubjectKind; id: string } {
  return subject.kind === "tool"
    ? { kind: "tool", id: `${subject.connectorId}:${subject.tool}` }
    : { kind: "skill-script", id: subject.skillId };
}

/** A stored subject, or null for one this version doesn't know (e.g. synced from a newer one). */
function subjectOf(row: Pick<PolicyRow, "subject_kind" | "subject_id">): ApprovalSubject | null {
  if (row.subject_kind === "tool") {
    const at = row.subject_id.indexOf(":");
    if (at <= 0 || at === row.subject_id.length - 1) return null;
    return {
      kind: "tool",
      connectorId: row.subject_id.slice(0, at),
      tool: row.subject_id.slice(at + 1),
    };
  }
  if (row.subject_kind === "skill-script") return { kind: "skill-script", skillId: row.subject_id };
  return null;
}

const isPolicy = (value: unknown): value is ApprovalPolicyValue =>
  value === "always" || value === "ask";

const sameSubject = (a: ApprovalSubject, b: ApprovalSubject) => {
  const [x, y] = [keyOf(a), keyOf(b)];
  return x.kind === y.kind && x.id === y.id;
};

const nonEmpty = (value: unknown, what: string): string => {
  if (typeof value !== "string" || value === "") {
    throw new InvalidInputError(`${what} must be a non-empty string.`);
  }
  return value;
};

/** A subject as the public interface takes it. */
function parseSubject(value: unknown): ApprovalSubject {
  if (!isRecord(value)) throw new InvalidInputError("subject must be an object.");
  if (value.kind === "tool") {
    return {
      kind: "tool",
      connectorId: nonEmpty(value.connectorId, "subject.connectorId"),
      tool: nonEmpty(value.tool, "subject.tool"),
    };
  }
  if (value.kind === "skill-script") {
    return { kind: "skill-script", skillId: nonEmpty(value.skillId, "subject.skillId") };
  }
  throw new InvalidInputError('subject.kind must be "tool" or "skill-script".');
}

export function createApprovals(options: ApprovalsOptions) {
  const { db, events, now, owners } = options;
  const pending = new Map<string, Pending>();
  let closed = false;

  const rowFor = (subject: ApprovalSubject) => {
    const { kind, id } = keyOf(subject);
    return db.get<PolicyRow>(
      `SELECT ${COLUMNS} FROM approval_policies
       WHERE subject_kind = ? AND subject_id = ? AND deleted_at IS NULL`,
      [kind, id],
    );
  };

  /** The User's policy for a subject, or null for the default. */
  const policyOf = (subject: ApprovalSubject): ApprovalPolicyValue | null => {
    const policy = rowFor(subject)?.policy;
    return isPolicy(policy) ? policy : null;
  };

  /** Live policies whose subject still exists, by owner name, then Tool name. */
  const list = (): ApprovalPolicy[] => {
    const connectors = owners.connectors();
    const skills = owners.skills();
    const policies: ApprovalPolicy[] = [];
    for (const row of db.all<PolicyRow>(
      `SELECT ${COLUMNS} FROM approval_policies WHERE deleted_at IS NULL ORDER BY created_at, rowid`,
    )) {
      const subject = subjectOf(row);
      if (!subject || !isPolicy(row.policy)) continue;
      const ownerName =
        subject.kind === "tool" ? connectors.get(subject.connectorId) : skills.get(subject.skillId);
      if (ownerName === undefined) continue;
      policies.push({
        id: row.id,
        subject,
        policy: row.policy,
        ownerName,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
      });
    }
    const label = (policy: ApprovalPolicy) =>
      policy.subject.kind === "tool" ? policy.subject.tool : "";
    return policies.sort(
      (a, b) =>
        a.ownerName.localeCompare(b.ownerName, undefined, { sensitivity: "base" }) ||
        label(a).localeCompare(label(b)),
    );
  };

  const changed = () => {
    if (!closed) events.emit("approvals.changed", list());
  };

  /** Writes a policy (null removes it); true if anything changed. Doesn't announce it. */
  const store = (subject: ApprovalSubject, policy: ApprovalPolicyValue | null): boolean =>
    db.transaction(() => {
      const row = rowFor(subject);
      const at = now();
      if (policy === null) {
        if (!row) return false;
        db.run("UPDATE approval_policies SET deleted_at = ?, updated_at = ? WHERE id = ?", [
          at,
          at,
          row.id,
        ]);
        return true;
      }
      if (row) {
        if (row.policy === policy) return false;
        db.run("UPDATE approval_policies SET policy = ?, updated_at = ? WHERE id = ?", [
          policy,
          at,
          row.id,
        ]);
        return true;
      }
      const { kind, id } = keyOf(subject);
      db.run(
        `INSERT INTO approval_policies (id, subject_kind, subject_id, policy, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
        [randomUUID(), kind, id, policy, at, at],
      );
      return true;
    });

  /** Takes a request off the waiting list and tells every window; false if it wasn't waiting. */
  const withdraw = (requestId: string, decision: ApprovalDecision): boolean => {
    if (!pending.delete(requestId)) return false;
    if (!closed) events.emit("approval.resolved", { requestId, decision });
    return true;
  };

  /** Ends a waiting request with a decision. */
  const settle = (entry: Pending, decision: ApprovalDecision) => {
    if (withdraw(entry.request.requestId, decision)) entry.settle(decision);
  };

  /** Soft-deletes the policies matching `where`, and denies the requests matching `waiting`. */
  const forget = (
    matches: (subject: ApprovalSubject) => boolean,
    waiting: (request: ApprovalRequest) => boolean,
  ) => {
    const at = now();
    let removed = false;
    db.transaction(() => {
      for (const row of db.all<PolicyRow>(
        `SELECT ${COLUMNS} FROM approval_policies WHERE deleted_at IS NULL`,
      )) {
        const subject = subjectOf(row);
        if (!subject || !matches(subject)) continue;
        db.run("UPDATE approval_policies SET deleted_at = ?, updated_at = ? WHERE id = ?", [
          at,
          at,
          row.id,
        ]);
        removed = true;
      }
    });
    for (const entry of [...pending.values()]) if (waiting(entry.request)) settle(entry, "deny");
    if (removed) changed();
  };

  return {
    list,

    /**
     * Whether a call of a Connector's Tool must ask first: the User's policy
     * for it, or else its Connector's read-only claim.
     */
    toolNeedsApproval(connectorId: string, tool: string, readOnly: boolean): boolean {
      const policy = policyOf({ kind: "tool", connectorId, tool });
      return policy === "ask" || (policy !== "always" && !readOnly);
    },

    /**
     * Asks the User about a Tool call, and waits: resolves true if they allow
     * it, false if they deny it (or the core closed). If `signal` aborts first,
     * e.g. the User stopped the Answer, the request is withdrawn as denied and
     * the promise rejects with the signal's reason.
     */
    request(call: ToolCallToApprove, signal: AbortSignal): Promise<boolean> {
      if (signal.aborted) return Promise.reject(signal.reason);
      if (closed) return Promise.resolve(false);
      const request: ApprovalRequest = {
        ...call,
        requestId: randomUUID(),
        subject: { kind: "tool", connectorId: call.connector.id, tool: call.tool },
      };
      return new Promise<boolean>((resolve, reject) => {
        const onAbort = () => {
          if (withdraw(request.requestId, "deny")) reject(signal.reason);
        };
        pending.set(request.requestId, {
          request,
          settle: (decision) => {
            signal.removeEventListener("abort", onAbort);
            resolve(decision !== "deny");
          },
        });
        signal.addEventListener("abort", onAbort, { once: true });
        events.emit("approval.requested", request);
      });
    },

    /** Requests still waiting, oldest first. */
    requests: (): ApprovalRequest[] => [...pending.values()].map((entry) => entry.request),

    respond(requestId: unknown, decision: unknown): void {
      if (typeof requestId !== "string") throw new InvalidInputError("requestId must be text.");
      if (!decisions.includes(decision as ApprovalDecision)) {
        throw new InvalidInputError('decision must be "allow-once", "always-allow" or "deny".');
      }
      const entry = pending.get(requestId);
      if (!entry) return; // Already decided, e.g. in another window, or its Answer stopped.
      const chosen = decision as ApprovalDecision;
      if (chosen !== "always-allow") {
        settle(entry, chosen);
        return;
      }
      const { subject } = entry.request;
      const stored = store(subject, "always");
      // Calls of the same Tool waiting elsewhere, e.g. in another Answer, are allowed too.
      for (const other of [...pending.values()]) {
        if (sameSubject(other.request.subject, subject)) settle(other, "always-allow");
      }
      if (stored) changed();
    },

    /** Sets or removes the User's policy for a subject that exists; returns it, or null for the default. */
    set(input: unknown): ApprovalPolicy | null {
      if (!isRecord(input)) throw new InvalidInputError("setApprovalPolicy expects an object.");
      const subject = parseSubject(input.subject);
      const { policy } = input;
      if (policy !== null && !isPolicy(policy)) {
        throw new InvalidInputError('policy must be "always", "ask" or null.');
      }
      if (subject.kind === "tool" && !owners.connectors().has(subject.connectorId)) {
        throw new NotFoundError("That Connector doesn't exist.");
      }
      if (subject.kind === "skill-script") {
        if (!owners.skills().has(subject.skillId))
          throw new NotFoundError("That Skill doesn't exist.");
        if (policy === "ask") {
          throw new InvalidInputError('A Skill\'s scripts always ask unless set to "always".');
        }
      }
      if (store(subject, policy)) changed();
      if (policy === null) return null;
      return list().find((each) => sameSubject(each.subject, subject)) ?? null;
    },

    /** Removes a policy by its id. One that's gone, or never was, does nothing. */
    revoke(policyId: unknown): void {
      if (typeof policyId !== "string" || policyId === "") {
        throw new InvalidInputError("A policy id must be a non-empty string.");
      }
      const at = now();
      const live = db.get<{ id: string }>(
        "SELECT id FROM approval_policies WHERE id = ? AND deleted_at IS NULL",
        [policyId],
      );
      if (!live) return;
      db.run("UPDATE approval_policies SET deleted_at = ?, updated_at = ? WHERE id = ?", [
        at,
        at,
        policyId,
      ]);
      changed();
    },

    /**
     * A Connector was deleted: its policies go with it (added again, it is a
     * new Connector), and calls of its Tools still waiting are denied.
     */
    forgetConnector(connectorId: string): void {
      forget(
        (subject) => subject.kind === "tool" && subject.connectorId === connectorId,
        (request) => request.connector.id === connectorId,
      );
    },

    /** A Skill was removed: its policy goes with it. */
    forgetSkill(skillId: string): void {
      forget(
        (subject) => subject.kind === "skill-script" && subject.skillId === skillId,
        () => false,
      );
    },

    /** Denies every request still waiting, e.g. when the app quits. */
    close(): void {
      for (const entry of [...pending.values()]) settle(entry, "deny");
      closed = true;
    },
  };
}
