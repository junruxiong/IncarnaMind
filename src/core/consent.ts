/**
 * Data-flow consent: nothing leaves the machine on a flow until the User has
 * accepted that flow for that service (ADR-0022).
 *
 * Core modules register their external flows here and call `ensure` before
 * every request. Choosing a service where a line says what goes to it (the
 * chat model's provider in setup or Settings, a search service, a Connector)
 * allows its flows at once: `choose`. A flow the User hasn't allowed asks
 * once: `ensure` raises a "consent.requested" event and waits for the UI to
 * answer with `respondToConsent`, except on a background flow, which never
 * asks and throws `ConsentNeededError`, so its work waits. Decisions are kept
 * per flow and service. A flow that starts sending a new kind of data asks
 * again; a declined flow stays unused until the User revokes the decision or
 * allows it in Settings (the Privacy page lists every registered flow).
 */
import { randomUUID } from "node:crypto";
import {
  type ConsentRequest,
  type DataFlow,
  type DataFlowId,
  type DataFlowStatus,
  type DataKind,
  dataFlowIds,
  dataKinds,
  type ExternalService,
  type RegisteredDataFlow,
} from "./api";
import {
  ConsentDeclinedError,
  ConsentNeededError,
  InvalidInputError,
  NotFoundError,
} from "./errors";
import type { createEventHub } from "./events";
import type { Database } from "./storage";

export interface DataFlowDefinition {
  id: DataFlowId;
  /** Every kind of data the flow sends. When this grows, the User is asked again. */
  sends: readonly DataKind[];
  /** What it sends to one service besides (GovInfo is sent the User's key), asked about for that service alone. */
  sendsTo?(serviceId: string): readonly DataKind[];
  /** The external services the flow currently goes to; empty when everything stays on this computer. */
  services(): Promise<ExternalService[]>;
  /** Background work, which never asks (ADR-0022): it waits until the User allows the flow. */
  background?: boolean;
}

/** The registry of every external data flow. Registering an id again replaces it. */
export interface DataFlowRegistry {
  register(definition: DataFlowDefinition): void;
  get(id: DataFlowId): DataFlowDefinition | undefined;
  list(): DataFlowDefinition[];
}

type Decision = "accepted" | "declined";

interface ConsentRow {
  decision: Decision;
  data_kinds: string;
  chosen: number;
  updated_at: string;
}

interface Pending {
  key: string;
  request: ConsentRequest;
  promise: Promise<void>;
  settle(): void;
  fail(error: Error): void;
}

export type ConsentStatus = DataFlowStatus["consent"];

export type Consent = ReturnType<typeof createConsent>;

const isDataKind = (value: unknown): value is DataKind => dataKinds.some((kind) => kind === value);
const isDataFlowId = (value: unknown): value is DataFlowId =>
  dataFlowIds.some((id) => id === value);

export function createConsent(
  db: Database,
  events: ReturnType<typeof createEventHub>,
  now: () => string,
) {
  const definitions = new Map<DataFlowId, DataFlowDefinition>();
  const pending = new Map<string, Pending>();

  const registry: DataFlowRegistry = {
    register(definition) {
      definitions.set(definition.id, { ...definition, sends: [...definition.sends] });
    },
    get: (id) => definitions.get(id),
    list: () => [...definitions.values()],
  };

  const flowFor = (id: DataFlowId, service: ExternalService): DataFlow => {
    const definition = definitions.get(id);
    if (!definition) throw new Error(`No data flow "${id}" is registered.`);
    const sends = [...new Set([...definition.sends, ...(definition.sendsTo?.(service.id) ?? [])])];
    return { id, service: { id: service.id, name: service.name }, sends };
  };

  const find = (flow: DataFlowId, serviceId: string) =>
    db.get<ConsentRow>(
      `SELECT decision, data_kinds, chosen, updated_at FROM data_flow_consents
       WHERE flow = ? AND service_id = ? AND deleted_at IS NULL`,
      [flow, serviceId],
    );

  const acceptedKinds = (row: ConsentRow): DataKind[] => {
    try {
      const kinds: unknown = JSON.parse(row.data_kinds);
      return Array.isArray(kinds) ? kinds.filter(isDataKind) : [];
    } catch {
      return [];
    }
  };

  /** The User's decision on a flow, and what it sends that they haven't accepted. */
  const statusOf = (flow: DataFlow) => {
    const row = find(flow.id, flow.service.id);
    if (!row) {
      return { consent: "not-asked" as const, decidedAt: null, chosen: false, missing: flow.sends };
    }
    if (row.decision === "declined") {
      return {
        consent: "declined" as const,
        decidedAt: row.updated_at,
        chosen: false,
        missing: [],
      };
    }
    const accepted = acceptedKinds(row);
    const missing = flow.sends.filter((kind) => !accepted.includes(kind));
    const chosen = row.chosen === 1;
    return missing.length === 0
      ? { consent: "accepted" as const, decidedAt: row.updated_at, chosen, missing }
      : // Allowed before it sent more: when, so the Privacy page can say it asks again.
        { consent: "not-asked" as const, decidedAt: row.updated_at, chosen, missing };
  };

  /** `chosen`: allowed by choosing the service (ADR-0022), not in a dialog or in Settings › Privacy. */
  const record = (
    flow: DataFlow,
    decision: Decision,
    kinds: readonly DataKind[],
    chosen = false,
  ) => {
    const at = now();
    const json = JSON.stringify(kinds);
    const chosenValue = chosen ? 1 : 0;
    db.transaction(() => {
      if (find(flow.id, flow.service.id)) {
        db.run(
          `UPDATE data_flow_consents
           SET decision = ?, data_kinds = ?, chosen = ?, service_name = ?, updated_at = ?
           WHERE flow = ? AND service_id = ? AND deleted_at IS NULL`,
          [decision, json, chosenValue, flow.service.name, at, flow.id, flow.service.id],
        );
      } else {
        db.run(
          `INSERT INTO data_flow_consents
             (id, flow, service_id, service_name, decision, data_kinds, chosen, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          [
            randomUUID(),
            flow.id,
            flow.service.id,
            flow.service.name,
            decision,
            json,
            chosenValue,
            at,
            at,
          ],
        );
      }
    });
  };

  const keyOf = (flow: DataFlow) => `${flow.id}\n${flow.service.id}`;

  /** The request already waiting for this flow and service, if any. */
  const waitingFor = (flow: DataFlow): Pending | undefined => {
    const key = keyOf(flow);
    for (const entry of pending.values()) if (entry.key === key) return entry;
    return undefined;
  };

  /** Raises a consent request, or joins the one already waiting for this flow and service. */
  const ask = (flow: DataFlow, newKinds: DataKind[]): Pending => {
    const waiting = waitingFor(flow);
    if (waiting) return waiting;
    const key = keyOf(flow);

    let settle!: () => void;
    let fail!: (error: Error) => void;
    const promise = new Promise<void>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    const request: ConsentRequest = { requestId: randomUUID(), flow, newKinds };
    const entry: Pending = { key, request, promise, settle, fail };
    pending.set(request.requestId, entry);
    events.emit("consent.requested", request);
    return entry;
  };

  /** Accepts everything the flow sends, keeping the kinds accepted before. */
  const recordAccepted = (flow: DataFlow, chosen = false) => {
    const row = find(flow.id, flow.service.id);
    const previous = row?.decision === "accepted" ? acceptedKinds(row) : [];
    record(flow, "accepted", [...new Set([...previous, ...flow.sends])], chosen);
  };

  /** Records the User's answer to a waiting request, and lets the requests waiting on it go on. */
  const answer = (entry: Pending, accepted: boolean, chosen = false) => {
    const { requestId, flow } = entry.request;
    pending.delete(requestId);
    if (accepted) recordAccepted(flow, chosen);
    else record(flow, "declined", []);
    events.emit("consent.resolved", { requestId, accepted });
    entry.settle();
  };

  /**
   * The services a flow goes to now, then every other service the User has
   * decided on (e.g. a provider tested but not kept), each once.
   */
  const servicesOf = async (definition: DataFlowDefinition): Promise<ExternalService[]> => {
    const decided = db.all<{ service_id: string; service_name: string }>(
      `SELECT service_id, service_name FROM data_flow_consents
       WHERE flow = ? AND deleted_at IS NULL ORDER BY created_at, rowid`,
      [definition.id],
    );
    const services = [
      ...(await definition.services()),
      ...decided.map((row) => ({ id: row.service_id, name: row.service_name })),
    ];
    const seen = new Set<string>();
    return services.filter((service) => {
      if (seen.has(service.id)) return false;
      seen.add(service.id);
      return true;
    });
  };

  /** Every registered flow, including one going nowhere now, with each service and its decision. */
  const listRegistered = async (): Promise<RegisteredDataFlow[]> => {
    const result: RegisteredDataFlow[] = [];
    for (const definition of definitions.values()) {
      const services = (await servicesOf(definition)).map((service): DataFlowStatus => {
        const flow = flowFor(definition.id, service);
        const { consent, decidedAt, chosen } = statusOf(flow);
        return { flow, consent, decidedAt, chosen };
      });
      result.push({ id: definition.id, sends: [...definition.sends], services });
    }
    return result;
  };

  return {
    registry,

    /** The User's decision on a flow to a service. */
    status(flowId: DataFlowId, service: ExternalService): ConsentStatus {
      return statusOf(flowFor(flowId, service)).consent;
    },

    /** Whether the flow is background work, which never asks. */
    isBackground: (flowId: DataFlowId) => definitions.get(flowId)?.background === true,

    /**
     * Resolves once the User has accepted everything the flow sends to the
     * service, asking them first if needed. Throws `ConsentDeclinedError` if
     * they declined, and, on a background flow, which never asks,
     * `ConsentNeededError` if they haven't allowed it: the caller must not
     * send anything.
     */
    async ensure(flowId: DataFlowId, service: ExternalService): Promise<void> {
      for (;;) {
        const flow = flowFor(flowId, service);
        const { consent, missing } = statusOf(flow);
        if (consent === "accepted") return;
        if (consent === "declined") throw new ConsentDeclinedError(flow);
        if (definitions.get(flowId)?.background) throw new ConsentNeededError(flow);
        // The decision is recorded before the request settles, so the loop re-checks it.
        await ask(flow, missing).promise;
      }
    },

    /**
     * The User chose `service` where a line said what goes to it (ADR-0022):
     * each of `flowIds` to it is allowed, as chosen, and requests waiting for
     * one are answered. A flow the User declined stays declined; one that
     * isn't registered is left out.
     */
    choose(flowIds: readonly DataFlowId[], service: ExternalService): void {
      for (const flowId of flowIds) {
        if (!definitions.has(flowId)) continue;
        const flow = flowFor(flowId, service);
        if (statusOf(flow).consent === "declined") continue;
        const waiting = waitingFor(flow);
        if (waiting) answer(waiting, true, true);
        else recordAccepted(flow, true);
      }
    },

    respond(requestId: unknown, accept: unknown): void {
      if (typeof requestId !== "string") throw new InvalidInputError("requestId must be text.");
      if (typeof accept !== "boolean") throw new InvalidInputError("accept must be true or false.");
      const entry = pending.get(requestId);
      if (!entry) return; // Already answered, e.g. in another window.
      answer(entry, accept);
    },

    /**
     * The User allows a flow to a service from Settings, without being asked.
     * The service must be one the flow goes to now, or one already decided on.
     */
    async allow(flowId: unknown, serviceId: unknown): Promise<void> {
      if (!isDataFlowId(flowId)) throw new InvalidInputError(`Unknown data flow "${flowId}".`);
      if (typeof serviceId !== "string") throw new InvalidInputError("serviceId must be text.");
      const definition = definitions.get(flowId);
      const services = definition ? await servicesOf(definition) : [];
      const service = services.find((each) => each.id === serviceId);
      if (!service)
        throw new NotFoundError(`The "${flowId}" data flow doesn't go to ${serviceId}.`);
      const flow = flowFor(flowId, service);
      const waiting = waitingFor(flow);
      if (waiting) answer(waiting, true);
      else recordAccepted(flow);
    },

    revoke(flowId: unknown, serviceId: unknown): void {
      if (!isDataFlowId(flowId)) throw new InvalidInputError(`Unknown data flow "${flowId}".`);
      if (typeof serviceId !== "string") throw new InvalidInputError("serviceId must be text.");
      const at = now();
      db.run(
        `UPDATE data_flow_consents SET deleted_at = ?, updated_at = ?
         WHERE flow = ? AND service_id = ? AND deleted_at IS NULL`,
        [at, at, flowId, serviceId],
      );
    },

    listRegistered,

    /**
     * Each flow to every service it currently goes to, then every other
     * service the User has decided on, so every decision can be seen and
     * revoked: `listRegistered`, as one list.
     */
    async list(): Promise<DataFlowStatus[]> {
      return (await listRegistered()).flatMap((flow) => flow.services);
    },

    requests(): ConsentRequest[] {
      return [...pending.values()].map((entry) => entry.request);
    },

    /** Fails every request still waiting, so nothing is left hanging after the core closes. */
    close(): void {
      for (const entry of pending.values()) entry.fail(new Error("IncarnaMind closed."));
      pending.clear();
    },
  };
}
