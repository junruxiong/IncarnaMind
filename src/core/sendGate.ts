/**
 * The last check before anything leaves this computer: every request to a
 * service passes it at the moment it is sent, not when its work began. Local
 * mode sends nothing to any cloud service, and a flow the User turned off
 * sends nothing more, whatever was allowed before (ADR-0022). A prepared
 * model, a waiting consent prompt or a paced lookup can outlive both.
 *
 * Background work is never asked (ADR-0022): a flow turned off under it
 * throws `ConsentNeededError`, so it waits as it does before it starts.
 */
import { wrapLanguageModel } from "ai";
import type { DataFlowId, ExternalService } from "./api";
import type { Consent } from "./consent";
import { ConsentNeededError, ConsentRevokedError, LocalModeError } from "./errors";
import type { ChatLanguageModel } from "./providers/models";

export function createSendGate(consent: Consent, localOnly: () => boolean) {
  /** Throws `LocalModeError`, `ConsentRevokedError` or `ConsentNeededError` when nothing may go to `service` on `flowId` now. */
  const check = (flowId: DataFlowId, service: ExternalService): void => {
    if (localOnly() && !consent.onThisComputer(flowId, service.id)) throw new LocalModeError();
    if (consent.allows(flowId, service)) return;
    const flow = consent.dataFlow(flowId, service);
    throw consent.isBackground(flowId)
      ? new ConsentNeededError(flow)
      : new ConsentRevokedError(flow);
  };

  return {
    check,

    /** The model, with `check` before each of its requests; one on this computer (no service) as it is. */
    model(
      model: ChatLanguageModel,
      flowId: DataFlowId,
      service: ExternalService | null,
    ): ChatLanguageModel {
      if (!service) return model;
      return wrapLanguageModel({
        model,
        middleware: {
          specificationVersion: "v4",
          wrapGenerate: ({ doGenerate }) => {
            check(flowId, service);
            return doGenerate();
          },
          wrapStream: ({ doStream }) => {
            check(flowId, service);
            return doStream();
          },
        },
      });
    },
  };
}
