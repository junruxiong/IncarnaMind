import { useEffect, useState } from "react";
import type { ExternalService } from "../../../core/api";
import { core } from "../core";
import { useAppStore } from "../store";

/**
 * The service tagging waits for the User to allow, or null. Tagging is
 * background work and never asks (ADR-0022): where its Documents wait says
 * so, and offers Allow in place. Read only while `waiting`.
 */
export function useTaggingPermission(waiting: boolean): ExternalService | null {
  const [service, setService] = useState<ExternalService | null>(null);

  useEffect(() => {
    if (!waiting) {
      setService(null);
      return;
    }
    let stopped = false;
    const load = () => {
      core.listRegisteredDataFlows().then(
        (flows) => {
          if (stopped) return;
          // The flow goes only to the service the Tagging setting tags with now.
          const unallowed = flows
            .find((flow) => flow.id === "tagging")
            ?.services.find((each) => each.consent !== "accepted");
          setService(unallowed ? unallowed.flow.service : null);
        },
        (error: unknown) => useAppStore.getState().reportError(error),
      );
    };
    load();
    const stops = [
      core.on("dataFlows.changed", load),
      core.on("jev.changed", load),
      core.on("chatReadiness.changed", load),
      core.on("library.changed", load),
    ];
    return () => {
      stopped = true;
      for (const stop of stops) stop();
    };
  }, [waiting]);

  return service;
}
