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
  const readiness = useAppStore((state) => state.chatReadiness);
  const [service, setService] = useState<ExternalService | null>(null);

  useEffect(() => {
    if (!waiting) {
      setService(null);
      return;
    }
    let stopped = false;
    const load = () => {
      Promise.all([core.listRegisteredDataFlows(), core.getJevSettings()]).then(
        ([flows, jev]) => {
          if (stopped) return;
          const tagger = jev.enabled
            ? jev.service
            : readiness && "provider" in readiness
              ? readiness.provider.service
              : null;
          const decision =
            tagger &&
            flows
              .find((flow) => flow.id === "tagging")
              ?.services.find((each) => each.flow.service.id === tagger.id);
          setService(decision && decision.consent !== "accepted" ? tagger : null);
        },
        (error: unknown) => useAppStore.getState().reportError(error),
      );
    };
    load();
    const stops = [core.on("dataFlows.changed", load), core.on("jev.changed", load)];
    return () => {
      stopped = true;
      for (const stop of stops) stop();
    };
  }, [waiting, readiness]);

  return service;
}
