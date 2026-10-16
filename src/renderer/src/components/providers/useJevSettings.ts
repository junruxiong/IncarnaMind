import { useEffect, useState } from "react";
import type { JevSettings } from "../../../../core/api";
import { core } from "../../core";

export function useJevSettings(): [JevSettings | null, (jev: JevSettings) => void] {
  const [jev, setJev] = useState<JevSettings | null>(null);
  useEffect(() => {
    core.getJevSettings().then(setJev, () => undefined);
    return core.on("jev.changed", setJev);
  }, []);
  return [jev, setJev];
}
