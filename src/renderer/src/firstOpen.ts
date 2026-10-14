import { firstOpenOf } from "../../shared/firstOpen";
import { useAppStore } from "./store";

export { firstOpenOf };

export const useFirstOpen = (): boolean =>
  useAppStore((state) => firstOpenOf(state.settings?.device.gettingStarted));
