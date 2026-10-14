import { type RefObject, useLayoutEffect, useState } from "react";
import { fitName } from "../../shared/naming";

const TEXT_STYLES = [
  "fontFamily",
  "fontSize",
  "fontWeight",
  "fontStyle",
  "letterSpacing",
  "fontVariationSettings",
  "fontFeatureSettings",
  "fontOpticalSizing",
] as const;

/**
 * `name` cut to what fits on one line of `field`, in the field's own type
 * (`fitName`), and fitted again as its width or the fonts change (#769).
 */
export function useFittedName(field: RefObject<HTMLElement | null>, name: string): string {
  const [fitted, setFitted] = useState(name);

  useLayoutEffect(() => {
    const element = field.current;
    if (!element) return;
    const refit = () => {
      const style = getComputedStyle(element);
      const probe = document.createElement("span");
      probe.style.cssText = "position:absolute;visibility:hidden;white-space:pre";
      for (const property of TEXT_STYLES) probe.style[property] = style[property];
      element.parentElement?.append(probe);
      const room = element.clientWidth;
      setFitted(
        fitName(name, (text) => {
          probe.textContent = text;
          return probe.getBoundingClientRect().width <= room;
        }),
      );
      probe.remove();
    };
    refit();
    const resizes = new ResizeObserver(refit);
    resizes.observe(element);
    void document.fonts.ready.then(refit);
    return () => resizes.disconnect();
  }, [field, name]);

  return fitted;
}
