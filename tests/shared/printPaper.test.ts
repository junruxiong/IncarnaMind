import { describe, expect, test } from "vitest";
import { systemRegion } from "../../src/core/language";
import { paperFor } from "../../src/shared/printDocument";

describe("the paper a PDF is printed on", () => {
  test.each(["US", "CA", "MX", "PH", "CL", "CO", "PR", "us"])("%s prints on Letter", (region) => {
    expect(paperFor(region)).toBe("letter");
  });

  test.each(["GB", "IE", "AU", "IN", "NZ", "ZA", "CN", "TW", "DE", "FR", "JP", ""])(
    "“%s” prints on A4",
    (region) => {
      expect(paperFor(region)).toBe("a4");
    },
  );

  test("the region is the system's own, else its first language's, else the likeliest for that language", () => {
    expect(systemRegion("GB", ["en-US"])).toBe("GB");
    expect(systemRegion("", ["en-GB", "en-US"])).toBe("GB");
    expect(systemRegion(undefined, ["zh-Hans-CN"])).toBe("CN");
    expect(systemRegion(undefined, ["en"])).toBe("US");
    expect(systemRegion(undefined, ["zh_TW"])).toBe("TW");
    expect(systemRegion(undefined, [])).toBe("");
    expect(systemRegion(undefined, ["not a language"])).toBe("");
  });
});
