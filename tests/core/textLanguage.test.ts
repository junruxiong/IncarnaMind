import { describe, expect, test } from "vitest";
import { detectLanguage, languageList } from "../../src/core/documents/textLanguage";

describe("The language a Document is written in", () => {
  test("Chinese, Japanese and Korean by their scripts, whatever Latin words they mix in", () => {
    expect(
      detectLanguage(
        "梯度下降法是一个一阶最优化算法，通常也称为最陡下降法。Haskell Curry 在1944年首先研究了该方法的收敛性。",
      ),
    ).toBe("Chinese");
    expect(
      detectLanguage("勾配降下法は、関数の最小値を探すための一次の反復最適化アルゴリズムです。"),
    ).toBe("Japanese");
    expect(
      detectLanguage("경사 하강법은 함수의 최솟값을 찾는 일차 반복 최적화 알고리즘이다."),
    ).toBe("Korean");
  });

  test("Latin-script languages by their common words", () => {
    expect(
      detectLanguage(
        "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output.",
      ),
    ).toBe("English");
    expect(
      detectLanguage(
        "Le Transformer est le premier modèle qui repose entièrement sur l'attention pour calculer les représentations des entrées et des sorties.",
      ),
    ).toBe("French");
    expect(
      detectLanguage(
        "Der Transformer ist das erste Modell, das sich ganz auf die Aufmerksamkeit stützt und nicht auf Rekurrenz.",
      ),
    ).toBe("German");
    expect(
      detectLanguage(
        "El Transformer es el primer modelo que se basa por completo en la atención para calcular las representaciones de la entrada y la salida.",
      ),
    ).toBe("Spanish");
  });

  test("an English Document with a few Chinese terms is English", () => {
    expect(
      detectLanguage(
        "The report covers sustainable finance (可持续金融) and the goals the bank set for the decade, with the figures for each of its regions.",
      ),
    ).toBe("English");
  });

  test("too little to tell, or a script it doesn't name, is no language", () => {
    expect(detectLanguage("")).toBeNull();
    expect(detectLanguage("12 34 5.6 — 7 %")).toBeNull();
    expect(detectLanguage("Revenue Q3 EBITDA")).toBeNull();
    expect(detectLanguage("Привет, как дела? Это текст на русском языке.")).toBeNull();
  });

  test("languages are listed most common first, with how many Documents are in each", () => {
    expect(
      languageList(["English", "Chinese", "English", null, "English", "Chinese", "French"]),
    ).toEqual([
      { language: "English", documents: 3 },
      { language: "Chinese", documents: 2 },
      { language: "French", documents: 1 },
    ]);
  });
});
