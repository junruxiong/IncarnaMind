/**
 * Generates the scanned-page test set next to this script (#397): one JPEG
 * per page, drawn at 200 dpi on an A4 sheet, and `set.json` with each page's
 * expected text, line by line in reading order. From the repository root, on
 * macOS (it draws with the system's fonts):
 *
 *     node tests/fixtures/scans/make-scans.ts
 *
 * All text is synthetic, written for this project (see ATTRIBUTION.md). The
 * output is the same for the same @napi-rs/canvas and fonts: noise, blur and
 * rotation come from a seeded generator.
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { type Canvas, createCanvas, GlobalFonts, type SKRSContext2D } from "@napi-rs/canvas";

const here = import.meta.dirname;

/** A4 at 200 dpi. */
const WIDTH = 1654;
const HEIGHT = 2339;
const MARGIN = 170;
/** 11 pt at 200 dpi. */
const BODY = 31;

const FONTS: Record<string, string> = {
  Times: "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
  TimesBold: "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
  Georgia: "/System/Library/Fonts/Supplemental/Georgia.ttf",
  GeorgiaBold: "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
  Arial: "/System/Library/Fonts/Supplemental/Arial.ttf",
  ArialBold: "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
  Song: "/System/Library/Fonts/Supplemental/Songti.ttc",
  Hei: "/System/Library/Fonts/Hiragino Sans GB.ttc",
  HeitiLight: "/System/Library/Fonts/STHeiti Light.ttc",
};
for (const [alias, path] of Object.entries(FONTS)) {
  if (!GlobalFonts.registerFromPath(path, alias)) throw new Error(`Couldn't load ${path}`);
}

/** mulberry32: a small seeded generator, so the noise is the same at every run. */
function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Line {
  text: string;
  font: string;
  size?: number;
  /** Extra space before the line, in lines. */
  gap?: number;
}

interface Page {
  id: string;
  language: "en" | "zh-Hans" | "zh-Hant" | "en+zh-Hans";
  condition: "clean" | "noisy";
  kind: "text" | "table";
  /** What the page is, for the README. */
  about: string;
  /** Drawn after the reading was tuned on the other pages: a check that it holds. */
  heldout?: boolean;
  /** Draws the page; returns its lines of text, in reading order, as drawn. */
  draw(context: SKRSContext2D): string[];
  noise?: {
    seed: number;
    degrees: number;
    blur: number;
    sigma: number;
    speckle: number;
    quality: number;
  };
}

function drawLines(context: SKRSContext2D, lines: readonly Line[], top = MARGIN + 40): string[] {
  let y = top;
  const drawn: string[] = [];
  for (const line of lines) {
    const size = line.size ?? BODY;
    y += (line.gap ?? 0) * size * 1.45;
    context.font = `${size}px ${line.font}`;
    context.fillStyle = "#111";
    context.fillText(line.text, MARGIN, y);
    drawn.push(line.text);
    y += size * 1.45;
  }
  return drawn;
}

const body = (font: string, texts: readonly string[], first: Partial<Line> = {}): Line[] =>
  texts.map((text, index) => ({ text, font, ...(index === 0 ? first : {}) }));

const pages: Page[] = [
  {
    id: "en-clean",
    language: "en",
    condition: "clean",
    kind: "text",
    about: "A contract's payment and termination clauses, in Times New Roman",
    draw: (context) =>
      drawLines(context, [
        { text: "MASTER SERVICES AGREEMENT", font: "TimesBold", size: 38 },
        ...body("Times", ["between Aldermoor Logistics Ltd and Kestrel Data Systems plc"], {
          gap: 0.2,
        }),
        { text: "4. Payment Terms", font: "TimesBold", gap: 1 },
        ...body(
          "Times",
          [
            "4.1 The Customer shall pay each undisputed invoice within thirty (30) days",
            "of the date of receipt, by bank transfer to the account named in Schedule 2.",
            "4.2 Any amount not paid when due shall bear interest at two per cent (2%)",
            "per annum above the base rate of the Bank of England from time to time.",
            "4.3 The Supplier may suspend the Services on fourteen (14) days' written",
            "notice if an invoice remains unpaid sixty (60) days after its due date.",
          ],
          { gap: 0.3 },
        ),
        { text: "9. Termination", font: "TimesBold", gap: 1 },
        ...body(
          "Times",
          [
            "9.1 Either party may terminate this Agreement for convenience by giving",
            "not less than ninety (90) days' written notice to the other party.",
            "9.2 Either party may terminate this Agreement with immediate effect if the",
            "other party commits a material breach which is not remedied within",
            "thirty (30) days of receiving notice specifying the breach.",
            "9.3 On termination the Supplier shall return all Customer Data within",
            "ten (10) Business Days and certify the deletion of any remaining copies.",
          ],
          { gap: 0.3 },
        ),
      ]),
  },
  {
    id: "en-noisy",
    language: "en",
    condition: "noisy",
    kind: "text",
    about:
      "A field report, in Georgia, rotated 1.3°, blurred, with noise, uneven light and heavy JPEG",
    noise: { seed: 7, degrees: 1.3, blur: 0.9, sigma: 16, speckle: 0.004, quality: 38 },
    draw: (context) =>
      drawLines(context, [
        { text: "Coastal Monitoring Report, Spring 2025", font: "GeorgiaBold", size: 35 },
        ...body(
          "Georgia",
          [
            "Survey teams visited eleven stations along the northern estuary between",
            "3 March and 28 April. Mean water temperature rose to 9.4 degrees Celsius,",
            "about 1.1 degrees above the ten-year average for the same weeks.",
            "Sediment samples from stations 4 and 7 contained more fine silt than in",
            "any earlier survey, which the team links to the dredging works upstream.",
          ],
          { gap: 0.8 },
        ),
        { text: "Findings", font: "GeorgiaBold", gap: 1 },
        ...body(
          "Georgia",
          [
            "Salt marsh cover at station 2 fell by 6 per cent since the autumn count.",
            "Seagrass beds near the harbour wall recovered after the winter storms.",
            "Two new colonies of oystercatchers were recorded on the shingle spit.",
            "The tide gauge at station 9 failed on 17 April and was replaced on 22 April;",
            "readings for those five days are estimated from stations 8 and 10.",
          ],
          { gap: 0.3 },
        ),
        { text: "Recommendations", font: "GeorgiaBold", gap: 1 },
        ...body(
          "Georgia",
          [
            "Repeat the silt sampling monthly until the dredging contract ends,",
            "and move the spare tide gauge to station 9 before the autumn survey.",
          ],
          { gap: 0.3 },
        ),
      ]),
  },
  {
    id: "zh-clean",
    language: "zh-Hans",
    condition: "clean",
    kind: "text",
    about: "A lease's rent and deposit clauses, in Songti",
    draw: (context) =>
      drawLines(context, [
        { text: "房屋租赁合同（节选）", font: "Hei", size: 40 },
        { text: "第三条 租金及支付方式", font: "Hei", gap: 1 },
        ...body(
          "Song",
          [
            "一、该房屋月租金为人民币捌仟伍佰元整（￥8,500.00），按季度支付。",
            "二、乙方应于每季度首月的五日前将租金汇入甲方指定的银行账户。",
            "三、乙方逾期支付租金的，每逾期一日按应付租金的万分之五支付违约金。",
          ],
          { gap: 0.4 },
        ),
        { text: "第四条 押金", font: "Hei", gap: 1 },
        ...body(
          "Song",
          [
            "一、乙方应于签订本合同之日向甲方支付押金人民币壹万柒仟元整。",
            "二、租赁期满或合同解除后，甲方应在十五个工作日内无息退还押金，",
            "但乙方拖欠的租金、水电费及损坏房屋设施的赔偿费用可从押金中扣除。",
          ],
          { gap: 0.4 },
        ),
        { text: "第五条 维修责任", font: "Hei", gap: 1 },
        ...body(
          "Song",
          [
            "一、房屋主体结构及供水、供电设施的维修由甲方负责。",
            "二、因乙方使用不当造成的损坏，由乙方负责修复或赔偿。",
            "三、甲方应在接到乙方维修通知后七日内完成维修。",
          ],
          { gap: 0.4 },
        ),
      ]),
  },
  {
    id: "zh-noisy",
    language: "zh-Hans",
    condition: "noisy",
    kind: "text",
    about:
      "A monthly operations report, in Hiragino Sans GB, rotated −1.1°, blurred, with noise and heavy JPEG",
    noise: { seed: 11, degrees: -1.1, blur: 0.9, sigma: 16, speckle: 0.004, quality: 38 },
    draw: (context) =>
      drawLines(context, [
        { text: "三月份运营情况报告", font: "Hei", size: 40 },
        ...body(
          "Hei",
          [
            "本月门店总客流量为十二万三千人次，环比增长百分之八点六。",
            "线上订单占全部订单的百分之三十七，较上月提高四个百分点。",
            "华东区销售额最高，达到四百六十二万元；西南区增速最快。",
          ],
          { gap: 0.8 },
        ),
        { text: "存在的问题", font: "Hei", gap: 1 },
        ...body(
          "Hei",
          [
            "一是部分门店库存周转天数超过四十五天，占用资金较多。",
            "二是配送准时率下降至百分之九十一，主要受雨季天气影响。",
            "三是新员工流失率偏高，培训计划需要进一步完善。",
          ],
          { gap: 0.4 },
        ),
        { text: "下月工作安排", font: "Hei", gap: 1 },
        ...body(
          "Hei",
          [
            "下月将清理滞销商品，并在华南区试行夜间配送。",
            "人力资源部将于四月十日前提交新的培训方案。",
          ],
          { gap: 0.4 },
        ),
      ]),
  },
  {
    id: "table",
    language: "en+zh-Hans",
    condition: "clean",
    kind: "table",
    about:
      "Two ruled tables of figures, one in English (Arial) and one in Chinese (Hiragino Sans GB)",
    draw: drawTables,
  },
  {
    id: "zh-hant",
    language: "zh-Hant",
    condition: "clean",
    kind: "text",
    about: "A short notice in Traditional Chinese, in Hiragino Sans GB",
    draw: (context) =>
      drawLines(context, [
        { text: "圖書館開放時間調整通知", font: "Hei", size: 40 },
        // Songti's first face is Simplified only: its traditional characters would be blank.
        ...body(
          "Hei",
          [
            "自十月一日起，本館週一至週五開放時間調整為上午九時至晚上九時。",
            "週六及週日開放時間為上午十時至下午六時，國定假日休館。",
            "讀者如需延長借閱期限，請於到期前透過網站或櫃檯辦理續借。",
          ],
          { gap: 0.8 },
        ),
      ]),
  },
  {
    id: "en-small",
    language: "en",
    condition: "noisy",
    kind: "text",
    heldout: true,
    about: "A memo in 9 pt Arial, rotated 0.6°, lightly blurred, with noise",
    noise: { seed: 23, degrees: 0.6, blur: 0.6, sigma: 10, speckle: 0.002, quality: 50 },
    draw: (context) =>
      drawLines(context, [
        { text: "MEMORANDUM", font: "ArialBold", size: 30 },
        ...body(
          "Arial",
          [
            "To: All project leads    From: Finance office    Date: 14 May 2025",
            "Subject: Travel claims for the second quarter",
          ],
          { gap: 0.5 },
        ).map((line) => ({ ...line, size: 25 })),
        ...body(
          "Arial",
          [
            "Claims for travel between April and June must reach the finance office by 5 July.",
            "Attach a receipt for every item over 25 pounds, and the booking confirmation for",
            "rail and air fares. Claims without receipts will be returned unpaid.",
            "Mileage is paid at 45 pence a mile for the first 10,000 miles in the tax year,",
            "and at 25 pence a mile after that. Taxis are paid only where no train or bus",
            "runs within an hour of the journey, or where the claimant carries equipment.",
            "Questions about a claim should go to the finance office, room 2.14, extension 4471.",
          ],
          { gap: 0.8 },
        ).map((line) => ({ ...line, size: 25 })),
      ]),
  },
  {
    id: "zh-light",
    language: "zh-Hans",
    condition: "noisy",
    kind: "text",
    heldout: true,
    about: "A meeting notice in 10 pt STHeiti Light, rotated −0.7°, lightly blurred, with noise",
    noise: { seed: 31, degrees: -0.7, blur: 0.6, sigma: 10, speckle: 0.002, quality: 50 },
    draw: (context) =>
      drawLines(context, [
        { text: "关于召开年度总结会议的通知", font: "HeitiLight", size: 36 },
        ...body(
          "HeitiLight",
          [
            "各部门：",
            "公司定于十二月二十日下午两点在三楼会议室召开年度总结会议。",
            "请各部门负责人准备本部门全年工作总结，时间控制在十分钟以内。",
            "会议材料请于十二月十八日前发送至行政部邮箱，逾期不再安排发言。",
            "如有特殊情况不能参加，请提前向行政部请假，并指定一名代表出席。",
            "会后将安排部门之间的交流，请各位预留一个小时的时间。",
          ],
          { gap: 0.8 },
        ).map((line) => ({ ...line, size: 28 })),
        { text: "行政部", font: "HeitiLight", size: 28, gap: 1 },
        { text: "二〇二五年十二月一日", font: "HeitiLight", size: 28 },
      ]),
  },
];

/** A ruled table: its rows of cells, each column's width, drawn from `top`. Returns its rows' text. */
function drawTable(
  context: SKRSContext2D,
  rows: readonly (readonly string[])[],
  widths: readonly number[],
  top: number,
  font: string,
): string[] {
  const rowHeight = 64;
  const left = MARGIN;
  const width = widths.reduce((sum, each) => sum + each, 0);
  context.strokeStyle = "#222";
  context.lineWidth = 2;
  rows.forEach((row, index) => {
    const y = top + index * rowHeight;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(left + width, y);
    context.stroke();
    let x = left;
    row.forEach((cell, column) => {
      const columnWidth = widths[column] ?? 0;
      // Hiragino Sans GB has no bold face here: its header row is set like the rest.
      context.font = index === 0 && font === "Arial" ? "29px ArialBold" : `28px ${font}`;
      context.fillStyle = "#111";
      const measured = context.measureText(cell).width;
      // Labels left-aligned, figures right-aligned, as a printed table sets them.
      const textX = column === 0 ? x + 16 : x + columnWidth - 16 - measured;
      context.fillText(cell, textX, y + rowHeight / 2 + 10);
      x += columnWidth;
    });
  });
  const bottom = top + rows.length * rowHeight;
  context.beginPath();
  context.moveTo(left, bottom);
  context.lineTo(left + width, bottom);
  let x = left;
  for (const columnWidth of [0, ...widths]) {
    x += columnWidth;
    context.moveTo(x, top);
    context.lineTo(x, bottom);
  }
  context.stroke();
  return rows.map((row) => row.join(" "));
}

function drawTables(context: SKRSContext2D): string[] {
  const lines: string[] = [];
  context.font = "36px ArialBold";
  context.fillStyle = "#111";
  context.fillText("Table 3. Revenue by product line (USD thousands)", MARGIN, MARGIN + 40);
  lines.push("Table 3. Revenue by product line (USD thousands)");
  lines.push(
    ...drawTable(
      context,
      [
        ["Product line", "2024", "2025", "Change"],
        ["Freight software", "4,812", "5,330", "+10.8%"],
        ["Warehouse sensors", "2,147", "1,985", "-7.5%"],
        ["Fleet analytics", "3,906", "4,412", "+13.0%"],
        ["Support contracts", "1,268", "1,301", "+2.6%"],
        ["Total", "12,133", "13,028", "+7.4%"],
      ],
      [520, 260, 260, 270],
      MARGIN + 90,
      "Arial",
    ),
  );
  context.font = "38px Hei";
  context.fillStyle = "#111";
  context.fillText("表 5  2025年分地区收入（万元）", MARGIN, MARGIN + 640);
  lines.push("表 5  2025年分地区收入（万元）");
  lines.push(
    ...drawTable(
      context,
      [
        ["地区", "上半年", "下半年"],
        ["华东", "1,286", "1,452"],
        ["华南", "974", "1,031"],
        ["西南", "512", "698"],
        ["合计", "2,772", "3,181"],
      ],
      [420, 300, 300],
      MARGIN + 690,
      "Hei",
    ),
  );
  return lines;
}

/** Spoils a clean page the way a poor scan does: rotated, blurred, uneven light, noise, specks. */
function degrade(clean: Canvas, noise: NonNullable<Page["noise"]>): Canvas {
  const random = seeded(noise.seed);
  const out = createCanvas(WIDTH, HEIGHT);
  const context = out.getContext("2d");
  context.fillStyle = "#e9e6df";
  context.fillRect(0, 0, WIDTH, HEIGHT);
  context.translate(WIDTH / 2, HEIGHT / 2);
  context.rotate((noise.degrees * Math.PI) / 180);
  context.translate(-WIDTH / 2, -HEIGHT / 2);
  context.filter = `blur(${noise.blur}px)`;
  context.drawImage(clean, 0, 0);
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.filter = "none";
  const image = context.getImageData(0, 0, WIDTH, HEIGHT);
  const { data } = image;
  for (let y = 0; y < HEIGHT; y++) {
    // Light falls off towards the bottom right, as under a lid that didn't close.
    for (let x = 0; x < WIDTH; x++) {
      const at = (y * WIDTH + x) * 4;
      const light = 1 - 0.16 * ((x / WIDTH + y / HEIGHT) / 2);
      // Roughly Gaussian: the sum of three uniforms.
      const grain = (random() + random() + random() - 1.5) * noise.sigma * 2;
      let value = (data[at] as number) * light + grain;
      const speck = random();
      if (speck < noise.speckle / 2) value = 30;
      else if (speck < noise.speckle) value = 250;
      const clamped = Math.max(0, Math.min(255, Math.round(value)));
      data[at] = clamped;
      data[at + 1] = clamped;
      data[at + 2] = clamped;
    }
  }
  context.putImageData(image, 0, 0);
  return out;
}

/** A page as `set.json` lists it: what it is, its file, and its lines of text in reading order. */
type ListedPage = Omit<Page, "draw" | "noise"> & { file: string; lines: string[] };

const set: { dpi: number; pages: ListedPage[] } = { dpi: 200, pages: [] };

for (const page of pages) {
  const canvas = createCanvas(WIDTH, HEIGHT);
  const context = canvas.getContext("2d");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, WIDTH, HEIGHT);
  const lines = page.draw(context);
  const final = page.noise ? degrade(canvas, page.noise) : canvas;
  const file = `${page.id}.jpg`;
  writeFileSync(join(here, file), final.encodeSync("jpeg", page.noise?.quality ?? 70));
  const { id, language, condition, kind, about, heldout } = page;
  set.pages.push({
    id,
    language,
    condition,
    kind,
    about,
    ...(heldout && { heldout }),
    file,
    lines,
  });
}

writeFileSync(join(here, "set.json"), `${JSON.stringify(set, null, 2)}\n`);
console.log(`Wrote ${set.pages.length} pages to ${here}`);
