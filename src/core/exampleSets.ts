/**
 * What each example Mind says (src/core/examples.ts): its folder of shipped
 * Documents, its Question and its Answer, whose `[^n]` markers are Citations
 * of the nth quote. Every quote is word for word from a shipped Document;
 * the core test runs the Citation check over every one of them.
 *
 * The Documents of each group and language are listed, with their sources and
 * licences, in resources/examples/LICENSE. The group examples have Documents
 * of their own in each language, not translations.
 */
import type { ExampleGroup } from "./api";
import type { Language } from "./language";

export interface ExampleQuote {
  /** The Document's name: its file name without the extension. */
  document: string;
  quote: string;
}

export interface ExampleSet {
  /** The shipped folder to copy, relative to the examples folder. */
  source: string;
  /** The folder made in the data folder, and linked. */
  folder: string;
  title: string;
  intro: string;
  question: string;
  /** Markdown; `[^n]` is the nth quote's Citation. */
  answer: string;
  quotes: readonly ExampleQuote[];
}

const TEA_QUOTES = [
  {
    document: "Tea · Wikipedia",
    quote:
      "Tea drinking may have begun in the region of Yunnan, where it was used for medicinal purposes.",
  },
  { document: "茶 · 维基百科", quote: "18世纪，英国已成为欧洲最大的茶叶消费国。" },
] as const;

export const EXAMPLE_SETS: Record<ExampleGroup, Record<Language, ExampleSet>> = {
  tea: {
    en: {
      source: "tea",
      folder: "Examples",
      title: "Where tea comes from",
      intro:
        "A short reading note on the history of tea. Two articles are linked as Documents: one in English, one in Chinese.",
      question: "Where does tea come from, and how did it spread around the world?",
      answer: [
        "Tea was probably first drunk in Yunnan, in southwest China, where it was used as a medicine [^1].",
        "",
        "It spread with trade, across Asia and later by sea to Europe, and by the 18th century Britain drank more tea than any other country in Europe [^2].",
      ].join("\n"),
      quotes: TEA_QUOTES,
    },
    "zh-CN": {
      source: "tea",
      folder: "示例文档",
      title: "茶从哪里来",
      intro: "一篇关于茶的历史的简短阅读笔记。两篇文章作为文档链接在这里：一篇英文，一篇中文。",
      question: "茶起源于哪里？又是如何传遍世界的？",
      answer: [
        "人们最早饮茶可能是在中国西南的云南，当时茶被用作药物 [^1]。",
        "",
        "茶随着贸易传遍亚洲，后来经海路传入欧洲；到 18 世纪，英国已成为欧洲饮茶最多的国家 [^2]。",
      ].join("\n"),
      quotes: TEA_QUOTES,
    },
  },

  papers: {
    en: {
      source: "papers/en",
      folder: "Example papers",
      title: "Retrieval for language models: a short literature review",
      intro:
        "Three open-access papers on retrieval-augmented generation (RAG), the way language models look things up before they answer. They are linked as Documents.",
      question:
        "What do these papers say about when a language model should retrieve, whether retrieval beats fine-tuning, and how to evaluate it?",
      answer: [
        "Retrieving a fixed number of passages every time can hurt: Self-RAG argues that retrieval should happen on demand, because indiscriminate retrieval reduces versatility and can lead to unhelpful answers [^1].",
        "",
        "For adding knowledge to a model, a Microsoft study found that retrieval beat unsupervised fine-tuning, for knowledge the model had seen in training and for entirely new knowledge [^2].",
        "",
        "Evaluating a RAG pipeline does not have to wait for human labels: Ragas offers a set of metrics that work without ground truth annotations [^3].",
      ].join("\n"),
      quotes: [
        {
          document: "Self-RAG (Asai et al., 2023)",
          quote:
            "However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation.",
        },
        {
          document: "Fine-Tuning or Retrieval (Ovadia et al., 2023)",
          quote:
            "RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge.",
        },
        {
          document: "Ragas (Es et al., 2023)",
          quote:
            "With Ragas, we put forward a suite of metrics which can be used to evaluate these different dimensions without having to rely on ground truth human annotations.",
        },
      ],
    },
    "zh-CN": {
      source: "papers/zh-CN",
      folder: "示例论文",
      title: "从中药里筛选活性成分：一份简短的文献综述",
      intro:
        "三篇发表在《色谱》上的开放获取综述，讲的是怎样从中药和天然产物里找出有活性的成分。它们作为文档链接在这里。",
      question: "这几篇综述分别介绍了哪些筛选活性成分的方法？各自靠什么起作用？",
      answer: [
        "亲和超滤-液相色谱-质谱的思路，是利用药用植物或中草药提取物与靶标之间的亲和力差异，从复杂的成分里把有活性的挑出来 [^1]。",
        "",
        "生物亲和垂钓是另一类靶向筛选技术，它把复杂体系中的分离和活性筛选结合在一起 [^2]。",
        "",
        "而这些方法能不能做好，关键在色谱分离材料：第三篇综述认为，材料的发展是提升中药复杂体系分离能力的关键驱动因素 [^3]。",
      ].join("\n"),
      quotes: [
        {
          document: "亲和超滤色谱-质谱筛选天然活性物质(徐勇兵等,2026)",
          quote: "该技术通过利用药用植物或中草药提取物与靶标之间的亲和力差异",
        },
        {
          document: "生物亲和垂钓发现天然产物活性成分(曲清莉等,2026)",
          quote: "以生物亲和垂钓为代表的靶向筛选技术因其能够将复杂体系中的分离与活性筛选相结合",
        },
        {
          document: "高效色谱材料在本草物质组中的应用(丰静等,2026)",
          quote: "色谱分离材料的发展是提升中药复杂体系分离能力的关键驱动因素",
        },
      ],
    },
  },

  reports: {
    en: {
      source: "reports/en",
      folder: "Example reports",
      title: "Three regions, one spring: comparing World Bank outlooks",
      intro:
        "The executive summaries of three World Bank regional economic updates from April 2026: Africa, Latin America and the Caribbean, and South Asia. They are linked as Documents.",
      question:
        "How do the growth outlooks for 2026 compare across the three regions, and which of them leans most on industrial policy?",
      answer: [
        "Sub-Saharan Africa is projected to grow 4.1 percent in 2026, the same as in 2025, with downside risks rising [^1].",
        "",
        "Latin America and the Caribbean is slower, at 2.1 percent in 2026, slightly below the 2.4 percent of 2025 [^2].",
        "",
        "South Asia is the fastest at 6.3 percent, though it is expected to slow amid global energy market dislocations [^3].",
        "",
        "South Asia also stands out for industrial policy: its countries use it at about twice the rate of other emerging market and developing economies [^4].",
      ].join("\n"),
      quotes: [
        {
          document: "Africa Economic Update, April 2026 (executive summary)",
          quote:
            "Economic growth in Sub-Saharan Africa is projected to remain at 4.1 percent in 2026, unchanged from 2025, but downside risks are increasing.",
        },
        {
          document:
            "Latin America and the Caribbean Economic Update, April 2026 (executive summary)",
          quote:
            "Regional GDP growth is projected to reach 2.1 percent in 2026—slightly below the 2.4 percent recorded in 2025",
        },
        {
          document: "South Asia Economic Update, April 2026 (executive summary)",
          quote:
            "South Asia’s growth again surprised on the upside but is expected to slow to 6.3 percent in 2026 amid headwinds from global energy market dislocations.",
        },
        {
          document: "South Asia Economic Update, April 2026 (executive summary)",
          quote:
            "South Asian countries make proactive use of industrial policies, at about twice the rate of other emerging market and developing economies (EMDEs).",
        },
      ],
    },
    "zh-CN": {
      source: "reports/zh-CN",
      folder: "示例报告",
      title: "两年的省会城市品牌报告：谁在前列？",
      intro:
        "同一个研究团队连续两年发布的省会城市及计划单列市城市品牌国际传播影响力报告，2023 年和 2024 年各一份。它们作为文档链接在这里。",
      question: "对比 2023 年和 2024 年的报告，国际传播影响力靠前的城市有什么变化？",
      answer: [
        "2023 年报告里，影响力一区的城市是杭州、深圳、成都、广州、西安、南京、厦门和福州 [^1]。",
        "",
        "2024 年报告里，名单变成了深圳、广州、杭州、成都、西安、南京、厦门和哈尔滨：福州退出，哈尔滨进入 [^2]。",
        "",
        "两年合起来看，报告称哈尔滨成为继深圳、杭州、成都、广州、西安、厦门之后，第七个进入第一方阵的城市 [^3]。",
      ].join("\n"),
      quotes: [
        {
          document: "2023年省会城市品牌国际传播影响力报告",
          quote: "杭州、深圳、成都、广州、西安、南京、厦门、福州",
        },
        {
          document: "2024年省会城市品牌国际传播影响力报告",
          quote: "深圳、广州、杭州、成都、西安、南京、厦门、哈尔滨",
        },
        {
          document: "2024年省会城市品牌国际传播影响力报告",
          quote: "继深圳、杭州、成都、广州、西安、厦门之后，哈尔滨进入第一方阵",
        },
      ],
    },
  },

  contracts: {
    en: {
      source: "contracts/en",
      folder: "Example contracts",
      title: "Three agreements: term, ending and liability",
      intro:
        "Three commercial agreements filed with the US Securities and Exchange Commission and shared in the CUAD dataset (CC BY 4.0). They are linked as Documents.",
      question:
        "How long does each agreement last, how can it be ended, and who limits their liability?",
      answer: [
        "The hosting agreement runs for six months from 1 April 1999 [^1], and either party may end it without cause on thirty days' written notice [^2].",
        "",
        "The Snotarator distributor agreement ends on 31 May 2015 unless ended sooner [^3], while the Erchonia distributor agreement has an initial term of three years [^4].",
        "",
        "On liability, the hosting company excludes lost profits and other consequential damages under any circumstances [^5].",
      ].join("\n"),
      quotes: [
        {
          document: "Web Site Hosting Agreement (i-on interactive and Centrack)",
          quote:
            "The term of this Agreement for the Hosted Site shall commence upon April 1, 1999 and shall continue for a period of six (6) months",
        },
        {
          document: "Web Site Hosting Agreement (i-on interactive and Centrack)",
          quote:
            "Either party may terminate this Agreement without cause at any time effective upon thirty (30) days' written notice.",
        },
        {
          document: "Distributor Agreement (Snotarator and SMSA Ballinger)",
          quote:
            "The term of this Agreement shall terminate on May 31, 2015, unless sooner terminated.",
        },
        {
          document: "Exclusive Distributor Agreement (Erchonia and InnerScope)",
          quote: "this Agreement shall have an initial term of three (3) years.",
        },
        {
          document: "Web Site Hosting Agreement (i-on interactive and Centrack)",
          quote:
            "i-on will not be liable under any circumstances for any lost profits or other consequential damages",
        },
      ],
    },
    "zh-CN": {
      source: "contracts/zh-CN",
      folder: "示例合同",
      title: "三份合同：期限、解除和责任限制",
      intro:
        "三份为这个示例写的合同：设备租赁、软件开发服务和货物采购。公司和人物都是虚构的，内容不是法律意见。它们作为文档链接在这里。",
      question: "这三份合同的期限、解除和责任限制各是怎么约定的？",
      answer: [
        "设备租赁合同的期限是十二个月，从 2026 年 4 月 1 日到 2027 年 3 月 31 日 [^1]；乙方连续两个月不付租金，甲方就可以解除合同并收回设备 [^2]。",
        "",
        "软件开发服务合同里，乙方逾期交付超过三十日，甲方有权解除合同 [^3]。",
        "",
        "货物采购合同的期限是两年 [^4]；卖方交付的货物不合格并造成损失时，赔偿责任累计不超过当年已付价款的百分之二十 [^5]。",
      ].join("\n"),
      quotes: [
        {
          document: "设备租赁合同",
          quote: "租赁期限为十二个月，自2026年4月1日起至2027年3月31日止。",
        },
        {
          document: "设备租赁合同",
          quote: "乙方连续两个月未支付租金的，甲方有权书面通知乙方解除本合同，并收回设备。",
        },
        {
          document: "软件开发服务合同",
          quote: "乙方逾期交付超过三十日的，甲方有权书面通知乙方解除本合同",
        },
        {
          document: "货物采购合同",
          quote: "本合同的期限为两年，自2026年7月1日起至2028年6月30日止。",
        },
        {
          document: "货物采购合同",
          quote: "赔偿责任累计不超过当年度已经支付价款的百分之二十",
        },
      ],
    },
  },

  meetings: {
    en: {
      source: "meetings/en",
      folder: "Example meeting",
      title: "Launch sync, 12 March",
      intro:
        "A transcript and agenda written for this example. Fernwood Pantry is a made-up company, and the people in the call are made up too. They are linked as Documents.",
      question: "What did the launch sync decide, and what has to happen by when?",
      answer: [
        "The starter box costs 34 euros, and new customers get 10 euros off their first box with a welcome code [^1].",
        "",
        "The 7 April launch holds as long as the payment provider approves the account by 27 March [^2].",
        "",
        "The first 500 people on the waiting list get the app on 30 March, and everyone else on launch day [^3].",
        "",
        "The supplier's request for more vegetables waits until the waiting list shows real demand, and comes back on 8 April [^4].",
      ].join("\n"),
      quotes: [
        {
          document: "Launch sync transcript · 12 March",
          quote:
            "Decision, then: the starter box costs 34 euros, and new customers get 10 euros off their first box with a welcome code.",
        },
        {
          document: "Launch sync transcript · 12 March",
          quote: "Yes, as long as the payment provider approves our account by 27 March.",
        },
        {
          document: "Launch sync transcript · 12 March",
          quote: "First 500 people on 30 March, everyone else on launch day.",
        },
        {
          document: "Launch sync transcript · 12 March",
          quote:
            "Not until the waiting list shows real demand. Tell them 800, and we revisit on 8 April, the day after launch.",
        },
      ],
    },
    "zh-CN": {
      source: "meetings/zh-CN",
      folder: "示例会议",
      title: "大促备货会，9月17日",
      intro:
        "为这个示例写的会议记录和议程。澄溪文具是虚构的公司，会上的人也都是虚构的。它们作为文档链接在这里。",
      question: "这次备货会定了哪些事？谁要在什么时候做完？",
      answer: [
        "首批备货 6.5 万件，另外留 5000 件的追加额度，由唐小满和供应商落实 [^1]。",
        "",
        "客服加 6 名临时人员，10 月 28 日前到岗培训，自动回复的话术由沈嘉树在 10 月 15 日前写好 [^2]。",
        "",
        "推广预算按 8 万元算，超出的部分要先报林慧同意 [^3]。",
        "",
        "做决定的依据之一是去年的数字：备货 6 万件，卖出 5.2 万件，还剩 8000 件 [^4]。",
      ].join("\n"),
      quotes: [
        {
          document: "大促备货会议记录 · 9月17日",
          quote: "决定：首批备货 6.5 万件，另有 5000 件的追加额度，由唐小满和供应商落实。",
        },
        {
          document: "大促备货会议记录 · 9月17日",
          quote:
            "决定：加 6 名临时客服，10 月 28 日前到岗培训，自动回复的话术由沈嘉树 10 月 15 日前写好。",
        },
        {
          document: "大促备货会议记录 · 9月17日",
          quote: "推广预算按 8 万元算，超过的部分必须先报我，我同意了才能花。",
        },
        {
          document: "大促备货会议议程",
          quote: "去年备货 6 万件，销售 5.2 万件，库存剩余 8000 件",
        },
      ],
    },
  },
};
