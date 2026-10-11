/**
 * Whether a Question asks for published works: papers, the literature,
 * preprints, a DOI, or sources it asks to find. Only such a Question is
 * offered Find sources' Tool (ADR-0018). Offered to every Answer, a model
 * could send searches out (and put a consent request in front of the User)
 * for a Question about their own Documents, and a small local model's window
 * would carry the Tool for nothing. The words are the ones that name
 * scholarly works and little else, and words like "sources" or "articles"
 * only after a verb that asks to find them, in English and Chinese, so "When
 * are the tide tables published?" or a contract's "Article 5" isn't one; the
 * Library's Find sources is always there for anything this misses.
 */
const ENGLISH =
  /\b(?:papers?|literature|preprints?|publications|studies|scholarly|arxiv|pubmed|openalex|crossref|doi)\b/i;

/** "Find sources on …", "search for articles about …", "suggest some references". */
const ASKED_TO_FIND =
  /\b(?:find|search(?:\s+for)?|look\s+(?:for|up)|recommend|suggest|list)\s+(?:(?:me|some|more|any|a\s+few|good|relevant|recent|other)\s+)*(?:sources|articles|references|citations|readings)\b/i;

/** Chinese puts the topic between the verb and what is found: "查找一些关于城市热岛的资料". */
const CHINESE =
  /论文|文献|预印本|综述|(?:查找|搜索|找|推荐)[^。？！?!\n]{0,20}?(?:资料|文章|参考资料)/;

export function asksForSources(question: string): boolean {
  return ENGLISH.test(question) || ASKED_TO_FIND.test(question) || CHINESE.test(question);
}
