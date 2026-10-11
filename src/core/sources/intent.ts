/**
 * Whether a Question asks for published sources: papers, articles, studies,
 * the literature, a DOI. Only such a Question is offered Find sources' Tool
 * (ADR-0018). Offered to every Answer, a model could send searches out (and
 * put a consent request in front of the User) for a Question about their own
 * Documents, and a small local model's window would carry the Tool for
 * nothing. Words in English and Chinese, as Questions are written; the
 * Library's Find sources is always there for anything this misses.
 */
const ENGLISH =
  /\b(?:papers?|articles?|stud(?:y|ies)|literature|publications?|published|preprints?|sources?|references?|citations?|journals?|reviews?|arxiv|pubmed|openalex|crossref|doi)\b/i;

const CHINESE = /论文|文献|文章|研究|资料|期刊|出处|来源|预印本|综述|引用/;

export function asksForSources(question: string): boolean {
  return ENGLISH.test(question) || CHINESE.test(question);
}
