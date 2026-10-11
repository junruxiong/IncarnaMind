/**
 * Whether a Question asks for published works: papers, the literature,
 * preprints, a DOI. Only such a Question is offered Find sources' Tool
 * (ADR-0018). Offered to every Answer, a model could send searches out (and
 * put a consent request in front of the User) for a Question about their own
 * Documents, and a small local model's window would carry the Tool for
 * nothing. The words are the ones that name scholarly works and little else,
 * in English and Chinese, so "When are the tide tables published?" or a
 * contract's "Article 5" isn't one; the Library's Find sources is always
 * there for anything this misses.
 */
const ENGLISH =
  /\b(?:papers?|literature|preprints?|publications|studies|scholarly|arxiv|pubmed|openalex|crossref|doi)\b/i;

const CHINESE = /论文|文献|预印本|综述/;

export function asksForSources(question: string): boolean {
  return ENGLISH.test(question) || CHINESE.test(question);
}
