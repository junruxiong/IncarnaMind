/**
 * Whether a Question asks for the web (#128): only such a Question is offered
 * web search's Tools, as only one that asks for papers is offered Find
 * sources' (../../sources/intent). Offered to every Answer, a model could
 * search the web, at a price and behind a consent request, for a Question
 * about the User's own Documents. The words name the web, a search engine or
 * a site, in English and Chinese.
 */
const ENGLISH =
  /\b(?:web|online|internet|google|bing|duckduckgo|websites?|webpages?|web\s+pages?|search\s+engines?)\b/i;

const CHINESE = /网上|网络上|联网|上网|网页|网站|互联网|在线搜索|搜索引擎|谷歌|百度|必应/;

export function asksForWeb(question: string): boolean {
  return ENGLISH.test(question) || CHINESE.test(question);
}
