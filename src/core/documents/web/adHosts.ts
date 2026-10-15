/**
 * Hosts whose pictures a saved web page never fetches (#756): ad servers,
 * trackers and analytics, whose "pictures" are beacons, not the article's.
 * A host matches itself and every host under it. Add to the list as new
 * ones turn up; the article's own pictures come from the site and its image
 * servers, never from these.
 */
const AD_HOSTS = [
  "adnxs.com",
  "adform.net",
  "adroll.com",
  "adsafeprotected.com",
  "adservice.google.com",
  "advertising.com",
  "amazon-adsystem.com",
  "analytics.google.com",
  "analytics.twitter.com",
  "bat.bing.com",
  "bluekai.com",
  "casalemedia.com",
  "chartbeat.net",
  "criteo.com",
  "criteo.net",
  "demdex.net",
  "doubleclick.net",
  "everesttech.net",
  "google-analytics.com",
  "googleadservices.com",
  "googlesyndication.com",
  "googletagmanager.com",
  "googletagservices.com",
  "hotjar.com",
  "krxd.net",
  "moatads.com",
  "mathtag.com",
  "omtrdc.net",
  "outbrain.com",
  "outbrainimg.com",
  "pubmatic.com",
  "quantserve.com",
  "rubiconproject.com",
  "scorecardresearch.com",
  "sharethrough.com",
  "taboola.com",
  "tapad.com",
  "teads.tv",
  "yieldmo.com",
] as const;

/** Whether a picture from `hostname` is an ad's or a tracker's. */
export function isAdHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/\.$/, "");
  return AD_HOSTS.some((ad) => host === ad || host.endsWith(`.${ad}`));
}
