import * as cheerio from 'cheerio';

const url = 'https://manhwazone.com/search?keyword=Gekkou';
const res = await fetch(url, {
  headers: {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    Accept: 'text/html,application/xhtml+xml,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9'
  }
});
const html = await res.text();
const $ = cheerio.load(html);
const base = 'https://manhwazone.com';

function isHost(input) {
  try {
    const domain = new URL(input).hostname.replace(/^www\./, '');
    return domain === 'manhwazone.com' || domain.endsWith('.manhwazone.com');
  } catch {
    return false;
  }
}

function isPreview(input) {
  try {
    const parsed = new URL(input);
    return isHost(parsed.toString()) && parsed.pathname.startsWith('/preview/');
  } catch {
    return false;
  }
}

function isSeries(input) {
  try {
    const parsed = new URL(input);
    if (!isHost(parsed.toString()) || isPreview(parsed.toString())) {
      return false;
    }
    const segments = parsed.pathname.split('/').filter(Boolean);
    const first = (segments[0] ?? '').toLowerCase();
    if (segments.length === 0 || first === 'search') {
      return false;
    }
    return !['account', 'auth', 'cdn-cgi', 'contact', 'dmca', 'login', 'privacy', 'register', 'terms'].includes(first);
  } catch {
    return false;
  }
}

const rows = [];
for (const element of $('a[href]').toArray()) {
  const anchor = $(element);
  const href = anchor.attr('href') ?? '';
  const resolved = new URL(href, base).toString();
  if (!isSeries(resolved)) {
    continue;
  }
  const text = anchor.text().replace(/\s+/g, ' ').trim();
  rows.push({ href: resolved, text: text.slice(0, 140) });
}

console.log(JSON.stringify(rows.slice(0, 40), null, 2));
