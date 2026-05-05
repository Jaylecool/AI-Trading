"""
News Data Fetcher
-----------------
Retrieves financial news headlines for a given stock symbol from multiple
free / optional sources, deduplicates by URL hash, and caches results to
  data/{SYMBOL}_news_cache.json

Sources (in priority order):
  1. yfinance Ticker.news  — completely free, no API key required
  2. Finnhub               — free tier, set FINNHUB_API_KEY in .env
  3. NewsAPI               — free tier, set NEWSAPI_KEY in .env

Usage (standalone test):
    python news_data_fetcher.py          # fetches news for all default symbols
    python news_data_fetcher.py AAPL     # single symbol
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config as cfg

DATA_DIR = cfg.DATA_DIR
DEFAULT_SYMBOLS = cfg.SUPPORTED_SYMBOLS
NEWS_LOOKBACK_DAYS = cfg.NEWS_LOOKBACK_DAYS


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

def _article_id(url: str) -> str:
    """Stable 12-char hash used as the deduplication key."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _make_article(title: str, url: str, published_at: str, source: str,
                  summary: str = '') -> Dict:
    return {
        'id': _article_id(url),
        'title': title.strip(),
        'summary': summary.strip(),
        'url': url,
        'published_at': published_at,
        'source': source,
        'sentiment_score': None,   # filled by nlp_sentiment_service
        'sentiment_label': None,
    }


# ---------------------------------------------------------------------------
# Source: yfinance (free, no key)
# ---------------------------------------------------------------------------

def _fetch_yfinance_news(symbol: str) -> List[Dict]:
    """
    Pull news from yfinance.Ticker.news.
    Returns a list of normalised article dicts.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
    except Exception as e:
        print(f"  [news/yfinance] {symbol}: {e}")
        return []

    articles = []
    for item in raw:
        try:
            # yfinance news schema varies; handle both older and newer versions
            content = item.get('content', item)
            title = (
                content.get('title')
                or item.get('title', '')
            )
            url = (
                content.get('canonicalUrl', {}).get('url')
                or content.get('clickThroughUrl', {}).get('url')
                or item.get('link', '')
            )
            if not title or not url:
                continue

            # published_at: prefer ISO string
            pub = content.get('pubDate') or item.get('providerPublishTime', 0)
            if isinstance(pub, (int, float)) and pub > 0:
                pub = datetime.utcfromtimestamp(pub).isoformat()
            else:
                pub = str(pub)

            summary = content.get('summary', '')
            source = content.get('provider', {}).get('displayName', 'Yahoo Finance')

            articles.append(_make_article(title, url, pub, source, summary))
        except Exception:
            continue

    return articles


# ---------------------------------------------------------------------------
# Source: Finnhub (free tier — optional)
# ---------------------------------------------------------------------------

def _fetch_finnhub_news(symbol: str, lookback_days: int = 7) -> List[Dict]:
    """
    Fetch news from Finnhub free API.
    Returns [] silently if FINNHUB_API_KEY is not set or package missing.
    """
    if not cfg.FINNHUB_API_KEY:
        return []
    try:
        import finnhub
    except ImportError:
        return []

    try:
        client = finnhub.Client(api_key=cfg.FINNHUB_API_KEY)
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        raw = client.company_news(symbol, _from=start, to=end) or []
    except Exception as e:
        print(f"  [news/finnhub] {symbol}: {e}")
        return []

    articles = []
    for item in raw:
        try:
            url = item.get('url', '')
            title = item.get('headline', '')
            if not url or not title:
                continue
            pub = item.get('datetime', 0)
            if isinstance(pub, (int, float)) and pub > 0:
                pub = datetime.utcfromtimestamp(pub).isoformat()
            articles.append(_make_article(
                title=title,
                url=url,
                published_at=str(pub),
                source=item.get('source', 'Finnhub'),
                summary=item.get('summary', ''),
            ))
        except Exception:
            continue

    return articles


# ---------------------------------------------------------------------------
# Source: NewsAPI (free tier — optional)
# ---------------------------------------------------------------------------

def _fetch_newsapi_news(symbol: str, company_name: str = '',
                        lookback_days: int = 7) -> List[Dict]:
    """
    Fetch news from NewsAPI.org free tier.
    Returns [] silently if NEWSAPI_KEY is not set or package missing.
    """
    if not cfg.NEWSAPI_KEY:
        return []
    try:
        from newsapi import NewsApiClient
    except ImportError:
        return []

    query = company_name if company_name else symbol
    try:
        client = NewsApiClient(api_key=cfg.NEWSAPI_KEY)
        from_dt = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        response = client.get_everything(
            q=query,
            from_param=from_dt,
            language='en',
            sort_by='publishedAt',
            page_size=50,
        )
        raw = response.get('articles', [])
    except Exception as e:
        print(f"  [news/newsapi] {symbol}: {e}")
        return []

    articles = []
    for item in raw:
        try:
            url = item.get('url', '')
            title = item.get('title', '')
            if not url or not title:
                continue
            articles.append(_make_article(
                title=title,
                url=url,
                published_at=item.get('publishedAt', ''),
                source=item.get('source', {}).get('name', 'NewsAPI'),
                summary=item.get('description', ''),
            ))
        except Exception:
            continue

    return articles


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(symbol: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f'{symbol}_news_cache.json')


def load_cached_news(symbol: str) -> List[Dict]:
    """Load news articles from local cache file."""
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_cache(symbol: str, articles: List[Dict]):
    """Persist deduplicated articles to cache (atomic write)."""
    path = _cache_path(symbol)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as e:
        print(f"  [news/cache] Failed to save {symbol} cache: {e}")


def _merge_deduplicate(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge two article lists, deduplicating by 'id' (URL hash)."""
    seen = {a['id'] for a in existing}
    merged = list(existing)
    for art in new:
        if art['id'] not in seen:
            seen.add(art['id'])
            merged.append(art)
    # Keep most-recent 500 articles; sort newest first
    merged.sort(key=lambda a: a.get('published_at', ''), reverse=True)
    return merged[:500]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_COMPANY_NAMES: Dict[str, str] = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google Alphabet',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'META': 'Meta Facebook',
    'NVDA': 'Nvidia',
}


def fetch_news(symbol: str,
               lookback_days: int = NEWS_LOOKBACK_DAYS,
               save: bool = True) -> List[Dict]:
    """
    Fetch latest news for *symbol* from all configured sources, merge with
    existing cache, and optionally save.

    Returns the merged (deduplicated) article list sorted newest-first.
    """
    symbol = symbol.upper()
    company = _COMPANY_NAMES.get(symbol, symbol)

    print(f"  [news] Fetching news for {symbol} …")

    articles: List[Dict] = []

    # 1. yfinance (always attempted — no key needed)
    yf_articles = _fetch_yfinance_news(symbol)
    articles.extend(yf_articles)
    print(f"    yfinance: {len(yf_articles)} articles")

    # 2. Finnhub (optional)
    fh_articles = _fetch_finnhub_news(symbol, lookback_days=lookback_days)
    articles.extend(fh_articles)
    if fh_articles:
        print(f"    Finnhub:  {len(fh_articles)} articles")

    # 3. NewsAPI (optional)
    na_articles = _fetch_newsapi_news(symbol, company, lookback_days=lookback_days)
    articles.extend(na_articles)
    if na_articles:
        print(f"    NewsAPI:  {len(na_articles)} articles")

    # Merge with existing cache
    existing = load_cached_news(symbol)
    merged = _merge_deduplicate(existing, articles)
    print(f"    Total cached: {len(merged)} articles ({len(merged) - len(existing)} new)")

    if save:
        _save_cache(symbol, merged)

    return merged


def refresh_news(symbol: str) -> List[Dict]:
    """
    Incremental update: only fetch if cache is older than NEWS_POLL_MINUTES
    or if there are fewer than 5 articles cached.
    """
    symbol = symbol.upper()
    existing = load_cached_news(symbol)

    # Check freshness from newest article timestamp
    if len(existing) >= 5:
        newest = existing[0].get('published_at', '')
        try:
            dt = datetime.fromisoformat(newest.replace('Z', '+00:00'))
            age_minutes = (datetime.now(dt.tzinfo) - dt).total_seconds() / 60
            if age_minutes < cfg.NEWS_POLL_MINUTES:
                return existing
        except (ValueError, TypeError):
            pass  # Can't parse timestamp — refetch

    return fetch_news(symbol)


def get_recent_articles(symbol: str, hours: int = 24) -> List[Dict]:
    """
    Return only articles published within the last *hours* hours.
    Uses cached data (does not trigger a network fetch).
    """
    cutoff = datetime.now() - timedelta(hours=hours)
    articles = load_cached_news(symbol)
    recent = []
    for art in articles:
        try:
            pub_str = art.get('published_at', '')
            # Handle timezone-aware ISO strings
            pub_str_clean = pub_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(pub_str_clean)
            # Make naive for comparison
            if dt.tzinfo is not None:
                from datetime import timezone
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            if dt >= cutoff:
                recent.append(art)
        except (ValueError, TypeError):
            continue
    return recent


def fetch_all_symbols_news(symbols: Optional[List[str]] = None,
                           lookback_days: int = NEWS_LOOKBACK_DAYS) -> Dict[str, List[Dict]]:
    """Fetch and cache news for multiple symbols. Returns {symbol: [articles]}."""
    symbols = symbols or DEFAULT_SYMBOLS
    results = {}
    for sym in symbols:
        try:
            results[sym] = fetch_news(sym, lookback_days=lookback_days)
            time.sleep(0.5)  # gentle rate-limiting between symbols
        except Exception as e:
            print(f"  [news] Error fetching {sym}: {e}")
            results[sym] = load_cached_news(sym)
    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ['AAPL']
    for sym in symbols:
        arts = fetch_news(sym.upper(), lookback_days=7)
        print(f"\n=== {sym}: {len(arts)} total articles ===")
        for a in arts[:5]:
            print(f"  [{a['published_at'][:10]}] {a['title'][:80]}")
        print()
