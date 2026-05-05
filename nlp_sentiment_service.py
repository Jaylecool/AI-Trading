"""
NLP Sentiment Service
---------------------
Computes financial sentiment scores for news headlines and article bodies.

Primary backend  : FinBERT (ProsusAI/finbert) — financial-domain BERT
                   ~420 MB first-time download; GPU-optional.
Fallback backend : VADER (vaderSentiment) — rule-based, instant, no download.

Backend selection (in priority order):
  1. Use 'finbert'  if NLP_BACKEND=finbert (default) AND torch/transformers available.
  2. Fall back to 'vader' automatically when torch is absent or on import error.
  3. Set NLP_BACKEND=vader in .env to skip FinBERT entirely.

Key public functions:
  analyze_sentiment(text)               → {positive, negative, neutral, compound, backend}
  score_news_batch(articles)            → float   (weighted aggregate score  -1…+1)
  compute_rolling_sentiment(symbol, ..) → {window_days: score, …}
  score_articles_inplace(articles)      → articles list with sentiment_score filled

Sentiment cache:
  Results are stored in  data/{SYMBOL}_sentiment_cache.json
  keyed by article['id'] so FinBERT inference is never repeated for the same article.

Usage (standalone test):
    python nlp_sentiment_service.py
"""

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config as cfg
from news_data_fetcher import load_cached_news, get_recent_articles, _cache_path

DATA_DIR = cfg.DATA_DIR

# ---------------------------------------------------------------------------
# FinBERT loader (lazy — downloaded on first use)
# ---------------------------------------------------------------------------

_finbert_lock = threading.Lock()
_finbert_pipeline = None        # transformers Pipeline object
_finbert_available: Optional[bool] = None   # None = not checked yet


def _load_finbert():
    """
    Load FinBERT once and cache it in _finbert_pipeline.
    Returns the pipeline or None if unavailable.
    Thread-safe via _finbert_lock.
    """
    global _finbert_pipeline, _finbert_available

    with _finbert_lock:
        if _finbert_available is not None:
            return _finbert_pipeline

        if cfg.NLP_BACKEND == 'vader':
            _finbert_available = False
            return None

        try:
            from transformers import pipeline as hf_pipeline
            import torch  # noqa: F401 — confirms torch is importable

            print("[NLP] Loading FinBERT (ProsusAI/finbert) — first run may download ~420 MB …")
            _finbert_pipeline = hf_pipeline(
                task='text-classification',
                model='ProsusAI/finbert',
                tokenizer='ProsusAI/finbert',
                top_k=None,         # return all three class scores
                device=-1,          # CPU; set to 0 for GPU if available
                truncation=True,
                max_length=512,
            )
            _finbert_available = True
            print("[NLP] FinBERT loaded successfully.")
        except Exception as e:
            print(f"[NLP] FinBERT unavailable ({e}) — falling back to VADER.")
            _finbert_pipeline = None
            _finbert_available = False

        return _finbert_pipeline


# ---------------------------------------------------------------------------
# VADER loader (instant, no download)
# ---------------------------------------------------------------------------

_vader_analyzer = None
_vader_lock = threading.Lock()


def _get_vader():
    global _vader_analyzer
    with _vader_lock:
        if _vader_analyzer is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                _vader_analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                raise RuntimeError(
                    "Neither FinBERT nor VADER is available. "
                    "Install with: pip install vaderSentiment"
                )
    return _vader_analyzer


# ---------------------------------------------------------------------------
# Core sentiment function
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> Dict:
    """
    Analyse a single piece of financial text.

    Returns:
        {
          'positive': float,   # 0.0–1.0
          'negative': float,   # 0.0–1.0
          'neutral':  float,   # 0.0–1.0
          'compound': float,   # -1.0–+1.0  (positive – negative)
          'backend':  str,     # 'finbert' or 'vader'
        }
    """
    if not text or not text.strip():
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34,
                'compound': 0.0, 'backend': 'none'}

    text = text.strip()[:1024]  # truncate to safe length

    # --- Try FinBERT first ---
    pipeline = _load_finbert()
    if pipeline is not None:
        try:
            results = pipeline(text)[0]  # list of {label, score}
            scores = {r['label'].lower(): r['score'] for r in results}
            pos = scores.get('positive', 0.0)
            neg = scores.get('negative', 0.0)
            neu = scores.get('neutral', 0.0)
            return {
                'positive': round(pos, 4),
                'negative': round(neg, 4),
                'neutral': round(neu, 4),
                'compound': round(pos - neg, 4),
                'backend': 'finbert',
            }
        except Exception as e:
            print(f"[NLP] FinBERT inference error: {e} — using VADER")

    # --- VADER fallback ---
    vader = _get_vader()
    scores = vader.polarity_scores(text)
    pos = scores['pos']
    neg = scores['neg']
    neu = scores['neu']
    compound = scores['compound']
    # Normalise compound (-1…+1) into the same pos/neg/neu split for consistency
    return {
        'positive': round(pos, 4),
        'negative': round(neg, 4),
        'neutral': round(neu, 4),
        'compound': round(compound, 4),
        'backend': 'vader',
    }


# ---------------------------------------------------------------------------
# Batch scoring helpers
# ---------------------------------------------------------------------------

def score_news_batch(articles: List[Dict],
                     recency_decay_hours: float = 48.0) -> float:
    """
    Compute a single sentiment score for a batch of articles.

    More-recent articles are weighted higher via exponential decay.
    Compound scores already in articles['sentiment_score'] are reused;
    otherwise analyze_sentiment() is called on title + summary.

    Returns a float in [-1, +1].
    """
    if not articles:
        return 0.0

    now = datetime.now()
    weighted_sum = 0.0
    weight_total = 0.0

    for art in articles:
        # Use cached sentiment if available
        compound = art.get('sentiment_score')
        if compound is None:
            text = f"{art.get('title', '')} {art.get('summary', '')}".strip()
            result = analyze_sentiment(text)
            compound = result['compound']

        # Recency weight
        try:
            pub_str = art.get('published_at', '')
            pub_str_clean = pub_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(pub_str_clean)
            if dt.tzinfo is not None:
                from datetime import timezone
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            age_hours = max(0.0, (now - dt).total_seconds() / 3600)
        except (ValueError, TypeError):
            age_hours = recency_decay_hours  # treat as old

        import math
        weight = math.exp(-age_hours / recency_decay_hours)
        weighted_sum += compound * weight
        weight_total += weight

    if weight_total == 0:
        return 0.0

    return round(weighted_sum / weight_total, 4)


def compute_rolling_sentiment(symbol: str,
                              windows_days: List[int] = None) -> Dict[str, float]:
    """
    Compute recency-weighted sentiment scores over multiple rolling windows.

    Args:
        symbol:       Stock ticker.
        windows_days: List of day windows, e.g. [1, 3, 7].

    Returns:
        { '1d': 0.12,  '3d': 0.07,  '7d': -0.03,
          'news_volume_7d': 14 }
    """
    if windows_days is None:
        windows_days = [1, 3, 7]

    result: Dict[str, float] = {}

    for days in windows_days:
        articles = get_recent_articles(symbol, hours=days * 24)
        score = score_news_batch(articles)
        result[f'{days}d'] = score

    # News volume (raw count over 7 days)
    vol_articles = get_recent_articles(symbol, hours=7 * 24)
    result['news_volume_7d'] = len(vol_articles)

    return result


# ---------------------------------------------------------------------------
# Sentiment cache (article-level — avoids re-running FinBERT on same article)
# ---------------------------------------------------------------------------

def _sentiment_cache_path(symbol: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f'{symbol}_sentiment_cache.json')


def _load_sentiment_cache(symbol: str) -> Dict[str, float]:
    """Load {article_id: compound_score} mapping from disk."""
    path = _sentiment_cache_path(symbol)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sentiment_cache(symbol: str, cache: Dict[str, float]):
    path = _sentiment_cache_path(symbol)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except OSError:
        pass


def score_articles_inplace(symbol: str, articles: List[Dict]) -> List[Dict]:
    """
    Fill in sentiment_score and sentiment_label for any article that doesn't
    have one yet, using the per-symbol disk cache to avoid recomputation.

    Modifies the list in-place AND persists updated scores.
    Returns the same list (for chaining).
    """
    cache = _load_sentiment_cache(symbol)
    dirty = False

    for art in articles:
        aid = art.get('id', '')
        if aid in cache:
            art['sentiment_score'] = cache[aid]
        elif art.get('sentiment_score') is None:
            text = f"{art.get('title', '')} {art.get('summary', '')}".strip()
            result = analyze_sentiment(text)
            compound = result['compound']
            art['sentiment_score'] = compound
            art['sentiment_label'] = (
                'positive' if compound > 0.05
                else 'negative' if compound < -0.05
                else 'neutral'
            )
            if aid:
                cache[aid] = compound
                dirty = True
        else:
            # Already has a score — keep it, but ensure label is set
            c = art['sentiment_score']
            if art.get('sentiment_label') is None:
                art['sentiment_label'] = (
                    'positive' if c > 0.05 else 'negative' if c < -0.05 else 'neutral'
                )

    if dirty:
        _save_sentiment_cache(symbol, cache)

    return articles


# ---------------------------------------------------------------------------
# Convenience: compute features for a single symbol from cached news
# ---------------------------------------------------------------------------

def get_sentiment_features(symbol: str) -> Dict:
    """
    Return the five sentiment-based ML features for *symbol*:
      Sentiment_1d, Sentiment_3d, Sentiment_7d,
      News_Volume_7d, Sentiment_Momentum
    These are the columns appended to the stock data CSV by data_fetcher.py.
    """
    rolling = compute_rolling_sentiment(symbol, windows_days=[1, 3, 7])
    s1d = rolling.get('1d', 0.0)
    s3d = rolling.get('3d', 0.0)
    s7d = rolling.get('7d', 0.0)
    vol7d = rolling.get('news_volume_7d', 0)
    momentum = round(s1d - s7d, 4)   # short-term vs long-term sentiment shift

    return {
        'Sentiment_1d': s1d,
        'Sentiment_3d': s3d,
        'Sentiment_7d': s7d,
        'News_Volume_7d': float(vol7d),
        'Sentiment_Momentum': momentum,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_headlines = [
        "Apple reports record-breaking quarterly profits, beats Wall Street estimates",
        "Tesla faces massive recall over safety concerns, stock plummets",
        "Microsoft announces new AI partnership, shares rise in after-hours trading",
        "Fed signals potential rate cuts amid cooling inflation data",
        "Markets mixed ahead of key earnings season",
    ]

    print("=== FinBERT / VADER Sentiment Test ===\n")
    for headline in test_headlines:
        result = analyze_sentiment(headline)
        bar_len = int((result['compound'] + 1) / 2 * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        print(f"[{result['backend']:7s}] {result['compound']:+.3f}  [{bar}]")
        print(f"  {headline[:70]}")
        print()
