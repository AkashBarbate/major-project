"""
revised_data_strategy.py
Combines HuggingFace datasets (historical) + RSS feeds (current Indian news)
Total target: ~2200 articles, 220 per category
"""

from datasets import load_dataset
import pandas as pd
import feedparser
import time
import logging
from newspaper import Article, Config

log = logging.getLogger(__name__)

CATEGORIES = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health',
    'Science/Other', 'Sports', 'Entertainment'
]

# ── Part A: RSS feeds (current, ~40% of data) ─────────────────────────────────
# Only sites confirmed to work without blocking

SAFE_RSS_SOURCES = {
    'Politics': [
        'https://feeds.feedburner.com/ndtvnews-india-news',
        'http://feeds.bbci.co.uk/news/politics/rss.xml',
        'https://www.thehindu.com/news/national/feeder/default.rss',
    ],
    'Technology': [
        'http://feeds.bbci.co.uk/news/technology/rss.xml',
        'https://feeds.feedburner.com/ndtvnews-tech',
        'https://www.thehindu.com/sci-tech/technology/feeder/default.rss',
    ],
    'Business': [
        'http://feeds.bbci.co.uk/news/business/rss.xml',
        'https://www.thehindu.com/business/feeder/default.rss',
        'https://feeds.feedburner.com/ndtvnews-business',
    ],
    'Economy': [
        'https://www.thehindu.com/business/Economy/feeder/default.rss',
        'https://feeds.feedburner.com/ndtvnews-economy',
    ],
    'International_Affairs': [
        'http://feeds.bbci.co.uk/news/world/rss.xml',
        'https://feeds.feedburner.com/ndtvnews-world-news',
        'https://www.thehindu.com/news/international/feeder/default.rss',
    ],
    'Education': [
        'https://feeds.feedburner.com/ndtvnews-education',
        'https://www.thehindu.com/education/feeder/default.rss',
    ],
    'Health': [
        'http://feeds.bbci.co.uk/news/health/rss.xml',
        'https://www.thehindu.com/sci-tech/health/feeder/default.rss',
        'https://feeds.feedburner.com/ndtvnews-health',
    ],
    'Science/Other': [
        'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
        'https://www.thehindu.com/sci-tech/science/feeder/default.rss',
    ],
    'Sports': [
        'http://feeds.bbci.co.uk/sport/rss.xml',
        'https://feeds.feedburner.com/ndtvnews-sports',
        'https://www.thehindu.com/sport/feeder/default.rss',
    ],
    'Entertainment': [
        'https://feeds.feedburner.com/ndtvnews-entertainment',
        'https://www.thehindu.com/entertainment/feeder/default.rss',
    ],
}

def get_config():
    config = Config()
    config.browser_user_agent = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
    config.request_timeout = 12
    config.fetch_images = False
    config.memoize_articles = False
    return config

def scrape_rss_entry(entry, category, config) -> dict | None:
    """
    First tries to get text from RSS entry itself (no HTTP request needed).
    Falls back to downloading the article page only if RSS text is too short.
    """
    # Try RSS content directly — avoids hitting article URL entirely
    rss_text = ''
    if hasattr(entry, 'content') and entry.content:
        rss_text = entry.content[0].get('value', '')
    elif hasattr(entry, 'summary'):
        rss_text = entry.summary or ''

    # Clean HTML tags from RSS text
    import re
    rss_text = re.sub(r'<[^>]+>', ' ', rss_text).strip()
    rss_text = re.sub(r'\s+', ' ', rss_text).strip()

    title = getattr(entry, 'title', '') or ''
    url   = getattr(entry, 'link',  '') or ''

    # If RSS gives enough text, use it directly (avoids scraping entirely)
    if len(rss_text.split()) >= 80:
        return {
            'title':        title,
            'text':         rss_text,
            'url':          url,
            'domain':       url.split('/')[2].replace('www.', '') if url else '',
            'word_count':   len(rss_text.split()),
            'source_section': category,
            'source_type':  'rss_content',
            'is_trusted':   1,
        }

    # RSS text too short — try downloading the article
    if not url:
        return None

    try:
        art = Article(url, config=config)
        art.download()
        art.parse()
        text = art.text or ''
        if len(text.split()) < 100:
            return None
        return {
            'title':        art.title or title,
            'text':         text[:2500],
            'url':          url,
            'domain':       url.split('/')[2].replace('www.', '') if url else '',
            'word_count':   len(text.split()),
            'source_section': category,
            'source_type':  'full_scrape',
            'is_trusted':   1,
        }
    except:
        return None


def collect_rss_articles(target_per_category: int = 90) -> pd.DataFrame:
    """Collect current articles from safe RSS feeds."""
    config = get_config()
    seen_urls = set()
    all_articles = []

    for category, feeds in SAFE_RSS_SOURCES.items():
        cat_articles = []
        log.info(f"\nRSS scraping: {category}")

        for feed_url in feeds:
            if len(cat_articles) >= target_per_category:
                break
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:60]:
                    url = getattr(entry, 'link', '')
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    art = scrape_rss_entry(entry, category, config)
                    if art:
                        cat_articles.append(art)

                    time.sleep(0.3)

            except Exception as e:
                log.warning(f"Feed failed: {feed_url}: {e}")

        log.info(f"  {category}: {len(cat_articles)} articles")
        all_articles.extend(cat_articles)

    return pd.DataFrame(all_articles)


# ── Part B: HuggingFace datasets (historical, ~60% of data) ──────────────────

def collect_historical_articles(target_per_category: int = 130) -> pd.DataFrame:
    """
    Pull historical articles from CC-News filtered to English.
    These will be labeled by Claude in the next step.
    Uses streaming to avoid downloading the entire dataset.
    """
    log.info("\nLoading CC-News dataset (streaming)...")

    # CC-News is large — stream it and stop when we have enough
    dataset = load_dataset(
        "cc_news",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Domains known to produce good quality English news
    preferred_domains = {
        'ndtv.com', 'thehindu.com', 'hindustantimes.com',
        'theprint.in', 'scroll.in', 'thewire.in',
        'bbc.com', 'bbc.co.uk', 'reuters.com', 'apnews.com',
        'aljazeera.com', 'theguardian.com',
    }

    # Rough keyword hints to pre-filter by likely category
    # These aren't labels — just to get a diverse initial sample
    # Claude will do the actual labeling
    category_hints = {
        'Politics':             ['parliament', 'minister', 'election', 'government', 'party', 'vote', 'political'],
        'Technology':           ['artificial intelligence', 'software', 'startup', 'app', 'cyber', 'smartphone', 'tech'],
        'Business':             ['company', 'stock', 'merger', 'ceo', 'revenue', 'profit', 'corporate'],
        'Economy':              ['gdp', 'inflation', 'interest rate', 'rbi', 'fiscal', 'budget', 'economic'],
        'International_Affairs':['bilateral', 'diplomatic', 'foreign', 'sanctions', 'un ', 'nato', 'treaty'],
        'Education':            ['university', 'school', 'student', 'exam', 'curriculum', 'teacher', 'admission'],
        'Health':               ['hospital', 'vaccine', 'disease', 'health', 'medicine', 'doctor', 'patient'],
        'Science/Other':        ['research', 'climate', 'space', 'environment', 'scientist', 'study finds'],
        'Sports':               ['cricket', 'football', 'tournament', 'match', 'player', 'championship', 'ipl'],
        'Entertainment':        ['bollywood', 'film', 'actor', 'movie', 'celebrity', 'series', 'box office'],
    }

    collected = {cat: [] for cat in CATEGORIES}
    total_target = target_per_category * len(CATEGORIES)
    checked = 0

    import re
    for item in dataset:
        # Stop when we have enough overall
        total_so_far = sum(len(v) for v in collected.values())
        if total_so_far >= total_target:
            break

        checked += 1
        if checked % 5000 == 0:
            log.info(f"  Scanned {checked} items, collected {total_so_far}/{total_target}")

        # Quality filter
        text = item.get('text', '') or ''
        title = item.get('title', '') or ''
        word_count = len(text.split())

        if word_count < 120 or word_count > 5000:
            continue
        if not title:
            continue

        # Language check (simple heuristic — mostly English words)
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.6:
            continue

        domain = item.get('domain', '') or ''

        # Assign a likely category using keyword hints
        # This is just for balancing the collection — Claude does final labeling
        text_lower = (title + ' ' + text[:500]).lower()
        matched_cat = None
        for cat, hints in category_hints.items():
            if any(hint in text_lower for hint in hints):
                if len(collected[cat]) < target_per_category:
                    matched_cat = cat
                    break

        if matched_cat is None:
            continue

        collected[matched_cat].append({
            'title':          title,
            'text':           text[:2500],
            'url':            item.get('url', ''),
            'domain':         domain,
            'word_count':     word_count,
            'source_section': matched_cat,  # tentative — Claude will relabel
            'source_type':    'historical',
            'is_trusted':     int(domain in preferred_domains),
        })

    all_historical = []
    for cat, arts in collected.items():
        log.info(f"  Historical {cat}: {len(arts)}")
        all_historical.extend(arts)

    return pd.DataFrame(all_historical)


# ── Part C: Merge and deduplicate ─────────────────────────────────────────────

def build_combined_dataset(
    output_path: str = 'raw_scraped_articles.csv',
    rss_target: int = 90,        # per category from RSS
    historical_target: int = 130  # per category from CC-News
) -> pd.DataFrame:
    """
    Combines RSS (current) + CC-News (historical).
    Deduplicates, shuffles, saves.
    """
    log.info("="*60)
    log.info("PHASE 1: Building combined dataset")
    log.info(f"RSS target:        {rss_target} per category")
    log.info(f"Historical target: {historical_target} per category")
    log.info(f"Total target:      ~{(rss_target + historical_target) * len(CATEGORIES)} articles")
    log.info("="*60)

    rss_df  = collect_rss_articles(target_per_category=rss_target)
    hist_df = collect_historical_articles(target_per_category=historical_target)

    combined = pd.concat([rss_df, hist_df], ignore_index=True)

    # Deduplicate by URL
    before = len(combined)
    combined.drop_duplicates(subset='url', inplace=True)

    # Deduplicate by title (catches same article from different URLs)
    combined['title_norm'] = combined['title'].str.lower().str.strip()
    combined.drop_duplicates(subset='title_norm', inplace=True)
    combined.drop(columns=['title_norm'], inplace=True)

    combined.reset_index(drop=True, inplace=True)

    log.info(f"\nCombined dataset: {len(combined)} articles "
             f"(removed {before - len(combined)} duplicates)")
    log.info("\nFinal distribution:")
    log.info(combined['source_section'].value_counts().to_string())
    log.info("\nSource type breakdown:")
    log.info(combined['source_type'].value_counts().to_string())

    combined.to_csv(output_path, index=False, encoding='utf-8')
    log.info(f"\nSaved → {output_path}")
    return combined


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    df = build_combined_dataset(
        output_path='raw_scraped_articles.csv',
        rss_target=90,        # ~900 current articles
        historical_target=130  # ~1300 historical articles
        # Total: ~2200 articles across 10 categories
    )