"""
adaptive_scrape.py
------------------
Unified adaptive scraper that:
  1. Reads current dataset counts
  2. Computes per-category scraping quotas using inverse-frequency weighting
  3. Scrapes from an expanded multi-source RSS list (all 10 categories)
  4. Pulls weighted historical data from CC-News
  5. Merges, deduplicates, and saves

Target: 300 articles per category = 3000 total
Run after revised_data_strategy.py to top up the full dataset.

Requirements:
    pip install newspaper4k feedparser lxml_html_clean pandas datasets
"""

import feedparser
import pandas as pd
import numpy as np
import time
import re
import os
import logging
from datetime import datetime
from newspaper import Article, Config
from urllib.parse import urlparse
from datasets import load_dataset

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/adaptive_scrape_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Global config ──────────────────────────────────────────────────────────────
EXISTING_CSV      = 'raw_scraped_articles.csv'
OUTPUT_CSV        = 'raw_scraped_articles.csv'
TARGET_PER_CAT    = 300     # final target per category
CRAWL_DELAY       = 0.4     # seconds between article requests
MAX_PER_FEED      = 100     # max entries to try per RSS feed
MIN_WORDS         = 100     # minimum word count
MAX_WORDS_STORED  = 2500    # cap stored text length

CATEGORIES = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health',
    'Science/Other', 'Sports', 'Entertainment'
]

TRUSTED_DOMAINS = {
    'bbc.com', 'bbc.co.uk', 'ndtv.com', 'thehindu.com',
    'timesofindia.indiatimes.com', 'hindustantimes.com',
    'indianexpress.com', 'theprint.in', 'scroll.in',
    'thewire.in', 'livemint.com', 'business-standard.com',
    'financialexpress.com', 'reuters.com', 'apnews.com',
    'techcrunch.com', 'theverge.com', 'wired.com',
    'arstechnica.com', 'espncricinfo.com', 'cricbuzz.com',
    'variety.com', 'hollywoodreporter.com',
    'universityworldnews.com', 'moneycontrol.com',
}

# ── Expanded RSS sources (all 10 categories) ───────────────────────────────────
ALL_SOURCES = {

    'Politics': [
        # Indian national politics
        'https://feeds.feedburner.com/ndtvnews-india-news',
        'https://www.thehindu.com/news/national/feeder/default.rss',
        'https://www.thehindu.com/news/national/other-states/feeder/default.rss',
        'https://theprint.in/politics/feed/',
        'https://scroll.in/politics/feed',
        'https://thewire.in/category/politics/feed',
        # International politics
        'http://feeds.bbci.co.uk/news/politics/rss.xml',
        'http://feeds.bbci.co.uk/news/uk_politics/rss.xml',
        # Indian state politics
        'https://timesofindia.indiatimes.com/rssfeeds/1221656.cms',
        'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
        'https://indianexpress.com/section/political-pulse/feed/',
        'https://www.deccanherald.com/india/rss',
    ],

    'Technology': [
        # International tech
        'https://techcrunch.com/feed/',
        'https://www.theverge.com/rss/index.xml',
        'https://feeds.arstechnica.com/arstechnica/index',
        'https://www.wired.com/feed/rss',
        'https://www.technologyreview.com/feed/',
        'https://venturebeat.com/feed/',
        # Indian tech
        'https://gadgets360.com/rss/news',
        'https://beebom.com/feed/',
        'https://feeds.feedburner.com/ndtvnews-tech',
        'https://www.thehindu.com/sci-tech/technology/feeder/default.rss',
        'https://indianexpress.com/section/technology/feed/',
        'https://timesofindia.indiatimes.com/rssfeeds/66949542.cms',
        'https://entrackr.com/feed/',           # Indian startup news
        'https://inc42.com/feed/',              # Indian startup ecosystem
    ],

    'Business': [
        'http://feeds.bbci.co.uk/news/business/rss.xml',
        'https://www.thehindu.com/business/feeder/default.rss',
        'https://www.livemint.com/rss/companies',
        'https://www.business-standard.com/rss/companies-10101.rss',
        'https://www.financialexpress.com/market/feed/',
        'https://feeds.feedburner.com/ndtvnews-business',
        'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'https://timesofindia.indiatimes.com/business/rssfeeds/1898055.cms',
        'https://www.moneycontrol.com/rss/business.xml',
        'https://www.deccanherald.com/business/rss',
        'https://qz.com/india/feed',            # Quartz India business
        'https://www.fortuneindia.com/rss/latest',
    ],

    'Economy': [
        # Macro economy — specifically NOT just business/markets
        'https://www.thehindu.com/business/Economy/feeder/default.rss',
        'https://www.livemint.com/rss/economy',
        'https://www.business-standard.com/rss/economy-policy-10601.rss',
        'https://www.financialexpress.com/economy/feed/',
        'https://www.moneycontrol.com/rss/economy.xml',
        'https://feeds.feedburner.com/ndtvnews-economy',
        'https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms',
        'https://www.reuters.com/business/finance/rss.xml',
        'https://www.imf.org/en/News/rss?language=eng',   # IMF news
        'https://timesofindia.indiatimes.com/business/india-business/rssfeeds/1898055.cms',
        'https://www.thehindu.com/business/Industry/feeder/default.rss',
        'https://thewire.in/category/economy/feed',
    ],

    'International_Affairs': [
        'http://feeds.bbci.co.uk/news/world/rss.xml',
        'https://feeds.feedburner.com/ndtvnews-world-news',
        'https://www.thehindu.com/news/international/feeder/default.rss',
        'https://indianexpress.com/section/world/feed/',
        'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
        'https://theprint.in/diplomacy/feed/',
        'https://thewire.in/category/world/feed',
        'https://www.aljazeera.com/xml/rss/all.xml',
        'https://feeds.reuters.com/Reuters/worldNews',
        'https://foreignpolicy.com/feed/',
        'https://www.thehindu.com/news/national/other-states/feeder/default.rss',
        'https://scroll.in/world/feed',
        'https://www.deccanherald.com/international/rss',
    ],

    'Education': [
        'https://feeds.feedburner.com/ndtvnews-education',
        'https://www.thehindu.com/education/feeder/default.rss',
        'https://timesofindia.indiatimes.com/rssfeeds/913168846.cms',
        'https://indianexpress.com/section/education/feed/',
        'https://www.hindustantimes.com/feeds/rss/education/rssfeed.xml',
        'https://www.universityworldnews.com/rss.php',    # global HE
        'https://scroll.in/education/feed',
        'https://www.careers360.com/rss/news',
        'https://www.jagranjosh.com/rss/education-news.xml',
        'https://www.edexlive.com/rss/news',
        'https://india.mongabay.com/feed/',               # science education
        'https://www.thehindu.com/education/colleges/feeder/default.rss',
        'https://www.thehindu.com/education/schools/feeder/default.rss',
        'https://thewire.in/category/education/feed',
        'https://theprint.in/campus-voice/feed/',
    ],

    'Health': [
        'http://feeds.bbci.co.uk/news/health/rss.xml',
        'https://www.thehindu.com/sci-tech/health/feeder/default.rss',
        'https://feeds.feedburner.com/ndtvnews-health',
        'https://timesofindia.indiatimes.com/rssfeeds/3908999.cms',
        'https://www.hindustantimes.com/feeds/rss/lifestyle/health/rssfeed.xml',
        'https://indianexpress.com/section/lifestyle/health/feed/',
        'https://thewire.in/category/health/feed',
        'https://scroll.in/health/feed',
        'https://www.deccanherald.com/tag/health/rss',
        'https://www.mohfw.gov.in/rss.xml',               # Indian MoHFW
    ],

    'Science/Other': [
        'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
        'https://www.thehindu.com/sci-tech/science/feeder/default.rss',
        'https://www.thehindu.com/sci-tech/energy-and-environment/feeder/default.rss',
        'https://timesofindia.indiatimes.com/rssfeeds/8347648.cms',
        'https://indianexpress.com/section/technology/science/feed/',
        'https://feeds.feedburner.com/ndtvnews-environment',
        'https://india.mongabay.com/feed/',               # environment/conservation
        'https://www.sciencedaily.com/rss/all.xml',       # global science
        'https://phys.org/rss-feed/',                     # physics/science
        'https://www.isro.gov.in/rss.xml',                # ISRO space news
        'https://thewire.in/category/science/feed',
        'https://scroll.in/science/feed',
        'https://www.downtoearth.org.in/rss/news',       # environment India
    ],

    'Sports': [
        # Cricket focused (India relevance)
        'https://feeds.feedburner.com/ndtvnews-sports',
        'https://www.thehindu.com/sport/feeder/default.rss',
        'https://timesofindia.indiatimes.com/rssfeeds/4719148.cms',
        'https://indianexpress.com/section/sports/feed/',
        'https://www.hindustantimes.com/feeds/rss/cricket/rssfeed.xml',
        'https://www.cricbuzz.com/rss',                   # dedicated cricket
        # Broader sports
        'http://feeds.bbci.co.uk/sport/rss.xml',
        'http://feeds.bbci.co.uk/sport/cricket/rss.xml',
        'http://feeds.bbci.co.uk/sport/football/rss.xml',
        'https://www.espncricinfo.com/rss/content/story/feeds/0.xml',
        'https://sports.ndtv.com/rss/all',
        'https://scroll.in/field/feed',                   # sports journalism
        'https://thewire.in/category/sport/feed',
    ],

    'Entertainment': [
        # Bollywood/Indian OTT
        'https://feeds.feedburner.com/ndtvnews-entertainment',
        'https://www.thehindu.com/entertainment/feeder/default.rss',
        'https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms',
        'https://indianexpress.com/section/entertainment/feed/',
        'https://www.hindustantimes.com/feeds/rss/entertainment/rssfeed.xml',
        'https://www.bollywoodhungama.com/rss/news.xml',
        'https://www.pinkvilla.com/rss.xml',
        'https://www.filmfare.com/rss/news.xml',
        # International entertainment
        'https://variety.com/feed/',
        'https://www.cinemablend.com/rss.xml',
        'https://www.hollywoodreporter.com/feed/',
        'https://collider.com/feed/',
        'https://scroll.in/reel/feed',                    # Indian film criticism
    ],
}

# Keywords for weighted CC-News collection
HISTORICAL_HINTS = {
    'Politics':             ['parliament', 'minister', 'election', 'government',
                             'political party', 'vote', 'modi', 'congress', 'bjp',
                             'lok sabha', 'rajya sabha', 'chief minister'],
    'Technology':           ['artificial intelligence', 'machine learning', 'software',
                             'startup', 'app', 'smartphone', 'semiconductor', 'tech',
                             'cybersecurity', 'cloud computing', 'chatgpt', 'openai'],
    'Business':             ['quarterly earnings', 'stock market', 'merger', 'acquisition',
                             'ceo', 'revenue', 'profit', 'ipo', 'company results',
                             'market cap', 'nse', 'bse', 'sensex', 'nifty'],
    'Economy':              ['gdp', 'inflation', 'interest rate', 'rbi', 'fiscal deficit',
                             'budget', 'monetary policy', 'unemployment', 'trade deficit',
                             'imf', 'world bank', 'economic growth', 'recession'],
    'International_Affairs':['bilateral', 'diplomatic relations', 'foreign minister',
                             'sanctions', 'united nations', 'nato', 'treaty', 'g20',
                             'india-china', 'india-pakistan', 'geopolitics', 'embassy'],
    'Education':            ['university', 'iit', 'iim', 'school education', 'neet',
                             'jee', 'student', 'curriculum', 'teacher', 'ugc',
                             'scholarship', 'board exam', 'higher education'],
    'Health':               ['hospital', 'vaccine', 'covid', 'disease outbreak',
                             'cancer', 'who', 'aiims', 'health ministry',
                             'clinical trial', 'public health', 'mental health'],
    'Science/Other':        ['research study', 'scientists discover', 'climate change',
                             'space mission', 'isro', 'nasa', 'environment',
                             'biodiversity', 'carbon emission', 'renewable energy'],
    'Sports':               ['cricket', 'ipl', 'bcci', 'virat kohli', 'rohit sharma',
                             'world cup', 'fifa', 'olympics', 'chess',
                             'badminton', 'kabaddi', 'pro kabaddi'],
    'Entertainment':        ['bollywood', 'film release', 'box office collection',
                             'ott platform', 'netflix india', 'amazon prime',
                             'actor', 'actress', 'director', 'movie review'],
}


# ── Utility functions ──────────────────────────────────────────────────────────

def get_config() -> Config:
    config = Config()
    config.browser_user_agent = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
    config.request_timeout   = 12
    config.fetch_images       = False
    config.memoize_articles   = False
    return config


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return ''


def clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def make_article_dict(title, text, url, category, source_type) -> dict:
    domain = get_domain(url)
    return {
        'title':          title.strip(),
        'text':           text[:MAX_WORDS_STORED * 6],
        'url':            url,
        'domain':         domain,
        'word_count':     len(text.split()),
        'source_section': category,
        'source_type':    source_type,
        'is_trusted':     int(domain in TRUSTED_DOMAINS),
    }


def scrape_entry(entry, category: str, config: Config) -> dict | None:
    title = clean_html(getattr(entry, 'title', '') or '')
    url   = getattr(entry, 'link',  '') or ''

    if not url or not title:
        return None

    # Try RSS content first — avoids hitting article server
    rss_text = ''
    if hasattr(entry, 'content') and entry.content:
        rss_text = clean_html(entry.content[0].get('value', ''))
    elif hasattr(entry, 'summary'):
        rss_text = clean_html(entry.summary or '')

    if len(rss_text.split()) >= MIN_WORDS:
        return make_article_dict(title, rss_text, url, category, 'rss_content')

    # Fall back to full download
    try:
        art = Article(url, config=config)
        art.download()
        art.parse()
        text = art.text or ''
        if len(text.split()) < MIN_WORDS:
            return None
        return make_article_dict(art.title or title, text, url, category, 'full_scrape')
    except:
        return None


# ── Step 1: Compute adaptive quotas ───────────────────────────────────────────

def compute_quotas(existing_df: pd.DataFrame) -> dict[str, int]:
    """
    Given current counts, compute how many more articles to scrape per category.
    
    Uses inverse-deficit weighting:
      deficit_i = max(0, TARGET - current_count_i)
      weight_i  = deficit_i / sum(all deficits)
    
    Total new articles to scrape = sum of all deficits (capped sensibly).
    Categories already at or above TARGET get quota = 0.
    """
    current_counts = existing_df['source_section'].value_counts().to_dict()

    # Ensure all categories are present even if count is 0
    for cat in CATEGORIES:
        if cat not in current_counts:
            current_counts[cat] = 0

    deficits = {}
    for cat in CATEGORIES:
        deficit = max(0, TARGET_PER_CAT - current_counts.get(cat, 0))
        deficits[cat] = deficit

    total_deficit = sum(deficits.values())

    log.info(f"\n{'='*60}")
    log.info(f"ADAPTIVE QUOTA CALCULATION")
    log.info(f"Target per category: {TARGET_PER_CAT}")
    log.info(f"Total articles needed: {total_deficit}")
    log.info(f"{'='*60}")
    log.info(f"\n{'Category':<25} {'Current':>8} {'Deficit':>8} {'Weight':>8} {'Quota':>8}")
    log.info(f"{'─'*60}")

    quotas = {}
    for cat in CATEGORIES:
        current = current_counts.get(cat, 0)
        deficit = deficits[cat]
        weight  = deficit / total_deficit if total_deficit > 0 else 0
        # Add 20% buffer on top of exact deficit to account for scraping failures
        quota   = int(deficit * 1.20) if deficit > 0 else 0
        quotas[cat] = quota

        log.info(
            f"{cat:<25} {current:>8} {deficit:>8} "
            f"{weight:>7.1%} {quota:>8}"
        )

    log.info(f"{'─'*60}")
    log.info(f"{'TOTAL':<25} {sum(current_counts.values()):>8} "
             f"{total_deficit:>8} {'100.0%':>8} {sum(quotas.values()):>8}")

    return quotas


# ── Step 2: RSS scraping with per-category quotas ─────────────────────────────

def scrape_with_quotas(
    quotas:        dict[str, int],
    existing_urls: set,
    config:        Config,
) -> list[dict]:
    """
    Scrapes RSS feeds for each category according to its quota.
    Skips categories with quota = 0.
    Skips URLs already in the dataset.
    """
    all_new = []

    for category in sorted(quotas.keys(), key=lambda c: quotas[c], reverse=True):
        quota = quotas[category]

        if quota == 0:
            log.info(f"\n{category}: already at target, skipping")
            continue

        feeds = ALL_SOURCES.get(category, [])
        if not feeds:
            log.warning(f"\n{category}: no feeds defined, skipping")
            continue

        log.info(f"\n{'─'*55}")
        log.info(f"{category}: quota = {quota} articles")

        cat_new   = []
        feed_idx  = 0

        while len(cat_new) < quota and feed_idx < len(feeds):
            feed_url = feeds[feed_idx]
            feed_idx += 1

            log.info(f"  [{feed_idx}/{len(feeds)}] {feed_url[:65]}")

            try:
                feed    = feedparser.parse(feed_url)
                entries = feed.entries[:MAX_PER_FEED]
            except Exception as e:
                log.warning(f"  Feed parse failed: {e}")
                continue

            feed_new = 0
            for entry in entries:
                if len(cat_new) >= quota:
                    break

                url = getattr(entry, 'link', '') or ''
                if url in existing_urls:
                    continue

                art = scrape_entry(entry, category, config)
                if art is None:
                    continue

                existing_urls.add(url)
                cat_new.append(art)
                feed_new += 1
                time.sleep(CRAWL_DELAY)

            log.info(f"  → {feed_new} articles "
                     f"(category total: {len(cat_new)}/{quota})")

        all_new.extend(cat_new)

        if len(cat_new) < quota:
            log.warning(
                f"  ⚠ Only got {len(cat_new)}/{quota} for {category}. "
                f"Consider adding more feed URLs."
            )
        else:
            log.info(f"  ✓ {category} quota filled: {len(cat_new)} articles")

    return all_new


# ── Step 3: Weighted historical collection from CC-News ───────────────────────

def collect_historical_weighted(
    quotas:        dict[str, int],
    existing_urls: set,
    historical_boost: float = 0.5,
) -> list[dict]:
    """
    Pulls historical articles from CC-News with quota-weighted collection.

    historical_boost: fraction of RSS quota to also collect historically.
      0.5 = collect 50% as many historical as RSS for each category.
      Set to 1.0 to match RSS collection exactly, 0.0 to skip.

    Categories with zero quota are skipped.
    Uses streaming to avoid downloading the full dataset.
    """
    # Compute how many historical articles to target per category
    hist_quotas = {
        cat: max(0, int(quota * historical_boost))
        for cat, quota in quotas.items()
    }

    total_hist_target = sum(hist_quotas.values())
    if total_hist_target == 0:
        log.info("All categories at target — skipping historical collection.")
        return []

    log.info(f"\n{'─'*55}")
    log.info(f"CC-News historical collection")
    log.info(f"Target: {total_hist_target} additional historical articles")
    log.info("Per-category historical quotas:")
    for cat, q in sorted(hist_quotas.items(), key=lambda x: x[1], reverse=True):
        if q > 0:
            log.info(f"  {cat:<25} {q}")

    dataset  = load_dataset("cc_news", split="train", streaming=True,
                            trust_remote_code=True)
    collected = {cat: [] for cat in CATEGORIES}
    scanned   = 0

    for item in dataset:
        # Stop when all historical quotas are filled
        if all(len(collected[cat]) >= hist_quotas[cat]
               for cat in CATEGORIES if hist_quotas[cat] > 0):
            break

        scanned += 1
        if scanned % 10000 == 0:
            totals = sum(len(v) for v in collected.values())
            log.info(f"  Scanned {scanned:,} CC-News items | "
                     f"Collected {totals}/{total_hist_target}")

        text      = item.get('text', '') or ''
        title     = item.get('title', '') or ''
        url       = item.get('url', '')  or ''
        domain    = item.get('domain', '') or ''

        # Basic quality filter
        word_count = len(text.split())
        if word_count < MIN_WORDS or word_count > 8000:
            continue
        if not title or not url:
            continue
        if url in existing_urls:
            continue

        # Check language heuristic
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars / max(len(text), 1) < 0.55:
            continue

        # Match to a category using keyword hints
        # Priority order: categories with highest remaining quota first
        text_lower = (title + ' ' + text[:600]).lower()

        matched_cat = None
        # Sort categories by remaining quota (highest first) → fills weak cats faster
        cats_by_priority = sorted(
            CATEGORIES,
            key=lambda c: hist_quotas[c] - len(collected[c]),
            reverse=True
        )

        for cat in cats_by_priority:
            if hist_quotas[cat] == 0:
                continue
            if len(collected[cat]) >= hist_quotas[cat]:
                continue
            hints = HISTORICAL_HINTS.get(cat, [])
            if any(hint.lower() in text_lower for hint in hints):
                matched_cat = cat
                break

        if matched_cat is None:
            continue

        existing_urls.add(url)
        collected[matched_cat].append(make_article_dict(
            title, text, url, matched_cat, 'historical'
        ))

    # Flatten and report
    all_historical = []
    log.info(f"\nHistorical collection results (scanned {scanned:,} items):")
    for cat in CATEGORIES:
        count = len(collected[cat])
        log.info(f"  {cat:<25} {count:>4} / {hist_quotas[cat]}")
        all_historical.extend(collected[cat])

    return all_historical


# ── Step 4: Merge, deduplicate, save ──────────────────────────────────────────

def merge_and_save(
    existing_df:   pd.DataFrame,
    rss_articles:  list[dict],
    hist_articles: list[dict],
    output_path:   str,
) -> pd.DataFrame:
    """Combines old + new RSS + new historical, deduplicates, saves."""

    rss_df  = pd.DataFrame(rss_articles)  if rss_articles  else pd.DataFrame()
    hist_df = pd.DataFrame(hist_articles) if hist_articles else pd.DataFrame()

    parts = [p for p in [existing_df, rss_df, hist_df] if not p.empty]
    combined = pd.concat(parts, ignore_index=True)

    # Deduplicate by URL
    before_url = len(combined)
    combined.drop_duplicates(subset='url', inplace=True)

    # Deduplicate by normalized title
    combined['_tnorm'] = combined['title'].str.lower().str.strip()
    combined.drop_duplicates(subset='_tnorm', inplace=True)
    combined.drop(columns=['_tnorm'], inplace=True)

    combined.reset_index(drop=True, inplace=True)

    removed = before_url - len(combined)

    log.info(f"\n{'='*60}")
    log.info(f"MERGE COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Existing:           {len(existing_df):>6}")
    log.info(f"New RSS:            {len(rss_df):>6}")
    log.info(f"New historical:     {len(hist_df):>6}")
    log.info(f"Duplicates removed: {removed:>6}")
    log.info(f"Final total:        {len(combined):>6}")

    log.info(f"\nFinal per-category distribution:")
    counts = combined['source_section'].value_counts()
    max_c  = counts.max()
    for cat in CATEGORIES:
        c   = counts.get(cat, 0)
        bar = '█' * int(c / max_c * 30)
        gap = ' ⚠ BELOW TARGET' if c < TARGET_PER_CAT else ''
        log.info(f"  {cat:<25} {c:>4}  {bar}{gap}")

    log.info(f"\nSource type breakdown:")
    log.info(combined['source_type'].value_counts().to_string())

    combined.to_csv(output_path, index=False, encoding='utf-8')
    log.info(f"\nSaved → {output_path}")
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def run_adaptive_scrape(
    historical_boost: float = 0.5
):
    """
    Full pipeline:
      1. Load existing data + compute adaptive quotas
      2. Scrape RSS feeds with weighted quotas
      3. Collect historical data from CC-News with weighted quotas
      4. Merge, deduplicate, save

    Args:
        historical_boost: For each 1 article scraped from RSS,
                          also collect `historical_boost` articles from CC-News.
                          0.5 = collect half as many historical as RSS.
                          1.0 = match RSS collection exactly.
                          0.0 = skip historical collection.
    """
    config = get_config()
    start  = datetime.now()

    # ── Load existing ──────────────────────────────────────────────────────────
    if os.path.exists(EXISTING_CSV):
        existing_df = pd.read_csv(EXISTING_CSV)
        log.info(f"Loaded existing dataset: {len(existing_df)} articles")
    else:
        existing_df = pd.DataFrame(columns=[
            'title', 'text', 'url', 'domain', 'word_count',
            'source_section', 'source_type', 'is_trusted'
        ])
        log.info("No existing dataset found — starting fresh")

    existing_urls = set(existing_df['url'].dropna().tolist())

    # ── Compute quotas ─────────────────────────────────────────────────────────
    quotas = compute_quotas(existing_df)

    if sum(quotas.values()) == 0:
        log.info("\nAll categories already at target. Nothing to scrape.")
        return existing_df

    # ── RSS scraping ───────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("PHASE A: RSS scraping (current articles)")
    log.info(f"{'='*60}")
    rss_articles = scrape_with_quotas(quotas, existing_urls, config)
    log.info(f"\nRSS phase complete: {len(rss_articles)} new articles")

    # ── Historical collection ──────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("PHASE B: Historical collection (CC-News)")
    log.info(f"{'='*60}")
    hist_articles = collect_historical_weighted(
        quotas, existing_urls, historical_boost
    )
    log.info(f"\nHistorical phase complete: {len(hist_articles)} new articles")

    # ── Merge and save ─────────────────────────────────────────────────────────
    final_df = merge_and_save(existing_df, rss_articles, hist_articles, OUTPUT_CSV)

    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    return final_df


if __name__ == '__main__':
    # historical_boost=0.5 means for every 2 RSS articles collected,
    # also pull 1 from CC-News history.
    # Increase to 1.0 for a 50/50 current/historical split.
    run_adaptive_scrape(historical_boost=0.5)