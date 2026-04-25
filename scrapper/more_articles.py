"""
targeted_topup.py
-----------------
Fills only the 5 weakest categories to reach 320 each.
Reads cleaned_articles.csv, appends new articles, saves back.

Run: python targeted_topup.py
"""

import feedparser
import pandas as pd
import time
import re
import os
import logging
import sys
import io
from datetime import datetime
from newspaper import Article, Config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('topup_log.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_CSV      = 'cleaned_articles.csv'   # output from preprocess_dataset.py
OUTPUT_CSV     = 'cleaned_articles.csv'   # overwrite in place
TARGET_PER_CAT = 320
CRAWL_DELAY    = 0.4
MAX_PER_FEED   = 100
MIN_WORDS      = 150

# Only top up these 5 — others are already fine
TOPUP_CATEGORIES = [
    'International_Affairs',
    'Entertainment',
    'Politics',
    'Sports',
    'Education',
]

# ── Fresh feeds specifically chosen for weak categories ───────────────────────
# Different from what was used before to get new articles
TOPUP_FEEDS = {

    'International_Affairs': [
        # Wire services — most reliable for international coverage
        'https://feeds.reuters.com/reuters/worldNews',
        'https://feeds.reuters.com/reuters/INtopNews',
        'https://www.aljazeera.com/xml/rss/all.xml',
        'https://foreignpolicy.com/feed/',
        'https://www.bbc.co.uk/news/world/rss.xml',
        # Indian international coverage
        'https://theprint.in/diplomacy/feed/',
        'https://theprint.in/world/feed/',
        'https://thewire.in/category/world/feed',
        'https://scroll.in/world/feed',
        'https://www.thehindu.com/news/international/feeder/default.rss',
        # Specific India-world topics
        'https://www.thehindu.com/news/national/feeder/default.rss',
        'https://indianexpress.com/section/world/feed/',
    ],

    'Entertainment': [
        # Bollywood — strong Indian relevance
        'https://www.filmfare.com/rss/news.xml',
        'https://www.bollywoodhungama.com/rss/news.xml',
        'https://www.pinkvilla.com/rss.xml',
        'https://www.bollywoodshaadis.com/rss/news.xml',
        # OTT/streaming — very current
        'https://www.gadgets360.com/entertainment/rss',
        'https://indianexpress.com/section/entertainment/bollywood/feed/',
        'https://indianexpress.com/section/entertainment/hollywood/feed/',
        # International entertainment
        'https://variety.com/v/film/feed/',
        'https://www.hollywoodreporter.com/c/movies/feed/',
        'https://collider.com/feed/',
        'https://www.cinemablend.com/rss.xml',
        'https://deadline.com/feed/',
    ],

    'Politics': [
        # State politics — India has 28 states, rich coverage
        'https://theprint.in/politics/feed/',
        'https://theprint.in/india/feed/',
        'https://thewire.in/category/politics/feed',
        'https://scroll.in/politics/feed',
        # Opposition/parliament coverage
        'https://www.thehindu.com/news/national/feeder/default.rss',
        'https://www.thehindu.com/opinion/editorial/feeder/default.rss',
        # State-level politics
        'https://www.deccanherald.com/india/rss',
        'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
        'https://indianexpress.com/section/political-pulse/feed/',
        # Regional English news
        'https://www.newindianexpress.com/nation/rssfeed/?id=1&getXmlFeed=true',
        'https://www.business-standard.com/rss/politics-10304.rss',
    ],

    'Sports': [
        # Cricket — dominant in India
        'https://www.cricbuzz.com/rss',
        'https://www.espncricinfo.com/rss/content/story/feeds/0.xml',
        'https://www.cricketworld.com/rss.xml',
        # Multi-sport Indian coverage
        'https://scroll.in/field/feed',
        'https://thewire.in/category/sport/feed',
        'https://timesofindia.indiatimes.com/rssfeeds/4719148.cms',
        'https://www.thehindu.com/sport/cricket/feeder/default.rss',
        'https://www.thehindu.com/sport/other-sports/feeder/default.rss',
        # International sports
        'https://www.bbc.co.uk/sport/cricket/rss.xml',
        'https://www.bbc.co.uk/sport/football/rss.xml',
        'https://www.bbc.co.uk/sport/tennis/rss.xml',
        # Kabaddi/Hockey/other Indian sports
        'https://sports.ndtv.com/rss/all',
    ],

    'Education': [
        # Exam results/boards — very specific Indian education content
        'https://www.careers360.com/rss/news',
        'https://www.jagranjosh.com/rss/education-news.xml',
        'https://school.careers360.com/rss/news',
        'https://university.careers360.com/rss/news',
        # Policy/higher education
        'https://www.universityworldnews.com/rss.php',
        'https://thewire.in/category/education/feed',
        'https://theprint.in/campus-voice/feed/',
        'https://scroll.in/education/feed',
        # Skill development / vocational
        'https://www.hindustantimes.com/feeds/rss/education/rssfeed.xml',
        'https://www.thehindu.com/education/colleges/feeder/default.rss',
        'https://www.thehindu.com/education/schools/feeder/default.rss',
        'https://indianexpress.com/section/education/feed/',
    ],
}

TRUSTED_DOMAINS = {
    'bbc.com', 'bbc.co.uk', 'reuters.com', 'thehindu.com',
    'ndtv.com', 'indianexpress.com', 'theprint.in', 'scroll.in',
    'thewire.in', 'cricbuzz.com', 'espncricinfo.com', 'aljazeera.com',
    'filmfare.com', 'careers360.com', 'universityworldnews.com',
    'hollywoodreporter.com', 'variety.com', 'foreignpolicy.com',
    'deccanherald.com', 'hindustantimes.com', 'timesofindia.indiatimes.com',
}


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace('www.', '')
    except:
        return ''


def scrape_entry(entry, category: str, config: Config) -> dict | None:
    """Try RSS content first, fall back to full download."""
    title = clean_html(getattr(entry, 'title', '') or '')
    url   = getattr(entry, 'link',  '') or ''

    if not url or not title:
        return None

    # Try RSS body first — faster, no server hit
    rss_text = ''
    if hasattr(entry, 'content') and entry.content:
        rss_text = clean_html(entry.content[0].get('value', ''))
    elif hasattr(entry, 'summary'):
        rss_text = clean_html(entry.summary or '')

    if len(rss_text.split()) >= MIN_WORDS:
        domain = get_domain(url)
        return {
            'title':          title,
            'text':           rss_text[:15000],
            'url':            url,
            'domain':         domain,
            'word_count':     len(rss_text.split()),
            'source_section': category,
            'source_type':    'rss_content',
            'is_trusted':     int(domain in TRUSTED_DOMAINS),
        }

    # Fall back to full article download
    try:
        art = Article(url, config=config)
        art.download()
        art.parse()
        text   = art.text or ''
        domain = get_domain(url)
        if len(text.split()) < MIN_WORDS:
            return None
        return {
            'title':          art.title or title,
            'text':           text[:15000],
            'url':            url,
            'domain':         domain,
            'word_count':     len(text.split()),
            'source_section': category,
            'source_type':    'full_scrape',
            'is_trusted':     int(domain in TRUSTED_DOMAINS),
        }
    except:
        return None


def apply_cleaning(title: str, text: str) -> tuple[str, str, int]:
    """
    Apply same cleaning as preprocess_dataset.py so new articles
    match the quality of existing ones.
    """
    # Strip HTML
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]{1,100}>', ' ', text)

    # HTML entities
    for old, new in [('&amp;','&'),('&lt;','<'),('&gt;','>'),
                     ('&nbsp;',' '),('&quot;','"')]:
        text = text.replace(old, new)

    # Boilerplate truncation
    lines = text.split('\n')
    clean_lines = []
    triggers = ['also read', 'read more', 'follow us on',
                'subscribe to', 'advertisement', 'share this']
    for line in lines:
        if any(t in line.lower() for t in triggers):
            break
        # Remove URLs from line
        line = re.sub(r'https?://\S+', ' ', line)
        clean_lines.append(line)
    text = ' '.join(clean_lines)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Truncate to 2000 words
    words = text.split()
    if len(words) > 2000:
        text = ' '.join(words[:2000])

    # Clean title
    title = re.sub(r'\s*[\|\-]\s*(bbc|reuters|ndtv|the hindu).*$',
                   '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+', ' ', title).strip()

    return title, text, len(text.split())


# ── Main top-up ────────────────────────────────────────────────────────────────

def run_targeted_topup():
    start = datetime.now()

    # Load existing cleaned dataset
    df = pd.read_csv(INPUT_CSV)
    log.info(f"Loaded: {len(df)} existing articles")

    existing_urls = set(df['url'].dropna().tolist())
    config        = get_config()

    # Compute exact needs per category
    current_counts = df['source_section'].value_counts().to_dict()
    needs = {}
    for cat in TOPUP_CATEGORIES:
        current = current_counts.get(cat, 0)
        needed  = max(0, TARGET_PER_CAT - current)
        needs[cat] = needed

    log.info(f"\nArticles needed per category:")
    for cat, n in sorted(needs.items(), key=lambda x: x[1], reverse=True):
        current = current_counts.get(cat, 0)
        log.info(f"  {cat:<25} {current:>4} -> need {n} more")

    total_needed = sum(needs.values())
    log.info(f"\nTotal to collect: {total_needed}")

    if total_needed == 0:
        log.info("All categories already at target. Nothing to do.")
        return df

    # Scrape per category
    all_new = []

    for category in sorted(needs.keys(), key=lambda c: needs[c], reverse=True):
        needed = needs[category]
        if needed == 0:
            continue

        # Add 25% buffer to account for scraping failures + duplicates
        quota  = int(needed * 1.25)
        feeds  = TOPUP_FEEDS.get(category, [])

        log.info(f"\n{'='*50}")
        log.info(f"{category}: collecting {quota} (need {needed}, +25% buffer)")

        cat_new  = []
        feed_idx = 0

        while len(cat_new) < quota and feed_idx < len(feeds):
            feed_url = feeds[feed_idx]
            feed_idx += 1

            log.info(f"  [{feed_idx}/{len(feeds)}] {feed_url[:65]}")

            try:
                feed    = feedparser.parse(feed_url)
                entries = feed.entries[:MAX_PER_FEED]
            except Exception as e:
                log.warning(f"  Feed failed: {e}")
                continue

            feed_new = 0
            for entry in entries:
                if len(cat_new) >= quota:
                    break

                url = getattr(entry, 'link', '') or ''
                if not url or url in existing_urls:
                    continue

                art = scrape_entry(entry, category, config)
                if art is None:
                    continue

                # Apply same cleaning pipeline
                title, text, wc = apply_cleaning(art['title'], art['text'])
                if wc < MIN_WORDS:
                    continue

                art['title']      = title
                art['text']       = text
                art['word_count'] = wc

                existing_urls.add(url)
                cat_new.append(art)
                feed_new += 1
                time.sleep(CRAWL_DELAY)

            log.info(f"  -> {feed_new} articles collected "
                     f"(category total: {len(cat_new)}/{quota})")

        # Trim to exact need (remove buffer excess)
        cat_new = cat_new[:needed]
        log.info(f"  Final: {len(cat_new)} new articles for {category}")

        if len(cat_new) < needed:
            log.warning(
                f"  Only got {len(cat_new)}/{needed}. "
                f"Add more feeds to TOPUP_FEEDS['{category}'] and re-run."
            )

        all_new.extend(cat_new)

    # ── Merge and save ─────────────────────────────────────────────────────────
    if not all_new:
        log.info("No new articles collected.")
        return df

    new_df   = pd.DataFrame(all_new)
    combined = pd.concat([df, new_df], ignore_index=True)

    # Final dedup
    combined.drop_duplicates(subset='url', inplace=True)
    combined['_tnorm'] = combined['title'].str.lower().str.strip()
    combined.drop_duplicates(subset='_tnorm', inplace=True)
    combined.drop(columns=['_tnorm'], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    # ── Final report ───────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    counts  = combined['source_section'].value_counts()

    log.info(f"\n{'='*50}")
    log.info(f"TOP-UP COMPLETE")
    log.info(f"{'='*50}")
    log.info(f"Previous total:  {len(df)}")
    log.info(f"New articles:    {len(all_new)}")
    log.info(f"Final total:     {len(combined)}")
    log.info(f"Time:            {elapsed/60:.1f} min")

    log.info(f"\nFinal distribution:")
    max_c = counts.max()
    for cat in sorted(combined['source_section'].unique()):
        c    = counts.get(cat, 0)
        bar  = 'X' * int(c / max_c * 30)
        flag = ' <- STILL LOW' if c < 280 else ''
        log.info(f"  {cat:<25} {c:>4}  {bar}{flag}")

    log.info(f"\nImbalance ratio: "
             f"{counts.max()/counts.min():.2f}x")

    log.info(f"\nSaved -> {OUTPUT_CSV}")
    return combined


if __name__ == '__main__':
    df = run_targeted_topup()

    # Quick verification
    print(f"\nFinal counts:\n{df['source_section'].value_counts().to_string()}")
    print(f"\nSource types:\n{df['source_type'].value_counts().to_string()}")
    print(f"\nImbalance ratio: "
          f"{df['source_section'].value_counts().max() / df['source_section'].value_counts().min():.2f}x")