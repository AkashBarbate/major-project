"""
boost_historical.py
-------------------
Specifically increases historical data from CC-News to reach
~550 articles per category (total ~5500).

Also adds two additional historical datasets:
  - AG News (topic-filtered, maps to your categories)
  - MIND (Microsoft News Dataset — high quality, pre-categorized)

Run after adaptive_scrape.py.

Requirements:
    pip install datasets pandas
"""

import pandas as pd
import numpy as np
import logging
import os
import re
from datetime import datetime
from datasets import load_dataset

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/boost_historical_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
EXISTING_CSV     = 'raw_scraped_articles.csv'
OUTPUT_CSV       = 'raw_scraped_articles.csv'
TARGET_PER_CAT   = 550     # final target per category → ~5500 total
MIN_WORDS        = 80
MAX_WORDS_STORED = 2500

CATEGORIES = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health',
    'Science/Other', 'Sports', 'Entertainment'
]

# ── Keyword hints for CC-News matching ────────────────────────────────────────
# More specific than before to reduce mis-categorization
HISTORICAL_HINTS = {
    'Politics': [
        'parliament', 'minister', 'election', 'government policy',
        'political party', 'lok sabha', 'rajya sabha', 'chief minister',
        'legislative', 'senator', 'congress party', 'bjp',
        'white house', 'prime minister', 'cabinet', 'opposition party',
        'referendum', 'campaign rally', 'political crisis',
    ],
    'Technology': [
        'artificial intelligence', 'machine learning', 'deep learning',
        'software update', 'tech startup', 'semiconductor chip',
        'cybersecurity breach', 'cloud platform', 'smartphone launch',
        'open source', 'developer tools', 'data center',
        'robotics', 'autonomous vehicle', 'neural network',
        'generative ai', 'large language model',
    ],
    'Business': [
        'quarterly earnings', 'revenue growth', 'profit margin',
        'merger acquisition', 'ipo listing', 'ceo appointed',
        'market share', 'supply chain', 'retail sales',
        'sensex nifty', 'stock rally', 'corporate results',
        'trade deal', 'business expansion', 'layoffs announced',
    ],
    'Economy': [
        'gdp growth', 'inflation rate', 'interest rate hike',
        'rbi policy', 'fiscal deficit', 'trade balance',
        'unemployment rate', 'monetary policy', 'consumer price',
        'economic slowdown', 'imf forecast', 'world bank report',
        'foreign exchange', 'current account deficit', 'tax revenue',
        'budget allocation', 'economic reform',
    ],
    'International_Affairs': [
        'bilateral talks', 'diplomatic relations', 'foreign minister',
        'economic sanctions', 'united nations resolution',
        'peace treaty', 'military alliance', 'g20 summit',
        'india china border', 'india pakistan', 'geopolitical',
        'nato expansion', 'un security council', 'foreign policy',
        'ambassador recalled', 'international summit', 'trade war',
    ],
    'Education': [
        'university admission', 'iit jee', 'neet exam',
        'school curriculum', 'ugc guidelines', 'board examination',
        'scholarship program', 'higher education reform',
        'digital classroom', 'student protest', 'faculty research',
        'education ministry', 'skill development', 'vocational training',
        'campus placement', 'academic research', 'phd fellowship',
    ],
    'Health': [
        'disease outbreak', 'vaccine approved', 'clinical trial results',
        'hospital capacity', 'public health emergency',
        'cancer treatment', 'mental health awareness',
        'health ministry', 'aiims hospital', 'who guidelines',
        'pandemic response', 'drug approval', 'healthcare policy',
        'medical research', 'epidemic control', 'nutrition study',
    ],
    'Science/Other': [
        'scientific discovery', 'research findings', 'climate change report',
        'space mission launch', 'isro satellite', 'nasa announcement',
        'environmental study', 'biodiversity loss', 'carbon emissions',
        'renewable energy project', 'ocean temperature',
        'archaeological discovery', 'genome sequencing',
        'physics breakthrough', 'chemistry research',
    ],
    'Sports': [
        'cricket match result', 'ipl auction', 'bcci announcement',
        'world cup qualifier', 'olympic preparation',
        'football transfer', 'tennis tournament',
        'chess championship', 'badminton open',
        'sports federation', 'athlete performance', 'pro kabaddi',
        'commonwealth games', 'asian games', 'national championship',
    ],
    'Entertainment': [
        'bollywood release', 'box office collection',
        'ott platform launch', 'netflix original', 'amazon prime',
        'film award', 'actor interview', 'director announcement',
        'music album', 'celebrity wedding', 'streaming debut',
        'film festival', 'television series', 'reality show',
        'movie review', 'entertainment industry',
    ],
}

# ── AG News category mapping ───────────────────────────────────────────────────
# AG News has 4 labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
AG_NEWS_MAP = {
    0: 'International_Affairs',  # World news
    1: 'Sports',
    2: 'Business',
    3: 'Technology',             # Sci/Tech → Technology
}
# AG News only covers these 4 — use it as supplementary for these categories only

# ── MIND dataset category mapping ─────────────────────────────────────────────
# MIND categories → your categories
MIND_CATEGORY_MAP = {
    'politics':     'Politics',
    'politicsNews': 'Politics',
    'news':         'International_Affairs',
    'newsworld':    'International_Affairs',
    'newsus':       'Politics',
    'sports':       'Sports',
    'sportsBasketball': 'Sports',
    'sportsSoccer': 'Sports',
    'sportsGolf':   'Sports',
    'sportsFootball':'Sports',
    'sportsMma':    'Sports',
    'sportsNfl':    'Sports',
    'sportsNba':    'Sports',
    'sportsNhl':    'Sports',
    'sportsMlb':    'Sports',
    'sportsTennis': 'Sports',
    'sportsCfb':    'Sports',
    'sportsCbb':    'Sports',
    'finance':      'Economy',
    'financenews':  'Economy',
    'financestock': 'Business',
    'entertainment':'Entertainment',
    'entertainmentmusic':    'Entertainment',
    'entertainmentmovieandtv':'Entertainment',
    'entertainmentcelebrity':'Entertainment',
    'health':       'Health',
    'healthNews':   'Health',
    'lifestyle':    'Entertainment',
    'lifestyleFood':'Entertainment',
    'travel':       'International_Affairs',
    'autos':        'Technology',
    'video':        'Entertainment',
    'tv':           'Entertainment',
    'music':        'Entertainment',
    'kids':         'Education',
    'weather':      'Science/Other',
    'science':      'Science/Other',
    'foodanddrink': 'Entertainment',
    'education':    'Education',
    'middleeast':   'International_Affairs',
    'northamerica': 'International_Affairs',
}


# ── Utility ────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def make_article(title, text, url, category, source_type) -> dict:
    return {
        'title':          str(title).strip()[:300],
        'text':           str(text)[:MAX_WORDS_STORED * 6],
        'url':            str(url),
        'domain':         url.split('/')[2].replace('www.', '') if '/' in url else '',
        'word_count':     len(str(text).split()),
        'source_section': category,
        'source_type':    source_type,
        'is_trusted':     1,
    }


def compute_deficits(existing_df: pd.DataFrame) -> dict[str, int]:
    """How many more articles each category needs to reach TARGET_PER_CAT."""
    counts = existing_df['source_section'].value_counts().to_dict()
    deficits = {}

    log.info(f"\nCurrent counts vs target ({TARGET_PER_CAT}):")
    for cat in CATEGORIES:
        current = counts.get(cat, 0)
        deficit = max(0, TARGET_PER_CAT - current)
        deficits[cat] = deficit
        status = '✓' if deficit == 0 else f'need {deficit} more'
        log.info(f"  {cat:<25} {current:>4} → {status}")

    return deficits


# ── Source A: CC-News (main historical source) ─────────────────────────────────

def collect_from_ccnews(
    deficits:      dict[str, int],
    existing_urls: set,
    boost_factor:  float = 1.4,
) -> list[dict]:
    """
    Streams CC-News and collects articles weighted by deficit.

    boost_factor: collect this multiple of deficit to account for
                  keyword matching failures. 1.4 = collect 40% more
                  than needed then trim to exact quota.
    """
    # Quotas with boost buffer
    quotas = {
        cat: int(deficit * boost_factor)
        for cat, deficit in deficits.items()
        if deficit > 0
    }

    total_target = sum(quotas.values())
    if total_target == 0:
        log.info("CC-News: all categories at target, skipping.")
        return []

    log.info(f"\nCC-News streaming — target {total_target} articles")
    log.info("Per-category quotas (with 40% boost buffer):")
    for cat, q in sorted(quotas.items(), key=lambda x: x[1], reverse=True):
        log.info(f"  {cat:<25} {q}")

    dataset   = load_dataset("cc_news", split="train",
                              streaming=True, trust_remote_code=True)
    collected = {cat: [] for cat in CATEGORIES}
    scanned   = 0

    for item in dataset:
        # Check if all quotas filled
        all_done = all(
            len(collected[cat]) >= quotas.get(cat, 0)
            for cat in CATEGORIES
            if quotas.get(cat, 0) > 0
        )
        if all_done:
            break

        scanned += 1
        if scanned % 20000 == 0:
            so_far = sum(len(v) for v in collected.values())
            log.info(f"  CC-News: scanned {scanned:,} | collected {so_far}/{total_target}")

        text  = clean_text(item.get('text', '') or '')
        title = clean_text(item.get('title', '') or '')
        url   = item.get('url', '') or ''

        # Quality filters
        words = len(text.split())
        if words < MIN_WORDS or words > 10000:
            continue
        if not title or not url or url in existing_urls:
            continue

        # Language heuristic
        if sum(c.isalpha() for c in text) / max(len(text), 1) < 0.55:
            continue

        text_lower = (title + ' ' + text[:800]).lower()

        # Find best matching category — prioritise highest deficit
        matched_cat  = None
        best_deficit = -1

        for cat in CATEGORIES:
            q = quotas.get(cat, 0)
            if q == 0 or len(collected[cat]) >= q:
                continue
            remaining = q - len(collected[cat])
            if remaining <= best_deficit:
                continue
            hints = HISTORICAL_HINTS.get(cat, [])
            if any(h.lower() in text_lower for h in hints):
                matched_cat  = cat
                best_deficit = remaining

        if matched_cat is None:
            continue

        existing_urls.add(url)
        collected[matched_cat].append(
            make_article(title, text, url, matched_cat, 'historical_ccnews')
        )

    # Trim to exact deficit (remove the boost buffer excess)
    result = []
    log.info(f"\nCC-News results (trimmed to exact deficits):")
    for cat in CATEGORIES:
        exact_need = deficits.get(cat, 0)
        trimmed    = collected[cat][:exact_need]
        log.info(f"  {cat:<25} collected {len(collected[cat]):>4} → using {len(trimmed):>4}")
        result.extend(trimmed)

    log.info(f"CC-News total: {len(result)} articles (scanned {scanned:,} items)")
    return result


# ── Source B: AG News ──────────────────────────────────────────────────────────

def collect_from_agnews(
    deficits:      dict[str, int],
    existing_urls: set,
) -> list[dict]:
    """
    AG News: 120,000 categorized news snippets.
    Only covers World/Sports/Business/Sci-Tech → maps to 4 of your categories.
    Good for supplementing Technology, Sports, Business, International_Affairs.
    """
    # Only categories AG News can help with
    relevant_cats = {'International_Affairs', 'Sports', 'Business', 'Technology'}
    relevant_deficits = {
        cat: deficits[cat] for cat in relevant_cats
        if deficits.get(cat, 0) > 0
    }

    if not relevant_deficits:
        log.info("AG News: relevant categories already at target, skipping.")
        return []

    log.info(f"\nAG News — supplementing: {list(relevant_deficits.keys())}")

    try:
        dataset = load_dataset("ag_news", split="train")
    except Exception as e:
        log.warning(f"AG News failed to load: {e}")
        return []

    # Allocate quota from AG News: up to 30% of each category's deficit
    ag_quotas = {
        cat: min(int(deficit * 0.30), 80)   # max 80 per category from AG News
        for cat, deficit in relevant_deficits.items()
    }

    collected  = {cat: [] for cat in CATEGORIES}
    seen_texts = set()  # AG News has no URLs, deduplicate by text hash

    import hashlib

    for item in dataset:
        if all(len(collected[cat]) >= ag_quotas.get(cat, 0)
               for cat in relevant_cats):
            break

        label    = item.get('label', -1)
        category = AG_NEWS_MAP.get(label)
        if category is None:
            continue
        if ag_quotas.get(category, 0) == 0:
            continue
        if len(collected[category]) >= ag_quotas[category]:
            continue

        title   = clean_text(item.get('title', '') or '')
        text    = clean_text(item.get('description', '') or '')
        full    = (title + ' ' + text).strip()

        if len(full.split()) < 20:    # AG News snippets are short — lower threshold
            continue

        text_hash = hashlib.md5(full.encode()).hexdigest()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)

        # Generate synthetic URL for deduplication compatibility
        url = f"agnews://item/{text_hash}"
        if url in existing_urls:
            continue
        existing_urls.add(url)

        collected[category].append(
            make_article(title, full, url, category, 'historical_agnews')
        )

    result = []
    log.info("AG News results:")
    for cat in relevant_cats:
        arts = collected[cat]
        log.info(f"  {cat:<25} {len(arts):>4}")
        result.extend(arts)

    return result


# ── Source C: MIND (Microsoft News Dataset) ───────────────────────────────────

def collect_from_mind(
    deficits:      dict[str, int],
    existing_urls: set,
) -> list[dict]:
    """
    MIND small: ~65,000 news articles with categories, titles, abstracts.
    High quality editorial content from MSN News.
    Maps ~18 MIND categories to your 10.
    """
    if not any(d > 0 for d in deficits.values()):
        log.info("MIND: all categories at target, skipping.")
        return []

    log.info(f"\nMIND (Microsoft News) — collecting supplementary articles")

    try:
        # MIND small is ~300MB, manageable
        dataset = load_dataset(
            "mind", "small",
            split="train",
            trust_remote_code=True
        )
    except Exception as e:
        log.warning(f"MIND dataset failed: {e}. Skipping.")
        log.warning("If this is a download issue, try: "
                    "pip install datasets --upgrade")
        return []

    # Allocate up to 20% of deficit from MIND per category
    mind_quotas = {
        cat: min(int(deficit * 0.20), 60)
        for cat, deficit in deficits.items()
        if deficit > 0
    }

    collected  = {cat: [] for cat in CATEGORIES}
    import hashlib

    for item in dataset:
        if all(len(collected[cat]) >= mind_quotas.get(cat, 0)
               for cat in CATEGORIES if mind_quotas.get(cat, 0) > 0):
            break

        # MIND structure: news_id, category, subcategory, title, abstract, url, ...
        raw_cat  = (item.get('category', '') or '').lower().replace(' ', '')
        category = MIND_CATEGORY_MAP.get(raw_cat)
        if category is None:
            continue
        if mind_quotas.get(category, 0) == 0:
            continue
        if len(collected[category]) >= mind_quotas[category]:
            continue

        title    = clean_text(item.get('title', '')    or '')
        abstract = clean_text(item.get('abstract', '') or '')
        url      = item.get('url', '') or ''

        full_text = (title + '. ' + abstract).strip()
        if len(full_text.split()) < 15:
            continue

        # Use URL if available, else hash-based synthetic URL
        if not url:
            h   = hashlib.md5(full_text.encode()).hexdigest()
            url = f"mind://item/{h}"

        if url in existing_urls:
            continue
        existing_urls.add(url)

        collected[category].append(
            make_article(title, full_text, url, category, 'historical_mind')
        )

    result = []
    log.info("MIND results:")
    for cat in CATEGORIES:
        arts = collected[cat]
        if arts:
            log.info(f"  {cat:<25} {len(arts):>4}")
        result.extend(arts)

    return result


# ── Merge and save ─────────────────────────────────────────────────────────────

def merge_and_save(
    existing_df:   pd.DataFrame,
    new_batches:   list[list[dict]],
    output_path:   str,
) -> pd.DataFrame:

    all_new = []
    for batch in new_batches:
        all_new.extend(batch)

    if not all_new:
        log.info("No new articles collected.")
        return existing_df

    new_df   = pd.DataFrame(all_new)
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    # Deduplicate by URL
    before = len(combined)
    combined.drop_duplicates(subset='url', inplace=True)

    # Deduplicate by title
    combined['_tnorm'] = combined['title'].str.lower().str.strip()
    combined.drop_duplicates(subset='_tnorm', inplace=True)
    combined.drop(columns=['_tnorm'], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    removed = before - len(combined)

    log.info(f"\n{'='*60}")
    log.info(f"FINAL DATASET")
    log.info(f"{'='*60}")
    log.info(f"Previous:           {len(existing_df):>6}")
    log.info(f"New articles added: {len(all_new):>6}")
    log.info(f"Duplicates removed: {removed:>6}")
    log.info(f"Final total:        {len(combined):>6}")

    log.info(f"\nPer-category distribution:")
    counts  = combined['source_section'].value_counts()
    max_cnt = counts.max()
    for cat in CATEGORIES:
        c    = counts.get(cat, 0)
        bar  = '█' * int(c / max_cnt * 35)
        flag = '' if c >= TARGET_PER_CAT else f'  ⚠ {TARGET_PER_CAT - c} short'
        log.info(f"  {cat:<25} {c:>5}  {bar}{flag}")

    log.info(f"\nSource type breakdown:")
    log.info(combined['source_type'].value_counts().to_string())

    log.info(f"\nImbalance ratio (max/min): "
             f"{counts.max()/counts.min():.2f}x  "
             f"(ideal: <1.5x)")

    combined.to_csv(output_path, index=False, encoding='utf-8')
    log.info(f"\nSaved → {output_path}")
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def run_boost():
    start = datetime.now()

    # Load existing
    existing_df   = pd.read_csv(EXISTING_CSV)
    existing_urls = set(existing_df['url'].dropna().tolist())
    log.info(f"Existing dataset: {len(existing_df)} articles")

    # Compute what's needed
    deficits = compute_deficits(existing_df)
    total_needed = sum(deficits.values())

    if total_needed == 0:
        log.info(f"\nAll categories already at {TARGET_PER_CAT}. Nothing to do.")
        return existing_df

    log.info(f"\nTotal articles needed: {total_needed}")
    log.info(f"Expected final total:  ~{len(existing_df) + total_needed}")

    # Collect from all three historical sources
    # CC-News handles the bulk
    # AG News and MIND supplement specific weak categories

    log.info(f"\n{'='*60}")
    log.info("SOURCE A: CC-News (primary)")
    log.info(f"{'='*60}")
    ccnews_articles = collect_from_ccnews(deficits, existing_urls)

    # Recompute deficits after CC-News
    # (CC-News fills most of the gap)
    temp_df      = pd.concat(
        [existing_df, pd.DataFrame(ccnews_articles)], ignore_index=True
    ) if ccnews_articles else existing_df
    deficits_2   = compute_deficits(temp_df)
    existing_urls.update(a['url'] for a in ccnews_articles)

    log.info(f"\n{'='*60}")
    log.info("SOURCE B: AG News (supplementary — 4 categories)")
    log.info(f"{'='*60}")
    agnews_articles = collect_from_agnews(deficits_2, existing_urls)
    existing_urls.update(a['url'] for a in agnews_articles)

    temp_df2     = pd.concat(
        [temp_df, pd.DataFrame(agnews_articles)], ignore_index=True
    ) if agnews_articles else temp_df
    deficits_3   = compute_deficits(temp_df2)

    log.info(f"\n{'='*60}")
    log.info("SOURCE C: MIND (Microsoft News — supplementary)")
    log.info(f"{'='*60}")
    mind_articles = collect_from_mind(deficits_3, existing_urls)

    # Merge everything
    final_df = merge_and_save(
        existing_df,
        [ccnews_articles, agnews_articles, mind_articles],
        OUTPUT_CSV
    )

    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    return final_df


if __name__ == '__main__':
    run_boost()