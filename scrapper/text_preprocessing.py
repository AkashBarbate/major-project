"""
preprocess_dataset.py
---------------------
Cleans raw_scraped_articles.csv before LLM labeling.
Fixes all identified issues:
  1. Taiwan News source capping (32.5% → max 200 articles)
  2. Short article removal (<150 words)
  3. HTML tag stripping
  4. Boilerplate removal (also read, advertisement, follow us, etc.)
  5. URL removal from body text
  6. Encoding artifact cleanup
  7. Near-duplicate removal
  8. Live blog / highlights truncation
  9. Text normalization for LLM input

Input:  raw_scraped_articles.csv
Output: cleaned_articles.csv  (ready for label_with_llm.py)
        preprocessing_report.txt

Run: python preprocess_dataset.py
"""

import pandas as pd
import numpy as np
import re
import os
import logging
import hashlib
from datetime import datetime

import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# File handler — always utf-8
fh = logging.FileHandler('preprocessing_report.txt', encoding='utf-8')
fh.setFormatter(fmt)
logger.addHandler(fh)

# Stream handler — force utf-8 on Windows
sh = logging.StreamHandler(stream=open(sys.stdout.fileno(), 
                                        mode='w', 
                                        encoding='utf-8', 
                                        closefd=False))
sh.setFormatter(fmt)
logger.addHandler(sh)

log = logger

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_CSV  = 'raw_scraped_articles.csv'
OUTPUT_CSV = 'cleaned_articles.csv'

MIN_WORDS            = 150   # drop articles below this
MAX_WORDS            = 2000  # truncate articles above this (live blogs etc.)
MAX_PER_DOMAIN       = 200   # cap any single domain at this count
TAIWAN_NEWS_CAP      = 200   # specific cap for the dominant source

CATEGORIES = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health',
    'Science/Other', 'Sports', 'Entertainment'
]

# ── Boilerplate patterns to strip from article body ───────────────────────────
# These are matched line-by-line and the line + everything after is removed
# (for navigation/pagination boilerplate that appears at the end)
END_BOILERPLATE_TRIGGERS = [
    r'^also read[:\-]',
    r'^read more[:\-]',
    r'^related[:\-\s]',
    r'^you may also like',
    r'^more from this section',
    r'^follow us on',
    r'^subscribe to',
    r'^sign up for',
    r'^newsletter signup',
    r'^get the latest',
    r'^download the app',
    r'^share this story',
    r'^share this article',
    r'^print this article',
    r'^advertisement$',
    r'^sponsored content',
    r'^tags:',
    r'^topics:',
    r'^keywords:',
    r'^filed under:',
]

# Inline patterns to remove from any line (not truncate at)
INLINE_REMOVE_PATTERNS = [
    r'https?://\S+',                          # URLs
    r'www\.\S+\.\S+',                         # www URLs without http
    r'\[object object\]',                      # JS rendering artifacts
    r'javascript:void\(0\)',
    r'<!--.*?-->',                             # HTML comments
    r'<script[^>]*>.*?</script>',             # script tags
    r'<style[^>]*>.*?</style>',              # style tags
    r'<[^>]{1,60}>',                          # HTML tags (short ones only, safe)
    r'&amp;',                                  # HTML entities
    r'&lt;', r'&gt;', r'&quot;', r'&nbsp;',
    r'&#\d{1,5};',                            # numeric HTML entities
    r'\{[^}]{0,50}\}',                        # template artifacts like {variable}
]

# Full-line removal (lines that are purely navigation/metadata)
REMOVE_FULL_LINE_PATTERNS = [
    r'^share$', r'^tweet$', r'^print$',
    r'^advertisement$', r'^sponsored$',
    r'^\s*\|+\s*$',                           # lines of just pipes
    r'^\s*_{3,}\s*$',                         # lines of underscores
    r'^\s*-{3,}\s*$',                         # lines of dashes
    r'^\s*\*{3,}\s*$',                        # lines of asterisks
    r'^\s*\d+\s*$',                           # standalone numbers (page numbers)
    r'^(home|news|sports|entertainment|about|contact|privacy policy|terms)$',
]


# ── Text cleaning functions ────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags safely."""
    # Remove script/style blocks first
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>',  ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove remaining tags
    text = re.sub(r'<[^>]{1,100}>', ' ', text)
    return text


def decode_entities(text: str) -> str:
    """Fix common HTML entities."""
    replacements = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&apos;': "'", '&nbsp;': ' ',
        '&#39;': "'", '&#x27;': "'", '&#x2F;': '/',
        '\u2019': "'", '\u2018': "'",   # curly apostrophes
        '\u201c': '"', '\u201d': '"',   # curly quotes
        '\u2013': '-', '\u2014': '--',  # en/em dashes
        '\u00a0': ' ',                  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove remaining numeric entities
    text = re.sub(r'&#\d{1,5};', ' ', text)
    text = re.sub(r'&#x[0-9a-fA-F]{1,4};', ' ', text)
    return text


def remove_boilerplate(text: str) -> str:
    """
    Process text line by line:
    - If a line matches an end-boilerplate trigger, truncate everything from that line
    - Remove full lines that match navigation/metadata patterns
    - Remove inline patterns from remaining lines
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line_lower = line.strip().lower()

        # Check if this line triggers end-truncation
        should_truncate = False
        for pattern in END_BOILERPLATE_TRIGGERS:
            if re.match(pattern, line_lower, re.IGNORECASE):
                should_truncate = True
                break

        if should_truncate:
            break  # stop here, everything below is boilerplate

        # Check if this whole line should be removed
        should_remove = False
        for pattern in REMOVE_FULL_LINE_PATTERNS:
            if re.match(pattern, line_lower, re.IGNORECASE):
                should_remove = True
                break

        if should_remove:
            continue

        # Apply inline removals
        cleaned_line = line
        for pattern in INLINE_REMOVE_PATTERNS:
            cleaned_line = re.sub(pattern, ' ', cleaned_line,
                                  flags=re.IGNORECASE | re.DOTALL)

        cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines, strip edges."""
    # Replace multiple newlines with double newline (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces on each line
    lines = [re.sub(r'  +', ' ', line).strip() for line in text.split('\n')]
    # Remove empty lines at start/end, preserve internal paragraph breaks
    text = '\n'.join(lines).strip()
    # Final collapse of multiple blank lines
    text = re.sub(r'\n\n\n+', '\n\n', text)
    return text


def truncate_to_max_words(text: str, max_words: int = MAX_WORDS) -> str:
    """
    Truncate text to max_words, breaking at sentence boundary.
    Used for live blogs and very long articles.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = ' '.join(words[:max_words])

    # Try to find the last sentence boundary
    last_period = max(
        truncated.rfind('. '),
        truncated.rfind('! '),
        truncated.rfind('? '),
    )

    if last_period > len(truncated) * 0.7:  # found a boundary in last 30%
        return truncated[:last_period + 1].strip()
    return truncated.strip()


def clean_title(title: str) -> str:
    """Clean title — remove site name suffixes, normalize."""
    title = str(title).strip()

    # Remove common site-name suffixes like "| BBC News", "- The Hindu", etc.
    title = re.sub(r'\s*[\|\-–—]\s*(bbc|the hindu|ndtv|reuters|ap|afp|pti|ani|times of india|hindustan times|indianexpress|economic times|livemint).*$',
                   '', title, flags=re.IGNORECASE)

    # Remove "- wave3.com-Louisville News, Weather & Sports" style suffixes
    title = re.sub(r'\s*-\s*\w+\.(com|org|net|co|in)\b.*$', '', title)

    title = decode_entities(title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def full_clean_pipeline(title: str, text: str) -> tuple[str, str]:
    """
    Apply full cleaning pipeline to title + text.
    Returns (cleaned_title, cleaned_text).
    """
    # Title
    title = clean_title(title)

    # Text: apply in order
    text = strip_html(text)
    text = decode_entities(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    text = truncate_to_max_words(text, MAX_WORDS)

    return title, text


# ── Domain capping ─────────────────────────────────────────────────────────────

def cap_domain(df: pd.DataFrame, domain_col: str = 'domain',
               max_per_domain: int = MAX_PER_DOMAIN) -> pd.DataFrame:
    log.info(f"\nApplying domain cap: max {max_per_domain} per domain")

    before = len(df)

    # Sort by word count descending so we keep the richest articles per domain
    df_sorted = df.sort_values('word_count', ascending=False).copy()

    # cumcount() assigns 0, 1, 2, ... within each domain group
    # Keep only rows where rank < max_per_domain
    df_sorted['_rank'] = df_sorted.groupby(domain_col).cumcount()
    df_capped = df_sorted[df_sorted['_rank'] < max_per_domain].copy()
    df_capped = df_capped.drop(columns=['_rank']).reset_index(drop=True)

    removed = before - len(df_capped)
    log.info(f"  Removed {removed} articles due to domain capping")

    # Report which domains were capped
    domain_counts_before = df[domain_col].value_counts()
    domain_counts_after  = df_capped[domain_col].value_counts()
    capped_domains = domain_counts_before[domain_counts_before > max_per_domain]

    if not capped_domains.empty:
        log.info(f"  Domains that were capped:")
        for domain in capped_domains.index:
            b = domain_counts_before[domain]
            a = domain_counts_after.get(domain, 0)
            log.info(f"    {domain}: {b} -> {a}")

    return df_capped

# ── Duplicate removal ──────────────────────────────────────────────────────────

def remove_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove near-duplicates using multiple signals:
    1. Exact URL duplicates (should already be gone, but safety check)
    2. Exact title duplicates (case-insensitive)
    3. Same text prefix (first 300 chars after cleaning)
    """
    before = len(df)

    # 1. URL dedup
    df = df.drop_duplicates(subset='url')

    # 2. Title dedup (case insensitive, stripped)
    df['_title_norm'] = df['title'].str.lower().str.strip()
    df['_title_norm'] = df['_title_norm'].str.replace(r'\s+', ' ', regex=True)
    df = df.drop_duplicates(subset='_title_norm')
    df = df.drop(columns=['_title_norm'])

    # 3. Text prefix dedup (catches same article from different URLs)
    df['_text_prefix'] = df['text'].str[:300].str.lower().str.strip()
    df['_text_prefix'] = df['_text_prefix'].str.replace(r'\s+', ' ', regex=True)
    df = df.drop_duplicates(subset='_text_prefix')
    df = df.drop(columns=['_text_prefix'])

    df = df.reset_index(drop=True)
    removed = before - len(df)
    log.info(f"  Near-duplicate removal: {removed} articles removed")
    return df


# ── Main pipeline ──────────────────────────────────────────────────────────────

def preprocess(input_csv: str, output_csv: str):
    start = datetime.now()

    df = pd.read_csv(input_csv)
    original_count = len(df)

    log.info(f"{'='*60}")
    log.info(f"PREPROCESSING PIPELINE")
    log.info(f"{'='*60}")
    log.info(f"Input: {original_count} articles")
    log.info(f"\nInitial distribution:")
    log.info(df['source_section'].value_counts().to_string())

    # ── Step 1: Domain capping ─────────────────────────────────────────────────
    log.info(f"\n-- Step 1: Domain capping --")
    df = cap_domain(df, max_per_domain=MAX_PER_DOMAIN)
    after_cap = len(df)
    log.info(f"After capping: {after_cap} articles ({original_count - after_cap} removed)")

    # ── Step 2: Text cleaning ──────────────────────────────────────────────────
    log.info(f"\n-- Step 2: Text cleaning (HTML, boilerplate, URLs, encoding) --")
    log.info(f"Processing {len(df)} articles...")

    cleaned_titles = []
    cleaned_texts  = []

    for i, (_, row) in enumerate(df.iterrows()):
        title, text = full_clean_pipeline(
            str(row['title']), str(row['text'])
        )
        cleaned_titles.append(title)
        cleaned_texts.append(text)

        if (i + 1) % 500 == 0:
            log.info(f"  Cleaned {i+1}/{len(df)} articles")

    df['title'] = cleaned_titles
    df['text']  = cleaned_texts

    # ── Step 3: Recompute word count after cleaning ────────────────────────────
    log.info(f"\n-- Step 3: Recomputing word counts after cleaning --")
    df['word_count'] = df['text'].str.split().str.len()

    # ── Step 4: Remove short articles ─────────────────────────────────────────
    log.info(f"\n-- Step 4: Removing short articles (<{MIN_WORDS} words) --")
    before_short = len(df)
    df = df[df['word_count'] >= MIN_WORDS].reset_index(drop=True)
    removed_short = before_short - len(df)
    log.info(f"  Removed {removed_short} short articles")

    # ── Step 5: Remove empty titles ───────────────────────────────────────────
    log.info(f"\n-- Step 5: Removing empty/stub titles --")
    before_title = len(df)
    df = df[df['title'].str.split().str.len() >= 3].reset_index(drop=True)
    log.info(f"  Removed {before_title - len(df)} stub-title articles")

    # ── Step 6: Remove near-duplicates ────────────────────────────────────────
    log.info(f"\n-- Step 6: Near-duplicate removal --")
    df = remove_near_duplicates(df)

    # ── Step 7: Remove live blogs (cap at MAX_WORDS already done in cleaning)
    # Flag any remaining very long articles for awareness
    very_long = (df['word_count'] > MAX_WORDS).sum()
    if very_long > 0:
        log.warning(f"  {very_long} articles still over {MAX_WORDS} words "
                    f"(truncation may have failed on some)")

    # ── Step 8: Final category balance check ──────────────────────────────────
    log.info(f"\n-- Step 8: Final balance check --")
    counts = df['source_section'].value_counts()
    max_c  = counts.max()
    min_c  = counts.min()
    ratio  = max_c / min_c

    log.info(f"\nFinal distribution:")
    for cat in CATEGORIES:
        c    = counts.get(cat, 0)
        bar  = '█' * int(c / max_c * 30)
        flag = ' ⚠ LOW' if c < 200 else ''
        log.info(f"  {cat:<25} {c:>4}  {bar}{flag}")

    log.info(f"\nImbalance ratio (max/min): {ratio:.2f}x  "
             f"({'OK' if ratio < 1.5 else 'Consider more scraping'})")

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False, encoding='utf-8')

    elapsed = (datetime.now() - start).total_seconds()

    log.info(f"\n{'='*60}")
    log.info(f"PREPROCESSING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Original:  {original_count:>6} articles")
    log.info(f"Final:     {len(df):>6} articles")
    log.info(f"Removed:   {original_count - len(df):>6} articles "
             f"({(original_count - len(df))/original_count*100:.1f}%)")
    log.info(f"Time:      {elapsed:.1f}s")
    log.info(f"Saved →    {output_csv}")

    return df


# ── Verification helper ────────────────────────────────────────────────────────

def verify_cleaning(output_csv: str, n: int = 5):
    """
    Spot-check the cleaned output.
    Prints random samples and re-checks for residual issues.
    """
    df = pd.read_csv(output_csv)

    print(f"\n{'='*60}")
    print(f"CLEANING VERIFICATION — {n} random samples per check")
    print(f"{'='*60}")

    # Check residual HTML
    html_left = df['text'].str.contains(r'<[a-zA-Z]+[\s>]', regex=True, na=False).sum()
    url_left  = df['text'].str.contains(r'https?://', regex=True, na=False).sum()
    also_read = df['text'].str.lower().str.contains('also read:', na=False).sum()
    ads_left  = df['text'].str.lower().str.contains('advertisement', na=False).sum()
    short_left = (df['word_count'] < MIN_WORDS).sum()

    print(f"\nResidual issues after cleaning:")
    print(f"  HTML tags remaining:     {html_left}")
    print(f"  URLs in body remaining:  {url_left}")
    print(f"  'Also read:' remaining:  {also_read}")
    print(f"  'Advertisement' left:    {ads_left}")
    print(f"  Articles below {MIN_WORDS}w:     {short_left}")

    print(f"\nWord count stats (should be min ~{MIN_WORDS}):")
    print(df['word_count'].describe().round(0).astype(int).to_string())

    print(f"\nRandom sample articles:")
    sample = df.sample(min(n, len(df)), random_state=99)
    for _, row in sample.iterrows():
        print(f"\n  Title:  {row['title'][:70]}")
        print(f"  Cat:    {row['source_section']} | "
              f"Words: {row['word_count']} | Domain: {row['domain']}")
        print(f"  Text:   {row['text'][:150].strip()}...")


if __name__ == '__main__':
    df = preprocess(INPUT_CSV, OUTPUT_CSV)
    verify_cleaning(OUTPUT_CSV, n=5)