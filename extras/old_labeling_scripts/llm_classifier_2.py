"""
gemini_batch_labeler.py
-----------------------
Labels 3 articles per API call — 3x faster than single-article approach.
Free tier: 15 RPM → at 3 articles/call = effectively 45 articles/minute
ETA for 3019 articles: ~67 minutes

Run on local machine (no GPU needed — it's an API call).
"""

import google.generativeai as genai
import json
import pandas as pd
import time
import os
import ast
import sys
import logging
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

load_dotenv()  # loads GOOGLE_API_KEY from .env file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('gemini_labeling.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_CSV       = 'cleaned_articles.csv'
OUTPUT_CSV      = 'gemini_labeled_articles.csv'
CHECKPOINT_CSV  = 'gemini_checkpoint.csv'

CATEGORIES = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health',
    'Science/Other', 'Sports', 'Entertainment'
]

BATCH_SIZE       = 3      # articles per API call
REQUEST_DELAY    = 4.2    # seconds between calls (free tier: 15 RPM)
                          # set to 0.5 if on paid tier
MAX_RETRIES      = 4
SNIPPET_LEN      = 500    # chars per article sent to API

# ── Gemini setup ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a precise multi-label news classifier.

Given multiple news articles, classify EACH ONE independently.
Assign ALL relevant labels from EXACTLY this list:
{', '.join(CATEGORIES)}

RULES:
1. Return a JSON array where each element is the label array for that article
2. Assign Business ONLY for company earnings, markets, mergers, corporate strategy
3. Do NOT assign Business to: pharma trials, sports contracts, tech research
4. Every article needs at least one label
5. Be conservative — fewer accurate labels beats many uncertain ones

EXAMPLES of what Business IS: quarterly earnings, IPO, CEO fired, merger, stock rally
EXAMPLES of what Business is NOT: drug approval, sports contract, AI research paper

Return ONLY valid JSON like:
[["Politics"], ["Technology", "Business"], ["Sports", "International_Affairs"]]"""

USER_TEMPLATE = """Classify these {n} articles:

{articles}

Return a JSON array of {n} label arrays."""


def setup_gemini() -> genai.GenerativeModel:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "Set your API key:\n"
            "  Windows: set GOOGLE_API_KEY=AIza...\n"
            "  Mac/Linux: export GOOGLE_API_KEY=AIza...\n"
            "  Get key: https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        generation_config=genai.GenerationConfig(
            temperature=0.05,
            max_output_tokens=300,
            response_mime_type='application/json',
        ),
        system_instruction=SYSTEM_PROMPT,
    )


def format_batch(rows: list[dict]) -> str:
    """Format multiple articles into numbered prompt."""
    parts = []
    for i, row in enumerate(rows, 1):
        parts.append(
            f"Article {i}:\n"
            f"Title: {str(row['title'])[:200]}\n"
            f"Text: {str(row['text'])[:SNIPPET_LEN]}"
        )
    return '\n\n'.join(parts)


def validate_batch_response(raw: str, expected_count: int) -> list[list[str]] | None:
    """
    Parse Gemini batch response.
    Expected: [["Label1"], ["Label1", "Label2"], ["Label1"]]
    """
    try:
        clean = raw.strip()
        if '```' in clean:
            parts = clean.split('```')
            for p in parts:
                if '[' in p:
                    clean = p.replace('json', '').strip()
                    break

        parsed = json.loads(clean)

        # Handle nested vs flat
        if not isinstance(parsed, list):
            return None

        # If it's a flat list of strings (single article response), wrap it
        if parsed and isinstance(parsed[0], str):
            parsed = [parsed]

        if len(parsed) != expected_count:
            log.warning(f"Expected {expected_count} results, got {len(parsed)}")
            # Pad or trim to expected count
            while len(parsed) < expected_count:
                parsed.append(['Science/Other'])
            parsed = parsed[:expected_count]

        # Validate each label list
        result = []
        for label_list in parsed:
            if not isinstance(label_list, list):
                result.append(['Science/Other'])
                continue
            valid = [str(l).strip() for l in label_list if str(l).strip() in CATEGORIES]
            result.append(valid if valid else ['Science/Other'])

        return result

    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def label_batch(
    model: genai.GenerativeModel,
    rows: list[dict],
    attempt: int = 0
) -> list[list[str]]:
    """Label a batch of articles. Returns list of label lists."""
    prompt = USER_TEMPLATE.format(
        n=len(rows),
        articles=format_batch(rows)
    )

    delays = [3, 8, 20, 45]

    for retry in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)

            if not response.candidates:
                raise ValueError("Empty response from Gemini")

            labels = validate_batch_response(response.text, len(rows))
            if labels is not None:
                return labels

            log.warning(f"Batch validation failed on retry {retry+1}. "
                        f"Raw: {response.text[:100]}")

        except Exception as e:
            err = str(e).lower()
            if '429' in err or 'quota' in err or 'rate' in err:
                wait = delays[min(retry, len(delays)-1)] * 4
                log.warning(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            elif '401' in err or 'api_key' in err:
                raise EnvironmentError("Invalid GOOGLE_API_KEY")
            else:
                wait = delays[min(retry, len(delays)-1)]
                log.warning(f"Error retry {retry+1}: {e}. Waiting {wait}s...")
                time.sleep(wait)

    # All retries failed — use source section as fallback
    log.error(f"All retries failed for batch. Using fallback labels.")
    return [[row.get('source_section', 'Science/Other')]
            for row in rows]


def load_checkpoint() -> tuple[set, list]:
    """Load already-labeled URLs and their results."""
    if not os.path.exists(CHECKPOINT_CSV):
        return set(), []
    ckpt = pd.read_csv(CHECKPOINT_CSV)
    labeled_urls = set(ckpt['url'].tolist())
    results = ckpt[['url', 'llm_labels', 'label_method']].to_dict('records')
    log.info(f"Resuming: {len(labeled_urls)} already labeled")
    return labeled_urls, results


def save_checkpoint(df_original: pd.DataFrame, results: list):
    """Save progress."""
    result_df = pd.DataFrame(results)
    merged = df_original.merge(
        result_df[['url', 'llm_labels', 'label_method']],
        on='url', how='inner'
    )
    merged.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8')


def run_labeling():
    model = setup_gemini()

    df = pd.read_csv(INPUT_CSV)
    log.info(f"Loaded {len(df)} articles")

    labeled_urls, results = load_checkpoint()
    to_label = df[~df['url'].isin(labeled_urls)].reset_index(drop=True)
    log.info(f"To label: {len(to_label)} articles")

    total      = len(to_label)
    start_time = datetime.now()
    batch_num  = 0

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = to_label.iloc[batch_start:batch_start + BATCH_SIZE]
        batch_dicts = batch_rows.to_dict('records')
        batch_num  += 1

        labels_batch = label_batch(model, batch_dicts)

        for row_dict, labels in zip(batch_dicts, labels_batch):
            results.append({
                'url':          row_dict['url'],
                'llm_labels':   labels,
                'label_method': 'gemini_batch',
            })

        # Progress every 10 batches (30 articles)
        articles_done = min(batch_start + BATCH_SIZE, total)
        if batch_num % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate    = articles_done / elapsed if elapsed > 0 else 0
            eta_min = (total - articles_done) / rate / 60 if rate > 0 else 0
            log.info(
                f"Progress: {articles_done}/{total} | "
                f"Speed: {rate:.2f} art/s | "
                f"ETA: {eta_min:.1f} min"
            )
            save_checkpoint(df, results)

        time.sleep(REQUEST_DELAY)

    # ── Final processing ───────────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    final_df  = df.merge(
        result_df[['url', 'llm_labels', 'label_method']],
        on='url', how='inner'
    )

    # Parse label lists
    final_df['llm_labels'] = final_df['llm_labels'].apply(
        lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
    )

    # Binary columns
    for cat in CATEGORIES:
        final_df[cat] = final_df['llm_labels'].apply(
            lambda x: 1 if cat in x else 0
        )

    # Quality check
    label_sums = final_df[CATEGORIES].sum(axis=1)
    zeros = (label_sums == 0).sum()
    if zeros:
        log.warning(f"Dropping {zeros} articles with no labels")
        final_df = final_df[label_sums > 0]

    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    # Report
    log.info(f"\n{'='*50}")
    log.info(f"LABELING COMPLETE: {len(final_df)} articles")
    log.info(f"\nLabel distribution:")
    counts = final_df[CATEGORIES].sum().sort_values(ascending=False)
    for cat, count in counts.items():
        pct = count / len(final_df) * 100
        log.info(f"  {cat:<25} {count:>4} ({pct:.1f}%)")

    log.info(f"\nMean labels per article: "
             f"{final_df[CATEGORIES].sum(axis=1).mean():.2f}")

    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)

    return final_df


if __name__ == '__main__':
    df = run_labeling()