"""
llm_classifier.py
-----------------
Uses a local HuggingFace NLI model (DeBERTa-v3-base) for multi-label 
zero-shot classification of news articles.

No API key needed. No rate limits. Runs entirely on CPU.

Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
  - 184M params, ~1.5 GB RAM
  - Trained on 4 NLI datasets: MNLI, FEVER, ANLI
  - Batches all 10 label hypotheses in a single forward pass for speed

Requirements:
    pip install transformers torch pandas

Input:  cleaned_articles.csv
Output: llm_labeled_articles.csv

Run: python scrapper/llm_classifier.py
"""

import os
os.environ['HF_HUB_DISABLE_XET'] = '1'  # avoid xet CDN timeout issues on Windows

import pandas as pd
import ast
import sys
import logging
import torch
import numpy as np
from datetime import datetime
from collections import Counter

# ── Logging (UTF-8 safe for Windows) ──────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

fh = logging.FileHandler('labeling_log.txt', encoding='utf-8')
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(stream=open(sys.stdout.fileno(),
                                       mode='w',
                                       encoding='utf-8',
                                       closefd=False))
sh.setFormatter(fmt)
logger.addHandler(sh)

log = logger

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_CSV      = 'cleaned_articles.csv'
OUTPUT_CSV     = 'llm_labeled_articles.csv'
CHECKPOINT_CSV = 'labeling_checkpoint.csv'

MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

CATEGORIES = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health',
    'Science/Other', 'Sports', 'Entertainment'
]

# Descriptive phrases for NLI hypotheses — dramatically improves zero-shot accuracy.
# The model sees these descriptions, but output uses original CATEGORIES names.
LABEL_DESCRIPTIONS = {
    'Politics':              'politics, government policy, elections, or political affairs',
    'Technology':            'technology, software, hardware, AI, or the tech industry',
    'Business':              'business, corporate earnings, markets, mergers, or company strategy',
    'Economy':               'economics, inflation, GDP, central bank policy, or fiscal policy',
    'International_Affairs': 'international affairs, foreign relations, diplomacy, or global events',
    'Education':             'education, schools, universities, exams, or academic policy',
    'Health':                'health, medicine, disease, hospitals, or public health',
    'Science/Other':         'science, research, environment, climate, or scientific discovery',
    'Sports':                'sports, cricket, football, athletics, or sporting events',
    'Entertainment':         'entertainment, movies, music, celebrities, or film industry',
}

# Zero-shot classification config
CONFIDENCE_THRESHOLD = 0.6    # entailment probability above this = label assigned
TEXT_SNIPPET_LEN     = 300    # chars of article text (title + snippet)
MAX_TOKEN_LEN        = 256    # max tokens — shorter = faster, 256 is plenty for classification
CHECKPOINT_EVERY     = 100    # save checkpoint every N articles

# Hypothesis template — the model evaluates: "premise entails hypothesis?"
HYPOTHESIS_TEMPLATE = "This text is about {}."


# ── Model setup ───────────────────────────────────────────────────────────────

_model = None
_tokenizer = None

def setup_model():
    """
    Loads the NLI model and tokenizer directly (not via pipeline).
    This allows us to batch all 10 label hypotheses in a single forward pass,
    which is 5-10x faster than the pipeline's sequential approach.
    """
    global _model, _tokenizer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    log.info(f"Loading model: {MODEL_NAME}")
    log.info("(First run will download ~350 MB, cached after that)")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.eval()  # inference mode

    log.info(f"Model loaded successfully on CPU")
    return (_tokenizer, _model)


def classify_nli(premise: str, labels: list[str]) -> dict[str, float]:
    """
    Run NLI inference: for each label, check if premise entails the hypothesis.
    Batches all labels into a single forward pass for speed.
    
    Uses descriptive phrases from LABEL_DESCRIPTIONS for better accuracy,
    but returns scores keyed by original category names.
    
    Returns dict of {label: entailment_probability}.
    """
    # Use descriptive hypotheses for better NLI accuracy
    hypotheses = [
        HYPOTHESIS_TEMPLATE.format(LABEL_DESCRIPTIONS.get(label, label))
        for label in labels
    ]

    # Tokenize all (premise, hypothesis) pairs at once
    inputs = _tokenizer(
        [premise] * len(hypotheses),
        hypotheses,
        return_tensors='pt',
        truncation='only_first',     # truncate only the premise, not hypothesis
        max_length=MAX_TOKEN_LEN,
        padding=True,
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits  # shape: (num_labels, 3) — contradiction, neutral, entailment

    # The model outputs 3 classes: 0=entailment, 1=neutral, 2=contradiction
    # We want P(entailment) for each hypothesis
    probs = torch.softmax(logits, dim=-1)
    entailment_probs = probs[:, 0].numpy()  # column 0 = entailment

    return {label: float(prob) for label, prob in zip(labels, entailment_probs)}


# ── Single article labeling ───────────────────────────────────────────────────

def label_article(
    model_tuple,
    title: str,
    text: str,
    source_section: str,
    article_idx: int,
    total: int,
) -> tuple[list[str], str]:
    """
    Labels a single article using batched NLI zero-shot classification.
    Returns (labels, method) where method is 'zeroshot' or 'fallback'.
    
    All 10 category hypotheses are evaluated in a single forward pass.
    """
    # Build input: title + text snippet
    title_clean = str(title).strip()[:200]
    text_clean  = str(text).strip()[:TEXT_SNIPPET_LEN]
    premise     = f"{title_clean}. {text_clean}"

    try:
        scores = classify_nli(premise, CATEGORIES)

        # Extract labels above threshold
        labels = [cat for cat, score in scores.items() if score >= CONFIDENCE_THRESHOLD]

        if labels:
            return labels, 'zeroshot'

        # If nothing passed threshold, take the top-scoring label
        top_label = max(scores, key=scores.get)
        log.debug(f"[{article_idx}/{total}] No label above threshold "
                  f"(top: {top_label}={scores[top_label]:.3f}), using top-1")
        return [top_label], 'zeroshot'

    except Exception as e:
        log.warning(f"[{article_idx}/{total}] Classification failed: {e}")
        fallback = [source_section] if source_section in CATEGORIES else ['Science/Other']
        return fallback, 'fallback'


# ── Checkpointing ──────────────────────────────────────────────────────────────

def load_checkpoint(input_df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """
    Loads checkpoint if it exists.
    Returns (labeled_so_far_df, set_of_already_labeled_urls)
    """
    if not os.path.exists(CHECKPOINT_CSV):
        return pd.DataFrame(), set()

    ckpt = pd.read_csv(CHECKPOINT_CSV)

    # Parse llm_labels column back to list
    ckpt['llm_labels'] = ckpt['llm_labels'].apply(
        lambda x: ast.literal_eval(str(x)) if pd.notna(x) else []
    )

    already_labeled = set(ckpt['url'].tolist())
    log.info(f"Resuming from checkpoint: {len(already_labeled)} articles already labeled")
    return ckpt, already_labeled


def save_checkpoint(labeled_rows: list[dict], original_df: pd.DataFrame):
    """Merges and saves progress to checkpoint CSV."""
    result_df = pd.DataFrame(labeled_rows)
    merged    = original_df.merge(result_df[['url', 'llm_labels', 'label_method']],
                                   on='url', how='inner')
    merged.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8')


# ── Main pipeline ──────────────────────────────────────────────────────────────

def label_all_articles(input_csv: str, output_csv: str):
    """
    Full labeling pipeline:
    1. Load cleaned articles
    2. Resume from checkpoint if available
    3. Label remaining articles with local NLI zero-shot classifier
    4. Post-process: binary columns, quality checks
    5. Save final CSV
    """
    model_tuple = setup_model()

    df = pd.read_csv(input_csv)
    log.info(f"\nLoaded {len(df)} articles from {input_csv}")
    log.info(f"Source distribution:\n{df['source_section'].value_counts().to_string()}")

    # Load checkpoint
    ckpt_df, already_labeled_urls = load_checkpoint(df)

    labeled_rows = []
    if not ckpt_df.empty:
        for _, row in ckpt_df.iterrows():
            labeled_rows.append({
                'url':          row['url'],
                'llm_labels':   row['llm_labels'],
                'label_method': row.get('label_method', 'zeroshot'),
            })

    # Filter to unlabeled articles
    to_label = df[~df['url'].isin(already_labeled_urls)].reset_index(drop=True)
    log.info(f"Articles to label: {len(to_label)}")

    total          = len(to_label)
    fallback_count = 0
    start_time     = datetime.now()

    for i, (_, row) in enumerate(to_label.iterrows(), start=1):
        labels, method = label_article(
            model_tuple    = model_tuple,
            title          = row.get('title', ''),
            text           = row.get('text', ''),
            source_section = row.get('source_section', 'Science/Other'),
            article_idx    = i,
            total          = total,
        )

        if method == 'fallback':
            fallback_count += 1

        labeled_rows.append({
            'url':          row['url'],
            'llm_labels':   labels,
            'label_method': method,
        })

        # Progress logging
        if i % 50 == 0:
            elapsed   = (datetime.now() - start_time).total_seconds()
            rate      = i / elapsed  # articles per second
            remaining = (total - i) / rate if rate > 0 else 0
            log.info(
                f"Progress: {i}/{total} | "
                f"Fallbacks: {fallback_count} | "
                f"Speed: {rate:.1f} art/s | "
                f"ETA: {remaining/60:.1f} min"
            )

        # Save checkpoint periodically
        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(labeled_rows, df)

    # ── Final merge ────────────────────────────────────────────────────────────
    result_df = pd.DataFrame(labeled_rows)
    final_df  = df.merge(
        result_df[['url', 'llm_labels', 'label_method']],
        on='url', how='inner'
    )

    # Parse label lists if they came through as strings
    final_df['llm_labels'] = final_df['llm_labels'].apply(
        lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
    )

    # ── Binary columns ─────────────────────────────────────────────────────────
    for cat in CATEGORIES:
        final_df[cat] = final_df['llm_labels'].apply(
            lambda x, c=cat: 1 if c in x else 0
        )

    # ── Quality checks ─────────────────────────────────────────────────────────
    # Drop rows with zero valid labels
    label_sums = final_df[CATEGORIES].sum(axis=1)
    zero_label_count = (label_sums == 0).sum()
    if zero_label_count > 0:
        log.warning(f"Dropping {zero_label_count} articles with no valid labels")
        final_df = final_df[label_sums > 0]

    final_df.reset_index(drop=True, inplace=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    final_df.to_csv(output_csv, index=False, encoding='utf-8')

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed_total = (datetime.now() - start_time).total_seconds()

    log.info(f"\n{'='*60}")
    log.info(f"LABELING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Model:           {MODEL_NAME}")
    log.info(f"Threshold:       {CONFIDENCE_THRESHOLD}")
    log.info(f"Total articles:  {len(final_df)}")
    log.info(f"Zero-shot:       {len(final_df[final_df['label_method']=='zeroshot'])}")
    log.info(f"Fallback:        {fallback_count}")
    log.info(f"Total time:      {elapsed_total/60:.1f} min")

    log.info(f"\nLabel distribution (articles per label):")
    label_counts = final_df[CATEGORIES].sum().sort_values(ascending=False)
    max_count = label_counts.max()
    for cat, count in label_counts.items():
        bar   = '#' * int(count / max_count * 35)
        alert = ' << LOW' if count < 80 else ''
        log.info(f"  {cat:<25} {count:>4}  {bar}{alert}")

    log.info(f"\nLabels per article distribution:")
    lpd = final_df[CATEGORIES].sum(axis=1).value_counts().sort_index()
    for n, count in lpd.items():
        log.info(f"  {n} label(s):  {count} articles")

    log.info(f"\nSaved -> {output_csv}")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)
        log.info("Checkpoint removed.")

    return final_df


# ── Spot-check helper ──────────────────────────────────────────────────────────

def verify_sample(csv_path: str, n: int = 25):
    """
    Prints n random labeled articles for manual quality inspection.
    Run this after labeling to catch systematic errors before training.

    Things to look for:
    - Sports articles mislabeled as Entertainment (cricket != Bollywood)
    - Economy vs Business confusion (RBI rate = Economy, company profit = Business)
    - International_Affairs only when India is not the actor
    - Science/Other as a catch-all being overused
    """
    df = pd.read_csv(csv_path)
    df['llm_labels'] = df['llm_labels'].apply(
        lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
    )

    print(f"\n{'='*70}")
    print(f"QUALITY CHECK -- {min(n, len(df))} random samples")
    print(f"{'='*70}")

    sample = df.sample(min(n, len(df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"\nTitle:   {str(row['title'])[:75]}")
        print(f"Source:  {row.get('source_section', '?')}")
        print(f"Labels:  {row['llm_labels']}")
        print(f"Method:  {row.get('label_method', '?')}")
        print(f"Domain:  {row.get('domain', '?')}")
        print('-' * 50)

    print(f"\nOverall label distribution:")
    all_labels = [l for sublist in df['llm_labels'] for l in sublist]
    for label, count in Counter(all_labels).most_common():
        print(f"  {label:<25} {count}")


def check_low_categories(csv_path: str, threshold: int = 80):
    """
    Identifies categories with too few articles.
    Call this after labeling -- if any category is below threshold,
    scrape more articles for that category before training.
    """
    df = pd.read_csv(csv_path)
    df['llm_labels'] = df['llm_labels'].apply(
        lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
    )

    label_counts = {cat: 0 for cat in CATEGORIES}
    for labels in df['llm_labels']:
        for l in labels:
            if l in label_counts:
                label_counts[l] += 1

    print(f"\n{'='*50}")
    print(f"CATEGORY HEALTH CHECK (threshold: {threshold})")
    print(f"{'='*50}")
    needs_more = []
    for cat, count in sorted(label_counts.items(), key=lambda x: x[1]):
        status = 'OK' if count >= threshold else 'NEEDS MORE DATA'
        print(f"  {cat:<25} {count:>4}  {status}")
        if count < threshold:
            needs_more.append((cat, threshold - count))

    if needs_more:
        print(f"\nAction required:")
        for cat, needed in needs_more:
            print(f"  Scrape ~{needed} more articles for: {cat}")
    else:
        print(f"\nAll categories above threshold. Ready for training.")

    return needs_more


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = label_all_articles(
        input_csv  = INPUT_CSV,
        output_csv = OUTPUT_CSV,
    )

    # Quality checks
    verify_sample(OUTPUT_CSV, n=25)
    check_low_categories(OUTPUT_CSV, threshold=80)