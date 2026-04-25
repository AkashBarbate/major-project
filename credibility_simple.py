"""
credibility_simple.py
---------------------
Credibility scorer using linguistic features only.
No LIAR dataset needed. No CLS embeddings needed.
Much more interpretable and actually works correctly.

Run: python credibility_simple.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import ast
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from language_feature import extract_linguistic_features, get_feature_names

# ── Paths ───────────────────────────────────────────────────────────────────────
DATASET_PATH = 'gemini_labeled_articles.csv'
SAVE_DIR     = 'models'

# Source domain → credibility score
# Based on known editorial standards — this is our ground truth proxy
DOMAIN_CREDIBILITY = {
    # High credibility (80-100)
    'bbc.com':                      95,
    'bbc.co.uk':                    95,
    'reuters.com':                  95,
    'apnews.com':                   95,
    'thehindu.com':                 88,
    'theprint.in':                  85,
    'scroll.in':                    83,
    'thewire.in':                   82,
    'ndtv.com':                     82,
    'livemint.com':                 82,
    'indianexpress.com':            80,
    'hindustantimes.com':           78,
    'timesofindia.indiatimes.com':  76,
    'aljazeera.com':                82,
    'theguardian.com':              88,
    'businessstandard.com':         80,
    'economictimes.indiatimes.com': 75,
    # Medium credibility (50-70)
    'deccanherald.com':             72,
    'newindianexpress.com':         70,
    'financialexpress.com':         72,
    'moneycontrol.com':             68,
    'careers360.com':               65,
    'jagranjosh.com':               60,
    # Historical CC-News domains (unknown = neutral)
    'default':                      50,
}


def get_domain_score(domain: str) -> float:
    """Return credibility score for a domain."""
    domain_clean = str(domain).lower().replace('www.', '').strip()
    return DOMAIN_CREDIBILITY.get(domain_clean, DOMAIN_CREDIBILITY['default'])


def compute_weak_credibility(row) -> float:
    """
    Generate a credibility pseudo-label for each article using:
    1. Domain reputation (primary signal)
    2. Linguistic quality signals (secondary signal)

    Returns score 0-100.
    """
    # Primary: domain score (60% weight)
    domain_score = get_domain_score(str(row.get('domain', '')))

    # Secondary: linguistic signals (40% weight)
    ling = extract_linguistic_features(
        title=str(row.get('title', '')),
        text=str(row.get('text', '')),
        url=str(row.get('url', '')),
    )

    # Inverted signals (high value = bad for credibility)
    subjectivity     = ling[0]
    caps_ratio       = ling[1]
    exclamation      = ling[2]
    clickbait        = ling[3]
    sensational      = ling[4]
    # Positive signals (high value = good for credibility)
    length_score     = ling[5]
    question_ratio   = ling[6]  # inverted — questions = uncertainty
    source_score     = ling[7]

    # Compute linguistic credibility (0-100)
    ling_score = (
        (1 - subjectivity)  * 20 +
        (1 - caps_ratio)    * 15 +
        (1 - exclamation)   * 10 +
        (1 - clickbait)     * 20 +
        (1 - sensational)   * 15 +
        length_score        * 15 +
        (1 - question_ratio)* 5,
    )[0] * (100 / 100)

    # Weighted combination
    final_score = (domain_score * 0.60) + (ling_score * 0.40)
    return float(np.clip(final_score, 0, 100))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load your labeled news articles ──────────────────────────────────────
    print(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} articles")

    # ── Generate weak credibility labels ─────────────────────────────────────
    print("\nGenerating credibility labels...")
    df['credibility_score'] = df.apply(compute_weak_credibility, axis=1)

    print(f"\nCredibility score distribution:")
    print(f"  Mean:   {df['credibility_score'].mean():.1f}")
    print(f"  Std:    {df['credibility_score'].std():.1f}")
    print(f"  Min:    {df['credibility_score'].min():.1f}")
    print(f"  Max:    {df['credibility_score'].max():.1f}")

    # Score buckets
    buckets = pd.cut(df['credibility_score'],
                     bins=[0, 30, 50, 70, 90, 100],
                     labels=['0-30 (Very Low)', '30-50 (Low)',
                             '50-70 (Moderate)', '70-90 (High)',
                             '90-100 (Very High)'])
    print(f"\n  Score distribution:")
    print(buckets.value_counts().sort_index().to_string())

    # ── Extract linguistic features ───────────────────────────────────────────
    print("\nExtracting linguistic features...")
    X = np.array([
        extract_linguistic_features(
            title=str(row['title']),
            text=str(row['text']),
            url=str(row.get('url', '')),
        )
        for _, row in df.iterrows()
    ])
    y = df['credibility_score'].values

    print(f"Feature matrix: {X.shape}")

    # ── Train/val split ───────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train regressor ───────────────────────────────────────────────────────
    print("\nTraining Ridge Regressor on linguistic features...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    val_preds = np.clip(pipeline.predict(X_val), 0, 100)
    mae       = mean_absolute_error(y_val, val_preds)
    r2        = r2_score(y_val, val_preds)

    print(f"\nResults:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R2:  {r2:.4f}")

    # ── Feature weights ───────────────────────────────────────────────────────
    ridge     = pipeline.named_steps['ridge']
    coef      = ridge.coef_
    feat_names = get_feature_names()

    print(f"\nFeature weights (Ridge):")
    for name, w in zip(feat_names, coef):
        direction = 'good for credibility' if w > 0 else 'bad for credibility'
        print(f"  {name:<25} {w:+.4f}  ({direction})")

    # Sample predictions
    print(f"\nSample predictions:")
    sample_df = df.sample(8, random_state=42)
    sample_X  = np.array([
        extract_linguistic_features(
            str(row['title']), str(row['text']), str(row.get('url', ''))
        )
        for _, row in sample_df.iterrows()
    ])
    sample_preds = np.clip(pipeline.predict(sample_X), 0, 100)
    for (_, row), pred in zip(sample_df.iterrows(), sample_preds):
        actual = row['credibility_score']
        print(f"  [{row['domain'][:25]:<25}]  "
              f"actual={actual:.0f}  pred={pred:.0f}  "
              f"| {str(row['title'])[:45]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(SAVE_DIR, 'credibility_regressor.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"\nSaved -> {out_path}")
    print("\nCredibility module ready.")
    print("Next step: streamlit run app.py")


if __name__ == '__main__':
    main()