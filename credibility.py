"""
Trains credibility regressor using LIAR dataset.
Supports CSV (Kaggle) and TSV (UCSB) formats.
Place dataset files in: liar_dataset/ folder
Run: python credibility.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import csv
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from language_feature import extract_linguistic_features, get_feature_names

# Paths
MODEL_PATH     = 'models/improved_model/best_model'
TOKENIZER_PATH = 'models/improved_model/best_tokenizer'
LIAR_DIR       = 'Dataset/liar_dataset/'
SAVE_DIR       = 'models'
EMB_CACHE      = 'models/credibility_embeddings_cache.npz'

# LIAR 6-class → credibility score (0-100)
LABEL_TO_SCORE = {
    # TSV string format (UCSB)
    'pants-fire':   0,
    'false':        20,
    'barely-true':  40,
    'half-true':    60,
    'mostly-true':  80,
    'true':         100,
    # Sometimes appear with hyphens replaced by spaces
    'pants fire':   0,
    'barely true':  40,
    'half true':    60,
    'mostly true':  80,
    # Numeric format (some Kaggle versions use 0-5)
    '0': 0,
    '1': 20,
    '2': 40,
    '3': 60,
    '4': 80,
    '5': 100,
}


# Load LIAR dataset from local files

def detect_format() -> tuple[str, str, str]:
    """Detect whether files are CSV or TSV and return paths."""
    # Check for CSV (Kaggle format)
    train_csv = os.path.join(LIAR_DIR, 'train.csv')
    valid_csv = os.path.join(LIAR_DIR, 'valid.csv')
    if os.path.exists(train_csv) and os.path.exists(valid_csv):
        print("Detected: Kaggle CSV format")
        return train_csv, valid_csv, 'csv'

    # Check for TSV (UCSB format)
    train_tsv = os.path.join(LIAR_DIR, 'train.tsv')
    valid_tsv = os.path.join(LIAR_DIR, 'valid.tsv')
    if os.path.exists(train_tsv) and os.path.exists(valid_tsv):
        print("Detected: UCSB TSV format")
        return train_tsv, valid_tsv, 'tsv'

    # Try alternate names some Kaggle downloads use
    alternates = [
        ('liar_train.csv', 'liar_valid.csv', 'csv'),
        ('train_data.csv', 'val_data.csv',   'csv'),
        ('TRAIN.csv',      'VALID.csv',       'csv'),
    ]
    for t, v, fmt in alternates:
        tp = os.path.join(LIAR_DIR, t)
        vp = os.path.join(LIAR_DIR, v)
        if os.path.exists(tp) and os.path.exists(vp):
            print(f"Detected alternate format: {t}")
            return tp, vp, fmt

    # List what's actually in the folder to help debug
    print(f"\nERROR: Could not find LIAR dataset files in '{LIAR_DIR}/'")
    print(f"Files found in {LIAR_DIR}/:")
    if os.path.exists(LIAR_DIR):
        for f in os.listdir(LIAR_DIR):
            print(f"  {f}")
    else:
        print(f"  Folder '{LIAR_DIR}' does not exist!")
    print("\nCreate the folder and place dataset files inside.")
    raise FileNotFoundError("LIAR dataset files not found.")


def parse_label(label_str: str) -> int:
    """Convert label string to 0-100 credibility score."""
    label_clean = str(label_str).strip().lower().replace('"', '')
    score = LABEL_TO_SCORE.get(label_clean)
    if score is None:
        # Try numeric
        try:
            num = int(float(label_clean))
            score = LABEL_TO_SCORE.get(str(num), 60)
        except (ValueError, TypeError):
            score = 60   # default to half-true if unknown
    return score


def load_liar_tsv(filepath: str) -> pd.DataFrame:
    """Parse UCSB TSV format."""
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 3:
                continue
            label_str  = row[1].strip()
            statement  = row[2].strip()
            speaker    = row[4].strip() if len(row) > 4 else ''

            if not statement:
                continue

            rows.append({
                'text':              statement,
                'credibility_score': parse_label(label_str),
                'speaker':           speaker,
                'raw_label':         label_str,
            })
    return pd.DataFrame(rows)


def load_liar_csv(filepath: str) -> pd.DataFrame:
    """Parse Kaggle CSV format."""
    # First peek at the file to detect if it has a header
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline().strip()

    has_header = 'label' in first_line.lower() or 'statement' in first_line.lower()

    df_raw = pd.read_csv(
        filepath,
        header=0 if has_header else None,
        encoding='utf-8',
        on_bad_lines='skip',
    )

    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Columns: {list(df_raw.columns)[:6]}...")
    print(f"  First row sample: {df_raw.iloc[0].tolist()[:4]}")

    rows = []
    for _, row in df_raw.iterrows():
        values = row.tolist()
        if len(values) < 3:
            continue

        # Detect label and statement columns
        # LIAR always has label at position 1, statement at position 2
        # (position 0 is ID)
        if has_header:
            # Try to find by column name
            label_col = next(
                (c for c in df_raw.columns
                 if 'label' in str(c).lower()), None
            )
            text_col = next(
                (c for c in df_raw.columns
                 if 'statement' in str(c).lower()
                 or 'text' in str(c).lower()), None
            )
            label_str = str(row[label_col]).strip() if label_col else str(values[1])
            statement = str(row[text_col]).strip() if text_col else str(values[2])
        else:
            # No header — use positional
            label_str = str(values[1]).strip()
            statement = str(values[2]).strip()

        if not statement or statement == 'nan':
            continue

        rows.append({
            'text':              statement,
            'credibility_score': parse_label(label_str),
            'speaker':           str(values[4]).strip() if len(values) > 4 else '',
            'raw_label':         label_str,
        })

    return pd.DataFrame(rows)


def load_liar() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, val_path, fmt = detect_format()

    print(f"Loading train: {train_path}")
    print(f"Loading val:   {val_path}")

    if fmt == 'tsv':
        train_df = load_liar_tsv(train_path)
        val_df   = load_liar_tsv(val_path)
    else:
        train_df = load_liar_csv(train_path)
        val_df   = load_liar_csv(val_path)

    # Validate
    assert len(train_df) > 100, f"Train set too small: {len(train_df)}"
    assert len(val_df)   > 50,  f"Val set too small: {len(val_df)}"

    print(f"\nLIAR loaded successfully:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Score distribution (train):")
    print(train_df['credibility_score'].value_counts().sort_index().to_string())

    # Quick label sanity check
    sample_labels = train_df['raw_label'].value_counts().head(6)
    print(f"\n  Raw label counts: {sample_labels.to_dict()}")

    return train_df, val_df


# Extract CLS embeddings

def extract_cls_embeddings(
    texts: list[str],
    tokenizer,
    backbone,
    batch_size: int = 32,
) -> np.ndarray:
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            max_length=256, padding='max_length',
            truncation=True, return_tensors='pt',
        )
        with torch.no_grad():
            out      = backbone(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
            )
            cls_vecs = out.last_hidden_state[:, 0, :].numpy()

        all_embeddings.append(cls_vecs)

        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings)


# Extract linguistic features

def extract_ling_features(df: pd.DataFrame) -> np.ndarray:
    features = []
    for _, row in df.iterrows():
        text = str(row['text'])
        f    = extract_linguistic_features(
            title=text[:80],   # LIAR has no title — use first 80 chars
            text=text,
            url='',            # LIAR has no URL
        )
        features.append(f)
    return np.array(features)


# Main

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load backbone
    print("Loading RoBERTa backbone...")
    tokenizer  = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    full_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    backbone   = full_model.roberta
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("Backbone loaded and frozen.\n")

    # Load dataset
    train_df, val_df = load_liar()

    # Embeddings (with cache)
    if os.path.exists(EMB_CACHE):
        print(f"\nLoading cached embeddings from {EMB_CACHE}")
        cache        = np.load(EMB_CACHE)
        train_emb    = cache['train_emb']
        val_emb      = cache['val_emb']
        print(f"Cache loaded — train: {train_emb.shape}, val: {val_emb.shape}")
    else:
        print("\nExtracting train CLS embeddings (~10-15 min on CPU)...")
        train_emb = extract_cls_embeddings(
            train_df['text'].tolist(), tokenizer, backbone
        )

        print("\nExtracting val CLS embeddings...")
        val_emb = extract_cls_embeddings(
            val_df['text'].tolist(), tokenizer, backbone
        )

        np.savez(
            EMB_CACHE,
            train_emb=train_emb,
            val_emb=val_emb,
        )
        print(f"\nEmbeddings cached -> {EMB_CACHE}")

    # Linguistic features
    print("\nExtracting linguistic features...")
    train_ling = extract_ling_features(train_df)
    val_ling   = extract_ling_features(val_df)
    print(f"Linguistic features: {train_ling.shape[1]}d per article")

    # Labels
    train_labels = train_df['credibility_score'].values.astype(float)
    val_labels   = val_df['credibility_score'].values.astype(float)

    # Combine CLS + linguistic
    X_train = np.concatenate([train_emb, train_ling], axis=1)  # (N, 776)
    X_val   = np.concatenate([val_emb,   val_ling],   axis=1)
    print(f"\nFeature shape: {X_train.shape} (768 CLS + 8 linguistic = 776d)")

    # Train Ridge Regressor
    print("\nTraining Ridge Regressor...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=10.0)),
    ])
    pipeline.fit(X_train, train_labels)

    val_preds = np.clip(pipeline.predict(X_val), 0, 100)
    mae       = mean_absolute_error(val_labels, val_preds)
    r2        = r2_score(val_labels, val_preds)

    print(f"\nCredibility Regressor Results:")
    print(f"  MAE:  {mae:.2f}  (on 0-100 scale)")
    print(f"  R2:   {r2:.4f}")
    print(f"  Note: MAE 18-25 is expected on LIAR — statement-level labels")
    print(f"        R2 of 0.20-0.40 is state-of-the-art for this task")

    # Feature importance
    scaler    = pipeline.named_steps['scaler']
    ridge     = pipeline.named_steps['ridge']
    ling_coef = ridge.coef_[-8:]   # last 8 = linguistic features

    print(f"\nLinguistic feature weights:")
    for name, coef in zip(get_feature_names(), ling_coef):
        direction = 'increases credibility' if coef > 0 else 'decreases credibility'
        print(f"  {name:<25} {coef:+.4f}  ({direction})")

    # Save
    out_path = os.path.join(SAVE_DIR, 'credibility_regressor.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"\nSaved -> {out_path}")
    print("Credibility module ready. Run: streamlit run app.py")


if __name__ == '__main__':
    main()