"""
gemini_labeler_v3.py
--------------------
Fixes all issues:
  1. Uses new google-genai package (not deprecated google.generativeai)
  2. Rotates across 4 free models — 6000 requests/day total
  3. Proper daily quota tracking per model
  4. Checkpoint saves every batch — safe to restart anytime
  5. Batch size 3 — 3x faster than single article

Install:
    pip install google-genai pandas

Run:
    set GOOGLE_API_KEY=AIza...
    python gemini_labeler_v3.py
"""
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
import json, os, ast, time, sys, logging
import pandas as pd
from datetime import datetime, date
from collections import defaultdict

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('labeling_v3.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────────
INPUT_CSV      = 'cleaned_articles.csv'
OUTPUT_CSV     = 'gemini_labeled_articles.csv'
CHECKPOINT_CSV = 'labeling_checkpoint_v3.csv'
QUOTA_FILE     = 'daily_quota_tracker.json'

CATEGORIES = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health',
    'Science/Other', 'Sports', 'Entertainment'
]

BATCH_SIZE    = 3      # articles per API call
SNIPPET_LEN   = 500    # chars per article

# Model rotation — each has independent daily quota of 1500 RPD
# Order = priority (best quality first)
MODELS = [
    'gemini-2.0-flash',
    'gemini-1.5-flash',
    'gemini-2.0-flash-lite',
    'gemini-1.5-flash-8b',
]

DAILY_LIMIT_PER_MODEL = 1490   # use 1490 to stay safely under 1500
RPM_DELAY             = 4.5    # seconds between requests (15 RPM safe)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a precise multi-label news classifier.

Classify each article using ONLY these labels:
{', '.join(CATEGORIES)}

CRITICAL RULES FOR BUSINESS LABEL:
- Business = company earnings, IPO, merger/acquisition, stock markets, CEO changes, corporate strategy
- Business is NOT: drug approvals, sports player contracts, AI research papers, crime news
- When unsure about Business, do NOT assign it

OTHER RULES:
- Assign multiple labels only when genuinely applicable
- Every article needs at least 1 label
- Return ONLY a valid JSON array of arrays, nothing else

EXAMPLE OUTPUT for 3 articles:
[["Politics", "Economy"], ["Sports"], ["Technology", "Business"]]"""

# ── Daily quota tracker ───────────────────────────────────────────────────────
class QuotaTracker:
    """
    Tracks requests per model per day.
    Persists to disk so it survives script restarts.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data     = self._load()

    def _load(self) -> dict:
        today = str(date.today())
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                data = json.load(f)
            # Reset if it's a new day
            if data.get('date') != today:
                log.info(f"New day detected — resetting quota counters")
                return {'date': today, 'counts': {m: 0 for m in MODELS}}
            return data
        return {'date': today, 'counts': {m: 0 for m in MODELS}}

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def use(self, model: str) -> bool:
        """Record one request. Returns False if daily limit exceeded."""
        count = self.data['counts'].get(model, 0)
        if count >= DAILY_LIMIT_PER_MODEL:
            return False
        self.data['counts'][model] = count + 1
        self.save()
        return True

    def remaining(self, model: str) -> int:
        return max(0, DAILY_LIMIT_PER_MODEL - self.data['counts'].get(model, 0))

    def get_available_model(self) -> str | None:
        """Returns first model with remaining quota, None if all exhausted."""
        for model in MODELS:
            if self.remaining(model) > 0:
                return model
        return None

    def status(self):
        log.info("Daily quota status:")
        total_remaining = 0
        for m in MODELS:
            used = self.data['counts'].get(m, 0)
            rem  = self.remaining(m)
            total_remaining += rem
            log.info(f"  {m:<30} used={used:>4}  remaining={rem:>4}")
        log.info(f"  Total remaining today: {total_remaining}")
        return total_remaining


# ── Gemini client ─────────────────────────────────────────────────────────────

def setup_client() -> genai.Client:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "Set API key:\n"
            "  Windows: set GOOGLE_API_KEY=AIza...\n"
            "  Mac/Linux: export GOOGLE_API_KEY=AIza...\n"
            "  Get key at: https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


# ── Label validation ──────────────────────────────────────────────────────────

def validate_batch_response(raw: str, n: int) -> list[list[str]] | None:
    try:
        clean = raw.strip()
        # Strip markdown fences
        if '```' in clean:
            for part in clean.split('```'):
                part = part.replace('json', '').strip()
                if part.startswith('['):
                    clean = part
                    break

        parsed = json.loads(clean)

        # Handle flat list (single article response unwrapped)
        if parsed and isinstance(parsed[0], str):
            parsed = [parsed]

        if not isinstance(parsed, list):
            return None

        # Pad/trim to expected count
        while len(parsed) < n:
            parsed.append(['Science/Other'])
        parsed = parsed[:n]

        result = []
        for item in parsed:
            if not isinstance(item, list):
                result.append(['Science/Other'])
                continue
            valid = [str(l).strip() for l in item if str(l).strip() in CATEGORIES]
            result.append(valid if valid else ['Science/Other'])

        return result

    except Exception:
        return None


def fallback_labels(rows: list[dict]) -> list[list[str]]:
    """Use source section as fallback when all models fail."""
    return [
        [r['source_section']] if r.get('source_section') in CATEGORIES
        else ['Science/Other']
        for r in rows
    ]


# ── Batch labeling with model rotation ───────────────────────────────────────

def label_batch(
    client:  genai.Client,
    quota:   QuotaTracker,
    rows:    list[dict],
) -> tuple[list[list[str]], str]:
    """
    Labels a batch of articles.
    Automatically rotates to next available model if current is exhausted.
    Returns (labels, model_used).
    """
    # Format prompt
    articles_text = '\n\n'.join([
        f"Article {i+1}:\nTitle: {str(r['title'])[:200]}\n"
        f"Text: {str(r['text'])[:SNIPPET_LEN]}"
        for i, r in enumerate(rows)
    ])
    user_prompt = (
        f"Classify these {len(rows)} articles.\n\n"
        f"{articles_text}\n\n"
        f"Return a JSON array of {len(rows)} label arrays."
    )

    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        # Get available model
        model_name = quota.get_available_model()
        if model_name is None:
            log.error("ALL MODELS EXHAUSTED for today. Run again tomorrow.")
            log.info("Quota resets at midnight Pacific Time (UTC-8).")
            quota.status()
            return fallback_labels(rows), 'fallback_quota_exhausted'

        # Check quota allows this request
        if not quota.use(model_name):
            continue  # try next model

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.05,
                    max_output_tokens=300,
                    response_mime_type='application/json',
                ),
            )

            raw    = response.text.strip() if response.text else ''
            labels = validate_batch_response(raw, len(rows))

            if labels is not None:
                return labels, model_name

            log.warning(f"Validation failed on {model_name}, attempt {attempt+1}")

        except Exception as e:
            err = str(e).lower()

            if 'resource_exhausted' in err or '429' in err or 'quota' in err:
                # This model's daily quota is truly exhausted
                log.warning(f"{model_name} daily quota exhausted — "
                            f"marking as used up and switching model")
                # Force exhaust this model in tracker so we skip it
                self_exhausted = DAILY_LIMIT_PER_MODEL - quota.data['counts'].get(model_name, 0)
                quota.data['counts'][model_name] = DAILY_LIMIT_PER_MODEL
                quota.save()
                # Don't sleep long — just switch model immediately
                continue

            elif 'rate' in err or 'per_minute' in err:
                # Per-minute rate limit — wait and retry same model
                log.warning(f"Per-minute rate limit on {model_name}. Waiting 65s...")
                time.sleep(65)
                # Refund the quota use since we'll retry
                quota.data['counts'][model_name] = max(
                    0, quota.data['counts'].get(model_name, 0) - 1
                )
                quota.save()

            elif '401' in err or 'api_key' in err or 'unauthorized' in err:
                raise EnvironmentError("Invalid GOOGLE_API_KEY")

            else:
                wait = [5, 15, 30][min(attempt, 2)]
                log.warning(f"Error attempt {attempt+1}: {str(e)[:80]}. "
                            f"Waiting {wait}s...")
                time.sleep(wait)

    return fallback_labels(rows), 'fallback_retry_exhausted'


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> tuple[set, list]:
    if not os.path.exists(CHECKPOINT_CSV):
        return set(), []
    ckpt = pd.read_csv(CHECKPOINT_CSV)
    urls = set(ckpt['url'].tolist())
    results = ckpt[['url', 'llm_labels', 'label_method']].to_dict('records')
    log.info(f"Resuming from checkpoint: {len(urls)} already labeled")
    return urls, results


def save_checkpoint(df_orig: pd.DataFrame, results: list):
    result_df = pd.DataFrame(results)
    merged = df_orig.merge(
        result_df[['url', 'llm_labels', 'label_method']],
        on='url', how='inner'
    )
    merged.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8')


# ── Main ─────────────────────────────────────────────────────────────────────

def run_labeling():
    client = setup_client()
    quota  = QuotaTracker(QUOTA_FILE)

    quota.status()
    total_remaining = sum(quota.remaining(m) for m in MODELS)
    if total_remaining == 0:
        log.error("All daily quotas exhausted. Wait until midnight Pacific Time (UTC-8).")
        return None

    df = pd.read_csv(INPUT_CSV)
    log.info(f"Loaded {len(df)} articles")

    labeled_urls, results = load_checkpoint()
    to_label = df[~df['url'].isin(labeled_urls)].reset_index(drop=True)
    log.info(f"To label: {len(to_label)} (already done: {len(labeled_urls)})")

    if len(to_label) == 0:
        log.info("All articles already labeled. Saving final output.")
    else:
        total     = len(to_label)
        start     = datetime.now()
        batch_num = 0

        for batch_start in range(0, total, BATCH_SIZE):
            batch_rows  = to_label.iloc[batch_start:batch_start+BATCH_SIZE]
            batch_dicts = batch_rows.to_dict('records')
            batch_num  += 1

            labels_batch, model_used = label_batch(client, quota, batch_dicts)

            if 'quota_exhausted' in model_used:
                log.warning("All models exhausted mid-run. "
                            "Saving progress. Run again tomorrow.")
                save_checkpoint(df, results)
                quota.status()
                return None

            for row_dict, labels in zip(batch_dicts, labels_batch):
                results.append({
                    'url':          row_dict['url'],
                    'llm_labels':   labels,
                    'label_method': f'gemini_{model_used}',
                })

            done = min(batch_start + BATCH_SIZE, total)
            if batch_num % 10 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate    = done / elapsed if elapsed > 0 else 0.001
                eta     = (total - done) / rate / 60 if rate > 0 else 0
                log.info(
                    f"Progress: {done}/{total} | "
                    f"Model: {model_used} | "
                    f"Speed: {rate:.2f} art/s | "
                    f"ETA: {eta:.1f} min"
                )
                save_checkpoint(df, results)

            time.sleep(RPM_DELAY)

    # ── Final output ──────────────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    final_df  = df.merge(
        result_df[['url', 'llm_labels', 'label_method']],
        on='url', how='inner'
    )

    final_df['llm_labels'] = final_df['llm_labels'].apply(
        lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
    )
    for cat in CATEGORIES:
        final_df[cat] = final_df['llm_labels'].apply(lambda x: 1 if cat in x else 0)

    final_df = final_df[final_df[CATEGORIES].sum(axis=1) > 0]
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    log.info(f"\n{'='*50}")
    log.info(f"DONE: {len(final_df)} articles labeled")
    log.info(f"Label distribution:")
    counts = final_df[CATEGORIES].sum().sort_values(ascending=False)
    for cat, count in counts.items():
        log.info(f"  {cat:<25} {count} ({count/len(final_df)*100:.1f}%)")

    methods = final_df['label_method'].value_counts()
    log.info(f"\nModel usage:\n{methods.to_string()}")

    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)
    if os.path.exists(QUOTA_FILE):
        os.remove(QUOTA_FILE)

    return final_df


if __name__ == '__main__':
    df = run_labeling()