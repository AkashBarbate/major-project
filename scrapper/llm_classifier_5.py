"""
gemini_master_labeler_v2.py
---------------------------
Updated based on actual AI Studio dashboard limits.

Free tier capacity (from dashboard):
  gemini-3.1-flash-lite : RPD 500,   RPM 15  ← Primary (best Gemini quality)
  gemma-4-31b-it        : RPD 1500,  RPM 15  ← Secondary (high capacity)
  gemma-3-27b-it        : RPD 14400, RPM 30  ← Tertiary (massive backup)
  gemini-2.5-flash      : RPD 20,    RPM 5   ← Last resort
  gemini-2.5-flash-lite : RPD 20,    RPM 10  ← Last resort

Total daily capacity: ~16,440 calls
Calls needed for 3019 articles (batch=3): ~1,007
Estimated completion: 75-90 minutes in one run

Key differences from v1:
  - Correct model names/RPDs from actual dashboard
  - Gemma models use JSON prompt (no Pydantic schema — not supported)
  - Gemini models use Pydantic response_schema (guaranteed structure)
  - Per-model RPM delay to avoid minute-level rate limits
  - Fixed fallback category mapping for 8-category system
"""

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from pydantic import BaseModel
import json, os, ast, time, sys, logging, re, html
import pandas as pd
from datetime import datetime, date

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('labeling_v2.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_CSV      = 'cleaned_articles.csv'
OUTPUT_CSV     = 'gemini_labeled_articles.csv'
CHECKPOINT_CSV = 'checkpoint_v2.csv'
QUOTA_FILE     = 'quota_tracker_v2.json'

BATCH_SIZE  = 3
SNIPPET_LEN = 1000    # chars per article sent to API

# 8 consolidated categories — reduces label confusion
CATEGORIES = [
    'Politics',
    'Science and Technology',
    'Business and Economy',
    'International Affairs',
    'Education',
    'Health',
    'Sports',
    'Entertainment'
]

# Mapping from old source_section values → new 8 categories
# Used for fallback when API fails
SOURCE_TO_CATEGORY = {
    'Politics':             'Politics',
    'Technology':           'Science and Technology',
    'Business':             'Business and Economy',
    'Economy':              'Business and Economy',
    'International_Affairs':'International Affairs',
    'Education':            'Education',
    'Health':               'Health',
    'Science/Other':        'Science and Technology',
    'Sports':               'Sports',
    'Entertainment':        'Entertainment',
}

# ── Model config based on ACTUAL dashboard limits ──────────────────────────────
# priority: lower = tried first
# supports_schema: True = use Pydantic response_schema
#                  False = use JSON prompt + manual parse (Gemma models)
# rpm_delay: seconds to wait between calls to respect RPM limit safely
MODELS_CONFIG = {
    'gemini-3.1-flash-lite': {
        'rpd': 500,
        'rpm': 15,
        'rpm_delay': 4.5,       # 60s/15rpm = 4s, use 4.5 for safety
        'supports_schema': True,
        'priority': 1,
    },
    'gemma-4-31b-it': {
        'rpd': 1500,
        'rpm': 15,
        'rpm_delay': 4.5,
        'supports_schema': False,  # Gemma: use JSON prompt
        'priority': 2,
    },
    'gemma-3-27b-it': {
        'rpd': 14400,
        'rpm': 30,
        'rpm_delay': 2.5,       # 60s/30rpm = 2s, use 2.5 for safety
        'supports_schema': False,
        'priority': 3,
    },
    'gemini-2.5-flash': {
        'rpd': 20,
        'rpm': 5,
        'rpm_delay': 13.0,      # 60s/5rpm = 12s, use 13 for safety
        'supports_schema': True,
        'priority': 4,
    },
    'gemini-2.5-flash-lite': {
        'rpd': 20,
        'rpm': 10,
        'rpm_delay': 7.0,       # 60s/10rpm = 6s, use 7 for safety
        'supports_schema': True,
        'priority': 5,
    },
}

# Sorted by priority — used for model selection
MODELS_BY_PRIORITY = sorted(MODELS_CONFIG.keys(), key=lambda m: MODELS_CONFIG[m]['priority'])

# ── Pydantic schema for Gemini models ─────────────────────────────────────────
class ArticleLabel(BaseModel):
    labels: list[str]

class BatchResponse(BaseModel):
    articles: list[ArticleLabel]

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a precise multi-label news classifier for a research project.

Classify each article using ONLY these 8 labels:
{', '.join(CATEGORIES)}

LABEL DEFINITIONS:
- Politics: government policy, elections, parliament, ministers, political parties, legislation
- Science and Technology: AI/ML, software, gadgets, space missions, environment, scientific research, medical technology
- Business and Economy: company earnings, IPO, mergers, stock markets, GDP, inflation, RBI policy, fiscal deficit
- International Affairs: diplomacy, bilateral relations, sanctions, UN/NATO, geopolitical events between countries
- Education: schools, universities, board exams, NEET/JEE, UGC, student/teacher issues, curriculum
- Health: disease, vaccines, hospitals, public health, mental health, medical research, nutrition
- Sports: cricket, football, Olympics, IPL, tournaments, athletes, sports governance
- Entertainment: films, OTT/streaming, music, celebrity news, box office, awards, TV shows

CRITICAL RULES:
1. Maximum 2 labels per article — assign the most dominant topics only
2. Every article must get at least 1 valid label
3. Business and Economy covers BOTH corporate news AND macro-economic news
4. Science and Technology covers BOTH tech products AND scientific research
5. International Affairs only when article is ABOUT relations between countries"""

# Gemma-specific prompt (no schema support — plain JSON instruction)
GEMMA_PROMPT_TEMPLATE = """Classify these {n} news articles. 

Valid labels (use ONLY these exact strings):
{categories}

Rules:
- Maximum 2 labels per article
- At least 1 label per article  
- Return ONLY valid JSON, nothing else

Return this exact JSON structure:
{{"articles": [{{"labels": ["Label1"]}}, {{"labels": ["Label1", "Label2"]}}, ...]}}

Articles to classify:

{articles_text}"""


# ── Text preprocessing ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip HTML, URLs, normalize whitespace — reduces tokens."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_articles(rows: list[dict]) -> str:
    """Format batch of articles into numbered text block."""
    parts = []
    for i, row in enumerate(rows, 1):
        title = clean_text(str(row.get('title', '')))[:200]
        text  = clean_text(str(row.get('text',  '')))[:SNIPPET_LEN]
        parts.append(f"Article {i}:\nTitle: {title}\nText: {text}")
    return '\n\n'.join(parts)


# ── Quota tracker ──────────────────────────────────────────────────────────────
class QuotaTracker:
    """
    Tracks requests per model per day.
    Persists to disk — survives script restarts.
    Resets automatically at start of each new day.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data     = self._load()

    def _load(self) -> dict:
        today = str(date.today())
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                data = json.load(f)
            if data.get('date') != today:
                log.info("New day — resetting all quota counters")
                return {'date': today, 'counts': {m: 0 for m in MODELS_CONFIG}}
            return data
        return {'date': today, 'counts': {m: 0 for m in MODELS_CONFIG}}

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def remaining(self, model: str) -> int:
        return max(0, MODELS_CONFIG[model]['rpd'] - self.data['counts'].get(model, 0))

    def use(self, model: str) -> bool:
        """Consume one request from quota. Returns False if exhausted."""
        if self.remaining(model) == 0:
            return False
        self.data['counts'][model] = self.data['counts'].get(model, 0) + 1
        self.save()
        return True

    def refund(self, model: str):
        """Return one request back — called on rate limit retry."""
        self.data['counts'][model] = max(0, self.data['counts'].get(model, 0) - 1)
        self.save()

    def exhaust(self, model: str):
        """Mark model as fully used — called on 404 or permanent failure."""
        self.data['counts'][model] = MODELS_CONFIG[model]['rpd']
        self.save()

    def get_best_model(self) -> str | None:
        """Return highest-priority model that still has quota."""
        for model in MODELS_BY_PRIORITY:
            if self.remaining(model) > 0:
                return model
        return None

    def total_remaining(self) -> int:
        return sum(self.remaining(m) for m in MODELS_CONFIG)

    def print_status(self):
        log.info("\nQuota status:")
        log.info(f"  {'Model':<30} {'Used':>6} {'Remaining':>10} {'Type'}")
        log.info(f"  {'-'*60}")
        for model in MODELS_BY_PRIORITY:
            cfg   = MODELS_CONFIG[model]
            used  = self.data['counts'].get(model, 0)
            rem   = self.remaining(model)
            mtype = 'Gemini' if cfg['supports_schema'] else 'Gemma'
            log.info(f"  {model:<30} {used:>6} {rem:>10}  {mtype}")
        log.info(f"  Total remaining: {self.total_remaining()}")


# ── Label parsing & validation ─────────────────────────────────────────────────
def validate_labels(raw_labels: list) -> list[str]:
    """Keep only valid category strings, max 2, min 1."""
    valid = [str(l).strip() for l in raw_labels if str(l).strip() in CATEGORIES]
    return valid[:2] if valid else []


def parse_gemma_response(raw: str, n: int) -> list[list[str]] | None:
    """
    Parse Gemma's plain-text JSON response.
    Expected: {"articles": [{"labels": [...]}, ...]}
    """
    try:
        clean = raw.strip()

        # Strip markdown fences if present
        if '```' in clean:
            for part in clean.split('```'):
                part = part.strip().lstrip('json').strip()
                if part.startswith('{') or part.startswith('['):
                    clean = part
                    break

        parsed = json.loads(clean)

        # Handle {"articles": [...]} wrapper
        if isinstance(parsed, dict) and 'articles' in parsed:
            parsed = parsed['articles']

        # Handle flat list of label lists (no wrapper)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict) and 'labels' in item:
                    result.append(validate_labels(item['labels']))
                elif isinstance(item, list):
                    result.append(validate_labels(item))
                elif isinstance(item, str) and item in CATEGORIES:
                    result.append([item])
                else:
                    result.append([])

            # Pad to expected count
            while len(result) < n:
                result.append([])
            return result[:n]

    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        pass

    return None


def fallback_for_batch(rows: list[dict]) -> list[list[str]]:
    """
    When all API retries fail, use source_section → 8-category mapping.
    Never returns empty — always at least one valid label.
    """
    result = []
    for row in rows:
        src  = row.get('source_section', '')
        cat  = SOURCE_TO_CATEGORY.get(src, 'Science and Technology')
        result.append([cat])
    return result


# ── Client setup ───────────────────────────────────────────────────────────────
def setup_client() -> genai.Client:
    api_key = os.environ.get('GOOGLE_API_KEY', '').strip()
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found.\n"
            "Add it to your .env file:\n"
            "GOOGLE_API_KEY=AIza..."
        )
    return genai.Client(api_key=api_key)


# ── Single batch labeling ──────────────────────────────────────────────────────
def label_batch_gemini(
    client:     genai.Client,
    model_name: str,
    rows:       list[dict],
) -> list[list[str]] | None:
    """
    Label using Gemini model with Pydantic response_schema.
    Returns validated label lists or None on failure.
    """
    response = client.models.generate_content(
        model=model_name,
        contents=f"Classify these {len(rows)} articles.\n\n{format_articles(rows)}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.05,
            max_output_tokens=300,
            response_mime_type='application/json',
            response_schema=BatchResponse,
        ),
    )

    if not response.parsed or not response.parsed.articles:
        return None

    result = []
    for art in response.parsed.articles:
        valid = validate_labels(art.labels)
        result.append(valid if valid else [SOURCE_TO_CATEGORY.get(
            rows[len(result)].get('source_section', ''), 'Science and Technology'
        )])

    while len(result) < len(rows):
        result.append(['Science and Technology'])

    return result[:len(rows)]


def label_batch_gemma(
    client:     genai.Client,
    model_name: str,
    rows:       list[dict],
) -> list[list[str]] | None:
    """
    Label using Gemma model with plain JSON prompt (no schema support).
    Returns validated label lists or None on failure.
    """
    prompt = GEMMA_PROMPT_TEMPLATE.format(
        n=len(rows),
        categories=', '.join(CATEGORIES),
        articles_text=format_articles(rows),
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.05,
            max_output_tokens=400,
        ),
    )

    if not response.text:
        return None

    return parse_gemma_response(response.text, len(rows))


def label_batch(
    client:    genai.Client,
    quota:     QuotaTracker,
    rows:      list[dict],
    batch_num: int,
) -> tuple[list[list[str]], str]:
    """
    Main entry point for labeling one batch.
    Handles model selection, rotation, retries, and all error cases.
    Returns (labels, model_used_or_fallback_reason).
    """
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        model_name = quota.get_best_model()

        if model_name is None:
            log.error("ALL MODELS EXHAUSTED. Run again tomorrow after midnight PT.")
            quota.print_status()
            return fallback_for_batch(rows), 'fallback_quota_exhausted'

        if not quota.use(model_name):
            continue

        cfg            = MODELS_CONFIG[model_name]
        supports_schema = cfg['supports_schema']

        try:
            if supports_schema:
                result = label_batch_gemini(client, model_name, rows)
            else:
                result = label_batch_gemma(client, model_name, rows)

            if result is not None:
                # Fill any empty label lists with source-based fallback
                for i, labels in enumerate(result):
                    if not labels:
                        src = rows[i].get('source_section', '')
                        result[i] = [SOURCE_TO_CATEGORY.get(src, 'Science and Technology')]
                return result, model_name

            log.warning(f"Batch {batch_num}: Parse failed on {model_name} "
                        f"attempt {attempt+1}")

        except Exception as e:
            err = str(e).lower()

            # Model doesn't exist or isn't accessible
            if '404' in err or 'not found' in err or 'model' in err and 'not' in err:
                log.warning(f"{model_name}: Not accessible (404). "
                            f"Removing from rotation.")
                quota.exhaust(model_name)
                continue

            # Daily quota exhausted (from API side)
            elif 'resource_exhausted' in err or ('quota' in err and 'exceeded' in err):
                log.warning(f"{model_name}: Daily quota exhausted by API. "
                            f"Marking as used up.")
                quota.exhaust(model_name)
                continue

            # Per-minute rate limit — wait and retry same model
            elif '429' in err or 'too many requests' in err or 'rate' in err:
                wait = 65
                log.warning(f"{model_name}: Per-minute rate limit. "
                            f"Waiting {wait}s then retrying...")
                quota.refund(model_name)
                time.sleep(wait)
                # Don't increment attempt — rate limit isn't a real failure

            # Transient server error
            elif '500' in err or '503' in err or 'unavailable' in err:
                wait = [5, 15, 30][min(attempt, 2)]
                log.warning(f"{model_name}: Server error attempt {attempt+1}. "
                            f"Waiting {wait}s... | {str(e)[:60]}")
                time.sleep(wait)

            # Network error
            elif 'getaddrinfo' in err or 'connection' in err or 'timeout' in err:
                wait = [10, 20, 40][min(attempt, 2)]
                log.warning(f"{model_name}: Network error. Waiting {wait}s...")
                time.sleep(wait)

            # Unknown error
            else:
                wait = [5, 10, 20][min(attempt, 2)]
                log.warning(f"{model_name}: Unknown error attempt {attempt+1}: "
                            f"{str(e)[:80]}. Waiting {wait}s...")
                time.sleep(wait)

    log.error(f"Batch {batch_num}: All retries exhausted. Using fallback.")
    return fallback_for_batch(rows), 'fallback_retry_exhausted'


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def load_checkpoint() -> tuple[set, list]:
    if not os.path.exists(CHECKPOINT_CSV):
        return set(), []
    try:
        ckpt    = pd.read_csv(CHECKPOINT_CSV)
        urls    = set(ckpt['url'].tolist())
        results = ckpt[['url', 'llm_labels', 'label_method']].to_dict('records')
        log.info(f"Checkpoint loaded: {len(urls)} articles already labeled")
        return urls, results
    except Exception as e:
        log.warning(f"Could not load checkpoint: {e}. Starting fresh.")
        return set(), []


def save_checkpoint(df_orig: pd.DataFrame, results: list):
    if not results:
        return
    try:
        result_df = pd.DataFrame(results)
        merged    = df_orig.merge(
            result_df[['url', 'llm_labels', 'label_method']],
            on='url', how='inner'
        )
        merged.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8')
    except Exception as e:
        log.warning(f"Checkpoint save failed: {e}")


# ── Final output processing ────────────────────────────────────────────────────
def build_output(df_orig: pd.DataFrame, results: list) -> pd.DataFrame:
    result_df = pd.DataFrame(results)
    final_df  = df_orig.merge(
        result_df[['url', 'llm_labels', 'label_method']],
        on='url', how='inner'
    )

    # Parse label lists from string representation
    final_df['llm_labels'] = final_df['llm_labels'].apply(
        lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else
                  (x if isinstance(x, list) else ['Science and Technology'])
    )

    # Binary columns for training
    for cat in CATEGORIES:
        final_df[cat] = final_df['llm_labels'].apply(
            lambda x: 1 if cat in x else 0
        )

    # Drop articles with no valid labels (safety check)
    valid_mask = final_df[CATEGORIES].sum(axis=1) > 0
    dropped    = (~valid_mask).sum()
    if dropped > 0:
        log.warning(f"Dropping {dropped} articles with no valid labels")
    final_df = final_df[valid_mask].reset_index(drop=True)
    return final_df


def print_report(df: pd.DataFrame):
    total = len(df)
    log.info(f"\n{'='*55}")
    log.info(f"LABELING COMPLETE — {total} articles")
    log.info(f"{'='*55}")

    log.info(f"\nLabel distribution:")
    counts = df[CATEGORIES].sum().sort_values(ascending=False)
    for cat, count in counts.items():
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        log.info(f"  {cat:<28} {count:>4} ({pct:>5.1f}%)  {bar}")

    lpa = df[CATEGORIES].sum(axis=1)
    log.info(f"\nLabels per article:")
    log.info(f"  Mean: {lpa.mean():.2f} | "
             f"Single: {(lpa==1).sum()} | "
             f"Multi: {(lpa>1).sum()}")

    log.info(f"\nModel usage:")
    for method, cnt in df['label_method'].value_counts().items():
        log.info(f"  {method:<35} {cnt}")

    fallback_count = df['label_method'].str.startswith('fallback').sum()
    if fallback_count > 0:
        pct = fallback_count / total * 100
        log.warning(f"\nFallback-labeled articles: {fallback_count} ({pct:.1f}%)")
        if pct > 5:
            log.warning("High fallback rate — consider re-labeling these articles")


# ── Main ───────────────────────────────────────────────────────────────────────
def run_labeling():
    client = setup_client()
    quota  = QuotaTracker(QUOTA_FILE)

    quota.print_status()

    if quota.total_remaining() == 0:
        log.error("All daily quotas exhausted. Resets at midnight Pacific Time.")
        return None

    # Load data
    df = pd.read_csv(INPUT_CSV)
    log.info(f"\nLoaded {len(df)} articles from {INPUT_CSV}")

    # Cost/time estimate
    labeled_urls, results = load_checkpoint()
    to_label = df[~df['url'].isin(labeled_urls)].reset_index(drop=True)
    n_batches = (len(to_label) + BATCH_SIZE - 1) // BATCH_SIZE
    primary_delay = MODELS_CONFIG['gemini-3.1-flash-lite']['rpm_delay']
    eta_min = (n_batches * primary_delay) / 60

    log.info(f"To label: {len(to_label)} | Already done: {len(labeled_urls)}")
    log.info(f"Estimated time: ~{eta_min:.0f} minutes "
             f"(at {primary_delay}s delay/call)")

    if len(to_label) == 0:
        log.info("All articles already labeled. Building final output...")
    else:
        total     = len(to_label)
        start     = datetime.now()
        batch_num = 0
        current_model = None  # track current model for delay

        for batch_start in range(0, total, BATCH_SIZE):
            batch_rows  = to_label.iloc[batch_start : batch_start + BATCH_SIZE]
            batch_dicts = batch_rows.to_dict('records')
            batch_num  += 1

            labels, model_used = label_batch(
                client, quota, batch_dicts, batch_num
            )

            # If quota fully exhausted mid-run, save and exit
            if 'quota_exhausted' in model_used:
                log.warning("Quota exhausted mid-run. Progress saved.")
                save_checkpoint(df, results)
                return None

            for row_dict, label_list in zip(batch_dicts, labels):
                results.append({
                    'url':          row_dict['url'],
                    'llm_labels':   str(label_list),
                    'label_method': model_used,
                })

            # Progress report every 15 batches (45 articles)
            articles_done = min(batch_start + BATCH_SIZE, total)
            if batch_num % 15 == 0 or articles_done == total:
                elapsed = (datetime.now() - start).total_seconds()
                rate    = articles_done / elapsed if elapsed > 0 else 0.001
                eta     = (total - articles_done) / rate / 60
                log.info(
                    f"Progress: {articles_done}/{total} | "
                    f"Model: {model_used} | "
                    f"Speed: {rate:.2f} art/s | "
                    f"ETA: {eta:.1f} min"
                )
                save_checkpoint(df, results)

            # Use delay appropriate for current model's RPM limit
            delay = MODELS_CONFIG.get(model_used, {}).get('rpm_delay', 4.5)
            time.sleep(delay)

    # Build and save final output
    log.info("\nBuilding final labeled dataset...")
    final_df = build_output(df, results)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    print_report(final_df)

    # Cleanup
    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)

    log.info(f"\nSaved → {OUTPUT_CSV}")
    log.info("Next step: python train_improved.py")
    return final_df


if __name__ == '__main__':
    df = run_labeling()