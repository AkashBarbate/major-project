"""
gemini_master_labeler.py
------------------------
Optimized for Google GenAI SDK with Pydantic Structured Outputs.
Features: 
1. Guaranteed JSON formatting via response_schema
2. Pre-processing for HTML/URL stripping
3. Dynamic quota tracking for current Gemini 2.5/1.5 Free Tier limits
"""

from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from pydantic import BaseModel
import json, os, ast, time, sys, logging, re, html
import pandas as pd
from datetime import datetime, date

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('labeling_master.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_CSV      = 'cleaned_articles.csv'
OUTPUT_CSV     = 'gemini_labeled_articles.csv'
CHECKPOINT_CSV = 'labeling_checkpoint_master.csv'
QUOTA_FILE     = 'daily_quota_tracker.json'

# Consolidated to 8 Categories to prevent the "Waterbed Effect"
CATEGORIES = [
    'Politics', 'Science and Technology', 'Business and Economy',
    'International_Affairs', 'Education', 'Health',
    'Sports', 'Entertainment'
]

BATCH_SIZE  = 3      
SNIPPET_LEN = 1200     # Increased since we are cleaning text
RPM_DELAY   = 4.5      

# Updated Free Tier limits (April 2026 configuration)
# Updated Free Tier limits (April 2026 configuration)
# Updated Free Tier limits (April 2026 generation)
# The final, stable configuration based on your AI Studio dashboard
MODELS_CONFIG = {'gemini-1.5-flash': {'rpd': 1490},
    'gemini-1.5-flash-8b': {'rpd': 1490},
    'gemini-3.1-flash-lite': {'rpd': 500},
    'gemini-3.0-flash': {'rpd': 20},
    'gemini-2.5-flash-lite': {'rpd': 20},
    'gemini-2.5-flash': {'rpd': 20},
    
}

# ── Structured Outputs (Pydantic) ──────────────────────────────────────────────
# This replaces complex regex parsing. The API is forced to return this exact shape.
class ArticleLabel(BaseModel):
    labels: list[str]

class BatchResponse(BaseModel):
    articles: list[ArticleLabel]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a precise multi-label news classifier.
Classify each article using ONLY these labels: {', '.join(CATEGORIES)}

CRITICAL RULES:
1. Assign multiple labels only when genuinely applicable.
2. Every article needs at least 1 valid label.
3. 'Business and Economy' = company earnings, IPOs, markets, corporate strategy, fiscal policy.
4. 'Science and Technology' = software, AI, gadgets, medical research, space.
"""

# ── Text Preprocessing ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Removes HTML, URLs, and normalizes spacing to save tokens."""
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── Daily quota tracker ────────────────────────────────────────────────────────
class QuotaTracker:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data     = self._load()

    def _load(self) -> dict:
        today = str(date.today())
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                data = json.load(f)
            if data.get('date') != today:
                log.info("New day detected — resetting quota counters")
                return {'date': today, 'counts': {m: 0 for m in MODELS_CONFIG.keys()}}
            return data
        return {'date': today, 'counts': {m: 0 for m in MODELS_CONFIG.keys()}}

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def use(self, model: str) -> bool:
        count = self.data['counts'].get(model, 0)
        limit = MODELS_CONFIG[model]['rpd']
        if count >= limit:
            return False
        self.data['counts'][model] = count + 1
        self.save()
        return True

    def get_available_model(self) -> str | None:
        for model, config in MODELS_CONFIG.items():
            if self.data['counts'].get(model, 0) < config['rpd']:
                return model
        return None

    def status(self):
        log.info("Daily quota status:")
        total_remaining = 0
        for m, config in MODELS_CONFIG.items():
            used = self.data['counts'].get(m, 0)
            rem  = max(0, config['rpd'] - used)
            total_remaining += rem
            log.info(f"  {m:<25} used={used:>4}  remaining={rem:>4}")
        log.info(f"  Total remaining today: {total_remaining}")
        return total_remaining

# ── Client Setup ───────────────────────────────────────────────────────────────
def setup_client() -> genai.Client:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is missing from environment/dotenv.")
    return genai.Client(api_key=api_key)

def fallback_labels(rows: list[dict]) -> list[list[str]]:
    return [[r.get('source_section', 'Science/Other')] if r.get('source_section') in CATEGORIES else ['Science/Other'] for r in rows]

# ── Batch labeling ─────────────────────────────────────────────────────────────
def label_batch(client: genai.Client, quota: QuotaTracker, rows: list[dict]) -> tuple[list[list[str]], str]:
    articles_text = '\n\n'.join([
        f"Article {i+1}:\nTitle: {clean_text(str(r['title']))[:200]}\nText: {clean_text(str(r['text']))[:SNIPPET_LEN]}"
        for i, r in enumerate(rows)
    ])
    user_prompt = f"Classify these {len(rows)} articles.\n\n{articles_text}"

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        model_name = quota.get_available_model()
        if not model_name:
            log.error("ALL MODELS EXHAUSTED for today. Run again tomorrow.")
            return fallback_labels(rows), 'fallback_quota_exhausted'

        if not quota.use(model_name):
            continue 

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.05,
                    response_mime_type='application/json',
                    response_schema=BatchResponse, # Magic bullet: Enforces Pydantic schema
                ),
            )

            # Map the parsed Pydantic objects back to our clean lists
            final_labels = []
            for art in response.parsed.articles:
                valid = [lbl for lbl in art.labels if lbl in CATEGORIES]
                final_labels.append(valid if valid else ['Science/Other'])

            # Pad if the LLM returned fewer articles than requested
            while len(final_labels) < len(rows):
                final_labels.append(['Science/Other'])

            return final_labels[:len(rows)], model_name

        except Exception as e:
            err = str(e).lower()
            
            # 1. Catch 404 - Model doesn't exist or is offline
            if '404' in err or 'not found' in err:
                log.warning(f"Model {model_name} is offline (404). Removing from rotation.")
                quota.data['counts'][model_name] = MODELS_CONFIG[model_name]['rpd']
                quota.save()
                continue
                
            # 2. Catch 429 - Rate Limit or Quota Exhausted
            elif '429' in err or 'exhausted' in err or 'too many requests' in err:
                log.warning(f"Rate limit hit on {model_name}. Waiting 65s...")
                time.sleep(65)
                # Refund the attempt so we can try again after cooldown
                quota.data['counts'][model_name] = max(0, quota.data['counts'].get(model_name, 0) - 1)
                quota.save()
                
            # 3. Standard Network Errors (like your getaddrinfo drop)
            else:
                wait = [5, 15, 30][min(attempt, 2)]
                log.warning(f"API Error on {model_name}: {str(e)[:80]}... Waiting {wait}s")
                time.sleep(wait)

    return fallback_labels(rows), 'fallback_retry_exhausted'

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
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
    merged = df_orig.merge(result_df[['url', 'llm_labels', 'label_method']], on='url', how='inner')
    merged.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8')

# ── Main ───────────────────────────────────────────────────────────────────────
def run_labeling():
    client = setup_client()
    quota  = QuotaTracker(QUOTA_FILE)

    if quota.status() == 0:
        log.error("All daily quotas exhausted. Wait until midnight Pacific Time (UTC-8).")
        return None

    df = pd.read_csv(INPUT_CSV)
    labeled_urls, results = load_checkpoint()
    to_label = df[~df['url'].isin(labeled_urls)].reset_index(drop=True)
    
    log.info(f"Loaded {len(df)} articles. To label: {len(to_label)}")

    if len(to_label) > 0:
        total     = len(to_label)
        start     = datetime.now()
        batch_num = 0

        for batch_start in range(0, total, BATCH_SIZE):
            batch_rows  = to_label.iloc[batch_start:batch_start+BATCH_SIZE]
            batch_dicts = batch_rows.to_dict('records')
            batch_num  += 1

            labels_batch, model_used = label_batch(client, quota, batch_dicts)

            if 'quota_exhausted' in model_used:
                log.warning("All models exhausted mid-run. Saving progress.")
                save_checkpoint(df, results)
                quota.status()
                return None

            for row_dict, labels in zip(batch_dicts, labels_batch):
                results.append({
                    'url':          row_dict['url'],
                    'llm_labels':   str(labels), # Save as string for CSV
                    'label_method': f'gemini_{model_used}',
                })

            done = min(batch_start + BATCH_SIZE, total)
            if batch_num % 10 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate    = done / elapsed if elapsed > 0 else 0.001
                eta     = (total - done) / rate / 60 if rate > 0 else 0
                log.info(f"Progress: {done}/{total} | Model: {model_used} | Speed: {rate:.2f} art/s | ETA: {eta:.1f} min")
                save_checkpoint(df, results)

            time.sleep(RPM_DELAY)

    # ── Final output & Binary Columns ──────────────────────────────────────────
    result_df = pd.DataFrame(results)
    final_df  = df.merge(result_df[['url', 'llm_labels', 'label_method']], on='url', how='inner')

    # Unpack strings back to lists, then create Binary Columns
    final_df['llm_labels'] = final_df['llm_labels'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
    for cat in CATEGORIES:
        final_df[cat] = final_df['llm_labels'].apply(lambda x: 1 if cat in x else 0)

    # Keep only articles that successfully received at least one valid label
    final_df = final_df[final_df[CATEGORIES].sum(axis=1) > 0]
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    log.info(f"\n{'='*50}\nDONE: {len(final_df)} articles labeled\nLabel distribution:")
    counts = final_df[CATEGORIES].sum().sort_values(ascending=False)
    for cat, count in counts.items():
        log.info(f"  {cat:<25} {count} ({count/len(final_df)*100:.1f}%)")

    log.info(f"\nModel usage:\n{final_df['label_method'].value_counts().to_string()}")

    if os.path.exists(CHECKPOINT_CSV): os.remove(CHECKPOINT_CSV)
    if os.path.exists(QUOTA_FILE): os.remove(QUOTA_FILE)

    return final_df

if __name__ == '__main__':
    df = run_labeling()