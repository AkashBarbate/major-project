"""
push_to_hf.py
-------------
Push the trained RoBERTa model, tokenizer, and credibility regressor
to a Hugging Face Hub repository.

Prerequisites:
    pip install huggingface_hub

Usage:
    1. Create a free account at https://huggingface.co
    2. Create a new model repo at https://huggingface.co/new
    3. Get your access token from https://huggingface.co/settings/tokens
    4. Run: python push_to_hf.py
"""

from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

load_dotenv()
# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Update these before running
# ══════════════════════════════════════════════════════════════════════════════
HF_REPO_ID = "Akash-barbate/news-classifier-roberta"   # <-- change to your HF username/repo
HF_TOKEN   = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # loaded from .env file

# Local paths to upload
MODEL_DIR       = "models/improved_model/best_model"
TOKENIZER_DIR   = "models/improved_model/best_tokenizer"
CRED_REGRESSOR  = "models/credibility_regressor.pkl"

# ══════════════════════════════════════════════════════════════════════════════


def main():
    # --- Login ---
    token = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        print("No token found. Logging in interactively...")
        login()
    else:
        login(token=token)

    api = HfApi()

    # --- Create repo (if it doesn't exist) ---
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        exist_ok=True,  # won't error if already exists
    )
    print(f"✓ Repo ready: https://huggingface.co/{HF_REPO_ID}")

    # --- Upload model weights ---
    print("\n📦 Uploading model weights (model.safetensors + config.json)...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        path_in_repo="model",        # stored under model/ in the HF repo
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print("✓ Model weights uploaded.")

    # --- Upload tokenizer ---
    print("\n📦 Uploading tokenizer (tokenizer.json + tokenizer_config.json)...")
    api.upload_folder(
        folder_path=TOKENIZER_DIR,
        path_in_repo="tokenizer",    # stored under tokenizer/ in the HF repo
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print("✓ Tokenizer uploaded.")

    # --- Upload credibility regressor ---
    if os.path.exists(CRED_REGRESSOR):
        print("\n📦 Uploading credibility regressor (credibility_regressor.pkl)...")
        api.upload_file(
            path_or_fileobj=CRED_REGRESSOR,
            path_in_repo="credibility_regressor.pkl",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print("✓ Credibility regressor uploaded.")
    else:
        print(f"⚠ Credibility regressor not found at {CRED_REGRESSOR}, skipping.")

    # --- Done ---
    print(f"""
{'='*60}
✅ All files uploaded successfully!

Your friend can download them with:

    from huggingface_hub import hf_hub_download, snapshot_download

    # Download model + tokenizer
    snapshot_download(
        repo_id="{HF_REPO_ID}",
        local_dir="models/improved_model"
    )

Or load directly in code:

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "{HF_REPO_ID}", subfolder="model"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "{HF_REPO_ID}", subfolder="tokenizer"
    )

Repo URL: https://huggingface.co/{HF_REPO_ID}
{'='*60}
""")


if __name__ == "__main__":
    main()
