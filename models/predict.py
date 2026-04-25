"""
predict.py
----------
News article classifier using the trained 8-category RoBERTa model.
All values updated from Optuna + calibration outputs.

Usage:
    from predict import NewsClassifier
    clf    = NewsClassifier()
    result = clf.predict(title="...", text="...")
"""

import os
import re
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Category definitions ────────────────────────────────────────────────────────
CATEGORIES = [
    'Politics',
    'Science and Technology',
    'Business and Economy',
    'International Affairs',
    'Education',
    'Health',
    'Sports',
    'Entertainment',
]

# Exact Optuna-tuned thresholds from your run
OPTUNA_THRESHOLDS = [
    0.6251138177843646,   # Politics
    0.7554300915494968,   # Science and Technology
    0.47825148961746794,  # Business and Economy
    0.7679008210439427,   # International Affairs
    0.5961489563153012,   # Education
    0.7344918680570672,   # Health
    0.6282793716255531,   # Sports
    0.7476509053277701,   # Entertainment
]

# Default paths — update if your folder structure differs
DEFAULT_MODEL_PATH     = 'models/improved_model/best_model'
DEFAULT_TOKENIZER_PATH = 'models/improved_model/best_tokenizer'
DEFAULT_TEMPERATURE    = 1.0    # update after running calibrate.py


@dataclass
class PredictionResult:
    labels:       list[str]         # predicted categories
    probabilities: dict[str, float] # prob for each predicted label
    all_scores:   dict[str, float]  # raw sigmoid score for all 8 labels
    confidence:   float             # mean prob of predicted labels
    is_uncertain: bool              # True if max prob < 0.15


class NewsClassifier:
    """
    Loads once, predicts many times.
    Handles single article, batch, and URL-based prediction.
    """

    def __init__(
        self,
        model_path:     str   = DEFAULT_MODEL_PATH,
        tokenizer_path: str   = DEFAULT_TOKENIZER_PATH,
        temperature:    float = DEFAULT_TEMPERATURE,
        thresholds:     list  = None,
        max_length:     int   = 256,
        device:         str   = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.max_length  = max_length
        self.temperature = temperature
        self.thresholds  = thresholds or OPTUNA_THRESHOLDS

        # Try loading temperature from file if it exists
        temp_path = os.path.join(
            os.path.dirname(model_path), '..', 'temperature.json'
        )
        if os.path.exists(temp_path):
            with open(temp_path) as f:
                self.temperature = json.load(f)['temperature']
            print(f"Loaded temperature: {self.temperature:.4f}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
                             model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model ready on {self.device}")
        print(f"Categories: {CATEGORIES}")
        print(f"Thresholds: {[round(t, 3) for t in self.thresholds]}")

    def _clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ''
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @torch.no_grad()
    def predict(
        self,
        title:                str,
        text:                 str,
        uncertainty_threshold: float = 0.15,
    ) -> PredictionResult:

        combined = self._clean(title) + ' [SEP] ' + self._clean(text)

        enc = self.tokenizer(
            combined,
            max_length            = self.max_length,
            padding               = 'max_length',
            truncation            = True,
            return_tensors        = 'pt',
            return_attention_mask = True,
        )

        logits = self.model(
            input_ids      = enc['input_ids'].to(self.device),
            attention_mask = enc['attention_mask'].to(self.device),
        ).logits.squeeze(0)

        # Apply temperature calibration
        calibrated = logits / self.temperature
        probs      = torch.sigmoid(calibrated).cpu().numpy()

        # Per-label threshold
        predicted_idx = [
            i for i, (p, t) in enumerate(zip(probs, self.thresholds))
            if p > t
        ]

        # Always return at least one label
        if not predicted_idx:
            predicted_idx = [int(np.argmax(probs))]

        predicted_labels = [CATEGORIES[i] for i in predicted_idx]

        all_scores = {
            cat: round(float(probs[i]), 4)
            for i, cat in enumerate(CATEGORIES)
        }

        return PredictionResult(
            labels        = predicted_labels,
            probabilities = {cat: all_scores[cat] for cat in predicted_labels},
            all_scores    = all_scores,
            confidence    = round(float(np.mean([probs[i] for i in predicted_idx])), 4),
            is_uncertain  = float(np.max(probs)) < uncertainty_threshold,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        articles:   list[dict],
        batch_size: int = 32,
    ) -> list[PredictionResult]:

        results = []

        for start in range(0, len(articles), batch_size):
            batch = articles[start : start + batch_size]
            texts = [
                self._clean(a.get('title', '')) + ' [SEP] ' +
                self._clean(a.get('text', ''))
                for a in batch
            ]

            enc = self.tokenizer(
                texts,
                max_length            = self.max_length,
                padding               = 'max_length',
                truncation            = True,
                return_tensors        = 'pt',
                return_attention_mask = True,
            )

            logits_batch = self.model(
                input_ids      = enc['input_ids'].to(self.device),
                attention_mask = enc['attention_mask'].to(self.device),
            ).logits

            probs_batch = torch.sigmoid(
                logits_batch / self.temperature
            ).cpu().numpy()

            for probs in probs_batch:
                predicted_idx = [
                    i for i, (p, t) in enumerate(zip(probs, self.thresholds))
                    if p > t
                ] or [int(np.argmax(probs))]

                predicted_labels = [CATEGORIES[i] for i in predicted_idx]
                all_scores = {
                    cat: round(float(probs[i]), 4)
                    for i, cat in enumerate(CATEGORIES)
                }

                results.append(PredictionResult(
                    labels        = predicted_labels,
                    probabilities = {c: all_scores[c] for c in predicted_labels},
                    all_scores    = all_scores,
                    confidence    = round(float(np.mean(
                        [probs[i] for i in predicted_idx]
                    )), 4),
                    is_uncertain  = float(np.max(probs)) < 0.15,
                ))

        return results

    def predict_from_url(self, url: str) -> PredictionResult:
        try:
            from newspaper import Article, Config
            cfg = Config()
            cfg.fetch_images      = False
            cfg.request_timeout   = 12
            cfg.browser_user_agent = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
            )
            art = Article(url, config=cfg)
            art.download()
            art.parse()
            if len(art.text.split()) < 50:
                raise ValueError("Article too short — may be paywalled.")
            return self.predict(title=art.title or '', text=art.text)
        except ImportError:
            raise ImportError(
                "Run: pip install newspaper4k lxml_html_clean"
            )