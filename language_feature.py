"""
Extracts 8 linguistic credibility signals from article text.
Used by credibility.py and the API.
"""

import re
import numpy as np
from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    'bbc.com', 'bbc.co.uk', 'ndtv.com', 'thehindu.com',
    'timesofindia.indiatimes.com', 'hindustantimes.com',
    'indianexpress.com', 'theprint.in', 'scroll.in',
    'thewire.in', 'livemint.com', 'reuters.com', 'apnews.com',
    'aljazeera.com', 'theguardian.com', 'businessstandard.com',
}

UNTRUSTED_DOMAINS = {
    'postcard.news', 'opindia.com', 'sudarshannews.in',
    'naturalnews.com', 'infowars.com', 'beforeitsnews.com',
}

CLICKBAIT_PATTERNS = [
    r"you won'?t believe", r"shocking(ly)?", r"mind.?blow",
    r"what happened next", r"this is why", r"will make you",
    r"secret[s]? (they|the government)", r"doctors hate",
    r"one weird trick", r"\d+ reasons why", r"breaking:?",
]

SENSATIONAL_WORDS = {
    'explosive', 'bombshell', 'outrage', 'scandal', 'chaos',
    'catastrophe', 'disaster', 'crisis', 'slams', 'blasts',
    'destroys', 'stunning', 'incredible', 'unbelievable',
    'terrifying', 'shocking', 'exposed', 'leaked',
}

FEATURE_NAMES = [
    'Subjectivity',
    'ALL-CAPS ratio',
    'Exclamation density',
    'Clickbait signals',
    'Sensational vocabulary',
    'Article length',
    'Headline questions',
    'Source credibility',
]


def extract_linguistic_features(
    title: str, text: str, url: str = ''
) -> np.ndarray:
    """Returns 8-dimensional feature vector, all values in [0, 1]."""
    title = title or ''
    text  = text  or ''
    words       = text.split()
    title_words = title.split()

    features = {}

    # Subjectivity
    try:
        from textblob import TextBlob
        features['subjectivity'] = float(
            TextBlob(text[:3000]).sentiment.subjectivity
        )
    except Exception:
        features['subjectivity'] = 0.5

    # ALL-CAPS word ratio
    caps = sum(1 for w in words if w.isupper() and len(w) > 2)
    features['caps_ratio'] = min(caps / max(len(words), 1), 1.0)

    # Exclamation mark density
    excl = (title + ' ' + text).count('!')
    features['exclamation_density'] = min(
        excl / max(len(words), 1) * 10, 1.0
    )

    # Clickbait headline score
    hits = sum(
        1 for p in CLICKBAIT_PATTERNS
        if re.search(p, title.lower())
    )
    features['clickbait_score'] = min(hits / 3.0, 1.0)

    # Sensational vocabulary ratio
    full_lower = (title + ' ' + text).lower()
    sens = sum(
        1 for w in full_lower.split()
        if w.strip('.,!?') in SENSATIONAL_WORDS
    )
    features['sensational_vocab'] = min(
        sens / max(len(words), 1) * 20, 1.0
    )

    # Article length score (longer = more credible)
    features['length_score'] = min(
        max(len(words) - 50, 0) / 950, 1.0
    )

    # Question marks in title
    q_ratio = title.count('?') / max(len(title_words), 1)
    features['title_question_ratio'] = min(q_ratio * 5, 1.0)

    # Source domain credibility
    if url:
        try:
            domain = urlparse(url).netloc.replace('www.', '')
            if domain in TRUSTED_DOMAINS:
                features['source_score'] = 1.0
            elif domain in UNTRUSTED_DOMAINS:
                features['source_score'] = 0.0
            else:
                features['source_score'] = 0.5
        except Exception:
            features['source_score'] = 0.5
    else:
        features['source_score'] = 0.5

    order = [
        'subjectivity', 'caps_ratio', 'exclamation_density',
        'clickbait_score', 'sensational_vocab', 'length_score',
        'title_question_ratio', 'source_score',
    ]
    return np.array([features[k] for k in order], dtype=np.float32)


def get_feature_names() -> list[str]:
    return FEATURE_NAMES