from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import datetime
from urllib.parse import urlparse

@dataclass
class EvidenceResult:
    url: str
    title: str
    passage: str
    domain: str
    source_group: str
    source_type: str = "reporting"  # official, fact_checker, reporting, academic, commentary
    score: float = 0.0
    published_at: Optional[str] = None
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    provider: str = "unknown"


def extract_domain_info(url: str) -> tuple[str, str, str]:
    """
    Extract (domain, source_group, source_type) from URL.
    Identifies fact-checkers, official sources, and major news groups.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        domain = "unknown"

    # Known fact-checkers
    FACT_CHECKERS = {
        "factcheck.org", "politifact.com", "snopes.com", "altnews.in",
        "boomlive.in", "vishvasnews.com", "fullfact.org", "reuters.com/fact-check",
        "afp.com/en/fact-check", "apnews.com/hub/ap-fact-check"
    }

    # Official / Institutional domains
    OFFICIAL_TLDS = {".gov", ".gov.in", ".nic.in", ".gov.uk", ".int", ".mil", ".europa.eu"}

    # Source grouping for major syndicates
    SOURCE_GROUPS = {
        "bbc.com": "BBC Global",
        "bbc.co.uk": "BBC Global",
        "reuters.com": "Thomson Reuters",
        "apnews.com": "Associated Press",
        "thehindu.com": "Kasturi & Sons",
        "thehindubusinessline.com": "Kasturi & Sons",
        "timesofindia.indiatimes.com": "Times Group",
        "economictimes.indiatimes.com": "Times Group",
        "indianexpress.com": "Indian Express Group",
        "ndtv.com": "NDTV Network",
        "hindustantimes.com": "HT Media",
        "livemint.com": "HT Media",
    }

    # Determine source type
    if any(fc in domain for fc in FACT_CHECKERS):
        source_type = "fact_checker"
    elif any(domain.endswith(tld) for tld in OFFICIAL_TLDS):
        source_type = "official"
    elif domain.endswith(".edu") or domain.endswith(".ac.in") or "nature.com" in domain or "sciencedirect.com" in domain:
        source_type = "academic"
    else:
        source_type = "reporting"

    source_group = SOURCE_GROUPS.get(domain, domain)
    return domain, source_group, source_type


class WebSearchProvider(ABC):
    """Abstract interface for pluggable search engines (Tavily, DuckDuckGo)."""

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
        domains_whitelist: Optional[List[str]] = None,
    ) -> List[EvidenceResult]:
        """Perform search and return list of normalized EvidenceResult."""
        pass

