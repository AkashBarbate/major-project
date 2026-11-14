from typing import Optional, List
from .provider_base import WebSearchProvider, EvidenceResult, extract_domain_info

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


class DuckDuckGoSearchProvider(WebSearchProvider):
    """
    Zero-configuration live web evidence retriever using DuckDuckGo / DDGS.
    Free, no API key required.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def search(
        self,
        query: str,
        max_results: int = 5,
        domains_whitelist: Optional[List[str]] = None,
    ) -> List[EvidenceResult]:
        results: List[EvidenceResult] = []
        if not query or len(query.strip()) < 3:
            return results

        search_query = query
        if domains_whitelist:
            site_clauses = " OR ".join([f"site:{d}" for d in domains_whitelist[:4]])
            search_query = f"{query} ({site_clauses})"

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = list(ddgs.text(search_query, max_results=max_results))

            for r in raw_results:
                url = r.get("href") or r.get("link", "")
                title = r.get("title", "")
                body = r.get("body") or r.get("snippet", "")

                if not url or not body:
                    continue

                domain, source_group, source_type = extract_domain_info(url)

                results.append(
                    EvidenceResult(
                        url=url,
                        title=title,
                        passage=body,
                        domain=domain,
                        source_group=source_group,
                        source_type=source_type,
                        score=0.8,
                        provider="duckduckgo",
                    )
                )
        except Exception as e:
            print(f"[DDG Search Warning] Query '{query}' warning: {e}")

        return results
