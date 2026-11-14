import os
from typing import Optional, List
from .provider_base import WebSearchProvider, EvidenceResult, extract_domain_info


class TavilySearchProvider(WebSearchProvider):
    """
    Evidence retriever using Tavily AI Search API.
    Provides structured, clean research passages tailored for LLMs and verification.
    Requires TAVILY_API_KEY environment variable or constructor arg.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                print(f"[Tavily Warning] Failed to initialize TavilyClient: {e}")

    def is_available(self) -> bool:
        return self._client is not None

    def search(
        self,
        query: str,
        max_results: int = 5,
        domains_whitelist: Optional[List[str]] = None,
    ) -> List[EvidenceResult]:
        results: List[EvidenceResult] = []
        if not self._client or not query.strip():
            return results

        try:
            kwargs = {
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": False,
            }
            if domains_whitelist:
                kwargs["include_domains"] = domains_whitelist

            response = self._client.search(**kwargs)
            raw_results = response.get("results", [])

            for r in raw_results:
                url = r.get("url", "")
                title = r.get("title", "")
                passage = r.get("content", "")
                score = float(r.get("score", 0.85))
                published_date = r.get("published_date")

                if not url or not passage:
                    continue

                domain, source_group, source_type = extract_domain_info(url)

                results.append(
                    EvidenceResult(
                        url=url,
                        title=title,
                        passage=passage,
                        domain=domain,
                        source_group=source_group,
                        source_type=source_type,
                        score=score,
                        published_at=published_date,
                        provider="tavily",
                    )
                )
        except Exception as e:
            print(f"[Tavily Search Error] Query failed '{query}': {e}")

        return results

