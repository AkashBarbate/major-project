import os
from dotenv import load_dotenv

load_dotenv()

from .provider_base import WebSearchProvider
from .duckduckgo_provider import DuckDuckGoSearchProvider
from .tavily_provider import TavilySearchProvider


def get_web_search_provider(preferred: str = "auto") -> WebSearchProvider:
    """
    Factory to select the most appropriate search provider.
    - If preferred is 'tavily' or 'auto' and TAVILY_API_KEY is present: uses Tavily.
    - Otherwise falls back cleanly to DuckDuckGo (zero config, free).
    """
    tavily_key = os.getenv("TAVILY_API_KEY")

    if preferred == "tavily" and tavily_key:
        return TavilySearchProvider(api_key=tavily_key)
    elif preferred == "duckduckgo":
        return DuckDuckGoSearchProvider()
    elif preferred == "auto":
        if tavily_key:
            return TavilySearchProvider(api_key=tavily_key)
        return DuckDuckGoSearchProvider()
    else:
        return DuckDuckGoSearchProvider()

