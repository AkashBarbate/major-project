"""
NewsLens 3.0 — News Article Web Scraper
Uses newspaper3k with automatic fallback to requests + BeautifulSoup
to reliably extract headline and article body from complex news sites.
"""
import re
from typing import Tuple


def extract_article_content(url: str) -> Tuple[str, str]:
    """
    Scrapes a news article from a given URL.
    Returns: (title, body_text)
    Uses Newspaper3k with fallback to BeautifulSoup and LD-JSON parsing.
    """
    if not url or not url.strip():
        return "", ""

    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    title = ""
    body = ""

    # 1. Newspaper3k extraction
    try:
        from newspaper import Article as NArticle, Config
        cfg = Config()
        cfg.fetch_images = False
        cfg.request_timeout = 10
        cfg.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        art = NArticle(url, config=cfg)
        art.download()
        art.parse()
        title = art.title or ""
        body = art.text or ""
    except Exception:
        pass

    # 2. BeautifulSoup & LD-JSON fallback if body is sparse (< 35 words)
    if len(body.split()) < 35:
        try:
            import requests
            import json
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")

                # Try finding title from OpenGraph / meta if not already found
                if not title:
                    og_title = soup.find("meta", property="og:title")
                    if og_title and og_title.get("content"):
                        title = og_title["content"].strip()
                    elif soup.title and soup.title.string:
                        title = soup.title.string.strip()

                # A. Try JSON-LD articleBody (widely supported by news publishers)
                for script in soup.find_all("script", type="application/ld+json"):
                    try:
                        data = json.loads(script.string or "")
                        if isinstance(data, list):
                            data = data[0] if data else {}
                        if isinstance(data, dict):
                            # Handle @graph wrapper
                            if "@graph" in data and isinstance(data["@graph"], list):
                                for item in data["@graph"]:
                                    if isinstance(item, dict) and "articleBody" in item:
                                        data = item
                                        break
                            if not title and "headline" in data:
                                title = str(data["headline"]).strip()
                            if "articleBody" in data and len(str(data["articleBody"]).strip().split()) > len(body.split()):
                                body = str(data["articleBody"]).strip()
                    except Exception:
                        pass

                # B. HTML paragraph extraction if body is still sparse
                if len(body.split()) < 35:
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        tag.decompose()

                    # Look inside article container if available
                    container = soup.find("article") or soup.find("main") or soup.find(class_=re.compile(r"article-body|story-content|_s30J|content-body", re.I)) or soup
                    paragraphs = [p.get_text().strip() for p in container.find_all("p")]
                    filtered = [
                        p for p in paragraphs
                        if len(p.split()) > 6 and not re.search(r"cookie|subscribe|sign in|privacy policy|terms of service|newsletter|advertisement", p, re.I)
                    ]

                    if len(filtered) >= 2:
                        extracted_body = "\n\n".join(filtered)
                        if len(extracted_body.split()) > len(body.split()):
                            body = extracted_body
        except Exception:
            pass

    return title.strip(), body.strip()

