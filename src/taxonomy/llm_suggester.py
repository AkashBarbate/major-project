import os
import json
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

from src.graph.nodes.classify import CATEGORIES


def suggest_micro_label_with_llm(
    headline: str,
    body: str = "",
    nearest_category: str = "",
) -> Dict[str, Any]:
    """
    Uses LangChain with Google Generative AI (ChatGoogleGenerativeAI) to suggest:
    - A canonical micro-label topic name (2-3 words)
    - Macro category (one of 8 frozen categories)
    - Concise scope definition
    - 2-4 search aliases
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return _fallback_suggestion(headline, nearest_category)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            api_key=api_key,
            temperature=0.1,
            max_output_tokens=300,
            timeout=10,
        )

        sys_prompt = f"""You are a taxonomy expert for a news verification system.
The system uses exactly these 8 frozen macro categories:
{', '.join(CATEGORIES)}

Given an article headline and text, propose an approved micro-label candidate.
Respond ONLY with a valid JSON object matching this schema:
{{
  "canonical_name": "short 2-3 word lowercase topic name (e.g. semiconductor packaging, renewable energy tariffs)",
  "macro_category": "exact name from the 8 allowed categories above",
  "definition": "one concise sentence defining what this topic covers",
  "aliases": ["alias 1", "alias 2", "alias 3"]
}}"""

        user_content = f"Article Headline: {headline}\nArticle Excerpt: {body[:600]}"
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_content)])

        if isinstance(response.content, list):
            raw_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response.content
            )
        else:
            raw_text = str(response.content)

        raw_text = raw_text.strip()
        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)

        macro = data.get("macro_category", nearest_category)
        if macro not in CATEGORIES:
            macro = nearest_category or "Science and Technology"

        return {
            "canonical_name": data.get("canonical_name", "").strip().lower(),
            "macro_category": macro,
            "definition": data.get("definition", f"News and factual analysis regarding {data.get('canonical_name')}."),
            "aliases": data.get("aliases", []),
            "source": "langchain_gemini",
        }
    except Exception as e:
        print(f"[LLM Suggester Warning] {e}")
        return _fallback_suggestion(headline, nearest_category)


def discover_micro_labels_with_llm(
    headline: str,
    body: str = "",
    top_macro_categories: Optional[List[str]] = None,
    existing_micro_names: Optional[List[str]] = None,
    max_labels: int = 2,
) -> List[Dict[str, Any]]:
    """
    Called when an article matches <= 2 catalogue micro-labels:
    Sends full article context to LangChain Google Gemini to discover
    1 to max_labels new fine-grained micro-labels.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    top_macro_categories = top_macro_categories or []
    existing_micro_names = [e.lower() for e in (existing_micro_names or [])]

    if not api_key:
        return _fallback_multi_discovery(headline, top_macro_categories, max_labels)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            api_key=api_key,
            temperature=0.1,
            max_output_tokens=400,
            timeout=10,
        )

        macro_hint = (
            f"Predicted macro category hints: {', '.join(top_macro_categories)}."
            if top_macro_categories
            else ""
        )
        existing_hint = (
            f"Already assigned labels to NOT duplicate: {', '.join(existing_micro_names)}."
            if existing_micro_names
            else ""
        )

        sys_prompt = f"""You are a news taxonomy expert.
The system uses exactly these 8 frozen macro categories:
{', '.join(CATEGORIES)}

The article has very few matching topics. Analyze the text and propose up to {max_labels} fine-grained micro-label topics that best capture the article subject matter.
{macro_hint}
{existing_hint}

Respond ONLY with a valid JSON array of objects:
[
  {{
    "canonical_name": "short 2-3 word lowercase topic name (e.g. sovereign wealth funds, quantum encryption, semiconductor packaging)",
    "macro_category": "exact name from the 8 allowed categories above",
    "definition": "one concise sentence defining what this topic covers",
    "aliases": ["alias 1", "alias 2"]
  }}
]"""

        user_content = f"Article Headline: {headline}\nArticle Text: {body[:800]}"
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_content)])

        if isinstance(response.content, list):
            raw_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response.content
            )
        else:
            raw_text = str(response.content)

        raw_text = raw_text.strip()
        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)
        if isinstance(data, dict):
            # If model returned a single object or { "labels": [...] }
            data = data.get("suggested_labels") or data.get("labels") or [data]

        results = []
        for item in data[:max_labels]:
            cname = str(item.get("canonical_name", "")).strip().lower()
            if not cname or cname in existing_micro_names:
                continue

            macro = item.get("macro_category")
            if macro not in CATEGORIES:
                macro = top_macro_categories[0] if top_macro_categories else "Science and Technology"

            definition = item.get("definition") or f"Reporting on {cname}."
            aliases = item.get("aliases") or [cname]

            results.append({
                "canonical_name": cname,
                "macro_category": macro,
                "definition": definition,
                "aliases": aliases,
                "source": "langchain_gemini",
            })

        return results if results else _fallback_multi_discovery(headline, top_macro_categories, max_labels)

    except Exception as e:
        print(f"[LLM Multi-Discovery Warning] {e}")
        return _fallback_multi_discovery(headline, top_macro_categories, max_labels)


def _fallback_multi_discovery(headline: str, top_macros: List[str], max_labels: int = 2) -> List[Dict[str, Any]]:
    """Heuristic fallback for multi-label discovery."""
    words = [w.lower().strip(".,!?:;\"'") for w in headline.split() if len(w) > 4]
    macro = top_macros[0] if top_macros else "Politics"
    results = []
    for i in range(min(max_labels, max(1, len(words) // 2))):
        chunk = " ".join(words[i * 2 : (i + 1) * 2])
        if chunk:
            results.append({
                "canonical_name": chunk,
                "macro_category": macro,
                "definition": f"News coverage regarding {chunk}.",
                "aliases": [chunk],
                "source": "heuristic_fallback",
            })
    return results


def _fallback_suggestion(headline: str, nearest_category: str) -> Dict[str, Any]:
    """Heuristic fallback when LLM is unavailable."""
    words = [w.lower().strip(".,!?:;\"'") for w in headline.split() if len(w) > 4]
    candidate = " ".join(words[:2]) if len(words) >= 2 else (headline.lower()[:25] or "general news")
    return {
        "canonical_name": candidate,
        "macro_category": nearest_category or "Politics",
        "definition": f"Factual reporting on {candidate}.",
        "aliases": [candidate],
        "source": "heuristic_fallback",
    }
