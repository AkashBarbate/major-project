import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

api_key = os.getenv("GOOGLE_API_KEY")
print("API Key present:", bool(api_key))

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key, temperature=0.1)

sys_prompt = """You are an expert news taxonomy assistant.
Allowed macro categories: ['Politics', 'Science and Technology', 'Business and Economy', 'International Affairs', 'Education', 'Health', 'Sports', 'Entertainment'].
Propose 2 fine-grained micro-label topics that best describe the news article.
Respond ONLY with a valid JSON array:
[
  {
    "canonical_name": "short 2-3 word topic name",
    "macro_category": "exact allowed macro category",
    "definition": "one sentence definition",
    "aliases": ["alias1", "alias2"]
  }
]"""

user_msg = """Article Headline: Ted Turner family puts 20 landscape paintings from Montana ranch up for auction
Article Body: More than 20 landscape paintings from media mogul Ted Turner's Montana ranch are heading to auction at Christie's in New York, with an estimated value of over $20 million. The collection features rare 19th-century works depicting Western American landscapes, wildlife, and historical scenes by renowned artists."""

res = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
content = res.content
if isinstance(content, list):
    content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
print("Raw output:")
print(content)

