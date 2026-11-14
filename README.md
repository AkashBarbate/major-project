# 📰 NewsLens 3.0: Evidence-Aware News Classification, Fact Verification & Micro-Topic Discovery Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework-Streamlit](https://img.shields.io/badge/UI-Streamlit%201.35+-red.svg)](https://streamlit.io/)
[![Framework-FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.110+-teal.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%26%20Transformers-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **NewsLens 3.0** is an end-to-end, neuro-symbolic news intelligence platform. It fuses fine-tuned deep learning (RoBERTa), Hybrid Retrieval-Augmented Generation (BM25 + Dense Semantic Embeddings + Cross-Encoder Reranking), check-worthy claim extraction, multi-source evidence verification with Natural Language Inference (NLI), and a human-in-the-loop Out-of-Distribution (OOD) taxonomy expansion mechanism.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture & Flow](#-system-architecture--flow)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Applications](#-running-the-applications)
  - [1. Streamlit Interactive Dashboard](#1-streamlit-interactive-dashboard)
  - [2. FastAPI RESTful Service](#2-fastapi-restful-service)
- [API Endpoints Overview](#-api-endpoints-overview)
- [Verification & Automated Test Suite](#-verification--automated-test-suite)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [License](#-license)

---

## 🌟 Key Features

1. **Hierarchical Macro-Micro Classification**:
   - **Macro Classification**: Fine-tuned RoBERTa-base classifier evaluated across 8 frozen categories (*Politics, Technology, Business, Economy, International Affairs, Education, Health, Science/Other*) with per-class **Optuna-tuned thresholds** and temperature scaling.
   - **Micro-Topic Retrieval**: Hybrid Search fusing **BM25 Lexical Matching** ($35\%$) and **Dense Vector Cosine Similarity** ($65\%$, `all-MiniLM-L6-v2`), refined by **Cross-Encoder Reranking** (`ms-marco-MiniLM-L-6-v2`) to capture dynamic, fine-grained news sub-topics without rigid multi-label ceilings.

2. **Claim Extraction & Check-Worthiness Scoring**:
   - Parses news text into atomic claims filtered by factual density, Named Entity Recognition (NER), numerical assertions, and policy predicates, filtering out generic fluff and rhetorical statements.

3. **Multi-Source Evidence Verification (Live NLI)**:
   - Queries dual sources: a **Curated Offline Knowledge Base** and **Real-Time Live Web Search** (DuckDuckGo / Tavily API) executed with concurrent thread pooling.
   - Assigns premise-hypothesis stance scores (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`) using cross-encoder NLI and aggregates an overall **Credibility Score (0–100)** with source attribution.

4. **Human-in-the-Loop OOD Review & LLM Expansion**:
   - Detects Out-of-Distribution (OOD) articles falling below micro-similarity thresholds.
   - An LLM Assistant (Google Gemini Flash) automatically drafts proposed canonical names, definitions, and aliases.
   - Analysts review and approve novel topics with one click, persisting them to SQLite and dynamically registering them into vector embeddings and BM25 index on-the-fly.

5. **Production Latency Optimizations**:
   - Singleton model caching (`@st.cache_resource`), vectorized batch inference, and multi-threaded retrieval reduce cold-start latency from $>140\text{s}$ down to **~3–8s**.

---

## 🏗️ System Architecture & Flow

```
                      +---------------------------------------+
                      | Raw News Text / Live Article URL      |
                      +---------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| LangGraph Execution Pipeline                                                      |
|                                                                                   |
|  [1. Ingest & Scrape]  --> Newspaper3k + LD-JSON + BS4 Parsing                    |
|            |                                                                      |
|            v                                                                      |
|  [2. Macro Classify]   --> RoBERTa + Optuna Thresholds (8 Macro Classes)          |
|            |                                                                      |
|            v                                                                      |
|  [3. Micro Retrieval]  --> Hybrid Search: BM25 (35%) + Dense Cosine (65%)         |
|                            --> Cross-Encoder Reranker (Top 10 -> Multi-labels)    |
|                            --> Low-confidence OOD routing -> OOD Queue            |
|            |                                                                      |
|            v                                                                      |
|  [4. Claim Extraction] --> Check-worthiness filter & atomic claim parsing         |
|            |                                                                      |
|            v                                                                      |
|  [5. Evidence Verify]  --> Parallel retrieval: Local KB + Live Web Search        |
|                            --> Cross-Encoder NLI Stance Evaluation                |
|                            --> Aggregated Credibility Score (0-100)               |
|            |                                                                      |
|            v                                                                      |
|  [6. SQLite Persist]   --> Full article, claims, evidence, and audit trail        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
             +----------------------------+----------------------------+
             |                                                         |
             v                                                         v
  +----------------------+                                  +----------------------+
  | Streamlit Web UI     |                                  | FastAPI REST API     |
  | (app.py)             |                                  | (src/api/main.py)    |
  +----------------------+                                  +----------------------+
```

---

## 📁 Project Structure

```
Minor Project/
├── app.py                      # Main Streamlit Dashboard (UI Tabs 1-4)
├── requirements.txt            # Python dependencies
├── .env.example                # Template for API keys & configuration
├── .gitignore                  # Git ignore rules (protects keys, weights, db)
│
├── src/                        # NewsLens 3.0 Core Engine
│   ├── api/                    # FastAPI REST API
│   │   ├── main.py             # FastAPI entrypoint & router assembly
│   │   ├── schemas.py          # Pydantic request/response schemas
│   │   └── routers/            # Modular endpoint controllers
│   │       ├── articles.py     # Pipeline execution & history
│   │       ├── classify.py     # Fast standalone classification
│   │       ├── claims.py       # Standalone claim extraction & verification
│   │       └── taxonomy.py     # Catalogue & OOD review endpoints
│   │
│   ├── database/               # Relational Persistence
│   │   ├── models.py           # SQLAlchemy ORM models (Article, Claim, Evidence, etc.)
│   │   └── session.py          # SQLite engine & session management
│   │
│   ├── graph/                  # LangGraph Pipeline Definition
│   │   ├── pipeline.py         # StateGraph orchestration & node wiring
│   │   ├── state.py            # TypedDict state schema
│   │   └── nodes/              # Pipeline node implementations
│   │       ├── ingest.py       # Text cleaning & normalization
│   │       ├── classify.py     # RoBERTa macro classifier
│   │       ├── micro_retrieval.py # Hybrid BM25 + dense retrieval + reranking
│   │       ├── credibility.py  # Linguistic credibility scoring
│   │       ├── claims.py       # Check-worthy claim extraction
│   │       └── verify.py       # Parallel evidence search & NLI verification
│   │
│   ├── taxonomy/               # Taxonomy & Embedding Services
│   │   ├── embedding_service.py # Singleton all-MiniLM-L6-v2 embedding model
│   │   └── llm_suggester.py    # Gemini LLM topic discovery assistant
│   │
│   └── web_retrieval/          # Search & Scraper Modules
│       ├── factory.py          # Search provider factory (DuckDuckGo / Tavily)
│       ├── tavily_provider.py  # Tavily API search client
│       ├── duckduckgo_provider.py # Free fallback search client
│       └── scraper.py          # Newspaper3k + LD-JSON web scraper
│
├── data/                       # Taxonomy and Knowledge Base Assets
│   ├── catalogue/              # Approved micro-label seed catalogues
│   │   └── micro_labels.json
│   └── curated_kb/             # Local offline evidence corpus
│       └── evidence_snippets.json
│
├── tests/                      # Automated Test Suite & Benchmarks
│   ├── test_api_endpoints.py   # FastAPI HTTP endpoint tests
│   ├── test_multilabel_micro.py# Hybrid micro-retrieval tests
│   ├── test_micro_discovery.py # LLM topic discovery & SQLite persistence test
│   └── benchmark_speed.py      # Latency & performance benchmarks
│
├── models/                     # Tokenizers, configs & model architecture scripts
└── Docs/                       # Reports, presentation assets & documentation
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Python**: `3.10`, `3.11`, or `3.12`
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/AkashBarbate/minor-project.git
cd minor-project
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv myenv
.\myenv\Scripts\activate

# Linux / macOS
python3 -m venv myenv
source myenv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy the `.env.example` file to `.env`:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and insert your API keys:
- `GOOGLE_API_KEY`: *(Recommended)* Free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) for LLM-assisted novel micro-label discovery in Tab 3.
- `TAVILY_API_KEY`: *(Optional)* Free tier key from [Tavily](https://tavily.com/) for high-speed news fact verification. If omitted, NewsLens automatically falls back to DuckDuckGo search.

---

## 🖥️ Running the Applications

### 1. Streamlit Interactive Dashboard
Launch the web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Dashboard Capabilities**:
- **Tab 1: Article Analysis & Verification**: Paste news text or enter an article URL. Runs the entire pipeline and displays macro badges, micro-topics, credibility index dial, and claim verification cards.
- **Tab 2: Claim Verification Sandbox**: Test any individual statement against live web evidence with stance classification (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`).
- **Tab 3: Taxonomy & OOD Review Queue**: Inspect articles tagged as novel topics, ask Gemini for recommended taxonomy labels, and approve them directly into the vector database.
- **Tab 4: System Architecture & Metrics**: View database counts, Optuna calibration thresholds, and API documentation.

---

### 2. FastAPI RESTful Service
Launch the backend server:
```bash
uvicorn src.api.main:app --reload --port 8000
```
Interactive Swagger API documentation is available at: `http://localhost:8000/docs`.

---

## 🔌 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/articles` | Executes full LangGraph pipeline on submitted raw text |
| `POST` | `/articles/url` | Scrapes news article from URL and executes analysis |
| `GET` | `/articles` | Lists analyzed articles persisted in SQLite database |
| `POST` | `/classify` | Fast standalone macro & micro-topic classification |
| `POST` | `/claims/extract` | Extracts check-worthy claims from text |
| `POST` | `/verify` | Verifies a single claim against live evidence |
| `GET` | `/taxonomy` | Retrieves the active approved micro-label catalogue |
| `GET` | `/taxonomy/ood-queue` | Retrieves novel/OOD topics pending analyst review |
| `POST` | `/annotations` | Submits analyst feedback for inter-annotator audit |

---

## 🧪 Verification & Automated Test Suite

To verify the installation and system components, run the automated test suite:

```bash
# Test 1: Verify all 4 core FastAPI HTTP endpoints
python -m tests.test_api_endpoints

# Test 2: Verify Hybrid (BM25 + Dense) Micro-Retrieval & Cross-Encoder Reranking
python -m tests.test_multilabel_micro

# Test 3: Verify OOD Discovery, SQLite persistence & Dynamic Vector Indexing
python -m tests.test_micro_discovery

# Test 4: End-to-End Latency Benchmark
python -m tests.benchmark_speed
```

---

## ⚙️ Configuration & Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Optional | `""` | Powers Gemini LLM for novel topic suggestions in Tab 3 |
| `TAVILY_API_KEY` | Optional | `""` | High-accuracy web search for live claim verification |
| `NEWSLENS_DB_PATH` | Optional | `data/newslens.db` | Local SQLite database file path |

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
