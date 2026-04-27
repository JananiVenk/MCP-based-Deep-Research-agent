# MCP Research Agent

An agentic research assistant built with the Model Context Protocol (MCP), LangGraph, and Google Gemini. It queries multiple data sources in real time, embeds results into a vector store, synthesizes cited answers using Gemini 2.5 Flash, and caches results in Supabase to protect API rate limits.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-green)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-blue)](https://ai.google.dev)
[![Supabase](https://img.shields.io/badge/Supabase-cache-darkgreen)](https://supabase.com)
[![Tests](https://img.shields.io/badge/tests-11%20passed-brightgreen)](https://pytest.org)

---

## Architecture

```
User Query
    ↓
Check Supabase cache (TTL: 6 hour)
    ↓
┌─────────────────────────────────────┐
│  Cache hit? → return cached answer  │
│  Cache miss? → run full agent       │
└─────────────────────────────────────┘
    ↓
Streamlit UI (app.py)
    ↓
LangGraph Agent (src/agent.py)
    ↓
┌─────────────────────────────────────┐
│  fetch_node                         │
│  ├── MCP → news_server.py (NewsAPI) │
│  └── MCP → arxiv_server.py (arXiv)  │
└─────────────────────────────────────┘
    ↓
ChromaDB (vector store + embeddings)
    ↓
retrieve_node → relevance check (min distance > 1.0)
    ↓
┌──────────────────────────────────────────┐
│  Relevant?                               │
│  YES → synthesize_node (Gemini)          │
│  NO  → fallback_node (DuckDuckGo + BS4)  │
└──────────────────────────────────────────┘
    ↓
Gemini 2.5 Flash → cited answer → save to Supabase cache
```

---

## Features

- **3 MCP servers** — NewsAPI, arXiv, and DuckDuckGo, each exposing tools via the MCP protocol
- **LangGraph orchestration** — fetch → retrieve → relevance check → synthesize/fallback pipeline
- **Supabase query cache** — answers cached for 6 hour to protect free tier API limits; cache hits shown with a ⚡ badge in the UI
- **Smart fallback** — relevance-based DuckDuckGo fallback using min cosine distance threshold (empirically derived via A/B testing)
- **Full page content extraction** — BeautifulSoup scrapes full article text from DuckDuckGo results
- **Cited answers** — every claim is attributed to its source
- **11 tests** — 5 A/B tests validating fallback threshold, 6 unit tests for core logic

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent framework | LangGraph |
| MCP protocol | mcp Python SDK |
| Vector store | ChromaDB (in-memory) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Cache | Supabase (PostgreSQL) |
| News data | NewsAPI |
| Research data | arXiv |
| Web fallback | DuckDuckGo (ddgs) + BeautifulSoup |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |
| Testing | pytest |

---

## Project Structure

```
mcp_research_agent/
├── src/
│   ├── __init__.py
│   ├── news_server.py      # MCP server — NewsAPI tools
│   ├── arxiv_server.py     # MCP server — arXiv tools
│   ├── web_server.py       # MCP server — DuckDuckGo tools
│   ├── rag_pipeline.py     # Embedding + retrieval utilities
│   └── agent.py            # LangGraph agent (MCP client) + cache logic
├── tests/
│   ├── test_ab_threshold.py    # AB test: fallback threshold strategies
│   ├── test_agent.py           # Unit tests: agent logic
│   └── test_rag_pipeline.py    # Unit tests: chunking + retrieval
├── app.py                  # Streamlit UI
├── .env                    # API keys (not committed)
└── requirements.txt        # Dependencies
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/JananiVenk/MCP-based-Deep-Research-agent.git
cd mcp-research-agent
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add API keys** — create a `.env` file:
```
NEWSAPI_KEY=your_newsapi_key_here
GEMINI_KEY=your_gemini_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
```

- NewsAPI: [newsapi.org](https://newsapi.org) (free tier — 100 requests/day)
- Gemini: [ai.google.dev](https://ai.google.dev) (free tier available)
- Supabase: [supabase.com](https://supabase.com) (free tier available)
- DuckDuckGo: no key needed

**5. Set up Supabase cache table**

Run this in your Supabase SQL Editor:
```sql
create table query_cache (
  id bigserial primary key,
  query text not null,
  answer text not null,
  created_at timestamptz default now()
);

alter table query_cache disable row level security;
```

**6. Run the app**
```bash
streamlit run app.py
```

**7. Run tests**
```bash
pytest tests/ -v
```

---

## Deployment (Hugging Face Spaces)

This app is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/JananiVenk/mcp-research-agent).

To deploy your own:
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) with SDK set to **Streamlit**
2. Add your Space as a git remote: `git remote add space https://huggingface.co/spaces/YourUsername/mcp-research-agent`
3. Add secrets in Space Settings → Variables and secrets
4. Push: `git push space main`

---

## Caching

To protect free tier API limits, query answers are cached in Supabase for **6 hour**. If the same query is asked within that window, the cached answer is returned instantly without hitting NewsAPI, arXiv, or Gemini. Cached responses are shown with a ⚡ badge in the UI.

To change the TTL, edit this line in `src/agent.py`:
```python
CACHE_TTL_HOURS = 6  # change to 24 for longer caching
```

---

## Example Queries

- **News query** (uses NewsAPI + arXiv): *"What is happening with OpenAI?"*
- **Research query** (uses arXiv): *"Latest developments in large language models"*
- **General query** (triggers DuckDuckGo fallback): *"What are the best Python frameworks in 2025?"*

---

## A/B Testing — Fallback Threshold

| Strategy | Logic | Accuracy |
|---|---|---|
| Version A (baseline) | avg(distances) > 0.7 | 2/4 (50%) |
| Version B (current) | min(distances) > 1.0 | 3/4 (75%) |

Version B uses the minimum distance (best match) rather than the average, making it robust to irrelevant chunks inflating the score. The threshold of 1.0 was empirically derived from the distance distribution of the all-MiniLM-L6-v2 embedding model.

---

## License

MIT
