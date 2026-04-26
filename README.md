# MCP Research Agent

An agentic research assistant built with the Model Context Protocol (MCP), LangGraph, and Ollama. It queries multiple data sources in real time, embeds results into a vector store, and synthesizes cited answers using Mistral.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-green)
![Ollama](https://img.shields.io/badge/Ollama-Mistral-purple)
![Tests](https://img.shields.io/badge/tests-11%20passed-brightgreen)

---

## Architecture

```
User Query
    ↓
Streamlit UI (app.py)
    ↓
LangGraph Agent (src/agent.py) ← MCP Client
    ↓
┌─────────────────────────────────────┐
│  fetch_node                         │
│  ├── MCP → news_server.py (NewsAPI) │
│  └── MCP → arxiv_server.py (arXiv)  │
│  └── MCP → web_server.py (BS4)      │
└─────────────────────────────────────┘
    ↓
ChromaDB (vector store + embeddings)
    ↓
retrieve_node → relevance check (min distance > 1.1)
    ↓
┌─────────────────────────────────┐
│  Relevant?                      │
│  YES → synthesize_node (Ollama) │
│  NO  → fallback_node (DDG)      │
└─────────────────────────────────┘
    ↓
Mistral (via Ollama) → cited answer
```

---

## Features

- **3 MCP servers** — NewsAPI, arXiv, DuckDuckGo, each exposing tools via the MCP protocol
- **LangGraph orchestration** — fetch → retrieve → relevance check → synthesize/fallback
- **Smart fallback** — relevance-based DuckDuckGo fallback using min cosine distance threshold (empirically derived via AB testing)
- **Full page content extraction** — BeautifulSoup scrapes full article text from DuckDuckGo results for richer answers
- **Cited answers** — every claim is attributed to its source
- **11 tests** — 5 AB tests validating fallback threshold, 6 unit tests for core logic

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama (Mistral) |
| Agent framework | LangGraph |
| MCP protocol | mcp Python SDK |
| Vector store | ChromaDB |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| News data | NewsAPI |
| Research data | arXiv |
| Web fallback | DuckDuckGo (ddgs) |
| UI | Streamlit |
| Testing | pytest |

---

## Project Structure

```
mcp_research_agent/
├── src/
│   ├── news_server.py      # MCP server — NewsAPI tools
│   ├── arxiv_server.py     # MCP server — arXiv tools
│   ├── web_server.py       # MCP server — DuckDuckGo tools
│   ├── rag_pipeline.py     # Embedding + retrieval utilities
│   └── agent.py            # LangGraph agent (MCP client)
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

**4. Pull the Mistral model**
```bash
ollama pull mistral
```

**5. Add API keys**

Create a `.env` file:

```env
NEWSAPI_KEY=your_key_here
```

- NewsAPI key: [newsapi.org](https://newsapi.org) (free tier — 100 requests/day)
- DuckDuckGo: no key needed

**6. Run the app**
```bash
streamlit run app.py
```

**7. Run tests**
```bash
pytest tests/ -v
```

---

## Example Queries

**News query** — uses NewsAPI + arXiv
> "What is happening with OpenAI?"

**Research query** — uses arXiv
> "Latest developments in large language models"

**General query** — triggers DuckDuckGo fallback
> "What are the best Netflix shows to watch in 2026?"

---

## AB Testing — Fallback Threshold

During development, the fallback logic was triggering incorrectly for news queries. Two threshold strategies were AB tested across 4 query types:

| Strategy | Logic | Accuracy |
|---|---|---|
| Version A (baseline) | avg(distances) > 0.7 | 2/4 (50%) |
| Version B (current) | min(distances) > 1.1 | 3/4 (75%) |

Version B uses the minimum distance (best match) rather than the average, making it robust to irrelevant chunks inflating the score. The threshold of 1.1 was empirically derived from the distance distribution of the all-MiniLM-L6-v2 embedding model.

---

## License

MIT
