import os
import json
import asyncio
import chromadb
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from google import genai
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from datetime import datetime, timezone, timedelta

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

load_dotenv()
gemini = genai.Client(api_key=os.getenv("GEMINI_KEY"))

# Supabase client (used only for caching)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

CACHE_TTL_HOURS = 6  # answers older than this are considered stale

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("research_articles")

news_server_params = StdioServerParameters(
    command="python",
    args=["-m", "src.news_server"]
)

arxiv_server_params = StdioServerParameters(
    command="python",
    args=["-m", "src.arxiv_server"]
)

web_server_params = StdioServerParameters(
    command="python",
    args=["-m", "src.web_server"]
)

# ── Cache helpers ──────────────────────────────────────────────

def get_cached_answer(query: str) -> str | None:
    """Return a cached answer if one exists and is within TTL, else None."""
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS)).isoformat()
        result = supabase.table("query_cache") \
            .select("answer") \
            .eq("query", query) \
            .gte("created_at", cutoff) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        if result.data:
            print(">> Cache hit — returning cached answer.")
            return result.data[0]["answer"]
    except Exception as e:
        print(f"   Cache lookup failed: {e}")
    return None

def save_to_cache(query: str, answer: str):
    """Save a query+answer to Supabase cache."""
    try:
        supabase.table("query_cache").insert({
            "query": query,
            "answer": answer,
        }).execute()
        print("   Saved to cache.")
    except Exception as e:
        print(f"   Cache save failed: {e}")

# ── Agent state ────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    articles_ingested: bool
    chunks: list
    answer: str
    used_fallback: bool
    needs_fallback: bool
    avg_distance: float
    messages: Annotated[list, operator.add]

# ── Utilities ──────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ── MCP fetchers ───────────────────────────────────────────────

async def fetch_from_news(query: str) -> list[dict]:
    try:
        async with stdio_client(news_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("fetch_news", {"query": query, "num_articles": 10})
                text = result.content[0].text
                if not text or not text.strip():
                    return []
                return json.loads(text)
    except Exception as e:
        print(f"   NewsAPI fetch failed: {e}")
        return []

async def fetch_from_arxiv(query: str) -> list[dict]:
    try:
        async with stdio_client(arxiv_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("fetch_arxiv", {"query": query, "num_papers": 5})
                text = result.content[0].text
                if not text or not text.strip():
                    return []
                return json.loads(text)
    except Exception as e:
        print(f"   arXiv fetch failed: {e}")
        return []

async def fetch_from_web(query: str) -> list[dict]:
    try:
        async with stdio_client(web_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("fetch_web", {"query": query, "num_results": 5})
                text = result.content[0].text
                if not text or not text.strip():
                    return []
                return json.loads(text)
    except Exception as e:
        print(f"   DuckDuckGo fetch failed: {e}")
        return []

def ingest_to_chromadb(articles: list[dict], prefix: str):
    for i, article in enumerate(articles):
        content = f"{article['title']}. {article['description'] or ''} {article['content'] or ''}"
        chunks = chunk_text(content)
        for j, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": article["source"], "url": article["url"]}],
                ids=[f"{prefix}{i}_chunk{j}"]
            )

# ── Graph nodes ────────────────────────────────────────────────

def fetch_node(state: AgentState) -> AgentState:
    print(">> Fetching articles from NewsAPI...")
    news_articles = asyncio.run(fetch_from_news(state["query"]))
    print(f"   Got {len(news_articles)} news articles")
    if news_articles:
        ingest_to_chromadb(news_articles, prefix="news")

    print(">> Fetching papers from arXiv...")
    arxiv_papers = asyncio.run(fetch_from_arxiv(state["query"]))
    print(f"   Got {len(arxiv_papers)} arXiv papers")
    if arxiv_papers:
        ingest_to_chromadb(arxiv_papers, prefix="arxiv")

    return {**state, "articles_ingested": True, "used_fallback": False}

def retrieve_node(state: AgentState) -> AgentState:
    print(">> Retrieving relevant chunks from ChromaDB...")
    query_embedding = embedding_model.encode(state["query"]).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    distances = results["distances"][0]

    # If ChromaDB is empty (fetch failed), go straight to fallback
    if not distances:
        print("   ChromaDB is empty — triggering fallback.")
        return {**state, "chunks": [], "needs_fallback": True}

    print(f"   Minimum relevance distance: {min(distances):.3f}")

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"], "url": meta["url"]})

    needs_fallback = min(distances) > 1.1

    return {**state, "chunks": chunks, "needs_fallback": needs_fallback}

def fallback_node(state: AgentState) -> AgentState:
    print(">> Results not relevant enough, falling back to DuckDuckGo...")
    web_results = asyncio.run(fetch_from_web(state["query"]))
    print(f"   Got {len(web_results)} web results")

    if web_results:
        ingest_to_chromadb(web_results, prefix="web")

    query_embedding = embedding_model.encode(state["query"]).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"], "url": meta["url"]})

    return {**state, "chunks": chunks, "used_fallback": True}

def should_fallback(state: AgentState) -> str:
    if state.get("needs_fallback", False):
        return "fallback"
    return "synthesize"

def synthesize_node(state: AgentState) -> AgentState:
    print(">> Synthesizing answer with Gemini-2.5-flash...")
    if state["used_fallback"]:
        print("   (using DuckDuckgo fallback results)")

    context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in state["chunks"]])

    message = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Answer the following question using only the context provided.
Cite the source for each claim.

Context:
{context}

Question: {state['query']}"""
    )
    answer = message.text

    # Save fresh answer to cache
    save_to_cache(state["query"], answer)

    return {**state, "answer": answer}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("fetch", fetch_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "retrieve")
    graph.add_conditional_edges("retrieve", should_fallback, {
        "fallback": "fallback",
        "synthesize": "synthesize"
    })
    graph.add_edge("fallback", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()

# ── Entry point ────────────────────────────────────────────────

def run_agent(query: str) -> str:
    # Check cache before running the full agent
    cached = get_cached_answer(query)
    if cached:
        return cached

    graph = build_graph()
    result = graph.invoke({
        "query": query,
        "articles_ingested": False,
        "chunks": [],
        "answer": "",
        "used_fallback": False,
        "needs_fallback": False,
        "avg_distance": 0.0,
        "messages": []
    })
    return result["answer"]

if __name__ == "__main__":
    response = run_agent("What is latest happening in OpenAI?")
    print("\n=== ANSWER ===")
    print(response)
