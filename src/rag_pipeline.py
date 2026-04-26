import os
import json
import asyncio
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from ollama import chat
load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("news_articles")

server_params = StdioServerParameters(
    command="python",
    args=["-m", "src.news_server"],
    env={**os.environ, "PYTHONIOENCODING": "utf-8"}
)

def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

async def ingest_articles(query: str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("fetch_news", {"query": query, "num_articles": 10})
            articles = json.loads(result.content[0].text)

    for i, article in enumerate(articles):
        content = f"{article['title']}. {article['description'] or ''} {article['content'] or ''}"
        chunks = chunk_text(content)
        for j, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": article["source"], "url": article["url"]}],
                ids=[f"article_{i}_chunk_{j}"]
            )
    print(f"Ingested {len(articles)} articles into ChromaDB")

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"], "url": meta["url"]})
    return chunks

async def answer_question(query: str) -> str:
    
    await ingest_articles(query)
    chunks = retrieve(query)
    
    context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in chunks])
    
    response = chat(
        model='gemma3:4b',
        messages=[{
            "role": "user",
            "content": f"""Answer the following question using only the context provided. 
Cite the source for each claim.

Context:
{context}

Question: {query}"""
        }]
    )
    return response['message']['content']
if __name__ == "__main__":
    response = asyncio.run(answer_question("What is happening with OpenAI?"))
    print(response)