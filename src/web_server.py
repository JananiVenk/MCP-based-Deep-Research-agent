import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import json

app = Server("web-server")

def extract_page_content(url: str, max_chars: int = 2000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, timeout=5, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        return text[:max_chars] if text else ""
    except Exception:
        return ""

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_web",
            description="Fetches web search results from Internet with full page content extraction",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to fetch, default 5",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "fetch_web":
        query = arguments["query"]
        num_results = arguments.get("num_results", 5)

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                full_content = extract_page_content(r["href"])
                results.append({
                    "title": r["title"],
                    "source": "BeautifulSoup",
                    "description": r["body"],
                    "content": full_content if full_content else r["body"],
                    "url": r["href"],
                    "published_at": ""
                })

        return [types.TextContent(type="text", text=json.dumps(results))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
