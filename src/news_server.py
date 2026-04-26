import os
import asyncio
from dotenv import load_dotenv
from newsapi import NewsApiClient
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

load_dotenv()

newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
app = Server("news-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_news",
            description="Fetches recent news articles for a given search query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to fetch news for"
                    },
                    "num_articles": {
                        "type": "integer",
                        "description": "Number of articles to fetch, default 10",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "fetch_news":
        query = arguments["query"]
        num_articles = arguments.get("num_articles", 10)

        response = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=num_articles
        )

        articles = []
        for article in response["articles"]:
            articles.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "description": article["description"],
                "content": article["content"],
                "url": article["url"],
                "published_at": article["publishedAt"]
            })

        import json
        return [types.TextContent(type="text", text=json.dumps(articles))]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())