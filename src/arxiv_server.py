import asyncio
import arxiv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import json

app = Server("arxiv-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_arxiv",
            description="Fetches recent research papers from arXiv for a given search query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to fetch research papers for"
                    },
                    "num_papers": {
                        "type": "integer",
                        "description": "Number of papers to fetch, default 5",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "fetch_arxiv":
        query = arguments["query"]
        num_papers = arguments.get("num_papers", 5)

        arxiv_client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for result in arxiv_client.results(search):
            papers.append({
                "title": result.title,
                "source": "arXiv",
                "description": result.summary[:500],
                "content": result.summary,
                "url": result.entry_id,
                "published_at": str(result.published)
            })

        return [types.TextContent(type="text", text=json.dumps(papers))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())