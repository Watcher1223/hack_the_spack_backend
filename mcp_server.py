"""
MCP Server for Universal Adapter
Exposes the agent as a single MCP tool that any client can use
"""

import asyncio
import logging
from mcp.server.fastmcp import FastMCP
from services.llm import Agent
from services.env import FIRECRAWL_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("Universal Adapter")


@mcp.tool()
async def chat(
    message: str,
    conversation_id: str | None = None,
    model: str = "anthropic/claude-haiku-4.5",
    max_iterations: int = 25,
) -> dict:
    """
    Chat with the Universal Adapter agent.

    The agent has access to all marketplace tools including:
    - File operations (read, write, list)
    - Web scraping (firecrawl search, scrape, crawl)
    - Tool generation (generate new tools from API docs)
    - Tool search (semantic search of marketplace)
    - All dynamically generated tools (crypto prices, APIs, etc.)

    Args:
        message: Your question or command
        conversation_id: Optional conversation ID for multi-turn chat
        model: LLM model to use (default: claude-haiku-4.5)
        max_iterations: Maximum agent iterations (default: 25)

    Returns:
        dict: Response with agent output, conversation_id, usage stats
    """
    try:
        logger.info(f"MCP chat request: {message[:100]}")

        # Create agent with all tools
        agent = Agent(
            model=model,
            save_conversations=True,
            firecrawl_api_key=FIRECRAWL_API_KEY,
        )

        # Use existing conversation if provided
        if conversation_id:
            agent.conversation_id = conversation_id
            # Could optionally load conversation history from DB here

        # Run agent (it has access to all marketplace tools)
        result = await agent.run(message, max_iterations=max_iterations)

        # Extract tool calls from agent messages
        tool_calls = []
        for msg in agent.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_calls.append(tc["function"]["name"])

        return {
            "response": result.output,
            "conversation_id": agent.conversation_id,
            "usage": result.usage,
            "tool_calls": list(set(tool_calls)),  # Unique tool names
            "success": True,
        }

    except Exception as e:
        logger.exception(f"Error in chat tool: {e}")
        return {
            "response": f"Error: {str(e)}",
            "conversation_id": conversation_id,
            "usage": {},
            "tool_calls": [],
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def list_marketplace_tools(limit: int = 50) -> dict:
    """
    List available tools in the marketplace.

    Args:
        limit: Maximum number of tools to return

    Returns:
        dict: List of tools with names and descriptions
    """
    from services import db

    try:
        tools = db.list_tools()[:limit]
        return {
            "count": len(tools),
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "category": t.get("category", "general"),
                }
                for t in tools
            ],
            "success": True,
        }
    except Exception as e:
        return {"error": str(e), "success": False}


# Optional: Health check
@mcp.tool()
async def health() -> dict:
    """Check if the MCP server is healthy"""
    return {
        "status": "healthy",
        "service": "universal-adapter-mcp",
        "version": "1.0.0",
    }


def create_sse_app():
    """
    Create an ASGI app for HTTP/SSE transport.
    This allows the MCP server to be deployed on web platforms like Render.
    """
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    from sse_starlette import EventSourceResponse
    from mcp.server.sse import SseServerTransport
    import json

    async def handle_sse(request):
        """Handle SSE connections for MCP protocol"""
        async def event_generator():
            transport = SseServerTransport("/messages")

            async def write_message(message):
                """Send message via SSE"""
                yield {
                    "event": "message",
                    "data": json.dumps(message)
                }

            # Connect transport to MCP server
            async with transport.connect_sse(request.scope, request.receive, write_message):
                # Handle incoming messages
                async for message in transport:
                    await mcp._mcp_server.handle_message(message)

        return EventSourceResponse(event_generator())

    async def handle_messages(request):
        """Handle POST messages for MCP protocol"""
        body = await request.json()
        result = await mcp._mcp_server.handle_message(body)
        return JSONResponse(result)

    async def health_check(request):
        """HTTP health check endpoint"""
        return JSONResponse({
            "status": "healthy",
            "service": "universal-adapter-mcp",
            "version": "1.0.0",
            "transport": "sse"
        })

    async def root(request):
        """Root endpoint with server info"""
        return JSONResponse({
            "name": "Universal Adapter MCP Server",
            "version": "1.0.0",
            "protocol": "mcp",
            "transport": "sse",
            "endpoints": {
                "/sse": "SSE endpoint for MCP clients",
                "/messages": "POST endpoint for MCP messages",
                "/health": "Health check"
            }
        })

    app = Starlette(
        routes=[
            Route("/", root),
            Route("/health", health_check),
            Route("/sse", handle_sse),
            Route("/messages", handle_messages, methods=["POST"]),
        ]
    )

    return app


if __name__ == "__main__":
    import sys
    import os

    # Check if running in HTTP mode (for Render, Heroku, etc.)
    mode = os.getenv("MCP_TRANSPORT", "stdio")

    if "--http" in sys.argv or mode == "http":
        # HTTP/SSE mode for remote access
        logger.info("Starting MCP server with HTTP/SSE transport")
        import uvicorn
        app = create_sse_app()
        port = int(os.getenv("PORT", os.getenv("MCP_PORT", 8002)))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # stdio mode for local clients (Claude Desktop, etc.)
        logger.info("Starting MCP server with stdio transport")
        mcp.run()
