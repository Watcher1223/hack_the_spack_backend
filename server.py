from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel

from services.api import AgentRequest, AgentResponse, run_agent
from services import db


# Pydantic models for responses
class ToolResponse(BaseModel):
    name: str
    description: str
    parameters: dict
    code: Optional[str] = None
    created_at: Optional[str] = None
    similarity_score: Optional[float] = None


class ToolSearchResponse(BaseModel):
    query: str
    count: int
    tools: List[ToolResponse]


class ConversationSummary(BaseModel):
    id: str
    conversation_id: str
    start_time: str
    model: str
    final_output: str


# Create FastAPI app
app = FastAPI(
    title="Autonomous Agent API",
    description="AI agent that autonomously searches APIs, generates tools, and executes tasks",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Autonomous Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/agent": "POST - Run the autonomous agent",
            "/agent/simple": "POST - Run agent with simple query params",
            "/tools": "GET - List all tools",
            "/tools/search": "GET - Search tools with vector similarity",
            "/tools/{name}": "GET - Get specific tool by name",
            "/tools/{name}": "DELETE - Delete a tool",
            "/conversations": "GET - List recent conversations",
            "/conversations/{id}": "GET - Get specific conversation",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "autonomous-agent"}


@app.post("/agent", response_model=AgentResponse)
async def run_agent_endpoint(request: AgentRequest):
    """
    Run the autonomous agent with the given prompt.

    The agent will:
    1. Search for relevant API documentation
    2. Scrape the API specs
    3. Generate Python tools dynamically
    4. Execute the tools to complete the task
    5. Return the results

    Example request:
    ```json
    {
        "prompt": "Get the current Bitcoin price",
        "max_iterations": 25,
        "model": "anthropic/claude-haiku-4.5",
        "save_conversation": true
    }
    ```
    """
    try:
        result = await run_agent(
            prompt=request.prompt,
            max_iterations=request.max_iterations,
            model=request.model,
            save_conversation=request.save_conversation,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running agent: {str(e)}")


@app.post("/agent/simple")
async def run_agent_simple(prompt: str, max_iterations: int = 25):
    """
    Simplified endpoint - just send a prompt as a query parameter.

    Example:
    POST /agent/simple?prompt=Get%20Bitcoin%20price&max_iterations=10
    """
    try:
        result = await run_agent(prompt=prompt, max_iterations=max_iterations)

        if result.success:
            return {"output": result.output, "usage": result.usage}
        else:
            raise HTTPException(
                status_code=500, detail=result.error or "Agent failed to complete task"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============================================
# Tool Marketplace Endpoints
# ============================================


@app.get("/tools", response_model=List[ToolResponse])
async def list_tools(
    limit: int = Query(default=50, ge=1, le=500), skip: int = Query(default=0, ge=0)
):
    """
    List all tools in the marketplace.

    - **limit**: Maximum number of tools to return (1-500, default: 50)
    - **skip**: Number of tools to skip for pagination (default: 0)
    """
    try:
        tools = db.list_tools()

        # Apply pagination
        paginated_tools = tools[skip : skip + limit]

        # Format response
        response = []
        for tool in paginated_tools:
            response.append(
                ToolResponse(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["parameters"],
                    code=tool.get("code"),
                    created_at=tool.get("created_at", "").isoformat()
                    if tool.get("created_at")
                    else None,
                )
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")


@app.get("/tools/search", response_model=ToolSearchResponse)
async def search_tools(
    q: str = Query(..., description="Search query for tools"),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Search tools using vector similarity.
    Returns the most relevant tools based on semantic similarity.

    - **q**: Search query describing what you're looking for
    - **limit**: Maximum number of results (1-50, default: 10)

    Example: `/tools/search?q=download images&limit=5`
    """
    try:
        results = db.search_tools(q, limit=limit)

        # Format response
        tools = []
        for tool in results:
            tools.append(
                ToolResponse(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["parameters"],
                    code=tool.get("code"),
                    created_at=tool.get("created_at", "").isoformat()
                    if tool.get("created_at")
                    else None,
                    similarity_score=tool.get("similarity_score"),
                )
            )

        return ToolSearchResponse(query=q, count=len(tools), tools=tools)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching tools: {str(e)}")


@app.get("/tools/{name}", response_model=ToolResponse)
async def get_tool(name: str):
    """
    Get a specific tool by name.

    Returns the complete tool definition including code.
    """
    try:
        tool = db.get_tool(name)

        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        return ToolResponse(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
            code=tool.get("code"),
            created_at=tool.get("created_at", "").isoformat()
            if tool.get("created_at")
            else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tool: {str(e)}")


@app.delete("/tools/{name}")
async def delete_tool(name: str):
    """
    Delete a tool from the marketplace.

    Returns success status.
    """
    try:
        deleted = db.delete_tool(name)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        return {"success": True, "message": f"Tool '{name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting tool: {str(e)}")


# ============================================
# Conversation Endpoints
# ============================================


@app.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100), skip: int = Query(default=0, ge=0)
):
    """
    List recent conversations.

    - **limit**: Maximum number of conversations to return (1-100, default: 20)
    - **skip**: Number of conversations to skip for pagination (default: 0)
    """
    try:
        conversations = db.list_conversations(limit=limit, skip=skip)

        response = []
        for conv in conversations:
            response.append(
                ConversationSummary(
                    id=conv["_id"],
                    conversation_id=conv.get("conversation_id", ""),
                    start_time=conv.get("start_time", ""),
                    model=conv.get("model", ""),
                    final_output=conv.get("final_output", "")[:200] + "..."
                    if len(conv.get("final_output", "")) > 200
                    else conv.get("final_output", ""),
                )
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing conversations: {str(e)}"
        )


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID.

    Returns the complete conversation including all messages and tool calls.
    """
    try:
        conversation = db.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(
                status_code=404, detail=f"Conversation '{conversation_id}' not found"
            )

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting conversation: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
