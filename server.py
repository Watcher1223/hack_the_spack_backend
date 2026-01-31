from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.api import (
    AgentRequest,
    AgentResponse,
    run_agent
)

# Create FastAPI app
app = FastAPI(
    title="Autonomous Agent API",
    description="AI agent that autonomously searches APIs, generates tools, and executes tasks",
    version="1.0.0"
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
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "autonomous-agent"
    }


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
            save_conversation=request.save_conversation
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running agent: {str(e)}"
        )


@app.post("/agent/simple")
async def run_agent_simple(prompt: str, max_iterations: int = 25):
    """
    Simplified endpoint - just send a prompt as a query parameter.

    Example:
    POST /agent/simple?prompt=Get%20Bitcoin%20price&max_iterations=10
    """
    try:
        result = await run_agent(
            prompt=prompt,
            max_iterations=max_iterations
        )

        if result.success:
            return {
                "output": result.output,
                "usage": result.usage
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.error or "Agent failed to complete task"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
