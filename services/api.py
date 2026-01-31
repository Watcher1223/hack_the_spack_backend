import asyncio
from typing import Optional
from pydantic import BaseModel, Field

from services.llm import Agent


class AgentRequest(BaseModel):
    """Request model for agent execution"""
    prompt: str = Field(description="The question or task for the agent")
    max_iterations: int = Field(default=25, description="Maximum number of agent iterations")
    model: str = Field(default="anthropic/claude-haiku-4.5", description="LLM model to use")
    save_conversation: bool = Field(default=True, description="Whether to save conversation history")


class AgentResponse(BaseModel):
    """Response model from agent execution"""
    success: bool = Field(description="Whether the agent completed successfully")
    output: str = Field(description="The agent's final output/answer")
    usage: dict = Field(default_factory=dict, description="Token usage statistics")
    error: Optional[str] = Field(default=None, description="Error message if failed")


async def run_agent(
    prompt: str,
    max_iterations: int = 25,
    model: str = "anthropic/claude-haiku-4.5",
    save_conversation: bool = True
) -> AgentResponse:
    """
    Run the autonomous agent with the given prompt.

    Args:
        prompt: The question or task for the agent
        max_iterations: Maximum number of agent iterations (default: 25)
        model: LLM model to use (default: anthropic/claude-haiku-4.5)
        save_conversation: Whether to save conversation history (default: True)

    Returns:
        AgentResponse with success status, output, and usage stats
    """
    try:
        # Create agent instance
        agent = Agent(
            model=model,
            save_conversations=save_conversation
        )

        # Run the agent
        result = await agent.run(prompt, max_iterations=max_iterations)

        # Check if successful
        if result.output and not result.output.startswith("Max iterations"):
            return AgentResponse(
                success=True,
                output=result.output,
                usage=result.usage
            )
        else:
            return AgentResponse(
                success=False,
                output=result.output,
                usage=result.usage,
                error="Agent reached max iterations without completing"
            )

    except Exception as e:
        return AgentResponse(
            success=False,
            output="",
            error=f"Error running agent: {str(e)}"
        )


def run_agent_sync(
    prompt: str,
    max_iterations: int = 25,
    model: str = "anthropic/claude-haiku-4.5",
    save_conversation: bool = True
) -> AgentResponse:
    """
    Synchronous wrapper for run_agent.
    Useful when calling from non-async code.

    Args:
        prompt: The question or task for the agent
        max_iterations: Maximum number of agent iterations (default: 25)
        model: LLM model to use (default: anthropic/claude-haiku-4.5)
        save_conversation: Whether to save conversation history (default: True)

    Returns:
        AgentResponse with success status, output, and usage stats
    """
    return asyncio.run(run_agent(
        prompt=prompt,
        max_iterations=max_iterations,
        model=model,
        save_conversation=save_conversation
    ))


# Convenience function for quick usage
def ask(prompt: str, max_iterations: int = 25) -> str:
    result = run_agent_sync(prompt, max_iterations=max_iterations)
    return result.output if result.success else f"Error: {result.error}"


# For backwards compatibility / simple imports
run = run_agent_sync


