import asyncio
import sys

from services.llm import Agent


async def main(question: str):
    # Test question that should use the search tool
    # question = "Get me the real-time stream flow data for the Mississippi River."

    agent = Agent()
    res = await agent.run(question, max_iterations=25)

    print("\n" + "=" * 50)
    print("AGENT OUTPUT:")
    print("=" * 50)
    print(res.output)
    print("\n" + "=" * 50)
    print("USAGE:")
    print("=" * 50)
    print(res.usage)


if __name__ == "__main__":
    question = sys.argv[1]
    asyncio.run(main(question))
