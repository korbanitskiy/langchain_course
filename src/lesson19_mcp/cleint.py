import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

CURRENT_PATH = Path(os.path.dirname(__file__))

load_dotenv(dotenv_path=os.path.join(CURRENT_PATH.parent.parent, ".env"))
load_dotenv()


async def main():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(
        model=llm,
        tools=tools,
    )
    math_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]})
    print("Math Response:", math_response["messages"][-1].content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
