from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """calculate text length and return the value"""
    print("get_text_length called")
    return len(text)


def main():
    tools = [TavilySearch(), get_text_length]
    tools_map = {tool.name: tool for tool in tools}
    llm = ChatOpenAI(name="gpt-4")
    llm = llm.bind_tools(tools)

    msg_1 = "Calculate text length for this: 'London is the capital', 'CAT is not a dog'"
    msg_2 = "Convert 2000 UAH to USD using current currency rate"

    messages = [
        HumanMessage(msg_1),
    ]

    response = llm.invoke(messages)
    messages.append(response)

    while True:
        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            selected_tool = tools_map[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)

        response = llm.invoke(messages)

    print(f"{response=}")


if __name__ == "__main__":
    main()
