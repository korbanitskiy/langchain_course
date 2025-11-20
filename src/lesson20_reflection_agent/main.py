import asyncio
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

CURRENT_PATH = Path(os.path.dirname(__file__))

load_dotenv(dotenv_path=os.path.join(CURRENT_PATH.parent.parent, ".env"))
load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def create_reflection_agent_chain():
    llm = ChatOpenAI(model="o4-mini")
    reflect_chain = reflection_prompt | llm
    return reflect_chain


def create_generation_agent_chain():
    llm = ChatOpenAI(model="o4-mini")
    generate_chain = generation_prompt | llm
    return generate_chain


class Nodes(StrEnum):
    GENERATE = "generate"
    REFLECT = "reflect"
    END = "end"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def tweet_is_fine(state: State) -> Nodes:
    if len(state["messages"]) > 5:
        return Nodes.END
    return Nodes.REFLECT


def generation_node(state: State) -> list[BaseMessage]:
    chain = create_generation_agent_chain()
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": response}


def reflection_node(state: State) -> list[BaseMessage]:
    chain = create_reflection_agent_chain()
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


async def main():
    graph_builder = StateGraph(state_schema=State)
    graph_builder.add_node(Nodes.GENERATE, generation_node)
    graph_builder.add_node(Nodes.REFLECT, reflection_node)
    graph_builder.set_entry_point(Nodes.GENERATE)

    graph_builder.add_conditional_edges(
        Nodes.GENERATE,
        tweet_is_fine,
        path_map={
            Nodes.END: END,
            Nodes.REFLECT: Nodes.REFLECT,
        },
    )
    graph_builder.add_edge(Nodes.REFLECT, Nodes.GENERATE)
    graph = graph_builder.compile()
    # print(graph.get_graph().draw_mermaid())

    inputs = HumanMessage(content="Write a viral tweet about cooking pasta. 5-6 sentences max.")
    result = graph.invoke({"messages": [inputs]})
    print("Final Tweet:")
    for message in result["messages"]:
        print(message.content)


if __name__ == "__main__":

    asyncio.run(main())
