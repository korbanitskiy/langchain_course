import datetime
import os
import time
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

CURRENT_PATH = Path(os.path.dirname(__file__))
load_dotenv(dotenv_path=os.path.join(CURRENT_PATH.parent.parent, ".env"))


tavily_search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=5)


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(description="detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-2 search queries for researching improvements to address the critique of your current answer."
    )
    answer_is_final: bool = Field(
        description="Set to True when you believe the answer is complete and requires no further improvement."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,
    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: list[str] = Field(description="Citations motivating your updated answer.")


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        for attempt in range(3):
            time.sleep(2)  # brief pause between attempts
            response: AIMessage = self.runnable.invoke(
                {"messages": state["messages"]},
                {"tags": [f"attempt:{attempt}"]},
            )
            try:
                parsed = self.validator.invoke(response)
                return {"messages": response, "answer_is_final": parsed[0].answer_is_final}
            except ValidationError as e:
                state["messages"] += [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        else:
            raise Exception("Max retries exceeded")


def get_actor_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are expert researcher.Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            4. If you believe the answer is already complete and cannot be improved further,
                set answer_is_final=True and do NOT generate search queries.
                Otherwise, set answer_is_final=False.
            """,
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "\n\n<system>Reflect on the user's original question and the"
                " actions taken thus far. Respond using the {function_name} function.</reminder>",
            ),
        ]
    ).partial(
        time=lambda: datetime.datetime.now().isoformat(),
    )


def get_first_responder() -> ResponderWithRetries:
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    template = get_actor_prompt_template()
    validator = PydanticToolsParser(tools=[AnswerQuestion])

    chain = template.partial(
        first_instruction="Provide a detailed ~250 word answer.",
        function_name=AnswerQuestion.__name__,
    ) | llm.bind_tools(tools=[AnswerQuestion])

    return ResponderWithRetries(runnable=chain, validator=validator)


def get_revisor() -> ResponderWithRetries:
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    template = get_actor_prompt_template()
    validator = PydanticToolsParser(tools=[ReviseAnswer])

    revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer.
    """

    chain = template.partial(
        first_instruction=revise_instructions,
        function_name=ReviseAnswer.__name__,
    ) | llm.bind_tools(tools=[ReviseAnswer])

    return ResponderWithRetries(runnable=chain, validator=validator)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_search.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    answer_is_final: bool


def _get_num_iterations(messages: list):
    i = 0
    for m in messages[::-1]:
        if m.type not in ("tool", "ai"):
            break
        i += 1
    return i


def event_loop(state: dict, max_iterations: int = 20):
    if state.get("answer_is_final") is True:
        print("Answer marked as final.")
        return END
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > max_iterations:
        return END
    return "execute_tools"


def main(query: str):
    first_responder = get_first_responder()
    revisor = get_revisor()

    builder = StateGraph(State)
    builder.add_node("draft", first_responder.respond)
    builder.add_node("execute_tools", tool_node)
    builder.add_node("revise", revisor.respond)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")
    builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

    graph = builder.compile()

    events = graph.stream(
        {"messages": [("user", query)], "answer_is_final": False},
        stream_mode="values",
    )
    for i, step in enumerate(events):
        print(f"Step {i}")
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main("How to cook souffle?")
