from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()


def main():
    tools = [TavilySearch()]
    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    llm = ChatOpenAI(name="gpt-4")
    react_prompt = PromptTemplate(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
        input_variables=["input", "tool_names", "agent_scratchpad", "format_instructions", "tools"],
        output_parser=output_parser,
    )
    react_prompt = react_prompt.partial(
        format_instructions=output_parser.get_format_instructions(),
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    result = agent_executor.invoke(
        input={
            "input": "Convert 3000 USD to UAH using actual bank currency",
        }
    )
    print(result)
    print(output_parser.parse(result["output"]))


if __name__ == "__main__":
    main()
