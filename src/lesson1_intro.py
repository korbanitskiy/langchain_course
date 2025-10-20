from time import perf_counter as timer

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    template = PromptTemplate(
        input_variables=["question"],
        template="What is a good name for a company that makes {question}?. Write 5 names.",
    )

    goods = ["glass", "cars", "bikes", "phones", "books"]

    start = timer()
    llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

    print(f"Init: {timer() - start} seconds")

    chain = template | llm
    for good in goods:
        start_request = timer()
        response = chain.invoke(input={"question": good})
        print(f"Response: {timer() - start_request} seconds")

    stop = timer()
    print(f"Time: {stop - start} seconds")


if __name__ == "__main__":
    main()
