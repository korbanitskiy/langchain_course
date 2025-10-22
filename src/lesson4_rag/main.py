import os
from dotenv import load_dotenv
import asyncio
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pathlib import Path

CURRENT_PATH = Path(os.path.dirname(__file__))

load_dotenv(dotenv_path=os.path.join(CURRENT_PATH.parent.parent, ".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"{OPENAI_API_KEY=}")


embedding = OpenAIEmbeddings()
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
)


query = "What is the Pinecone in machine learning?"




def load_data(filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    loader = TextLoader(path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vector_store.add_documents(chunks)


def get_response_no_rag():
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(query)
    return response


def get_rag_response():
    llm = ChatOpenAI(temperature=0)
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    response = retrieval_chain.invoke({"input": query})
    return response


def main():
    load_data("about_vector_db.txt")
    rag_response = get_rag_response()
    print(rag_response)
    print("\n\n")

    # response_no_rag = get_response_no_rag()
    # print(response_no_rag.content)




if __name__ == "__main__":
    main()
