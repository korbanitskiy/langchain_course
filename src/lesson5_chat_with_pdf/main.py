import os
from pathlib import Path

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CURRENT_PATH = Path(os.path.dirname(__file__))
PDF_PATH = CURRENT_PATH / "django.pdf"


load_dotenv(dotenv_path=os.path.join(CURRENT_PATH.parent.parent, ".env"))


def main():
    embeddings = OpenAIEmbeddings()
    loader = PyPDFLoader(PDF_PATH)
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    documents = loader.load()
    documents = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(documents, embeddings)
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(
        llm=OpenAI(),
        prompt=qa_prompt,
    )
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=stuff_documents_chain,
    )
    response = retrieval_chain.invoke({"input": "get the code of RestrictToUserMixin"})

    print(response["answer"])


if __name__ == "__main__":
    main()
