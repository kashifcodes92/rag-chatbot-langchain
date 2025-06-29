# chatbot.py

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

import os

# Set your OpenAI API key via environment variable or directly (not recommended for prod)
openai_api_key = os.getenv("OPENAI_API_KEY", "sk-...")  # Replace with your key or set as env var

def generate_answer(query):
    # Example RAG setup
    loader = TextLoader("data.txt")  # You need a data.txt in root
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    return qa.run(query)
