# app.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# 📥 Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 📂 Load all PDFs from the "data" folder
pdf_folder = "data"
all_documents = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"✅ Loaded: {filename}")

# 📊 Convert documents into embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(all_documents, embeddings)
retriever = db.as_retriever()

# 🤖 Load free LLM from HuggingFace
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",  # More powerful than flan-t5-large
    huggingfacehub_api_token=hf_token
)

# 🔗 Build the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🧠 Start chatbot loop
print("\n📄 RAG Chatbot is ready! Ask anything based on your PDFs.")
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    try:
        response = qa_chain.invoke({"query": query})
        if not response or 'result' not in response:
            print("❌ Error: No answer returned from the model.\n")
        else:
            print(f"🤖 Bot: {response['result']}\n")
    except Exception as e:
        print(f"❌ Exception: {str(e)}\n")
# This code is a simple Retrieval-Augmented Generation (RAG) chatbot that uses LangChain to process PDF documents and answer user queries.
# It loads all PDFs from a specified folder, converts them into embeddings, and uses a free LLM from HuggingFace to answer questions based on the content of those PDFs.
# The chatbot runs in a loop, allowing users to ask questions until they type 'exit' or 'quit'.
# Make sure to have the required libraries installed and the HuggingFace API token set in your environment variables.
# Ensure you have the necessary packages installed: