# 📄 RAG Chatbot using LangChain + Hugging Face + FAISS

This project is a simple Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Hugging Face Transformers**, and **FAISS**. It allows you to ask questions from your own PDF documents using a local vectorstore and a free Hugging Face model.

---

## 🚀 Features

- Load and parse multiple PDF files
- Convert text into embeddings with Hugging Face models
- Store embeddings using FAISS vectorstore
- Query documents using LangChain’s RetrievalQA chain
- Powered by open-source Hugging Face models (no OpenAI API required)

---

## 📁 Folder Structure
rag-chatbot/
│
├── app.py # Main chatbot script
├── data/ # Folder to store your PDFs
├── .env # Hugging Face API token
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot

2. Create a virtual environment & activate it: 
python -m venv venv
venv\Scripts\activate  # On Windows

3. Install dependencies:
pip install -r requirements.txt

4. Add your Hugging Face API token:
Create a .env file and paste:
HUGGINGFACEHUB_API_TOKEN=your_token_here
You can get your free token from: https://huggingface.co/settings/tokens

5. Add your PDF files to the data/ folder.


💬 How to Use
Run the chatbot:
    python app.py

Then type questions like:

"What is LangChain used for?"

"Summarize the lecture on prompt engineering"

Type exit to quit.

📚 Credits
LangChain

Hugging Face Transformers

FAISS


🧠 License
This project is open-source and available under the MIT License.