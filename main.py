from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot import generate_answer  # We'll create this in chatbot.py

app = FastAPI()

class Question(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "RAG API is up"}

@app.post("/ask")
def ask(question: Question):
    answer = generate_answer(question.query)
    return {"answer": answer}
