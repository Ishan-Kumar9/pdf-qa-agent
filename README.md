# 📄 PDF Q&A Agent

A RAG-based PDF Question Answering application that allows users to
upload a PDF and ask questions about its content.

## Features

- Upload PDF documents
- Extract and chunk PDF text
- Generate embeddings
- Store embeddings in ChromaDB
- Semantic similarity search
- Generate answers using Qwen3 4B
- Local LLM inference using Ollama
- Display source pages

## Tech Stack

- Python
- Streamlit
- LangChain
- PyPDF
- Sentence Transformers
- ChromaDB
- Ollama
- Qwen3 4B

## Run

```bash
pip install -r requirements.txt
streamlit run app.py