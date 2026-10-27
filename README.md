# 📄 PDF Q&A Agent

Upload a PDF and ask questions about its content using an Agentic RAG pipeline.

## Features

- PDF upload
- Semantic search with ChromaDB
- Qwen3-based answers
- Agentic RAG with LangGraph
- Retrieval evaluation
- Query rewriting
- Retry mechanism
- Source page references
- No-answer fallback

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- PyPDF
- Sentence Transformers
- ChromaDB
- Ollama
- Qwen3 4B

## Run

```bash
pip install -r requirements.txt
ollama pull qwen3:4b
streamlit run app.py