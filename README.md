# 📄 PDF Q&A Agent — Agentic RAG with LangGraph

An AI-powered PDF Question Answering application built using **Retrieval-Augmented Generation (RAG)**, LangGraph, LangChain, Gemini, and ChromaDB. Upload a PDF, ask questions about its content, and receive context-grounded answers with source page references.

## ✨ Features

- **PDF Upload:** Upload a PDF directly through the Streamlit interface.
- **Semantic Search:** Retrieve relevant document chunks using sentence-transformer embeddings and ChromaDB.
- **Agentic RAG Pipeline:** Orchestrate retrieval, evaluation, query rewriting, and answer generation with LangGraph.
- **Retrieval Evaluation:** Evaluate whether retrieved context is sufficiently relevant to the user's question.
- **Query Rewriting:** Rewrite questions when the initial retrieval does not provide sufficient context.
- **Retry Mechanism:** Retry retrieval with the rewritten question.
- **Context-Grounded Answers:** Generate answers using the retrieved PDF context.
- **Source References:** Display page references for retrieved document chunks.
- **No-Answer Fallback:** Return a fallback response when the retrieved context remains insufficient.
- **Interactive Interface:** Use the application through a Streamlit web interface.

## 🧠 Agent Workflow

The application follows this workflow:

1. **Retrieve:** Search ChromaDB for the top 3 relevant document chunks.
2. **Evaluate:** Use Gemini to evaluate whether the retrieved context can answer the question.
3. **Generate:** If the context is sufficient, generate an answer using the original question and retrieved context.
4. **Rewrite:** If the context is insufficient, rewrite the question to improve semantic retrieval.
5. **Retry:** Retrieve and evaluate the context again using the rewritten question.
6. **No Answer:** If the context is still insufficient after the retry, return a fallback response.

```text
User Question
     |
     v
  Retrieve
     |
     v
  Evaluate
     |
     +------ Sufficient ------> Generate Answer
     |
     +------ Insufficient
                  |
                  v
                Rewrite
                  |
                  v
                Retrieve
                  |
                  v
                Evaluate
                  |
                  +---- Sufficient ----> Generate Answer
                  |
                  +---- Insufficient -> No-Answer Fallback
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface |
| LangChain | LLM and retrieval integrations |
| LangGraph | Agent workflow orchestration |
| Google Gemini API | Context evaluation, query rewriting, and answer generation |
| ChromaDB | Vector storage and semantic retrieval |
| Sentence Transformers | Text embeddings |
| Hugging Face | Embedding model distribution |
| PyPDF | PDF document processing |

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

### Language Model

Gemini model configured in `agent.py`.

## 📁 Project Structure

```text
pdf-qa-agent/
├── app.py                 # Streamlit application
├── agent.py               # LangGraph agent and workflow
├── vector_store.py        # PDF processing and vector store
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── test_agent_rag.py      # RAG workflow tests
├── test_gemini.py         # Gemini integration test
└── test_langgraph.py      # LangGraph tests
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ishan-Kumar9/pdf-qa-agent.git
cd pdf-qa-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a file named `.streamlit/secrets.toml` in the project directory.

Add the following configuration:

```toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your actual Gemini API key.

Keep this file private. It is excluded from version control through `.gitignore`.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## ☁️ Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with your GitHub account.
3. Create an app using this repository: `Ishan-Kumar9/pdf-qa-agent`.
4. Select the `main` branch.
5. Set the main file path to `app.py`.
6. Add the following under the app's **Secrets** settings:

```toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

7. Deploy the application.

Streamlit Community Cloud will install the dependencies listed in `requirements.txt`.

## 🔒 Security

- Never commit API keys or `.streamlit/secrets.toml` to GitHub.
- Store deployment secrets in Streamlit Community Cloud's Secrets settings.
- Do not commit virtual environments, local PDF files, or generated vector databases.

## 🚀 Future Improvements

- Conversation history and follow-up questions.
- Improved retrieval using reranking and hybrid search.
- Streaming answer generation.
- Better document citation with page-level excerpts.
- Support for multiple uploaded documents.
- Evaluation metrics for retrieval quality and answer faithfulness.

## 👨‍💻 Author

**Ishan Kumar**

GitHub: [Ishan-Kumar9](https://github.com/Ishan-Kumar9)

---

*Built with Python, LangChain, LangGraph, ChromaDB, Streamlit, and Google Gemini.*
