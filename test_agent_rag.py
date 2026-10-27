from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# 1. Define Graph State
# -----------------------------

class GraphState(TypedDict):
    original_question: str
    question: str
    context: list
    relevant: bool
    answer: str
    retry_count: int
    rewritten_question: str


# -----------------------------
# 2. Load Embedding Model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 3. Load Existing ChromaDB
# -----------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# -----------------------------
# 4. Load Qwen3
# -----------------------------

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# -----------------------------
# 5. Retriever Node
# -----------------------------

def retrieve(state: GraphState):

    question = state["question"]
    retry_count = state["retry_count"]

    print("\nQuestion:")
    print(question)

    print("Retrieval attempt:", retry_count + 1)

    documents = vectorstore.similarity_search(
        question,
        k=3
    )

    print(f"Retrieved {len(documents)} documents.")

    return {
        "context": documents,
        "retry_count": retry_count + 1
    }


# -----------------------------
# 6. Evaluation Node
# -----------------------------

def evaluate(state: GraphState):

    question = state["question"]
    documents = state["context"]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation system.

Your task is to determine whether the retrieved context is
SUFFICIENT and RELEVANT for answering the user's exact question.

Question:
{question}

Retrieved context:
{context}

Evaluation rules:

1. The context must contain information that can directly answer
   the user's question.

2. Do NOT return YES merely because the context is related to
   the same general topic.

3. If the question is vague, ambiguous, incomplete, or contains
   unclear references such as "it", "this", or "that", return NO
   unless the context clearly identifies what the user is referring to.

4. If the context only contains related information but does not
   provide enough information to answer the exact question, return NO.

5. Return YES only when the context is sufficient to answer the
   question accurately.

Respond with ONLY:
YES
or
NO
"""

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    print("\nEvaluation:")
    print(result)

    return {
        "relevant": result == "YES"
    }


# -----------------------------
# 7. Routing Function
# -----------------------------

def route_after_evaluation(state: GraphState):

    if state["relevant"]:
        return "generate"

    if state["retry_count"] < 2:
        return "rewrite"

    return "no_answer"

def no_answer(state: GraphState):

    print("\nRouting to NO ANSWER")

    return {
        "answer": (
            "I could not find sufficient information in the "
            "provided document to answer this question."
        )
    }


# -----------------------------
# 8. Generate Node
# -----------------------------

def generate(state: GraphState):

    question = state["original_question"]
    documents = state["context"]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    print("\nRouting to GENERATE")

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# -----------------------------
# 9. Rewrite Node
# -----------------------------

def rewrite(state: GraphState):

    question = state["question"]

    prompt = f"""
You are a query rewriting assistant for a PDF retrieval system.

Rewrite the user's question to make it clearer and more specific
for semantic search.

Do not answer the question.
Only return the rewritten question.

Original question:
{question}

Rewritten question:
"""

    print("\nRouting to REWRITE")

    response = llm.invoke(prompt)

    rewritten_question = response.content.strip()

    print("Rewritten question:")
    print(rewritten_question)

    return {
    "question": rewritten_question,
    "rewritten_question": rewritten_question
}


# -----------------------------
# 10. Create Graph
# -----------------------------

graph = StateGraph(GraphState)

# Add nodes
graph.add_node("retrieve", retrieve)
graph.add_node("evaluate", evaluate)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("no_answer", no_answer)


# -----------------------------
# 11. Create Edges
# -----------------------------

# START → Retrieve
graph.add_edge(START, "retrieve")

# Retrieve → Evaluate
graph.add_edge("retrieve", "evaluate")

graph.add_edge("no_answer", END)


# Evaluate → Conditional Routing
graph.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "no_answer": "no_answer"
    }
)


# Generate → END
graph.add_edge("generate", END)

# Rewrite → Retrieve
graph.add_edge("rewrite", "retrieve")


# -----------------------------
# 12. Compile Graph
# -----------------------------

app = graph.compile()


# -----------------------------
# 13. Run Graph
# -----------------------------

result = app.invoke({
    "original_question": "What is it?",
    "question": "What is it?",
    "context": [],
    "retry_count": 0,
    "rewritten_question": ""
})


# -----------------------------
# 14. Display Results
# -----------------------------

print("\nRetrieved Context:\n")

for i, document in enumerate(result["context"], 1):

    print(f"--- Result {i} ---")

    print("Page:", document.metadata.get("page"))

    print(document.page_content[:500])

    print()


# -----------------------------
# 15. Display Evaluation
# -----------------------------

print("\nEvaluation result:")
print(result["relevant"])


# -----------------------------
# 16. Display Answer
# -----------------------------

print("\nFinal answer:")
print(result["answer"])