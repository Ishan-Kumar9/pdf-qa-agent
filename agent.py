from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. GRAPH STATE
# ============================================================

class GraphState(TypedDict):

    original_question: str
    question: str
    context: list
    relevant: bool
    answer: str
    retry_count: int
    rewritten_question: str


# ============================================================
# 2. LOAD LLM
# ============================================================

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# ============================================================
# 3. RETRIEVE NODE
# ============================================================

def retrieve(state: GraphState, vector_store):

    question = state["question"]
    retry_count = state["retry_count"]

    documents = vector_store.similarity_search(
        question,
        k=3
    )

    print(
        f"Retrieval attempt: {retry_count + 1}"
    )

    print(
        f"Retrieved {len(documents)} documents."
    )

    return {
        "context": documents,
        "retry_count": retry_count + 1
    }


# ============================================================
# 4. EVALUATE NODE
# ============================================================

def evaluate(state: GraphState):

    question = state["question"]
    documents = state["context"]

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented
Generation system.

Your task is to determine whether the retrieved context
is SUFFICIENT and RELEVANT for answering the user's
exact question.

Question:
{question}

Retrieved context:
{context}

Evaluation rules:

1. The context must contain information that can directly
   answer the user's question.

2. Do NOT return YES merely because the context is related
   to the same general topic.

3. If the question is vague, ambiguous, incomplete, or
   contains unclear references such as "it", "this", or
   "that", return NO unless the context clearly identifies
   what the user is referring to.

4. If the context only contains related information but
   does not provide enough information to answer the exact
   question, return NO.

5. Return YES only when the context is sufficient to answer
   the question accurately.

Respond with ONLY:
YES
or
NO
"""

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    print("Evaluation:", result)

    return {
        "relevant": result == "YES"
    }


# ============================================================
# 5. ROUTING FUNCTION
# ============================================================

def route_after_evaluation(state: GraphState):

    if state["relevant"]:
        return "generate"

    if state["retry_count"] < 2:
        return "rewrite"

    return "no_answer"


# ============================================================
# 6. REWRITE NODE
# ============================================================

def rewrite(state: GraphState):

    question = state["question"]

    prompt = f"""
You are a query rewriting assistant for a PDF
retrieval system.

Rewrite the user's question to make it clearer and
more specific for semantic search.

Do not answer the question.

Only return the rewritten question.

Original question:
{question}

Rewritten question:
"""

    print("Routing to REWRITE")

    response = llm.invoke(prompt)

    rewritten_question = response.content.strip()

    print(
        "Rewritten question:",
        rewritten_question
    )

    return {
        "question": rewritten_question,
        "rewritten_question": rewritten_question
    }


# ============================================================
# 7. GENERATE NODE
# ============================================================

def generate(state: GraphState):

    question = state["original_question"]
    documents = state["context"]

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided
context.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I could not find this information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    print("Routing to GENERATE")

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# 8. NO ANSWER NODE
# ============================================================

def no_answer(state: GraphState):

    print("Routing to NO ANSWER")

    return {
        "answer": (
            "I could not find sufficient information "
            "in the provided PDF to answer this question."
        )
    }


# ============================================================
# 9. BUILD AGENT
# ============================================================

def build_agent(vector_store):

    graph = StateGraph(GraphState)


    # Add nodes

    graph.add_node(
        "retrieve",
        lambda state: retrieve(
            state,
            vector_store
        )
    )

    graph.add_node(
        "evaluate",
        evaluate
    )

    graph.add_node(
        "rewrite",
        rewrite
    )

    graph.add_node(
        "generate",
        generate
    )

    graph.add_node(
        "no_answer",
        no_answer
    )


    # START → RETRIEVE

    graph.add_edge(
        START,
        "retrieve"
    )


    # RETRIEVE → EVALUATE

    graph.add_edge(
        "retrieve",
        "evaluate"
    )


    # EVALUATE → CONDITIONAL ROUTING

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "no_answer": "no_answer"
        }
    )


    # REWRITE → RETRIEVE

    graph.add_edge(
        "rewrite",
        "retrieve"
    )


    # GENERATE → END

    graph.add_edge(
        "generate",
        END
    )


    # NO ANSWER → END

    graph.add_edge(
        "no_answer",
        END
    )


    return graph.compile()