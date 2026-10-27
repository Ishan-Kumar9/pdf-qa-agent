from typing import TypedDict
import time
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# 1. GRAPH STATE

class GraphState(TypedDict):

    original_question: str
    question: str
    context: list
    relevant: bool
    answer: str
    retry_count: int
    rewritten_question: str

# 2. LOAD LLM


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    try:
        secret_key = st.secrets.get("GOOGLE_API_KEY")
        if secret_key:
            api_key = secret_key
    except Exception:
        # Fall back to the environment variable
        pass

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Configure it in "
            "Streamlit Secrets or as an environment variable."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        thinking_level="low",
        max_output_tokens=512,
    )

llm = get_llm()

def get_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            else:
                block_text = getattr(block, "text", None)
                if block_text:
                    text_parts.append(block_text)

        return "\n".join(text_parts).strip()

    return str(content).strip()


# 3. RETRIEVE NODE

def retrieve(state: GraphState, vector_store):
    question = state["question"]
    retry_count = state["retry_count"]

    start = time.perf_counter()

    documents = vector_store.similarity_search(
        question,
        k=3
    )

    print(f"[TIMING] Retrieval: {time.perf_counter() - start:.2f}s")
    print(f"Retrieval attempt: {retry_count + 1}")
    print(f"Retrieved {len(documents)} documents.")

        
    return {
        "context": documents,
        "retry_count": retry_count + 1
    }




# 4. EVALUATE NODE

def evaluate(state: GraphState):
    start = time.perf_counter()

    question = state["question"]
    documents = state["context"]

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented
Generation system.

Determine whether the retrieved context is sufficient
and relevant to answer the user's exact question.

Question:
{question}

Retrieved context:
{context}

Rules:
1. The context must directly answer the question.
2. Related information alone is not sufficient.
3. If the question is vague or ambiguous, return NO
   unless the context clearly resolves the ambiguity.
4. Return YES only when the context is sufficient
   to answer accurately.

Respond with ONLY:
YES
or
NO
"""

    api_start = time.perf_counter()
    response = llm.invoke(prompt)
    api_elapsed = time.perf_counter() - api_start

    result = get_response_text(response).strip().upper()

    # Accept only an unambiguous YES or NO response.
    if result == "YES":
        relevant = True
    elif result == "NO":
        relevant = False
    else:
        relevant = False
        print(
            f"Unexpected evaluator response: {result!r}. "
            "Treating as NO."
        )

    print(
        f"[TIMING] Gemini evaluation API call: "
        f"{api_elapsed:.2f}s"
    )
    print("Evaluation:", repr(result))
    print(
        f"[TIMING] Evaluation node: "
        f"{time.perf_counter() - start:.2f}s"
    )

    return {
        "relevant": relevant
    }



# 5. ROUTING FUNCTION

def route_after_evaluation(state: GraphState):

    if state["relevant"]:
        return "generate"

    if state["retry_count"] < 2:
        return "rewrite"

    return "no_answer"


# 6. REWRITE NODE


def rewrite(state: GraphState):
    start = time.perf_counter()
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

    rewritten_question = get_response_text(response).strip()

    elapsed = time.perf_counter() - start

    print("Rewritten question:", rewritten_question)
    print(f"[TIMING] Rewrite node: {elapsed:.2f}s")

    return {
        "question": rewritten_question,
        "rewritten_question": rewritten_question
    }


# 7. GENERATE NODE


def generate(state: GraphState):
    question = state["original_question"]
    documents = state["context"]

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided context.
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

    start = time.perf_counter()
    response = llm.invoke(prompt)
    print(f"[TIMING] Gemini generation: {time.perf_counter() - start:.2f}s")

    answer = get_response_text(response)

    return {
        "answer": answer
    }



# 8. NO ANSWER NODE

def no_answer(state: GraphState):

    print("Routing to NO ANSWER")

    return {
        "answer": (
            "I could not find sufficient information "
            "in the provided PDF to answer this question."
        )
    }
    

# 9. BUILD AGENT

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