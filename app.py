
import hashlib
import streamlit as st

from agent import build_agent
from vector_store import create_vector_store
from langchain_huggingface import HuggingFaceEmbeddings


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Q&A Assistant")
st.write("Upload a PDF and ask questions about its content.")


# 2. INITIALIZE SESSION STATE

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# 3. LOAD EMBEDDING MODEL

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


try:
    embeddings = load_embeddings()
except Exception as e:
    st.error("Failed to load the embedding model.")
    st.exception(e)
    st.stop()


# 4. PDF UPLOAD

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    help="Upload a text-based PDF to ask questions about its content."
)


if uploaded_file is not None:

    # Identify the PDF by its content, not only its filename.
    pdf_bytes = uploaded_file.getvalue()
    current_file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Process only when the PDF content changes.
    if st.session_state.file_hash != current_file_hash:

        # Clear the previous document before processing the new one.
        st.session_state.vector_store = None
        st.session_state.file_name = None
        st.session_state.file_hash = None
        st.session_state.processed = False

        try:
            with st.spinner("Processing PDF and creating embeddings..."):

                vector_store = create_vector_store(
                    uploaded_file,
                    embeddings
                )

                if vector_store is None:
                    raise ValueError(
                        "The PDF could not be converted into a vector store."
                    )

                st.session_state.vector_store = vector_store
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_hash = current_file_hash
                st.session_state.processed = True

            st.success("✅ PDF processed successfully!")

        except Exception as e:
            st.session_state.vector_store = None
            st.session_state.file_name = None
            st.session_state.file_hash = None
            st.session_state.processed = False

            st.error("Failed to process the PDF.")
            st.exception(e)
            st.stop()


# 5. QUESTION-ANSWERING INTERFACE

if (
    st.session_state.processed
    and st.session_state.vector_store is not None
):

    st.info(f"📄 Current document: {st.session_state.file_name}")

    with st.form("question_form"):
        question = st.text_input(
            "Ask a question about the PDF:",
            placeholder="What is the main topic of this document?"
        )

        submitted = st.form_submit_button(
            "🔍 Ask",
            type="primary"
        )

    if submitted:

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Agent is searching and reasoning..."
                ):

                    agent = build_agent(
                        st.session_state.vector_store
                    )

                    initial_state = {
                        "original_question": question.strip(),
                        "question": question.strip(),
                        "context": [],
                        "relevant": False,
                        "answer": "",
                        "retry_count": 0,
                        "rewritten_question": ""
                    }

                    result = agent.invoke(initial_state)

                # 6. DISPLAY ANSWER

                st.subheader("💡 Answer")
                st.write(result.get(
                    "answer",
                    "No answer was returned by the agent."
                ))

                # 7. DISPLAY SOURCE PAGES

                st.subheader("📚 Sources")

                pages = []

                for document in result.get("context", []):
                    page = document.metadata.get("page")

                    if page is not None:
                        # PyPDF commonly stores pages as zero-based indices.
                        # Display them as one-based page numbers.
                        try:
                            display_page = int(page) + 1
                        except (TypeError, ValueError):
                            display_page = page

                        pages.append(display_page)

                pages = sorted(set(pages), key=lambda p: (str(type(p)), p))

                if pages:
                    for page in pages:
                        st.write(f"📄 Page {page}")
                else:
                    st.caption(
                        "Page references are not available for this answer."
                    )

                # 8. DISPLAY QUERY REWRITE, IF USED

                rewritten_question = result.get(
                    "rewritten_question",
                    ""
                )

                if rewritten_question:
                    with st.expander("🔄 Agent Query Rewrite"):
                        st.write(rewritten_question)

            except Exception as e:
                st.error(
                    "An error occurred while answering your question."
                )
                st.exception(e)


# 9. RESET DOCUMENT

if st.session_state.processed:

    st.divider()

    if st.button("🔄 Upload a different PDF"):

        st.session_state.vector_store = None
        st.session_state.file_name = None
        st.session_state.file_hash = None
        st.session_state.processed = False

        st.rerun()
