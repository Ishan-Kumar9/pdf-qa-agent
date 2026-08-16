import streamlit as st

from agent import build_agent
from vector_store import create_vector_store

from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Q&A Assistant")
st.write("Upload a PDF and ask questions about its content.")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.session_state.file_name != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            vector_store = create_vector_store(
                uploaded_file,
                embeddings
            )

            st.session_state.vector_store = vector_store

            st.session_state.file_name = uploaded_file.name

            st.session_state.processed = True

        st.success(
            "✅ PDF processed successfully!"
        )


    if st.session_state.processed:

        st.info(
            f"📄 Current document: "
            f"{st.session_state.file_name}"
        )

        question = st.text_input(
            "Ask a question about the PDF:"
        )

        if st.button(
            "🔍 Ask",
            type="primary"
        ):

            if not question:

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Agent is searching and reasoning..."
                ):

                    agent = build_agent(
                        st.session_state.vector_store
                    )

                    result = agent.invoke({
                        "original_question": question,
                        "question": question,
                        "context": [],
                        "relevant": False,
                        "answer": "",
                        "retry_count": 0,
                        "rewritten_question": ""
                    })


                st.subheader("💡 Answer")

                st.write(
                    result["answer"]
                )


                st.subheader("📚 Sources")

                pages = []

                for document in result["context"]:

                    page = document.metadata.get(
                        "page"
                    )

                    if page is not None:
                        pages.append(page)


                pages = sorted(set(pages))


                for page in pages:

                    st.write(
                        f"📄 Page {page}"
                    )


                if result["rewritten_question"]:

                    with st.expander(
                        "🔄 Agent Query Rewrite"
                    ):

                        st.write(
                            result["rewritten_question"]
                        )


if st.session_state.processed:

    st.divider()

    if st.button(
        "🔄 Upload a different PDF"
    ):

        st.session_state.vector_store = None
        st.session_state.file_name = None
        st.session_state.processed = False

        st.rerun()