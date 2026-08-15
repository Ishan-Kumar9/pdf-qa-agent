import streamlit as st

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# PAGE CONFIGURATION

st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Q&A Assistant")
st.write("Upload a PDF and ask questions about its content.")


# INITIALIZE SESSION STATE

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# LOAD EMBEDDING MODEL

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# LOAD LLM

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="qwen3:4b",
        temperature=0
    )


# LOAD MODELS

embeddings = load_embeddings()
llm = load_llm()


# PDF UPLOAD

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


# PROCESS PDF

if uploaded_file is not None:

    # Process only if a new PDF is uploaded
    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            # 1. Read PDF

            reader = PdfReader(
                uploaded_file
            )

            documents = []

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if text:

                    documents.append({
                        "text": text,
                        "page": page_number
                    })


            # 2. Create chunks

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = []
            metadatas = []

            for document in documents:

                page_chunks = splitter.split_text(
                    document["text"]
                )

                for chunk in page_chunks:

                    chunks.append(chunk)

                    metadatas.append({
                        "page": document["page"]
                    })


            # 3. Create vector store

            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                metadatas=metadatas
            )


            # 4. Store in session state

            st.session_state.vector_store = vector_store

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.processed = True


        st.success(
            "✅ PDF processed successfully!"
        )


    # DISPLAY PDF INFORMATION

    if st.session_state.processed:

        st.info(
            f"📄 Current document: "
            f"{st.session_state.file_name}"
        )


        # ASK QUESTION

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
                    "Searching the PDF and generating answer..."
                ):

                    # Retrieve relevant documents

                    results = (
                        st.session_state
                        .vector_store
                        .similarity_search(
                            question,
                            k=3
                        )
                    )


                    # Build context

                    context = "\n\n".join(
                        document.page_content
                        for document in results
                    )


                    # Create prompt

                    prompt = f"""
You are a helpful assistant answering questions
about an uploaded PDF.

Answer the question ONLY using the provided context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I could not find this information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


                    # Generate answer with Qwen3

                    response = llm.invoke(
                        prompt
                    )


                # DISPLAY ANSWER

                st.subheader("💡 Answer")

                st.write(
                    response.content
                )


                # DISPLAY SOURCES

                st.subheader("📚 Sources")

                pages = []

                for document in results:

                    page = document.metadata.get(
                        "page"
                    )

                    if page is not None:
                        pages.append(page)


                # Remove duplicate pages
                pages = sorted(
                    set(pages)
                )


                for page in pages:

                    st.write(
                        f"📄 Page {page}"
                    )


# RESET BUTTON

if st.session_state.processed:

    st.divider()

    if st.button("🔄 Upload a different PDF"):

        st.session_state.vector_store = None
        st.session_state.file_name = None
        st.session_state.processed = False

        st.rerun()