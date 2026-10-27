from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def create_vector_store(pdf_file, embeddings):

    reader = PdfReader(pdf_file)

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


    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vector_store