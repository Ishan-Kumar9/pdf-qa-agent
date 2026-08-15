from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load PDF

pdf_path = "data/notes.pdf"

reader = PdfReader(pdf_path)

documents = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if text:
        documents.append({
            "text": text,
            "page": page_number
        })


# 2. Split text into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []
metadatas = []

for document in documents:

    page_chunks = splitter.split_text(document["text"])

    for chunk in page_chunks:
        chunks.append(chunk)

        metadatas.append({
            "page": document["page"]
        })


print("Number of chunks:", len(chunks))


# 3. Create embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Store in Chroma

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./chroma_db"
)

print("Vector database created successfully!")