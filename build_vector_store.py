from pathlib import Path
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb

def build_vector_store():
    # Define base directory relative to the project
    base_dir = Path(__file__).resolve().parent

    # Define paths dynamically
    persist_path = base_dir / "vectorstore"
    file_paths = [
        base_dir / "data" / "two-wheeler-policy-bundled-policy-wordings.pdf",
    ]

    # Convert paths to strings if required by libraries
    persist_path = str(persist_path)
    file_paths = [str(path) for path in file_paths]

    # Initialize embedding model
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

    # Load documents
    loader = UnstructuredLoader(
        file_paths,
        post_processors=[clean_extra_whitespace],
    )
    docs = loader.load()

    # Combine all document content into a single string
    full_text = "\n\n".join([doc.page_content for doc in docs])

    # Split text into structured chunks
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text(full_text)

    # Further split chunks into smaller token-based segments
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, tokens_per_chunk=256
    )
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    # Generate embeddings for the tokenized text
    embeddings = embedding_model.embed_documents(token_split_texts)

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path=persist_path)
    chroma_collection = chroma_client.get_or_create_collection("acko")

    # Add documents and embeddings to the collection
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts, embeddings=embeddings)
    chroma_client.persist()
    print(f"Vector store successfully built and saved at {persist_path}")
