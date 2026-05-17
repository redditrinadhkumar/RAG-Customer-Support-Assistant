"""
ingest.py
---------
Document Ingestion Pipeline
Handles: PDF Loading → Chunking → Embedding → ChromaDB Storage
"""

import os
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from typing import List

# LangChain document loaders & splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "customer_support_kb"
CHUNK_SIZE         = 800     # tokens approx; balances context quality vs retrieval precision
CHUNK_OVERLAP      = 150     # overlap preserves context across chunk boundaries
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_pdf(pdf_path: str) -> List:
    """Load a PDF and return a list of LangChain Document objects (one per page)."""
    logging.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # Basic cleanup
    for page in pages:
        page.page_content = page.page_content.strip()
    logging.info(f"Loaded {len(pages)} pages.")
    return pages


def chunk_documents(pages: List) -> List:
    """
    Split pages into semantic chunks and enrich metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Add metadata enrichment
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["source_file"] = chunk.metadata.get("source", "unknown")

        if "page" not in chunk.metadata:
            chunk.metadata["page"] = "unknown"

    logging.info(f"Created {len(chunks)} enriched chunks.")

    return chunks


def get_embeddings():
    """Return the HuggingFace embedding model (local, no API key needed)."""
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def store_embeddings(chunks: List, embeddings) -> Chroma:

    # Clear old DB
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)

    logging.info(f"Storing {len(chunks)} chunks into ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    logging.info("Embeddings stored successfully.")

    return vectorstore


def load_vectorstore(embeddings) -> Chroma:
    """Load an existing ChromaDB vectorstore from disk."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest_pipeline(pdf_paths: List[str]) -> Chroma:
    """
    Full ingestion pipeline:
    Multiple PDFs → Pages → Chunks → Embeddings → ChromaDB
    """

    all_pages = []

    for pdf_path in pdf_paths:

        # Validate file existence
        if not os.path.exists(pdf_path):
            logging.error(f"File not found: {pdf_path}")
            continue

        try:
            pages = load_pdf(pdf_path)

            if not pages:
                logging.warning(f"No pages loaded from: {pdf_path}")
                continue

            all_pages.extend(pages)

        except Exception as e:
            logging.error(f"Error loading {pdf_path}: {e}")

    # Prevent empty ingestion
    if not all_pages:
        raise ValueError("No valid PDF documents were loaded.")

    # Chunk documents
    chunks = chunk_documents(all_pages)

    # Load embedding model
    embeddings = get_embeddings()

    # Store embeddings in ChromaDB
    vectorstore = store_embeddings(chunks, embeddings)

    logging.info("Ingestion pipeline completed successfully.")

    return vectorstore


def vectorstore_exists() -> bool:
    """Check if ChromaDB has already been populated."""
    db_path = Path(CHROMA_PERSIST_DIR)
    return db_path.exists() and any(db_path.iterdir())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf1> <pdf2> ...")
        sys.exit(1)
    ingest_pipeline(sys.argv[1:])
