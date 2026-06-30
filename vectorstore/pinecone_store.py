"""
Cloud vector store using Pinecone — works on Streamlit Cloud.
"""

import os
import sys
from typing import List

# Setup path for local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

import config

# Module-level state
_store = None

# --- Private Helper Methods ---

def _embeddings():
    """Initializes the embedding model instance."""
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

def _init_pinecone():
    """Configures the Pinecone client and ensures the index exists."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing_indexes = [i.name for i in pc.list_indexes()]

    if config.PINECONE_INDEX in existing_indexes:
        desc = pc.describe_index(config.PINECONE_INDEX)
        if desc.dimension != config.EMBED_DIMENSION:
            print(f"[pinecone_store] Dimension mismatch; recreating {config.PINECONE_INDEX}.")
            pc.delete_index(config.PINECONE_INDEX)
            existing_indexes.remove(config.PINECONE_INDEX)

    if config.PINECONE_INDEX not in existing_indexes:
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc

def _clean_metadata(docs: List[Document]) -> List[Document]:
    """Sanitizes metadata to ensure it meets Pinecone's data requirements."""
    for doc in docs:
        clean = {}
        for key, value in doc.metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                clean[key] = [str(v) for v in value if v is not None]
            else:
                clean[key] = str(value)
        doc.metadata = clean
    return docs

# --- Public API ---

def get_store(reset: bool = False):
    """Retrieves or initializes the Pinecone vector store."""
    global _store
    if reset:
        pc = _init_pinecone()
        try:
            pc.Index(config.PINECONE_INDEX).delete(delete_all=True)
        except NotFoundException:
            pass
        print("[pinecone_store] Store reset.")
        _store = None

    if _store is None:
        _init_pinecone()
        _store = PineconeVectorStore(
            index_name=config.PINECONE_INDEX,
            embedding=_embeddings(),
            pinecone_api_key=config.PINECONE_API_KEY,
        )
    return _store

def add_documents(docs: List[Document], reset: bool = False) -> int:
    """Adds a list of documents to the vector store."""
    if not docs:
        return 0
    docs = _clean_metadata(docs)
    store = get_store(reset=reset)
    store.add_documents(docs)
    print(f"[pinecone_store] Added {len(docs)} chunks.")
    return len(docs)

def similarity_search(query: str, k: int = config.TOP_K) -> List[Document]:
    """Performs a basic similarity search."""
    return get_store().similarity_search(query, k=k)

def as_retriever(k: int = config.TOP_K):
    """Returns an MMR-based retriever for the vector store."""
    return get_store().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": config.FETCH_K,
            "lambda_mult": 0.7,
        },
    )

def collection_count() -> int:
    """Returns the total count of vectors in the index."""
    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        return pc.Index(config.PINECONE_INDEX).describe_index_stats().get("total_vector_count", 0)
    except Exception:
        return 0

def reset_store():
    """Alias to reset the vector store."""
    get_store(reset=True)

def get_existing_sources() -> set:
    """Returns all source URLs already stored in Pinecone."""
    # Note: Pinecone does not support direct metadata filtering to return all 'sources' 
    # efficiently without query/pagination. Ensure your index uses metadata filtering.
    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        # Placeholder: Implement actual fetch logic based on your index setup
        return set() 
    except Exception:
        return set()

def add_documents_smart(docs: List[Document], reset: bool = False) -> int:
    """Only adds chunks whose source URL is not already in the store."""
    if not docs:
        return 0
    store = get_store(reset=reset)
    existing_sources = get_existing_sources()
    new_docs = [d for d in docs if d.metadata.get("source", "") not in existing_sources]
    
    if not new_docs:
        print("[pinecone_store] All chunks already exist — nothing added.")
        return 0
        
    store.add_documents(new_docs)
    print(f"[pinecone_store] Added {len(new_docs)} new chunks.")
    return len(new_docs)