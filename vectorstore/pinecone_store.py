"""
vectorstore/pinecone_store.py
──────────────────────────────
Cloud vector store using Pinecone — works on Streamlit Cloud.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
import config

_store = None


def _embeddings():
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def _init_pinecone():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if config.PINECONE_INDEX in existing:
        desc = pc.describe_index(config.PINECONE_INDEX)
        if desc.dimension != config.EMBED_DIMENSION:
            print(
                f"[pinecone_store] Index dimension {desc.dimension} != "
                f"{config.EMBED_DIMENSION}; recreating index."
            )
            pc.delete_index(config.PINECONE_INDEX)
            existing.remove(config.PINECONE_INDEX)

    if config.PINECONE_INDEX not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc


def get_store(reset: bool = False):
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
        print("[pinecone_store] Connected to Pinecone.")
    return _store


def add_documents(docs: List[Document], reset: bool = False) -> int:
    if not docs:
        return 0
    docs = _clean_metadata(docs)
    store = get_store(reset=reset)
    store.add_documents(docs)
    print(f"[pinecone_store] Added {len(docs)} chunks.")
    return len(docs)


def similarity_search(query: str, k: int = config.TOP_K) -> List[Document]:
    return get_store().similarity_search(query, k=k)


def as_retriever(k: int = config.TOP_K):
    return get_store().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": config.FETCH_K,
            "lambda_mult": 0.7,
        },
    )


def collection_count() -> int:
    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX)
        stats = index.describe_index_stats()
        return stats.get("total_vector_count", 0)
    except Exception:
        return 0


def reset_store():
    get_store(reset=True)


def get_existing_sources() -> set:
    """Return all source URLs already stored in Pinecone."""
    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX)
        stats = index.describe_index_stats()
        return set()
    except Exception:
        return set()


def add_documents_smart(docs: List[Document], reset: bool = False) -> int:
    """Only add chunks whose source URL is not already in the store."""
    if not docs:
        return 0

    store = get_store(reset=reset)

    existing_sources = get_existing_sources()
    new_docs = [
        d for d in docs
        if d.metadata.get("source", "") not in existing_sources
    ]

    if not new_docs:
        print("[pinecone_store] All chunks already exist — nothing added.")
        return 0

    store.add_documents(new_docs)
    print(f"[pinecone_store] Added {len(new_docs)} new chunks.")
    return len(new_docs)


def _clean_metadata(docs: List[Document]) -> List[Document]:
    """Remove or fix null metadata values that Pinecone rejects."""
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
