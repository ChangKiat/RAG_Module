"""
vectorstore/pinecone_store.py
──────────────────────────────
Cloud vector store using Pinecone — works on Streamlit Cloud.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import config

_store = None


def _embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )


def _init_pinecone():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if config.PINECONE_INDEX not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=3072,        # ← changed from 768 to 3072
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc


def get_store(reset: bool = False):
    global _store
    if reset:
        pc = _init_pinecone()
        pc.Index(config.PINECONE_INDEX).delete(delete_all=True)
        print("[pinecone_store] 🗑 Store reset.")
        _store = None

    if _store is None:
        _init_pinecone()
        _store = PineconeVectorStore(
            index_name=config.PINECONE_INDEX,
            embedding=_embeddings(),
            pinecone_api_key=config.PINECONE_API_KEY,
        )
        print("[pinecone_store] ✓ Connected to Pinecone.")
    return _store


def add_documents(docs: List[Document], reset: bool = False) -> int:
    if not docs:
        return 0
    store = get_store(reset=reset)
    store.add_documents(docs)
    print(f"[pinecone_store] ✓ Added {len(docs)} chunks.")
    return len(docs)


def similarity_search(query: str, k: int = config.TOP_K) -> List[Document]:
    return get_store().similarity_search(query, k=k)


def as_retriever(k: int = config.TOP_K):
    return get_store().as_retriever(search_kwargs={"k": k})


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