"""
vectorstore/chroma_store.py
───────────────────────────
Thin wrapper around ChromaDB + Ollama embeddings.
Handles: add, retrieve, delete, reset.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

import config


# ── singleton store ───────────────────────────────────────────────────────────

_store: Optional[Chroma] = None


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_store(reset: bool = False) -> Chroma:
    """Return (or create) the persistent Chroma vector store."""
    global _store
    if reset and os.path.exists(config.CHROMA_DIR):
        import shutil
        shutil.rmtree(config.CHROMA_DIR)
        print("[chroma_store] 🗑  Existing store deleted.")
        _store = None

    if _store is None:
        _store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=_embeddings(),
            persist_directory=config.CHROMA_DIR,
        )
        count = _store._collection.count()
        print(f"[chroma_store] Store ready — {count} vectors in collection.")

    return _store


# ── public API ────────────────────────────────────────────────────────────────

def add_documents(docs: List[Document], reset: bool = False) -> int:
    """
    Embed and add documents to the store.
    If reset=True the existing store is wiped first.
    Returns the number of chunks added.
    """
    if not docs:
        print("[chroma_store] No documents to add.")
        return 0

    store = get_store(reset=reset)
    print(f"[chroma_store] Embedding {len(docs)} chunks … (this may take a moment)")
    store.add_documents(docs)
    total = store._collection.count()
    print(f"[chroma_store] ✓ Done — total vectors in store: {total}")
    return len(docs)


def similarity_search(query: str, k: int = config.TOP_K) -> List[Document]:
    """Return the top-k most relevant chunks for *query*."""
    store = get_store()
    return store.similarity_search(query, k=k)


def as_retriever(k: int = config.TOP_K):
    """Return a LangChain retriever interface."""
    store = get_store()
    return store.as_retriever(search_kwargs={"k": k})


def collection_count() -> int:
    """Return how many vectors are stored."""
    store = get_store()
    return store._collection.count()


def reset_store() -> None:
    """Wipe the entire vector store."""
    get_store(reset=True)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Vectors in store: {collection_count()}")
