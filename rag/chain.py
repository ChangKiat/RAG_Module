"""
rag/chain.py
────────────
RAG pipeline: retrieval → prompt → Llama generation.
Supports streaming and returns source metadata.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Iterator, List, Optional

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
from vectorstore.chroma_store import as_retriever, collection_count


# ── prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's question
using ONLY the context passages provided below. If the answer cannot be found in
the context, say so clearly — do NOT make up information.

Be concise, accurate, and cite the source URLs when available.

Context:
{context}"""

HUMAN_PROMPT = "{question}"

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _llm() -> OllamaLLM:
    return OllamaLLM(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=512,      # max tokens in response
        num_ctx=4096,         # context window size
        repeat_penalty=1.1,   # reduces repetition
    )


# ── public API ────────────────────────────────────────────────────────────────

def ask(question: str, k: int = config.TOP_K) -> dict:
    """
    Run a RAG query and return:
        {
            "answer":  str,
            "sources": List[str],   # unique source URLs / file paths
            "chunks":  List[Document],
        }
    """
    if collection_count() == 0:
        return {
            "answer": "⚠  The knowledge base is empty. Please ingest some content first.",
            "sources": [],
            "chunks": [],
        }

    retriever = as_retriever(k=k)
    retrieved: List[Document] = retriever.invoke(question)

    chain = (
        {
            "context":  lambda _: _format_docs(retrieved),
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | _llm()
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    sources = list({d.metadata.get("source", "") for d in retrieved if d.metadata.get("source")})

    return {"answer": answer, "sources": sources, "chunks": retrieved}


def ask_stream(question: str, k: int = config.TOP_K) -> Iterator[str]:
    """
    Streaming version of ask().
    Yields answer tokens one by one.
    Use ask() for sources; this is UI-friendly streaming only.
    """
    if collection_count() == 0:
        yield "⚠  The knowledge base is empty. Please ingest some content first."
        return

    retriever = as_retriever(k=k)
    retrieved: List[Document] = retriever.invoke(question)

    chain = (
        {
            "context":  lambda _: _format_docs(retrieved),
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | _llm()
        | StrOutputParser()
    )

    for token in chain.stream(question):
        yield token


def get_sources_for(question: str, k: int = config.TOP_K) -> List[Document]:
    """Retrieve the raw context chunks for a question (no generation)."""
    retriever = as_retriever(k=k)
    return retriever.invoke(question)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = input("Ask a question: ").strip()
    result = ask(q)
    print("\n── Answer ──")
    print(result["answer"])
    print("\n── Sources ──")
    for s in result["sources"]:
        print(" •", s)
