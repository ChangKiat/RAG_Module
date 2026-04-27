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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
from vectorstore.chroma_store import as_retriever, collection_count


# ── prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's question
using ONLY the context passages provided below. 

IMPORTANT:
- The answer may be spread across multiple sections — read ALL context carefully
- If the information seems incomplete, say what you found and what might be missing
- Never stop mid-answer — always give a complete response

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


def _llm():
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.2,
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
