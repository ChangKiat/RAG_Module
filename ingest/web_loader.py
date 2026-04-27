"""
ingest/web_loader.py
────────────────────
Load content from one URL or recursively crawl an entire website.
Returns a list of LangChain Document objects ready for chunking.
"""

from __future__ import annotations

import sys
import os
os.environ["USER_AGENT"] = "rag-llama-bot/1.0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import config


# ── helpers ──────────────────────────────────────────────────────────────────

def _html_extractor(raw_html: str) -> str:
    """Strip tags, collapse whitespace."""
    try:
        soup = BeautifulSoup(raw_html, "lxml")   # changed from html.parser
    except Exception:
        soup = BeautifulSoup(raw_html, "html.parser")  # fallback
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── public API ────────────────────────────────────────────────────────────────

def load_single_url(url: str) -> List[Document]:
    """Fetch a single web page and return chunked Documents."""
    print(f"[web_loader] Loading single URL: {url}")
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {"timeout": 15}
    raw_docs = loader.load()

    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)
    print(f"[web_loader] → {len(chunks)} chunks from single page")
    return chunks


def load_website(url: str,
                 max_depth: int = config.MAX_CRAWL_DEPTH,
                 max_pages: int = config.MAX_CRAWL_PAGES) -> List[Document]:
    """
    Recursively crawl a website up to *max_depth* link levels deep.
    Returns chunked Documents.
    """
    print(f"[web_loader] Crawling website: {url}  (depth={max_depth}, max_pages={max_pages})")

    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=_html_extractor,
        timeout=15,
        prevent_outside=True,   # stay on same domain
    )

    raw_docs = loader.load()
    # Respect hard page cap
    raw_docs = raw_docs[:max_pages]
    print(f"[web_loader] Crawled {len(raw_docs)} pages")

    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)
    print(f"[web_loader] → {len(chunks)} total chunks after splitting")
    return chunks


def load_urls(urls: List[str]) -> List[Document]:
    """Load a list of individual URLs and return combined chunked Documents."""
    all_chunks: List[Document] = []
    for url in urls:
        all_chunks.extend(load_single_url(url))
    return all_chunks


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    docs = load_website(target, max_depth=1, max_pages=5)
    for i, d in enumerate(docs[:3]):
        print(f"\n── chunk {i} ──\n{d.page_content[:300]}")
