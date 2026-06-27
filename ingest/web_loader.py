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

import requests
import trafilatura
from langchain_core.documents import Document
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import config


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_main_content(raw_html: str, url: str = "") -> str:
    """Extract main article text using trafilatura; fall back to basic stripping."""
    text = trafilatura.extract(
        raw_html,
        url=url or None,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if text and text.strip():
        return text.strip()

    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except Exception:
        soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _fetch_html(url: str) -> str:
    response = requests.get(url, timeout=15, headers={"User-Agent": os.environ["USER_AGENT"]})
    response.raise_for_status()
    return response.text


def _html_extractor(raw_html: str) -> str:
    """Extractor callback for RecursiveUrlLoader (URL not available here)."""
    return _extract_main_content(raw_html)


def _make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _page_title(raw_html: str) -> str:
    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except Exception:
        soup = BeautifulSoup(raw_html, "html.parser")
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""


def _filter_chunks(chunks: List[Document]) -> List[Document]:
    filtered = [
        c for c in chunks
        if len(c.page_content.strip()) >= config.MIN_CHUNK_CHARS
    ]
    for i, chunk in enumerate(filtered):
        chunk.metadata["chunk_index"] = i
    return filtered


def _split_and_filter(raw_docs: List[Document]) -> List[Document]:
    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)
    return _filter_chunks(chunks)


# ── public API ────────────────────────────────────────────────────────────────

def load_single_url(url: str) -> List[Document]:
    """Fetch a single web page and return chunked Documents."""
    print(f"[web_loader] Loading single URL: {url}")
    html = _fetch_html(url)
    text = _extract_main_content(html, url=url)
    if not text:
        print(f"[web_loader] ⚠  No content extracted from {url}")
        return []

    raw_docs = [
        Document(
            page_content=text,
            metadata={
                "source": url,
                "title": _page_title(html),
                "doc_type": "web",
            },
        )
    ]
    chunks = _split_and_filter(raw_docs)
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
        prevent_outside=True,
    )

    raw_docs = loader.load()
    raw_docs = raw_docs[:max_pages]
    print(f"[web_loader] Crawled {len(raw_docs)} pages")

    for doc in raw_docs:
        doc.metadata.setdefault("doc_type", "web")
        doc.metadata.setdefault("title", "")

    chunks = _split_and_filter(raw_docs)
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
