"""
ingest/doc_loader.py
────────────────────
Load PDF documents and return chunked LangChain Documents ready for embedding.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from typing import List

import fitz  # pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_splitter(file_type: str = "default"):
    if file_type == "pdf":
        return RecursiveCharacterTextSplitter(
            chunk_size=config.PDF_CHUNK_SIZE,
            chunk_overlap=config.PDF_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _load_pdf(path: Path) -> List[Document]:
    """Extract text per page using PyMuPDF."""
    raw_docs: List[Document] = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            raw_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "page": page_num,
                        "doc_type": "pdf",
                    },
                )
            )
    return raw_docs


def _filter_chunks(chunks: List[Document]) -> List[Document]:
    """Drop empty or too-short chunks before embedding."""
    filtered = [
        c for c in chunks
        if len(c.page_content.strip()) >= config.MIN_CHUNK_CHARS
    ]
    for i, chunk in enumerate(filtered):
        chunk.metadata["chunk_index"] = i
    return filtered


# ── public API ────────────────────────────────────────────────────────────────

def load_file(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"[doc_loader] Loading file: {path.name}")

    if ext == ".pdf":
        raw_docs = _load_pdf(path)
        splitter = _make_splitter("pdf")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = splitter.split_documents(raw_docs)
    chunks = _filter_chunks(chunks)
    print(f"[doc_loader] → {len(chunks)} chunks from {path.name}")
    return chunks


def load_directory(dir_path: str,
                   recursive: bool = True) -> List[Document]:
    """
    Load all supported documents from a directory.
    Set recursive=True to traverse sub-folders.
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    pattern = "**/*" if recursive else "*"
    all_chunks: List[Document] = []

    for file in root.glob(pattern):
        if file.suffix.lower() in config.SUPPORTED_EXTENSIONS and file.is_file():
            try:
                all_chunks.extend(load_file(str(file)))
            except Exception as exc:
                print(f"[doc_loader] ⚠  Skipping {file.name}: {exc}")

    print(f"[doc_loader] Total chunks from directory: {len(all_chunks)}")
    return all_chunks


def load_files(file_paths: List[str]) -> List[Document]:
    """Load a list of file paths and return combined chunked Documents."""
    all_chunks: List[Document] = []
    for fp in file_paths:
        try:
            all_chunks.extend(load_file(fp))
        except Exception as exc:
            print(f"[doc_loader] ⚠  Skipping {fp}: {exc}")
    return all_chunks


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    docs = load_directory(target)
    print(f"\nLoaded {len(docs)} chunks total.")
    if docs:
        print(f"\nSample chunk:\n{docs[0].page_content[:400]}")
