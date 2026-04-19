"""
ingest/doc_loader.py
────────────────────
Load local documents (PDF, TXT, MD, HTML, CSV) and return chunked
LangChain Documents ready for embedding.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _loader_for(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    elif ext in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8")
    elif ext == ".html":
        return UnstructuredHTMLLoader(str(path))
    elif ext == ".csv":
        return CSVLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── public API ────────────────────────────────────────────────────────────────

def load_file(file_path: str) -> List[Document]:
    """Load a single file and return chunked Documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[doc_loader] Loading file: {path.name}")
    loader = _loader_for(path)
    raw_docs = loader.load()

    splitter = _make_splitter()
    chunks = splitter.split_documents(raw_docs)
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
