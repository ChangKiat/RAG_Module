"""
api/main.py
───────────
FastAPI REST API exposing ingest + query endpoints.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl

from ingest.web_loader import load_single_url, load_website, load_urls
from ingest.doc_loader  import load_file, load_directory
from vectorstore.chroma_store import add_documents, collection_count, reset_store
from rag.chain import ask, ask_stream

app = FastAPI(
    title="RAG-Llama API",
    description="Retrieval-Augmented Generation powered by Llama + ChromaDB",
    version="1.0.0",
)


# ── schemas ───────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str
    crawl: bool = False          # True = recursive crawl
    max_depth: int = 2
    max_pages: int = 30
    reset: bool = False          # wipe store before ingesting


class IngestURLsRequest(BaseModel):
    urls: List[str]
    reset: bool = False


class IngestFileRequest(BaseModel):
    path: str                    # local path
    reset: bool = False


class IngestDirRequest(BaseModel):
    path: str
    recursive: bool = True
    reset: bool = False


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    stream: bool = False


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "vectors": collection_count()}


@app.post("/ingest/url")
def ingest_url(req: IngestURLRequest):
    try:
        if req.crawl:
            docs = load_website(req.url, max_depth=req.max_depth, max_pages=req.max_pages)
        else:
            docs = load_single_url(req.url)
        added = add_documents(docs, reset=req.reset)
        return {"status": "ok", "chunks_added": added, "total_vectors": collection_count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/urls")
def ingest_urls(req: IngestURLsRequest):
    try:
        docs = load_urls(req.urls)
        added = add_documents(docs, reset=req.reset)
        return {"status": "ok", "chunks_added": added, "total_vectors": collection_count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
def ingest_file(req: IngestFileRequest):
    try:
        docs = load_file(req.path)
        added = add_documents(docs, reset=req.reset)
        return {"status": "ok", "chunks_added": added, "total_vectors": collection_count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/directory")
def ingest_directory(req: IngestDirRequest):
    try:
        from ingest.doc_loader import load_directory
        docs = load_directory(req.path, recursive=req.recursive)
        added = add_documents(docs, reset=req.reset)
        return {"status": "ok", "chunks_added": added, "total_vectors": collection_count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    if req.stream:
        def _gen():
            for token in ask_stream(req.question, k=req.k):
                yield token
        return StreamingResponse(_gen(), media_type="text/plain")

    result = ask(req.question, k=req.k)
    return {
        "answer":  result["answer"],
        "sources": result["sources"],
    }


@app.delete("/store")
def delete_store():
    reset_store()
    return {"status": "ok", "message": "Vector store has been reset."}
