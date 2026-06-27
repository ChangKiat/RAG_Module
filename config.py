# ─────────────────────────────────────────────
#  RAG-Llama  ·  Central Configuration
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv
load_dotenv()

# Chat (Gemini)
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL       = "gemini-2.5-flash"

# Embeddings (Ollama / Llama ecosystem)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = "nomic-embed-text"
EMBED_DIMENSION = 768

# ChromaDB persistence directory
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "rag_collection"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = "rag-llama"

# Chunking (web + default file types)
CHUNK_SIZE       = 1200
CHUNK_OVERLAP    = 200
# PDF-specific chunking (larger chunks for long-form docs)
PDF_CHUNK_SIZE   = 1500
PDF_CHUNK_OVERLAP = 250
MIN_CHUNK_CHARS  = 80

# Retrieval
TOP_K            = 5
FETCH_K          = 15

# Web crawler settings
MAX_CRAWL_DEPTH  = 3
MAX_CRAWL_PAGES  = 50

# Supported document extensions for doc_loader (PDF + web via web_loader)
SUPPORTED_EXTENSIONS = [".pdf"]
