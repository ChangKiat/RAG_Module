# ─────────────────────────────────────────────
#  RAG-Llama  ·  Central Configuration
# ─────────────────────────────────────────────

# Ollama model to use for generation & embeddings
#LLM_MODEL = "llama3.1:8b" 
#EMBED_MODEL      = "nomic-embed-text"          # or "nomic-embed-text" for faster embeddings
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL       = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# Ollama base URL (default local)
OLLAMA_BASE_URL  = "http://localhost:11434"

# ChromaDB persistence directory
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "rag_collection"

# Chunking settings
CHUNK_SIZE       = 1500
CHUNK_OVERLAP    = 200

# Retrieval settings
TOP_K            = 10                # number of chunks returned per query

# Web crawler settings
MAX_CRAWL_DEPTH  = 3                 # how deep to follow links
MAX_CRAWL_PAGES  = 50                # hard cap on pages crawled

# Supported document extensions for doc_loader
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".html", ".csv"]
