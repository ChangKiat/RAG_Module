# 🦙 RAG-Llama

A fully local Retrieval-Augmented Generation (RAG) system powered by **Llama** (via Ollama), **ChromaDB**, and **LangChain** — with a polished **Streamlit chat UI** and an optional **FastAPI REST API**.

---

## 📁 Project Structure

```
rag-llama/
├── app.py                   ← Streamlit chat UI (start here)
├── config.py                ← All settings in one place
├── requirements.txt
│
├── ingest/
│   ├── web_loader.py        ← Scrape single pages or crawl entire websites
│   └── doc_loader.py        ← Load PDFs, TXT, MD, HTML, CSV
│
├── vectorstore/
│   └── chroma_store.py      ← ChromaDB embed & retrieve wrapper
│
├── rag/
│   └── chain.py             ← RAG chain: retrieval → Llama → answer
│
└── api/
    └── main.py              ← FastAPI REST API (optional)
```

---

## ⚙️ Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Then pull the Llama model:

```bash
ollama pull llama3
```

> **Tip:** For faster embeddings, also pull `nomic-embed-text` and set `EMBED_MODEL = "nomic-embed-text"` in `config.py`.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Chat UI

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### What you can do in the UI:

| Action | How |
|--------|-----|
| Ingest a single web page | Sidebar → Website tab → paste URL → Ingest |
| Crawl an entire website | Sidebar → Website tab → select "Crawl" → set depth & page limit |
| Ingest a PDF / TXT / MD | Sidebar → File tab → upload → Ingest |
| Ask questions | Type in the chat box and hit Ask |
| See sources | Sources are shown as pills below each AI answer |
| Reset store | Sidebar → Danger Zone → Reset Vector Store |

---

## 🌐 Running the REST API

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Example API calls

**Ingest a URL:**
```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "crawl": true, "max_depth": 2}'
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this website about?", "k": 5}'
```

**Streaming answer:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the content", "stream": true}'
```

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `llama3` | Ollama model for generation |
| `EMBED_MODEL` | `llama3` | Ollama model for embeddings |
| `CHUNK_SIZE` | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |
| `MAX_CRAWL_DEPTH` | `3` | Link depth for website crawling |
| `MAX_CRAWL_PAGES` | `50` | Hard cap on pages crawled |

---

## 🐍 Using the modules in your own code

```python
from ingest.web_loader import load_website
from ingest.doc_loader import load_file
from vectorstore.chroma_store import add_documents
from rag.chain import ask

# Ingest a website
docs = load_website("https://example.com", max_depth=2)
add_documents(docs)

# Ingest a PDF
docs = load_file("report.pdf")
add_documents(docs)

# Ask a question
result = ask("What does this site say about pricing?")
print(result["answer"])
print(result["sources"])
```

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on Ollama | Run `ollama serve` first |
| Slow embeddings | Switch to `nomic-embed-text` in `config.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Empty answers | Make sure you've ingested content before querying |

---

## 📄 License

MIT — use freely, modify as needed.
