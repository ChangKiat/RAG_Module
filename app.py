"""
app.py  ·  RAG-Llama Chat UI
─────────────────────────────
Run with:
    streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from streamlit_extras.colored_header import colored_header   # pip install streamlit-extras

from ingest.web_loader   import load_single_url, load_website
from ingest.doc_loader   import load_file
from vectorstore.chroma_store import add_documents, collection_count, reset_store
from rag.chain           import ask_stream, ask

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · Llama",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── palette ─────────────────────────────────────────────── */
:root {
    --bg:       #0d0f14;
    --surface:  #161922;
    --border:   #2a2f3d;
    --accent:   #e8a245;
    --accent2:  #5b8dee;
    --text:     #e2e6f0;
    --muted:    #7a8199;
    --user-bg:  #1e2233;
    --ai-bg:    #13161e;
}
.stApp { background: var(--bg); color: var(--text); }

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── chat bubbles ────────────────────────────────────────── */
.chat-bubble {
    border-radius: 16px;
    padding: 14px 18px;
    margin: 6px 0;
    line-height: 1.65;
    font-size: 0.95rem;
}
.bubble-user {
    background: var(--user-bg);
    border: 1px solid var(--border);
    margin-left: 8%;
}
.bubble-ai {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    margin-right: 8%;
}
.bubble-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.label-user { color: var(--accent2); }
.label-ai   { color: var(--accent); }

/* source pill */
.source-pill {
    display: inline-block;
    background: #1e2435;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: var(--accent2);
    margin: 3px 4px 3px 0;
    word-break: break-all;
}

/* stat badge */
.stat-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e2435, #161922);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 16px;
    text-align: center;
    margin: 4px 0;
    width: 100%;
}
.stat-num {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .08em;
}

/* buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .04em;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: .85; }

/* danger button override */
.danger-btn > button {
    background: #3d1e1e !important;
    color: #e57373 !important;
    border: 1px solid #5a2828 !important;
}

/* input */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* divider */
hr { border-color: var(--border) !important; }

/* scrollable chat area */
.chat-area {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, sources}
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []  # display log


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar — Knowledge Base Management
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦙 RAG · Llama")
    st.markdown("<p style='color:#7a8199;font-size:.82rem;margin-top:-8px;'>Local AI · Private by default</p>", unsafe_allow_html=True)
    st.divider()

    # vector count badge
    count = collection_count()
    st.markdown(f"""
    <div class="stat-badge">
        <div class="stat-num">{count:,}</div>
        <div class="stat-label">vectors in store</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    # ── ingest tabs ──────────────────────────────────────────────────────────
    tab_web, tab_file = st.tabs(["🌐  Website", "📄  File"])

    with tab_web:
        url_input = st.text_input("URL", placeholder="https://example.com", key="url_input")
        crawl_mode = st.radio("Mode", ["Single page", "Crawl whole site"], horizontal=True)
        if crawl_mode == "Crawl whole site":
            crawl_depth = st.slider("Depth", 1, 5, 2)
            crawl_pages = st.slider("Max pages", 5, 100, 30)
        reset_on_web = st.checkbox("🗑 Reset store before ingesting", key="reset_web")

        if st.button("Ingest Website ▶", key="btn_web"):
            if not url_input.strip():
                st.warning("Please enter a URL.")
            else:
                with st.spinner("Loading website content …"):
                    try:
                        if crawl_mode == "Crawl whole site":
                            docs = load_website(url_input.strip(), max_depth=crawl_depth, max_pages=crawl_pages)
                        else:
                            docs = load_single_url(url_input.strip())
                        added = add_documents(docs, reset=reset_on_web)
                        st.success(f"✓ Added {added} chunks from {url_input}")
                        st.session_state.ingested_sources.append(f"🌐 {url_input}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_file:
        uploaded = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md", "html", "csv"],
            key="file_upload",
        )
        reset_on_file = st.checkbox("🗑 Reset store before ingesting", key="reset_file")

        if st.button("Ingest File ▶", key="btn_file"):
            if not uploaded:
                st.warning("Please upload a file first.")
            else:
                # save to temp and load
                import tempfile, pathlib
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=pathlib.Path(uploaded.name).suffix,
                ) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                with st.spinner(f"Processing {uploaded.name} …"):
                    try:
                        docs = load_file(tmp_path)
                        added = add_documents(docs, reset=reset_on_file)
                        st.success(f"✓ Added {added} chunks from {uploaded.name}")
                        st.session_state.ingested_sources.append(f"📄 {uploaded.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        os.unlink(tmp_path)

    st.divider()

    # ── ingested sources log ─────────────────────────────────────────────────
    if st.session_state.ingested_sources:
        st.markdown("**Ingested this session**")
        for s in st.session_state.ingested_sources:
            st.markdown(f"<div class='source-pill'>{s}</div>", unsafe_allow_html=True)
        st.write("")

    # ── danger zone ──────────────────────────────────────────────────────────
    with st.expander("⚠  Danger Zone"):
        st.markdown("<p style='font-size:.8rem;color:#e57373;'>This will permanently wipe the entire vector store.</p>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("Reset Vector Store 🗑", key="btn_reset"):
                reset_store()
                st.session_state.ingested_sources = []
                st.success("Store wiped.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Clear Chat History", key="btn_clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Main — Chat Interface
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2rem;font-weight:800;margin-bottom:0;'>
    Ask your Knowledge Base
</h1>
<p style='color:#7a8199;margin-top:4px;font-size:.9rem;'>
    Powered by Llama · ChromaDB · LangChain
</p>
""", unsafe_allow_html=True)
st.divider()

# render past messages
chat_html = '<div class="chat-area">'
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f"""
        <div class="chat-bubble bubble-user">
            <div class="bubble-label label-user">You</div>
            {msg["content"]}
        </div>"""
    else:
        sources_html = ""
        if msg.get("sources"):
            pills = "".join(f'<span class="source-pill">🔗 {s}</span>' for s in msg["sources"])
            sources_html = f"<div style='margin-top:10px;'>{pills}</div>"
        chat_html += f"""
        <div class="chat-bubble bubble-ai">
            <div class="bubble-label label-ai">🦙 Llama</div>
            {msg["content"]}{sources_html}
        </div>"""
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── input bar ────────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([9, 1])
with col_q:
    question = st.text_input(
        "question",
        label_visibility="collapsed",
        placeholder="Ask anything about your ingested content …",
        key="question_input",
    )
with col_btn:
    send = st.button("Ask", key="btn_ask", use_container_width=True)

# ── handle send ───────────────────────────────────────────────────────────────
if send and question.strip():
    st.session_state.messages.append({"role": "user", "content": question.strip(), "sources": []})

    with st.spinner("Thinking …"):
        result = ask(question.strip())
        answer  = result["answer"]
        sources = result["sources"]

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
