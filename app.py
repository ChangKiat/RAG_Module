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
from vectorstore.pinecone_store import add_documents, collection_count, reset_store
from rag.chain           import ask_stream, get_sources_for

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · AI",
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

/* ── native chat messages ─────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
}

[data-testid="stChatMessage"] {
    max-width: 820px;
    margin: 0 auto 1rem auto;
    padding: 0.25rem 0;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px 18px;
    line-height: 1.65;
    font-size: 0.95rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: var(--user-bg);
    border-color: #3a4560;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.5rem;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
    margin-bottom: 0;
}

[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    font-size: 1.1rem !important;
}

/* chat input composer */
[data-testid="stChatInput"] {
    max-width: 820px;
    margin: 0 auto;
}
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232, 162, 69, 0.2) !important;
}

/* scrollable chat container */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar {
    width: 6px;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}

/* empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 320px;
    padding: 2rem;
    margin: 0 auto;
    max-width: 480px;
    border: 1px dashed var(--border);
    border-radius: 20px;
    background: linear-gradient(160deg, #161922 0%, #12151c 100%);
}
.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.9;
}
.empty-state h3 {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.5rem 0;
}
.empty-state p {
    font-size: 0.88rem;
    color: var(--muted);
    margin: 0;
    line-height: 1.5;
}

/* source pills */
.sources-row {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
}
.sources-label {
    display: block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.source-pill {
    display: inline-block;
    background: #1e2435;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 11px;
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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, sources}
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []  # display log
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()  # track filenames already ingested
if "ingested_urls" not in st.session_state:
    st.session_state.ingested_urls = set()  # track URLs already ingested


def _render_sources(sources: list) -> None:
    if not sources:
        return
    pills = "".join(f'<span class="source-pill">{s}</span>' for s in sources)
    st.markdown(
        f'<div class="sources-row"><span class="sources-label">Sources</span>{pills}</div>',
        unsafe_allow_html=True,
    )


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
                url = url_input.strip()

                if url in st.session_state.ingested_urls and not reset_on_web:
                    st.info(f"⏭ Already ingested: {url} — tick 'Reset store' to force re-ingest.")
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
                            st.session_state.ingested_urls.add(url)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    with tab_file:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "html", "csv"],
            key="file_upload",
            accept_multiple_files=True,   # ← this is the key change
        )
        reset_on_file = st.checkbox("🗑 Reset store before ingesting", key="reset_file")

        if st.button("Ingest Files ▶", key="btn_file"):  # ← indented inside tab_file
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                import tempfile, pathlib
                all_docs = []
                failed = []
                skipped = []

                progress = st.progress(0, text="Starting …")

                for i, uploaded in enumerate(uploaded_files):
                    progress.progress(
                        int((i / len(uploaded_files)) * 100),
                        text=f"Processing {uploaded.name} …"
                    )
                    if uploaded.name in st.session_state.ingested_files:
                        skipped.append(uploaded.name)
                        continue

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=pathlib.Path(uploaded.name).suffix,
                    ) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    try:
                        docs = load_file(tmp_path)
                        all_docs.extend(docs)
                        st.session_state.ingested_sources.append(f"📄 {uploaded.name}")
                        st.session_state.ingested_files.add(uploaded.name)  # mark as done
                    except Exception as e:
                        failed.append(f"{uploaded.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

                progress.progress(100, text="Embedding …")

                if all_docs:
                    added = add_documents(all_docs, reset=reset_on_file)
                    progress.empty()
                    st.success(f"✓ Added {added} chunks from {len(uploaded_files) - len(failed)} file(s)")

                if skipped:
                    st.info(f"⏭ Skipped {len(skipped)} already ingested: {', '.join(skipped)}")

                if failed:
                    for f in failed:
                        st.error(f"⚠ Failed: {f}")


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
    Ollama embeddings · Gemini chat · Pinecone
</p>
""", unsafe_allow_html=True)
st.divider()

chat_container = st.container(height=520)
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <h3>Start a conversation</h3>
            <p>Ingest a PDF or website in the sidebar, then ask a question here.</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources(msg.get("sources", []))

if question := st.chat_input("Ask anything about your ingested content …", key="chat_input"):
    q = question.strip()
    if q:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="✨"):
            answer = st.write_stream(ask_stream(q))

        source_docs = get_sources_for(q)
        sources = list({
            d.metadata.get("source", "")
            for d in source_docs
            if d.metadata.get("source")
        })

        st.session_state.messages.append({"role": "user", "content": q, "sources": []})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
