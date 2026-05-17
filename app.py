"""
app.py
------
Streamlit Customer Support Assistant
Integrates RAG pipeline + LangGraph workflow + HITL escalation UI

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import time

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Customer Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .answer-box {
        background-color: #f3f4f6;
        color: #111827;
        border-left: 4px solid #2563eb;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        font-size: 16px;
        line-height: 1.6;
    }
    .escalation-box {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 16px;
        border-radius: 6px;
        margin: 12px 0;
    }
    .human-answer-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 16px;
        border-radius: 6px;
        margin: 12px 0;
    }
    .meta-tag {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 8px;
    }
    .score-pill {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ───────────────────────────────────────────────
def init_session():
    defaults = {
        "rag":              None,
        "ingested":         False,
        "chat_history":     [],    # [{role, content, meta}]
        "pending_escalation": None,  # state dict awaiting human input
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Lazy Imports (avoid slow imports on every rerun) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_engine(llm_provider: str):
    from rag_engine import RAGEngine
    return RAGEngine(llm_provider=llm_provider)


@st.cache_resource(show_spinner=False)
def run_ingestion(pdf_path: str):

    from ingest import ingest_pipeline

    return ingest_pipeline([pdf_path])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/robot-2.png", width=60)
    st.title("⚙️ Configuration")

    st.subheader("1. Upload Knowledge Base PDF")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    st.subheader("2. LLM Provider")
    llm_provider = st.selectbox(
        "Choose LLM",
        ["groq", "openai"],
        help="Groq (free, fast) uses LLaMA3. OpenAI uses GPT-3.5.",
    )

    st.subheader("3. API Key")
    if llm_provider == "groq":
        api_key = st.text_input("GROQ_API_KEY", type="password",
                                help="Get free key at console.groq.com")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        api_key = st.text_input("OPENAI_API_KEY", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    ingest_btn = st.button("🚀 Ingest PDF & Initialise", use_container_width=True,
                            disabled=(uploaded_pdf is None or not api_key))

    if ingest_btn and uploaded_pdf:
        with st.spinner("Loading PDF, chunking, embedding... this may take a minute."):
            # Save uploaded file temporarily
            import tempfile
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_pdf.read())

            run_ingestion(tmp_path)
            st.session_state["rag"]      = load_rag_engine(llm_provider)
            st.session_state["ingested"] = True
        st.success("✅ Knowledge base ready!")

    st.divider()
    st.subheader("ℹ️ System Info")
    st.caption("""
    **Architecture**
    - Embeddings: BAAI/bge-small-en-v1.5
    - Vector DB: ChromaDB
    - Workflow: LangGraph
    - HITL: Human escalation UI
    """)

    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        st.session_state["pending_escalation"] = None
        st.rerun()

    st.divider()
    st.subheader("📊 Analytics")

    total_msgs = len(st.session_state["chat_history"])

    escalations = sum(
        1
        for m in st.session_state["chat_history"]
        if "Escalating" in m.get("content", "")
    )

    st.metric("Messages", total_msgs)
    st.metric("Escalations", escalations)


# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 RAG Customer Support Assistant</p>', unsafe_allow_html=True)
st.caption("Powered by LangGraph · ChromaDB · LLM · Human-in-the-Loop")
with st.expander("💡 Example Questions"):

    st.markdown("""
    - How can I reset my password?
    - What payment methods are accepted?
    - How do I request a refund?
    - Why was my payment declined?
    - Can I cancel my subscription anytime?
    - What happens after escalation?
    """)
if not st.session_state["ingested"]:
    st.info("👈 Upload a PDF knowledge base and enter your API key in the sidebar to get started.")
    st.stop()

# ── Chat History Display ───────────────────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.markdown(f'<p class="meta-tag">{msg["meta"]}</p>', unsafe_allow_html=True)


# ── HITL: Pending Escalation Input ────────────────────────────────────────────
if st.session_state["pending_escalation"]:
    escalation_state = st.session_state["pending_escalation"]

    st.markdown("""
    <div class="escalation-box">
    <b>🔶 Human Review Required</b><br>
    The AI could not answer this with sufficient confidence.<br>
    Please provide a human response below.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Customer Question:** {escalation_state['question']}")
    reason_map = {
        "no_chunks_found": "No relevant information found in the knowledge base.",
        "low_confidence":  f"Low retrieval confidence (score: {escalation_state['top_score']:.2f}).",
        "complex_query":   "Query is complex and requires human judgment.",
        "llm_uncertain":   "The AI model expressed uncertainty in its answer.",
    }
    st.caption(f"Escalation reason: {reason_map.get(escalation_state['reason'], escalation_state['reason'])}")

    human_resp = st.text_area("✍️ Human Agent Response:", height=120,
                               placeholder="Type the human agent's answer here...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Submit Human Response", use_container_width=True):
            if human_resp.strip():
                # Integrate human response into chat history
                st.session_state["chat_history"].append({
                    "role":    "assistant",
                    "content": f"👤 **Human Agent:** {human_resp}",
                    "meta":    "✅ Handled by human agent via HITL escalation",
                })
                st.session_state["pending_escalation"] = None
                st.rerun()
            else:
                st.warning("Please enter a response before submitting.")
    with col2:
        if st.button("⏭️ Skip / Mark Unresolved", use_container_width=True):
            st.session_state["chat_history"].append({
                "role":    "assistant",
                "content": "⚠️ This query has been marked as unresolved and will be reviewed later.",
                "meta":    "Status: Escalated — Unresolved",
            })
            st.session_state["pending_escalation"] = None
            st.rerun()


# ── Chat Input ─────────────────────────────────────────────────────────────────
if not st.session_state["pending_escalation"]:
    user_input = st.chat_input("Ask a question about your product or service...")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["chat_history"].append({
            "role":    "user",
            "content": user_input,
        })

        # Run graph
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from graph import run_graph
                result = run_graph(user_input, st.session_state["rag"])

            if not result["escalated"]:

                # Streaming response
                placeholder = st.empty()

                partial = ""

                for word in result["answer"].split():

                    partial += word + " "

                    placeholder.markdown(f"""
                    <div class="answer-box">
                    {partial}
                    </div>
                    """, unsafe_allow_html=True)

                    time.sleep(0.02)

                # Metadata
                sources_str = (
                    ", ".join(result["sources"])
                    if result["sources"]
                    else "N/A"
                )

                meta = (
                    f"🔍 Retrieval score: {result['top_score']:.2f} · "
                    f"📚 Sources Used: {sources_str} · "
                    f"🏷️ Intent: {result['intent']}"
                )

                st.markdown(
                    f'<p class="meta-tag">{meta}</p>',
                    unsafe_allow_html=True
                )

                # Confidence bar
                confidence = min(result["top_score"], 1.0)

                st.progress(confidence)

                st.caption(
                    f"Confidence Score: {confidence:.2f}"
                )

                # Retrieved chunks
                with st.expander("📚 Retrieved Context Chunks"):

                    for idx, chunk in enumerate(
                        result["retrieved_chunks"],
                        start=1
                    ):

                        source = chunk.metadata.get(
                            "source_file",
                            "unknown"
                        )

                        page = chunk.metadata.get(
                            "page",
                            "?"
                        )

                        st.markdown(
                            f"### Chunk {idx} (Page {page})"
                        )

                        st.caption(source)

                        st.write(chunk.page_content)

                        st.divider()

                
                # Feedback buttons
                col1, col2 = st.columns(2)

                with col1:
                    st.button(
                        "👍 Helpful",
                        key=f"helpful_{len(st.session_state['chat_history'])}"
                    )

                with col2:
                    st.button(
                        "👎 Not Helpful",
                        key=f"not_helpful_{len(st.session_state['chat_history'])}"
                    )
                

                # Save chat history
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": meta,
                })

            else:
                # Escalation triggered
                st.markdown("""
                <div class="escalation-box">
                🔶 <b>Escalating to human agent...</b><br>
                This query requires human review. A support agent will respond shortly.
                </div>
                """, unsafe_allow_html=True)

                st.session_state["chat_history"].append({
                    "role":    "assistant",
                    "content": "🔶 **Escalating to human agent.** This query requires human review.",
                    "meta":    f"Reason: {result['reason']}",
                })
                st.session_state["pending_escalation"] = result
                st.rerun()
