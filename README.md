# RAG Customer Support Assistant
### AI-Powered Support System using LangGraph, ChromaDB, HITL & Streamlit

A production-grade Retrieval-Augmented Generation (RAG) system for customer support,
built as part of the Innomatics Research Labs internship project.

---

## Project Structure

```
rag_customer_support/
│
├── ingest.py          # PDF → Chunks → Embeddings → ChromaDB
├── rag_engine.py      # Retrieval + LLM answer generation
├── graph.py           # LangGraph workflow + conditional routing + HITL logic
├── app.py             # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Architecture Overview

```
User Query
    │
    ▼
[Streamlit UI]
    │
    ▼
[LangGraph Workflow]
    │
    ├─ process_query node
    │       ├── Detect intent (FAQ / complaint / complex / unknown)
    │       └── RAG Engine
    │               ├── ChromaDB retrieval (top-K chunks)
    │               ├── Confidence assessment (similarity score)
    │               └── LLM answer generation (Groq / OpenAI)
    │
    ▼ (conditional routing)
    │
    ├─── confident answer? ──► output_answer node ──► Display in UI
    │
    └─── low confidence?   ──► escalate_to_human node ──► HITL UI panel
                                                              │
                                                        Human agent types response
                                                              │
                                                        Integrated back into chat
```

---

## Setup

### 1. Clone / download this folder

```bash
cd rag_customer_support
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

---

## Running the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**In the sidebar:**
1. Upload your PDF knowledge base
2. Select LLM provider (Groq recommended)
3. Enter your API key
4. Click **Ingest PDF & Initialise**
5. Start chatting!

---

## How HITL Works

The Human-in-the-Loop escalation triggers when:

| Condition | Reason Code |
|-----------|-------------|
| No relevant chunks found in ChromaDB | `no_chunks_found` |
| Similarity score < 0.35 | `low_confidence` |
| LLM expresses uncertainty | `llm_uncertain` |
| Query detected as complex | `complex_query` |

When escalated:
1. The chat shows an orange escalation notice
2. A human agent input panel appears
3. Agent types a response and submits
4. The response is added to the chat as a human-verified answer

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Chunk size: 800 chars | Balances context richness vs retrieval precision |
| Overlap: 150 chars | Prevents losing context at chunk boundaries |
| BAAI/bge-small-en-v1.5 embeddings | Fast, lightweight, open-source semantic embedding model with strong retrieval quality |
| ChromaDB | Local persistence, zero infra, great LangChain integration |
| Groq + LLaMA3 | Free tier, ~300 tokens/sec, ideal for prototyping |
| Similarity threshold: 0.15 | Tuned to reduce unnecessary escalations while maintaining retrieval quality |

---

## Deliverables Map

| Deliverable | Covered In |
|------------|-----------|
| HLD — Architecture | This README + graph.py comments |
| LLD — Module design | All source files with inline documentation |
| Technical Documentation | Inline docstrings + this README |
| Working Project | `streamlit run app.py` |

---

## Final Features Implemented

- PDF Knowledge Base Ingestion
- Semantic Chunking
- Embedding Generation
- ChromaDB Vector Storage
- LangGraph Workflow Orchestration
- Intent Detection
- Conditional Routing
- Confidence-Based Escalation
- Human-in-the-Loop (HITL)
- Explainable Retrieval Viewer
- Streaming Responses
- Analytics Dashboard
- Multi-PDF Support
- Feedback Buttons
- Source Citations
