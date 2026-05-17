"""
rag_engine.py
-------------
Retrieval-Augmented Generation (RAG) Engine
Handles: Query → Retrieve relevant chunks → Build prompt → LLM → Answer
"""

import os
from typing import List, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from ingest import get_embeddings, load_vectorstore, COLLECTION_NAME, CHROMA_PERSIST_DIR
import logging
logging.basicConfig(level=logging.INFO)
# ── Configuration ──────────────────────────────────────────────────────────────
TOP_K                  = 6        # number of chunks to retrieve
SIMILARITY_THRESHOLD   = 0.15     # below this → low confidence → escalate
MAX_CONTEXT_CHARS      = 4000     # trim context if too long


# ── Prompt Template ────────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful customer support assistant.
Use ONLY the provided context to answer the question.
If the context does not contain enough information, say exactly:
"I don't have enough information to answer this confidently."

Context:
{context}

Customer Question: {question}

Answer (be concise and helpful):"""
)


def get_llm(provider: str = "groq"):
    """
    Return LLM instance.
    Supports: 'groq' (free, fast) or 'openai'
    Set GROQ_API_KEY or OPENAI_API_KEY in environment.
    """
    if provider == "groq":
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


class RAGEngine:
    """
    Core RAG engine.
    
    Flow:
    1. receive_query(question)
    2. retrieve_chunks(question) → top-K relevant chunks + similarity scores
    3. assess_confidence(scores) → decide route
    4. generate_answer(chunks, question) → LLM response
    """

    def __init__(self, llm_provider: str = "groq"):
        self.embeddings  = get_embeddings()
        self.vectorstore = load_vectorstore(self.embeddings)
        self.retriever   = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": TOP_K,
                "score_threshold": 0.0,   # get all scores; filter manually
            },
        )
        self.llm    = get_llm(llm_provider)
        

    def rewrite_query(self, question: str) -> str:
        """
        Normalize and improve user queries before retrieval.
        """

        question = question.strip().lower()

        replacements = {
            "cancel plan": "cancel subscription",
            "money back": "refund",
            "charged": "billing charge",
        }

        for k, v in replacements.items():
            question = question.replace(k, v)

        return question
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve top-K chunks with similarity scores."""
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=TOP_K
        )
        return results  # [(Document, score), ...]

    def assess_confidence(self, scored_results: List[Tuple[Document, float]]) -> dict:
        """
        Determine retrieval confidence.
        
        Returns:
            {
                "confident": bool,
                "reason": str,
                "top_score": float,
                "chunks": [Document]
            }
        """
        if not scored_results:
            return {
                "confident": False,
                "reason": "no_chunks_found",
                "top_score": 0.0,
                "chunks": [],
            }

        top_score = scored_results[0][1]
        chunks    = [doc for doc, _ in scored_results]

        if top_score < SIMILARITY_THRESHOLD:
            return {
                "confident": False,
                "reason": "low_confidence",
                "top_score": top_score,
                "chunks": chunks,
            }

        return {
            "confident": True,
            "reason": "retrieved",
            "top_score": top_score,
            "chunks": chunks,
        }

    def build_context(self, chunks: List[Document]) -> str:
        """
        Build formatted retrieval context.
        """

        context_parts = []

        for c in chunks:

            page = c.metadata.get("page", "?")
            source = c.metadata.get("source_file", "unknown")

            context_parts.append(
                f"[Source: {source} | Page: {page}]\n{c.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return context[:MAX_CONTEXT_CHARS]

    def generate_answer(self, chunks: List[Document], question: str) -> str:
        """
        Run RAG chain: context + question → LLM → answer string.
        """

        context = self.build_context(chunks)

        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        try:

            response = self.llm.invoke(prompt)

            answer = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            return answer

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            return "The AI system is temporarily unavailable."
    

    def requires_human(self, question: str) -> bool:

        sensitive_keywords = [
            "legal",
            "lawsuit",
            "court",
            "fraud",
            "sue",
            "compensation",
            "police",
            "compliance",
        ]

        return any(k in question.lower() for k in sensitive_keywords)
    
    def query(self, question: str) -> dict:
        """
        Full RAG query pipeline.
        """

        logging.info(f"Received Question: {question}")

        # Query rewriting
        question = self.rewrite_query(question)

        # Sensitive query routing
        if self.requires_human(question):

            logging.warning("Sensitive query detected.")

            return {
                "answer": None,
                "confident": False,
                "reason": "sensitive_query",
                "top_score": 0.0,
                "sources": [],
                "retrieved_chunks": [],
            }

        # Retrieve chunks
        scored_results = self.retrieve_with_scores(question)

        assessment = self.assess_confidence(scored_results)

        logging.info(f"Top similarity score: {assessment['top_score']}")

        # Escalate if low confidence
        if not assessment["confident"]:

            return {
                "answer": None,
                "confident": False,
                "reason": assessment["reason"],
                "top_score": assessment["top_score"],
                "sources": [],
                "retrieved_chunks": assessment["chunks"],
            }

        # Generate answer
        answer = self.generate_answer(
            assessment["chunks"],
            question
        )

        # Build source citations
        sources = list({
            f"{c.metadata.get('source_file', 'unknown')} | Page {c.metadata.get('page', '?')}"
            for c in assessment["chunks"]
        })

        # Detect LLM uncertainty
        uncertain_phrases = [
            "don't have enough information",
            "not in the context",
            "cannot answer",
            "unsure",
        ]

        llm_uncertain = any(
            p in answer.lower()
            for p in uncertain_phrases
        )

        return {
            "answer": answer,
            "confident": not llm_uncertain,
            "reason": "llm_uncertain" if llm_uncertain else "answered",
            "top_score": assessment["top_score"],
            "sources": sources,
            "retrieved_chunks": assessment["chunks"],
        }
