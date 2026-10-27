"""Customer Support RAG pipeline (from Experiment 6) with a pluggable LLM backend.

Retrieval (FAISS + MiniLM embeddings) is kept separate from generation. The backend is
chosen with LLM_BACKEND: "groq" (cloud API, default) or "mock" (offline, for testing).
"""
import logging
import os
from pathlib import Path
from typing import List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()
log = logging.getLogger("support_rag")

BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "support_faq.txt"
PAGE_MARKER = "=====PAGE====="

GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

FALLBACK_TEXT = (
    "I couldn't find a reliable answer in the available FAQ. "
    "Please contact the support team through the Help section. "
    "Do not share your password, one-time code, or full payment-card details."
)


class BackendError(Exception):
    """The LLM backend failed. The message is safe to log; never contains secrets."""


# ---------------------------------------------------------------- LLM backends
def groq_generate(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise BackendError("GROQ_API_KEY is not configured")
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1000,
            },
            timeout=LLM_TIMEOUT,
        )
    except requests.RequestException as e:
        raise BackendError(f"network error: {type(e).__name__}") from None
    if r.status_code != 200:
        # status code only: the response body could echo request details
        raise BackendError(f"provider returned HTTP {r.status_code}")
    try:
        return r.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError):
        raise BackendError("unexpected provider response format") from None


def mock_generate(prompt: str) -> str:
    """Offline stand-in: answers with the first context passage, or INSUFFICIENT_CONTEXT."""
    context = prompt.split("FAQ CONTEXT:")[1].split("CUSTOMER QUESTION:")[0].strip()
    question = prompt.split("CUSTOMER QUESTION:")[1].split("ANSWER:")[0].lower()
    if "insurance" in question or not context:
        return "INSUFFICIENT_CONTEXT"
    return "[mock backend] " + context[:200]


BACKENDS = {"groq": groq_generate, "mock": mock_generate}


def generate(prompt: str) -> str:
    return BACKENDS[os.getenv("LLM_BACKEND", "groq")](prompt)


# ----------------------------------------------------------- knowledge base
def load_documents() -> List[Document]:
    raw = KB_PATH.read_text(encoding="utf-8")
    docs = []
    for block in raw.split(PAGE_MARKER):
        block = block.strip()
        if not block:
            continue
        topic, source, *rest = block.split("\n")
        docs.append(Document(
            page_content=f"{topic}\n" + "\n".join(rest),
            metadata={"topic": topic, "source": source.removeprefix("Source: ").strip()},
        ))
    return docs


class SupportState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    context: str
    sources: List[str]
    draft: str
    retry_count: int
    final_answer: str


class SupportRAG:
    """Builds the embedding model, FAISS index and LangGraph once; reuse for every request."""

    def __init__(self, k: int = 3):
        chunks = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60).split_documents(
            load_documents())
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        self.num_chunks = len(chunks)
        self.graph = self._build_graph()
        log.info("RAG ready: %d chunks indexed", self.num_chunks)

    # --- nodes
    @staticmethod
    def classify_intent(state: SupportState):
        q = state["question"].lower()
        if any(w in q for w in ["refund", "return", "exchange"]):
            intent = "returns_refunds"
        elif any(w in q for w in ["track", "tracking", "shipment", "delivery", "order status"]):
            intent = "order_tracking"
        elif any(w in q for w in ["password", "sign in", "login", "account", "verification"]):
            intent = "account_access"
        elif any(w in q for w in ["charged", "payment", "billing", "invoice", "card"]):
            intent = "billing_payments"
        elif any(w in q for w in ["website", "error", "loading", "browser", "technical"]):
            intent = "technical_support"
        else:
            intent = "general_support"
        return {"intent": intent, "search_query": state["question"],
                "retry_count": state.get("retry_count", 0)}

    def retrieve_context(self, state: SupportState):
        docs = self.retriever.invoke(state["search_query"])
        return {"context": "\n\n".join(d.page_content for d in docs),
                "sources": [d.metadata.get("source", "unknown") for d in docs]}

    @staticmethod
    def draft_answer(state: SupportState):
        prompt = f"""You are a polite customer-support assistant.
Use only the FAQ CONTEXT to answer the CUSTOMER QUESTION.
Do not invent order status, refund approvals, timelines, or company policies.
If the context does not contain the answer, output exactly:
INSUFFICIENT_CONTEXT
Do not ask for passwords, one-time codes, or full payment-card details.
Keep the answer clear and concise.

FAQ CONTEXT:
{state.get("context", "")}

CUSTOMER QUESTION:
{state["question"]}

ANSWER:"""
        return {"draft": generate(prompt)}

    @staticmethod
    def review_answer(state: SupportState):
        draft = state.get("draft", "").strip()
        return {"final_answer": "" if "INSUFFICIENT_CONTEXT" in draft else draft}

    @staticmethod
    def route_after_review(state: SupportState):
        if state.get("final_answer"):
            return "finish"
        return "retry" if state.get("retry_count", 0) < 1 else "fallback"

    @staticmethod
    def rewrite_query(state: SupportState):
        terms = {
            "returns_refunds": "return eligibility refund process",
            "order_tracking": "order dispatch tracking shipment status",
            "account_access": "account sign in password reset",
            "billing_payments": "billing payment transaction",
            "technical_support": "website technical troubleshooting",
            "general_support": "customer support help contact",
        }
        return {"search_query": state["question"] + " " + terms.get(state.get("intent"), "customer support"),
                "retry_count": state.get("retry_count", 0) + 1}

    @staticmethod
    def fallback_answer(state: SupportState):
        return {"final_answer": FALLBACK_TEXT}

    def _build_graph(self):
        g = StateGraph(SupportState)
        g.add_node("classify_intent", self.classify_intent)
        g.add_node("retrieve", self.retrieve_context)
        g.add_node("draft_answer", self.draft_answer)
        g.add_node("review_answer", self.review_answer)
        g.add_node("rewrite_query", self.rewrite_query)
        g.add_node("fallback", self.fallback_answer)
        g.add_edge(START, "classify_intent")
        g.add_edge("classify_intent", "retrieve")
        g.add_edge("retrieve", "draft_answer")
        g.add_edge("draft_answer", "review_answer")
        g.add_conditional_edges("review_answer", self.route_after_review,
                                {"finish": END, "retry": "rewrite_query", "fallback": "fallback"})
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("fallback", END)
        return g.compile()

    def ask(self, question: str) -> dict:
        """Run the workflow. Raises BackendError if the LLM backend fails."""
        result = self.graph.invoke({"question": question, "retry_count": 0})
        return {
            "answer": result["final_answer"],
            "intent": result.get("intent"),
            "sources": sorted(set(result.get("sources", []))),
            "retries": result.get("retry_count", 0),
            "fallback": result["final_answer"] == FALLBACK_TEXT,
        }
