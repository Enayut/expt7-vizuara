"""Streamlit frontend for the Customer Support RAG API (run app.py first)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="Customer Support Assistant", page_icon="💬")
st.title("💬 Customer Support Assistant")
st.caption("RAG over public help-center pages · LangGraph + FAISS · Groq LLM · served by FastAPI")

try:
    ok = requests.get(f"{API_URL}/health", timeout=3).json().get("status") == "ok"
except requests.RequestException:
    ok = False
st.sidebar.markdown(f"**API status:** {'🟢 running' if ok else '🔴 unreachable'}")
st.sidebar.code(API_URL)

examples = [
    "How can I track my shipment?",
    "I cannot sign in to my account. What should I do?",
    "My payment seems to have been charged twice. What should I do?",
    "Do you sell home insurance?",
]
st.sidebar.markdown("**Example questions**")
for ex in examples:
    if st.sidebar.button(ex):
        st.session_state["q"] = ex

question = st.text_area("Your question", key="q", max_chars=1100, height=100)
if st.button("Ask", type="primary"):
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
    except requests.RequestException:
        st.error("Could not reach the API.")
    else:
        data = r.json()
        if r.status_code == 200:
            (st.warning if data["fallback"] else st.success)(data["answer"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Latency", f"{data['latency_ms']:.0f} ms")
            c2.metric("Intent", data["intent"])
            c3.metric("Retries", data["retries"])
            if data["sources"]:
                with st.expander("Sources"):
                    for s in data["sources"]:
                        st.write(s)
        elif r.status_code == 422:
            st.error("Invalid input: " + "; ".join(e["message"] for e in data["detail"]))
        else:
            st.error(f"{data['detail']} (request id: {data.get('request_id')})")
