"""Endpoint tests (offline: uses the mock backend, plus a simulated provider failure)."""
import os

os.environ["LLM_BACKEND"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

import support_rag  # noqa: E402
from app import app  # noqa: E402


def test_all():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"

        r = c.post("/ask", json={"question": "How do I request a refund?"})
        assert r.status_code == 200 and r.json()["latency_ms"] > 0 and not r.json()["fallback"]

        r = c.post("/ask", json={"question": "Do you sell home insurance?"})
        assert r.status_code == 200 and r.json()["fallback"]

        assert c.post("/ask", json={"question": "   "}).status_code == 422
        assert c.post("/ask", json={"question": "x" * 1001}).status_code == 422
        assert c.post("/ask", json={}).status_code == 422

        def boom(prompt):
            raise support_rag.BackendError("provider returned HTTP 401")
        support_rag.BACKENDS["mock"] = boom
        r = c.post("/ask", json={"question": "How do I request a refund?"})
        assert r.status_code == 503 and "401" not in r.text
