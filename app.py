"""FastAPI service exposing the Customer Support RAG pipeline."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from support_rag import BackendError, SupportRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

MAX_QUESTION_CHARS = 1000
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy objects (embedding model, FAISS index, graph) are built once at startup.
    t0 = time.perf_counter()
    state["rag"] = SupportRAG()
    log.info("startup complete in %.0f ms", (time.perf_counter() - t0) * 1000)
    yield
    state.clear()


app = FastAPI(title="Customer Support RAG API", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def check_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
        return v


class AskResponse(BaseModel):
    answer: str
    intent: str | None
    sources: list[str]
    retries: int
    fallback: bool
    latency_ms: float


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # Only field + message are returned (no echo of the submitted input).
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body",
               "message": e["msg"].removeprefix("Value error, ")} for e in exc.errors()]
    log.warning("rejected request: invalid input (%d error(s))", len(errors))
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health():
    return {"status": "ok", "rag_loaded": "rag" in state}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    req_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        result = state["rag"].ask(req.question)
    except BackendError as e:
        latency = (time.perf_counter() - start) * 1000
        log.error("req=%s status=failure reason=%s latency_ms=%.0f", req_id, e, latency)
        return JSONResponse(status_code=503, content={
            "detail": "The answer service is temporarily unavailable. Please try again later.",
            "request_id": req_id})
    except Exception:
        latency = (time.perf_counter() - start) * 1000
        log.exception("req=%s status=failure reason=unexpected latency_ms=%.0f", req_id, latency)
        return JSONResponse(status_code=500, content={
            "detail": "Internal server error.", "request_id": req_id})
    latency = (time.perf_counter() - start) * 1000
    log.info("req=%s status=success intent=%s fallback=%s question_chars=%d latency_ms=%.0f",
             req_id, result["intent"], result["fallback"], len(req.question), latency)
    return AskResponse(**result, latency_ms=round(latency, 1))
