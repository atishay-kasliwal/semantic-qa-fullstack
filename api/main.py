from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.rag import (
    build_prompt,
    load_config,
    ollama_embedding,
    ollama_generate_stream_sse,
    rag_answer,
    weaviate_near_vector_query,
)


app = FastAPI(title="Semantic QA API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    scripture_filter: Optional[str] = Field(
        default=None, description="Filters by Weaviate 'title' field (PDF basename)."
    )
    limit: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: int
    models: dict[str, str]


@app.get("/health")
async def health() -> dict[str, Any]:
    cfg = load_config()

    async with httpx.AsyncClient(timeout=5.0) as client:
        weaviate_ok = False
        ollama_ok = False
        try:
            r = await client.get(cfg.weaviate_url.rstrip("/") + "/v1/.well-known/ready")
            weaviate_ok = r.status_code == 200
        except Exception:
            weaviate_ok = False

        try:
            r = await client.get(cfg.ollama_url.rstrip("/") + "/api/tags")
            ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False

    return {"ok": bool(weaviate_ok and ollama_ok), "weaviate_ok": weaviate_ok, "ollama_ok": ollama_ok}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    result = await rag_answer(
        question=req.question,
        limit=req.limit,
        scripture_filter=req.scripture_filter,
    )
    return QueryResponse(**result)


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    SSE stream:
      - event: meta (JSON with sources, models)
      - event: token (JSON string token)
      - event: done
    """
    cfg = load_config()
    started = time.perf_counter()

    async def event_iter():
        async with httpx.AsyncClient(timeout=60.0) as client:
            query_vector = await ollama_embedding(client, cfg, req.question)
            chunks = await weaviate_near_vector_query(
                client,
                cfg,
                query_vector,
                limit=req.limit,
                scripture_filter=req.scripture_filter,
            )
            meta = {
                "sources": chunks,
                "models": {"embedding_model": cfg.embedding_model, "generate_model": cfg.generate_model},
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            prompt = build_prompt(chunks, req.question)
            async for evt in ollama_generate_stream_sse(client, cfg, prompt):
                yield evt

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        yield f"event: latency\ndata: {json.dumps({'latency_ms': elapsed_ms})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream")

