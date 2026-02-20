from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


@dataclass(frozen=True)
class RagConfig:
    ollama_url: str
    weaviate_url: str
    embedding_model: str
    generate_model: str
    collection_name: str


def load_config() -> RagConfig:
    return RagConfig(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8081"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma"),
        generate_model=os.getenv("GENERATE_MODEL", "gemma3:4b"),
        collection_name=os.getenv("WEAVIATE_COLLECTION", "BhagavadGitaChunks"),
    )


async def ollama_embedding(
    client: httpx.AsyncClient, cfg: RagConfig, text: str
) -> List[float]:
    url = cfg.ollama_url.rstrip("/") + "/api/embeddings"
    resp = await client.post(url, json={"model": cfg.embedding_model, "prompt": text})
    resp.raise_for_status()
    data = resp.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected embedding response from Ollama: {data}")
    return emb


async def weaviate_near_vector_query(
    client: httpx.AsyncClient,
    cfg: RagConfig,
    query_vector: List[float],
    limit: int,
    scripture_filter: Optional[str],
) -> List[Dict[str, Any]]:
    where_clause = ""
    if scripture_filter:
        # Filter by title (the PDF filename without extension in `embbeded.py`).
        where_clause = (
            ', where: {path: ["title"], operator: Equal, valueText: %s}'
            % json.dumps(scripture_filter)
        )

    graphql = """
    {
      Get {
        %s(
          nearVector: { vector: %s }
          limit: %d
          %s
        ) {
          title
          chunk
          page
        }
      }
    }
    """ % (
        cfg.collection_name,
        json.dumps(query_vector),
        int(limit),
        where_clause,
    )

    url = cfg.weaviate_url.rstrip("/") + "/v1/graphql"
    resp = await client.post(url, json={"query": graphql})
    resp.raise_for_status()
    data = resp.json()
    return (
        data.get("data", {})
        .get("Get", {})
        .get(cfg.collection_name, [])
        or []
    )


def build_prompt(context_chunks: List[Dict[str, Any]], question: str) -> str:
    if not context_chunks:
        context_str = "No relevant passages were found in the current knowledge base."
    else:
        formatted: List[str] = []
        for c in context_chunks:
            page = c.get("page")
            chunk = (c.get("chunk") or "").strip()
            if not chunk:
                continue
            formatted.append(f"(Page {page}) {chunk}")
        context_str = "\n\n".join(formatted) or "No relevant passages were found in the current knowledge base."

    return (
        "Context information is below.\n"
        "---------------------\n"
        f"{context_str}\n"
        "---------------------\n"
        "You are a helpful assistant answering questions about the documents.\n"
        "Base your answer strictly on the context above, not on outside knowledge.\n"
        "Generate clear, human-readable output.\n"
        "Do not mention that you used context or retrieval.\n"
        f"Question: {question}\n"
        "Answer:"
    )


async def ollama_generate(
    client: httpx.AsyncClient, cfg: RagConfig, prompt: str
) -> str:
    url = cfg.ollama_url.rstrip("/") + "/api/generate"
    resp = await client.post(
        url,
        json={"model": cfg.generate_model, "prompt": prompt, "stream": False},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


async def ollama_generate_stream_sse(
    client: httpx.AsyncClient, cfg: RagConfig, prompt: str
) -> AsyncIterator[str]:
    """
    Streams Ollama's JSONL responses as Server-Sent Events (SSE).
    Each yielded string is a complete SSE event (ending with \n\n).
    Note: this function does NOT emit a final "done" event; callers should do so.
    """
    url = cfg.ollama_url.rstrip("/") + "/api/generate"
    async with client.stream(
        "POST",
        url,
        json={"model": cfg.generate_model, "prompt": prompt, "stream": True},
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = obj.get("response") or ""
            if token:
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            if obj.get("done") is True:
                return


async def rag_answer(
    question: str,
    limit: int = 5,
    scripture_filter: Optional[str] = None,
    cfg: Optional[RagConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    started = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:
        query_vector = await ollama_embedding(client, cfg, question)
        chunks = await weaviate_near_vector_query(
            client, cfg, query_vector, limit=limit, scripture_filter=scripture_filter
        )
        prompt = build_prompt(chunks, question)
        answer = await ollama_generate(client, cfg, prompt)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "answer": answer or "I could not generate an answer.",
        "sources": chunks,
        "latency_ms": elapsed_ms,
        "models": {
            "embedding_model": cfg.embedding_model,
            "generate_model": cfg.generate_model,
        },
    }

