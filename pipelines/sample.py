from typing import List, Union, Generator, Iterator
import os
import json

import requests
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_URL: str
        WEAVIATE_URL: str
        EMBEDDING_MODEL: str
        GENERATE_MODEL: str

    def __init__(self):
        self.name = "RAG"

        self.valves = self.Valves(
            OLLAMA_URL=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            WEAVIATE_URL=os.getenv("WEAVIATE_URL", "http://weaviate:8080"),
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "embeddinggemma"),
            GENERATE_MODEL=os.getenv("GENERATE_MODEL", "gemma3:4b"),
        )





    async def on_startup(self):
        # No long-lived connections required; everything is HTTP-per-call.
        pass

    async def on_shutdown(self):
        pass

    

    def _get_ollama_embedding(self, text: str) -> List[float]:
        url = self.valves.OLLAMA_URL.rstrip("/") + "/api/embeddings"
        payload = {"model": self.valves.EMBEDDING_MODEL, "prompt": text}

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Unexpected embedding response from Ollama: {data}")

        return embedding

    def _weaviate_bhagavad_gita_query(self, query_vector: List[float], limit: int = 10) -> List[dict]:
        graphql = """
        {
          Get {
            BhagavadGitaChunks(
              nearVector: { vector: %s }
              limit: %d
            ) {
              title
              chunk
              page
            }
          }
        }
        """ % (json.dumps(query_vector), limit)

        url = self.valves.WEAVIATE_URL.rstrip("/") + "/v1/graphql"
        resp = requests.post(url, json={"query": graphql}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        return data.get("data", {}).get("Get", {}).get("BhagavadGitaChunks", []) or []

    def _ollama_generate(self, prompt: str) -> str:
        url = self.valves.OLLAMA_URL.rstrip("/") + "/api/generate"
        payload = {
            "model": self.valves.GENERATE_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/generate returns the text under "response"
        return data.get("response", "")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        RAG pipeline:
          1. Embed the user query with Ollama (`embeddinggemma`)
          2. Retrieve relevant Bhagavad Gita chunks from Weaviate
          3. Build a QA prompt with the context
          4. Call Ollama generation model (e.g. `gemma2:2b`) to answer
        """
        try:
            # 1) Embed query
            query_vector = self._get_ollama_embedding(user_message)

            # 2) Retrieve from Weaviate
            docs = self._weaviate_bhagavad_gita_query(query_vector, limit=5)

            if not docs:
                context_str = "No relevant passages were found in the current knowledge base."
            else:
                context_chunks = []
                for d in docs:
                    page = d.get("page")
                    chunk = d.get("chunk", "")
                    context_chunks.append(f"(Page {page}) {chunk}")
                context_str = "\n\n".join(context_chunks)

            # 3) Build QA prompt
            prompt = (
                "Context information is below.\n"
                "---------------------\n"
                f"{context_str}\n"
                "---------------------\n"
                "You are a helpful AI assistant answering questions about the Bhagavad Gita.\n"
                "Base your answer strictly on the context above, not on outside knowledge.\n"
                "Generate clear, human-readable output.\n"
                "Do not mention that you used context or retrieval.\n"
                "Never say that you are an AI assistant; just answer the question.\n"
                f"Question: {user_message}\n"
                "Answer:"
            )

            # 4) Generate answer with Ollama
            answer = self._ollama_generate(prompt)
            return answer or "I could not generate an answer."

        except requests.RequestException as e:
            return f"RAG pipeline error: HTTP request failed: {e}"
        except Exception as e:
            return f"RAG pipeline error: {e}"