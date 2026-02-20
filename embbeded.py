import os
from typing import List, Dict

import httpx
import fitz  # PyMuPDF
import weaviate


COLLECTION_NAME = "BhagavadGitaChunks"
PDF_PATH = os.getenv("PDF_PATH", "Files/The Bhagavad Gita.pdf")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8081")


def get_ollama_embedding(text: str) -> List[float]:
    """
    Call the local Ollama embeddings endpoint using the `embeddinggemma` model.
    """
    url = OLLAMA_URL.rstrip("/") + "/api/embeddings"
    payload = {"model": EMBEDDING_MODEL, "prompt": text}

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError(f"Unexpected embedding response from Ollama: {data}")

    return embedding


def connect_weaviate() -> weaviate.Client:
    """
    Connect to the local Weaviate instance via HTTP (no auth, per docker-compose).
    """
    return weaviate.Client(WEAVIATE_URL)


def ensure_collection(client: weaviate.Client) -> None:
    """
    Create a simple class in Weaviate that does NOT use any built-in vectorizer.
    We will supply vectors manually from Ollama (`embeddinggemma`).
    """
    class_obj = {
        "class": COLLECTION_NAME,
        "vectorizer": "none",
        "moduleConfig": {},
        "properties": [
            {"name": "title", "dataType": ["text"]},
            {"name": "chunk", "dataType": ["text"]},
            {"name": "page", "dataType": ["int"]},
        ],
    }

    schema = client.schema.get()
    existing_classes = {c["class"] for c in schema.get("classes", [])}
    if COLLECTION_NAME in existing_classes:
        return

    client.schema.create_class(class_obj)


def load_pdf_chunks(
    path: str,
    title: str,
    max_chars: int = 600,
    overlap: int = 100,
) -> List[Dict[str, str]]:
    """
    Read the Bhagavad Gita PDF and return text chunks with simple sliding-window
    chunking per page.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found at: {path}")

    doc = fitz.open(path)
    chunks: List[Dict[str, str]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        raw_text = page.get_text("text")
        raw_text = raw_text.strip()
        if not raw_text:
            continue

        # Basic cleaning: drop very short / non-alphabetic lines and page noise.
        cleaned_lines: List[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) < 20:
                continue
            alpha = sum(ch.isalpha() for ch in line)
            if alpha / max(len(line), 1) < 0.4:
                continue
            cleaned_lines.append(line)

        text = " ".join(cleaned_lines).strip()
        if len(text) < 80:
            # Skip pages that don't have enough clean content.
            continue

        start = 0
        while start < len(text):
            end = start + max_chars
            chunk_text = text[start:end].strip()
            if not chunk_text:
                break

            chunks.append(
                {
                    "title": title,
                    "chunk": chunk_text,
                    "page": page_index + 1,
                }
            )

            if end >= len(text):
                break

            # slide window with overlap
            start = end - overlap

    doc.close()
    return chunks


def populate_all_pdfs(client: weaviate.Client, root_dir: str = "Files") -> None:
    """
    Load chunks from all PDFs in the given directory and store them in Weaviate
    with embeddings from Ollama (`embeddinggemma`).
    """
    pdf_paths: List[str] = []
    for name in os.listdir(root_dir):
        if not name.lower().endswith(".pdf"):
            continue
        pdf_paths.append(os.path.join(root_dir, name))

    pdf_paths.sort()

    total_chunks = 0
    for path in pdf_paths:
        title = os.path.splitext(os.path.basename(path))[0]
        print(f"Loading PDF from: {path}")
        chunks = load_pdf_chunks(path, title=title)
        print(f"  Chunks to ingest from \"{title}\": {len(chunks)}")

        for i, obj in enumerate(chunks, start=1):
            vector = get_ollama_embedding(obj["chunk"])

            client.data_object.create(
                data_object=obj,
                class_name=COLLECTION_NAME,
                vector=vector,
            )

            total_chunks += 1
            if total_chunks % 50 == 0:
                print(f"Ingested {total_chunks} chunks so far...")

    print(f"Finished ingesting all PDFs. Total chunks ingested: {total_chunks}")


def run_vector_search(client: weaviate.Client) -> None:
    """
    Perform a vector search in Weaviate using a query embedded by Ollama.
    """
    query_text = "What does the Gita say about doing your duty without attachment to results?"
    query_vector = get_ollama_embedding(query_text)

    response = (
        client.query.get(COLLECTION_NAME, ["title", "chunk", "page"])
        .with_near_vector({"vector": query_vector})
        .with_limit(3)
        .do()
    )

    print("Top results:")
    for obj in response.get("data", {}).get("Get", {}).get(COLLECTION_NAME, []):
        title = obj.get("title")
        chunk = obj.get("chunk")
        page = obj.get("page")
        print(f"\nPage {page} - {title}:\n{chunk[:400]}...")


def main() -> None:
    """
    End-to-end local flow:
      1. Connect to local Weaviate
      2. Ensure BhagavadGitaChunks collection exists (no built-in vectorizer)
      3. Populate it with chunks from all PDFs in `Files/` using Ollama (`embeddinggemma`) vectors
      4. Run a vector search using an Ollama-embedded query (Bhagavad Gita example)
    """
    client = connect_weaviate()
    ensure_collection(client)
    populate_all_pdfs(client)
    run_vector_search(client)


if __name__ == "__main__":
    main()


