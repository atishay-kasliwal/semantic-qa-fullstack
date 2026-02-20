import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

import embbeded  # reuse PDF chunking


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)


def _load_all_chunks(sample_limit: int = 800) -> List[Dict[str, str]]:
    """
    Load chunks from all PDFs under Files/ using the existing chunker.
    """
    chunks: List[Dict[str, str]] = []
    root_dir = "Files"
    for name in os.listdir(root_dir):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(root_dir, name)
        title = os.path.splitext(name)[0]
        cs = embbeded.load_pdf_chunks(path, title=title)
        chunks.extend(cs)

    if sample_limit and len(chunks) > sample_limit:
        chunks = chunks[:sample_limit]

    return chunks


def _embed_with_ollama(texts: List[str]) -> np.ndarray:
    vectors: List[List[float]] = []
    for t in texts:
        vectors.append(embbeded.get_ollama_embedding(t))
    return np.asarray(vectors, dtype=np.float32)


def _embed_with_hf(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )


# Very generic questions that should touch almost all scriptures.
COVERAGE_QUESTIONS = [
    "What is the nature of the soul or self according to this scripture?",
    "What is the path to liberation or salvation according to this scripture?",
    "How should a person live a righteous life according to this scripture?",
    "How does this scripture describe the nature of God or the Supreme Being?",
]


def evaluate_coverage(
    name: str,
    embed_fn,
    chunks: List[Dict[str, str]],
    top_k: int = 40,
) -> None:
    print(f"\n=== Retrieval coverage for embedding model: {name} ===")

    texts = [c["chunk"] for c in chunks]
    titles = [c["title"] for c in chunks]
    embs = embed_fn(texts)

    avg_unique_titles = 0.0

    for q in COVERAGE_QUESTIONS:
        q_vec = embed_fn([q])[0:1]
        sims = _cosine_similarity(embs, q_vec)[:, 0]
        idx = sims.argsort()[::-1][:top_k]

        seen_titles = {}
        for rank, i in enumerate(idx, start=1):
            t = titles[i]
            seen_titles.setdefault(t, 0)
            seen_titles[t] += 1

        unique_count = len(seen_titles)
        avg_unique_titles += unique_count

        print(f"\nQuestion: {q}")
        print(f"  Unique titles in top {top_k}: {unique_count}")
        # Show brief distribution
        for t, cnt in sorted(seen_titles.items(), key=lambda x: -x[1]):
            print(f"    {t}: {cnt}")

    avg_unique_titles /= len(COVERAGE_QUESTIONS)
    print(f"\nAverage unique titles per question (top {top_k}) for {name}: {avg_unique_titles:.2f}")


def main() -> None:
    chunks = _load_all_chunks(sample_limit=800)
    print(f"Loaded {len(chunks)} chunks from all PDFs for coverage evaluation.")

    # 1) embeddinggemma via Ollama
    evaluate_coverage(
        "embeddinggemma (Ollama)",
        embed_fn=_embed_with_ollama,
        chunks=chunks,
        top_k=40,
    )

    # 2) Qwen and nomic via HuggingFace on CPU
    print("\nLoading HuggingFace models on CPU (Qwen, nomic)...")
    qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    nomic_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu"
    )

    evaluate_coverage(
        "Qwen3-Embedding-0.6B",
        embed_fn=lambda texts: _embed_with_hf(qwen_model, texts),
        chunks=chunks,
        top_k=40,
    )

    evaluate_coverage(
        "nomic-embed-text-v1.5",
        embed_fn=lambda texts: _embed_with_hf(nomic_model, texts),
        chunks=chunks,
        top_k=40,
    )


if __name__ == "__main__":
    main()


