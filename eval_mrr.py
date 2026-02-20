import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

import embbeded  # reuse PDF chunking


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)


def _load_gita_chunks(sample_limit: int = 400) -> List[Dict[str, str]]:
    chunks = embbeded.load_pdf_chunks(
        embbeded.PDF_PATH,
        title="The Bhagavad Gita",
    )
    if sample_limit and len(chunks) > sample_limit:
        return chunks[:sample_limit]
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


# Multilingual query groups for MRR evaluation.
MRR_GROUPS = [
    {
        "id": "duty_without_attachment",
        "expected_title": "The Bhagavad Gita",
        "queries": {
            "en": "What does the Gita say about doing your duty without attachment to results?",
            "hi": "गीता कर्म करते हुए फल की आसक्ति के बिना क्या कहती है?",
            "es": "¿Qué dice el Gita sobre cumplir tu deber sin apego a los resultados?",
        },
    },
    {
        "id": "nature_of_self",
        "expected_title": "The Bhagavad Gita",
        "queries": {
            "en": "How does the Bhagavad Gita describe the nature of the self (Atman)?",
            "hi": "भगवद् गीता आत्मा (आत्मन) के स्वरूप का वर्णन कैसे करती है?",
            "es": "¿Cómo describe el Bhagavad Gita la naturaleza del ser (Atman)?",
        },
    },
    {
        "id": "fear_and_doubt",
        "expected_title": "The Bhagavad Gita",
        "queries": {
            "en": "What guidance does Krishna give Arjuna about fear and doubt on the battlefield?",
            "hi": "कृष्ण अर्जुन को युद्धभूमि में भय और संदेह के बारे में क्या उपदेश देते हैं?",
            "es": "¿Qué consejo le da Krishna a Arjuna sobre el miedo y la duda en el campo de batalla?",
        },
    },
]


def evaluate_mrr(
    name: str,
    embed_fn,
    chunks: List[Dict[str, str]],
    groups: List[Dict],
    max_rank: int = 100,
) -> None:
    print(f"\n=== MRR evaluation for embedding model: {name} ===")

    chunk_texts = [c["chunk"] for c in chunks]
    chunk_titles = [c["title"] for c in chunks]
    chunk_embs = embed_fn(chunk_texts)

    reciprocal_ranks: List[float] = []

    for group in groups:
        expected_title = group["expected_title"]
        for lang, query in group["queries"].items():
            q_vec = embed_fn([query])[0:1]
            sims = _cosine_similarity(chunk_embs, q_vec)[:, 0]
            sorted_idx = sims.argsort()[::-1]

            rr = 0.0
            for rank, idx in enumerate(sorted_idx[:max_rank]):
                if chunk_titles[idx] == expected_title:
                    rr = 1.0 / (rank + 1)
                    break

            reciprocal_ranks.append(rr)
            print(
                f"Query group '{group['id']}' [{lang}] -> RR={rr:.3f}"
            )

    if reciprocal_ranks:
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        print(f"\nMRR for {name}: {mrr:.3f}")
    else:
        print("No queries evaluated.")


def main() -> None:
    chunks = _load_gita_chunks(sample_limit=400)
    print(f"Loaded {len(chunks)} chunks from Gita for MRR evaluation.")

    # 1) embeddinggemma via Ollama
    evaluate_mrr(
        "embeddinggemma (Ollama)",
        embed_fn=_embed_with_ollama,
        chunks=chunks,
        groups=MRR_GROUPS,
    )

    # 2) Qwen and nomic via HuggingFace on CPU
    print("\nLoading HuggingFace models on CPU (Qwen, nomic)...")
    qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    nomic_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu"
    )

    evaluate_mrr(
        "Qwen3-Embedding-0.6B",
        embed_fn=lambda texts: _embed_with_hf(qwen_model, texts),
        chunks=chunks,
        groups=MRR_GROUPS,
    )

    evaluate_mrr(
        "nomic-embed-text-v1.5",
        embed_fn=lambda texts: _embed_with_hf(nomic_model, texts),
        chunks=chunks,
        groups=MRR_GROUPS,
    )


if __name__ == "__main__":
    main()


