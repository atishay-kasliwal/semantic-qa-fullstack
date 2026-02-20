import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

import embbeded  # reuse PDF chunking


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)


def _load_balanced_chunks(max_chunks_per_title: int = 80) -> List[Dict[str, str]]:
    """
    Load up to `max_chunks_per_title` chunks from each PDF under Files/.
    This balances the corpus so one large book (like the Gita) doesn't dominate.
    """
    chunks: List[Dict[str, str]] = []
    root_dir = "Files"

    for name in sorted(os.listdir(root_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(root_dir, name)
        title = os.path.splitext(name)[0]
        cs = embbeded.load_pdf_chunks(path, title=title)
        if max_chunks_per_title and len(cs) > max_chunks_per_title:
            cs = cs[:max_chunks_per_title]
        chunks.extend(cs)

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


# One question per scripture, targeting a specific title.
TITLE_QUESTIONS = [
    {
        "title": "The Bhagavad Gita",
        "question": "According to the Bhagavad Gita, what is the nature of the soul and its relation to the Supreme?",
    },
    {
        "title": "The Bhagavad Gita",
        "question": "According to the Bhagavad Gita, how should a warrior like Arjuna perform his duty in battle?",
    },
    {
        "title": "CSB_Pew_Bible_2nd_Printing",
        "question": "According to the Christian Bible, how should a person love God and their neighbor?",
    },
    {
        "title": "CSB_Pew_Bible_2nd_Printing",
        "question": "According to the Christian Bible, what guidance is given in the Sermon on the Mount about righteous living?",
    },
    {
        "title": "The Talmud",
        "question": "According to the Talmud, what principles guide justice and ethical behavior?",
    },
    {
        "title": "The Talmud",
        "question": "According to the Talmud, what is the importance of study and debate in understanding the law?",
    },
    {
        "title": "17 Dhammapada (tr. Daw Mya Tin, by DPPS)_text",
        "question": "According to the Dhammapada, what is the path to the end of suffering?",
    },
    {
        "title": "17 Dhammapada (tr. Daw Mya Tin, by DPPS)_text",
        "question": "According to the Dhammapada, how should one relate to anger and hatred?",
    },
    {
        "title": "The-Upanishads-Translated-by-Swami-Paramananda",
        "question": "According to the Upanishads, how is the identity between Atman and Brahman described?",
    },
    {
        "title": "The-Upanishads-Translated-by-Swami-Paramananda",
        "question": "According to the Upanishads, what is the role of meditation and inward inquiry in realizing the Self?",
    },
    {
        "title": "Siri Guru Granth - English Translation (matching pages)",
        "question": "According to the Guru Granth Sahib, what does it mean to remember and meditate on the Divine Name?",
    },
    {
        "title": "Siri Guru Granth - English Translation (matching pages)",
        "question": "According to the Guru Granth Sahib, how does a person live in hukam (divine will)?",
    },
    {
        "title": "thetanakh",
        "question": "According to the Tanakh, what does it mean to walk humbly with God and do justice?",
    },
    {
        "title": "thetanakh",
        "question": "According to the Tanakh, what covenant obligations does Israel have toward God?",
    },
    {
        "title": "Four-Vedas-English-Translation",
        "question": "According to the Vedas, what rituals and duties uphold cosmic order (rta)?",
    },
    {
        "title": "Four-Vedas-English-Translation",
        "question": "According to the Vedas, what is the role of sacrifice (yajna) in maintaining harmony between humans and the gods?",
    },
    {
        "title": "The Complete Mahabharata ",
        "question": "According to the Mahabharata, what lessons does the epic teach about dharma in complex situations?",
    },
    {
        "title": "The Complete Mahabharata ",
        "question": "According to the Mahabharata, how do the stories of the Pandavas and Kauravas illustrate the consequences of adharma?",
    },
    {
        "title": "agama_023168_hr6",
        "question": "According to this Agama text, how should rituals be performed to uphold spiritual discipline?",
    },
    {
        "title": "agama_023168_hr6",
        "question": "According to this Agama text, what is the significance of purity and correct procedure in temple worship?",
    },
]


def evaluate_title_discrimination(
    name: str,
    embed_fn,
    chunks: List[Dict[str, str]],
    max_rank: int = 20,
) -> None:
    """
    For each (title, question) pair, compute the rank of the first retrieved chunk
    from that title. Report MRR and top-1 accuracy over all titles.
    """
    print(f"\n=== Title discrimination for embedding model: {name} ===")

    texts = [c["chunk"] for c in chunks]
    titles = [c["title"] for c in chunks]
    embs = embed_fn(texts)

    reciprocal_ranks: List[float] = []
    top1_hits = 0

    for spec in TITLE_QUESTIONS:
        expected = spec["title"]
        q = spec["question"]

        q_vec = embed_fn([q])[0:1]
        sims = _cosine_similarity(embs, q_vec)[:, 0]
        idx = sims.argsort()[::-1]

        rr = 0.0
        rank_found = None
        for rank, i in enumerate(idx[:max_rank], start=1):
            if titles[i] == expected:
                rr = 1.0 / rank
                rank_found = rank
                break

        reciprocal_ranks.append(rr)
        if rank_found == 1:
            top1_hits += 1

        rank_str = str(rank_found) if rank_found is not None else f">{max_rank}"
        print(f"Question for '{expected}' -> RR={rr:.3f} (rank={rank_str})")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    top1_acc = top1_hits / len(reciprocal_ranks)
    print(f"\nMRR@{max_rank} for {name}: {mrr:.3f}")
    print(f"Top-1 accuracy for {name}: {top1_acc:.3f}")


def main() -> None:
    chunks = _load_balanced_chunks(max_chunks_per_title=80)
    print(f"Loaded {len(chunks)} balanced chunks across titles for discrimination eval.")

    # 1) embeddinggemma via Ollama
    evaluate_title_discrimination(
        "embeddinggemma (Ollama)",
        embed_fn=_embed_with_ollama,
        chunks=chunks,
        max_rank=20,
    )

    # 2) Qwen and nomic via HuggingFace on CPU
    print("\nLoading HuggingFace models on CPU (Qwen, nomic)...")
    qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    nomic_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu"
    )

    evaluate_title_discrimination(
        "Qwen3-Embedding-0.6B",
        embed_fn=lambda texts: _embed_with_hf(qwen_model, texts),
        chunks=chunks,
        max_rank=20,
    )

    evaluate_title_discrimination(
        "nomic-embed-text-v1.5",
        embed_fn=lambda texts: _embed_with_hf(nomic_model, texts),
        chunks=chunks,
        max_rank=20,
    )


if __name__ == "__main__":
    main()


