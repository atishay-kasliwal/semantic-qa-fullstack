import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

import embbeded  # reuse PDF chunking and Ollama embedding

# Load environment variables from a local .env file, if present
load_dotenv()

# Optional: inline override for your OpenAI key.
# Set this to your real key locally if you prefer not to use .env.
# WARNING: Keep this empty in committed code; do NOT push real keys to git.
INLINE_OPENAI_API_KEY = ""  # e.g. "sk-..."

if INLINE_OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = INLINE_OPENAI_API_KEY

# Optional: set your OpenAI key in a local .env file instead of hard-coding it here.
# Example .env entry (do NOT commit your real key to git):
# OPENAI_API_KEY=sk-...


QUESTIONS: List[str] = [
    # Duty / karma-yoga
    "What does the Gita say about doing your duty without attachment to results?",
    "Why does Krishna tell Arjuna to focus on action, not on the fruits of action?",
    "How does the Gita define karma-yoga?",
    "What guidance does the Gita give on performing one’s svadharma (own duty) even if imperfectly?",
    "Why is performing your own duty considered better than doing another’s duty well?",
    "How should a person act in the world according to karma-yoga?",
    "What is the Gita’s view on inaction versus action?",
    "How does Krishna explain the relationship between duty and righteousness (dharma)?",
    "What advice does Krishna give about remaining even-minded in success and failure?",
    "How does the Gita describe the right attitude toward success, failure, gain, and loss?",

    # Atman / nature of the self
    "How does the Bhagavad Gita describe the nature of the self (Atman)?",
    "What does the Gita say about the immortality of the soul?",
    "How does Krishna explain the difference between the body and the self?",
    "Why does the Gita say the wise are not disturbed by birth and death?",
    "How is the Atman described in terms of being unborn and eternal?",
    "What does the Gita teach about the self being unaffected by weapons, fire, water, and wind?",
    "How does the Gita distinguish between the changing body and the unchanging self?",
    "What does Krishna tell Arjuna about the continuity of the self through different bodies?",
    "How does the Gita relate the Atman to the Supreme Self (Paramatma)?",
    "What is the Gita’s teaching on realizing the self beyond the three gunas?",

    # Yoga paths (karma, jnana, bhakti, dhyana)
    "How does the Gita compare karma-yoga and jnana-yoga?",
    "What does the Gita teach about jnana-yoga, the path of knowledge?",
    "How is dhyana-yoga (meditation) described in the Gita?",
    "What are the conditions for successful meditation according to the Gita?",
    "How does the Gita describe the mind of a yogi who is well established in meditation?",
    "What is the Gita’s definition of a sthitaprajna (one of steady wisdom)?",
    "How does Krishna define true renunciation (sannyasa) versus mere outer renunciation?",
    "What is the relationship between renunciation and yoga in the Gita?",
    "How does the Gita present bhakti-yoga as a path to liberation?",
    "What does Krishna say about different types of devotees in the Gita?",

    # Bhakti / devotion
    "How does the Bhagavad Gita describe pure devotion (shuddha-bhakti)?",
    "Why does Krishna say that even a person of sinful conduct can be considered righteous if they are devoted?",
    "What qualities of a true devotee does Krishna list in the Gita?",
    "How does the Gita describe God’s reciprocation with His devotees?",
    "What does Krishna mean when He says, ‘Offer Me a leaf, a flower, a fruit, or water with devotion’?",
    "How does the Gita describe remembering and surrendering to Krishna at the time of death?",
    "What does the Gita say about constantly remembering and worshiping Krishna?",
    "How does Krishna encourage Arjuna to take refuge in Him alone?",
    "What is the significance of ‘man-mana bhava mad-bhakto’ in the Gita?",
    "How does the Gita differentiate between worship of the personal form and the impersonal absolute?",

    # Detachment / control of mind and senses
    "What practical instructions does the Gita give for developing detachment?",
    "How does the Gita describe the mind as both friend and enemy?",
    "What guidance does Krishna give Arjuna about controlling the senses?",
    "How does the Gita explain the progression from sense contemplation to fall-down (from thought to ruin)?",
    "What is the Gita’s teaching on desire and anger as enemies of spiritual life?",
    "How does a person of steady wisdom relate to pleasure and pain?",
    "What does the Gita say about being equal in honor and dishonor, praise and blame?",
    "How does Krishna describe the person who remains calm amid dualities like heat and cold?",
    "What is the role of moderation in eating, sleeping, and recreation according to the Gita?",
    "How does the Gita advise dealing with constant mental disturbances and distractions?",

    # Arjuna’s doubt, fear, and surrender
    "What guidance does Krishna give Arjuna about fear and doubt on the battlefield?",
    "How does Arjuna describe his initial confusion and despair at the start of the Gita?",
    "What is the turning point where Arjuna surrenders and accepts Krishna as his teacher?",
    "How does Krishna address Arjuna’s fear of killing his relatives?",
    "What reassurance does Krishna give Arjuna about the ultimate welfare of the soul?",
    "Why does Krishna criticize Arjuna’s weakness and urge him to ‘stand up and fight’?",
    "How does the Gita encourage acting with faith in divine guidance despite uncertainty?",
    "What is the significance of Krishna revealing His universal form (Vishvarupa) to Arjuna?",
    "How does Arjuna respond after seeing the universal form?",
    "What is the final instruction Krishna gives Arjuna about surrendering all duties to Him?",
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)


def _load_chunks(sample_limit: int = 400) -> List[Dict[str, str]]:
    # For eval we focus on the Bhagavad Gita PDF.
    chunks = embbeded.load_pdf_chunks(embbeded.PDF_PATH, title="The Bhagavad Gita")
    if sample_limit and len(chunks) > sample_limit:
        return chunks[:sample_limit]
    return chunks


def _embed_with_ollama(texts: List[str]) -> np.ndarray:
    vectors: List[List[float]] = []
    for t in texts:
        vectors.append(embbeded.get_ollama_embedding(t))
    return np.asarray(vectors, dtype=np.float32)


def _embed_with_hf(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)


def evaluate_model(
    name: str,
    embed_fn,
    chunks: List[Dict[str, str]],
    top_k: int = 5,
) -> None:
    print(f"\n=== Evaluating embedding model: {name} ===")

    # Precompute embeddings for all chunks
    chunk_texts = [c["chunk"] for c in chunks]
    chunk_embs = embed_fn(chunk_texts)

    scores: List[float] = []

    for q in QUESTIONS:
        q_vec = embed_fn([q])[0:1]  # shape (1, dim)
        sims = _cosine_similarity(chunk_embs, q_vec)[:, 0]
        top_idx = sims.argsort()[-top_k:][::-1]
        retrieval_context = [chunk_texts[i] for i in top_idx]

        metric = ContextualRelevancyMetric(
            threshold=0.0,
            # Uses OpenAI (or other provider) via DeepEval; requires OPENAI_API_KEY.
            model=os.getenv("DEEPEVAL_MODEL", "gpt-4.1-mini"),
            include_reason=True,
        )
        test_case = LLMTestCase(
            input=q,
            actual_output="",  # not used for contextual relevancy
            retrieval_context=retrieval_context,
        )

        metric.measure(test_case)
        scores.append(metric.score)

        print(f"\nQuestion: {q}")
        print(f"Score: {metric.score:.3f}")
        if metric.reason:
            print(f"Reason: {metric.reason}")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nAverage contextual relevancy for {name}: {avg:.3f}")


def main() -> None:
    # Load a subset of chunks for faster evaluation
    # Lowered to 200 to reduce memory for large HF models on MPS/CPU.
    chunks = _load_chunks(sample_limit=200)
    print(f"Loaded {len(chunks)} chunks from PDF for evaluation.")

    # Prepare HuggingFace embedding models (force CPU to avoid MPS OOM)
    print("Loading HuggingFace models on CPU (this may take a while the first time)...")
    qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    nomic_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")

    # 1) embeddinggemma via Ollama
    evaluate_model(
        "embeddinggemma (Ollama)",
        embed_fn=_embed_with_ollama,
        chunks=chunks,
    )

    # 2) Qwen3-Embedding-0.6B via HuggingFace
    evaluate_model(
        "Qwen3-Embedding-0.6B",
        embed_fn=lambda texts: _embed_with_hf(qwen_model, texts),
        chunks=chunks,
    )

    # 3) nomic-embed-text-v1.5 via HuggingFace
    evaluate_model(
        "nomic-embed-text-v1.5",
        embed_fn=lambda texts: _embed_with_hf(nomic_model, texts),
        chunks=chunks,
    )


if __name__ == "__main__":
    main()


