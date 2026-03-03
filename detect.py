from __future__ import annotations

import os
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache"),
)

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "all-mpnet-base-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)


def embed_history(history: list[str]) -> np.ndarray:
    """Converts list of responses into embedding vectors."""
    print(f"Embedding {len(history)} responses ({EMBEDDING_MODEL})...")
    return embedder.encode(history, show_progress_bar=True)


def find_convergence(
    embeddings: np.ndarray,
    window: int = 5,
    threshold: float = 0.85,
    sustained: int = 3,
) -> int:
    """
    Anchor-based convergence detection with sustained confirmation.

    For each turn i, computes avg cosine similarity between turn i
    and the previous `window` turns. Convergence fires when this
    average exceeds `threshold` for `sustained` consecutive windows,
    filtering out one-off similarity spikes.

    Returns: convergence turn index (0-based), or last turn if none.
    """
    streak = 0
    for i in range(window, len(embeddings)):
        sims = [
            cosine_similarity([embeddings[j]], [embeddings[i]])[0][0]
            for j in range(i - window, i)
        ]
        avg_sim = float(np.mean(sims))

        if avg_sim > threshold:
            streak += 1
            if streak >= sustained:
                print(
                    f"✓ Converged at turn {i + 1} "
                    f"(sustained {sustained} windows above {threshold}, "
                    f"avg: {avg_sim:.3f})"
                )
                return i
        else:
            streak = 0

    print(
        f"No convergence detected across {len(embeddings)} turns "
        f"(threshold={threshold}, sustained={sustained})."
    )
    print("This is a valid finding — model may not have a strong attractor.")
    return len(embeddings) - 1


def analyze(history: list[str]) -> tuple[np.ndarray, int]:
    """Full analysis pipeline: embed → detect convergence."""
    embeddings = embed_history(history)
    convergence_turn = find_convergence(embeddings)

    print(f"\nAttractor text (turn {convergence_turn + 1}):")
    print(f"{history[convergence_turn][:400]}")
    print()

    return embeddings, convergence_turn


if __name__ == "__main__":
    from loop import run_loop
    history, labels = run_loop(turns=50)
    embeddings, convergence_turn = analyze(history)
