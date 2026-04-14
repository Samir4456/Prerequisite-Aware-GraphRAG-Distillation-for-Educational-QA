"""
src/retrieval/embedder.py
Sentence-transformers wrapper for encoding questions and corpus.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


_model = None  # singleton — load once, reuse


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading embedder: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def embed(texts: list[str], batch_size: int = 256, show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.

    Args:
        texts: List of strings to encode
        batch_size: Encoding batch size (256 is fast on GPU)
        show_progress: Show tqdm bar for large corpora

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 norm → cosine sim = dot product
    )
    return embeddings


def embed_single(text: str) -> np.ndarray:
    """Encode a single string. Returns shape (1, dim)."""
    return embed([text])


if __name__ == "__main__":
    sample = ["Who directed The Matrix?", "The Matrix directed_by Wachowski Sisters"]
    vecs = embed(sample)
    print(f"Embedding shape: {vecs.shape}")
    print(f"Similarity: {float(vecs[0] @ vecs[1]):.4f}")
