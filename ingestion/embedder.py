import os
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    embeddings: np.ndarray = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    # BGE models benefit from a query prefix for retrieval tasks
    prefixed = f"Represent this sentence: {query}"
    return embed_texts([prefixed])[0]
