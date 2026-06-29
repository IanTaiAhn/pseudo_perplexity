import os
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# BGE models expect a task prefix; MiniLM and most others do not.
_BGE_MODELS = {"BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"}

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
    if _MODEL_NAME in _BGE_MODELS:
        query = f"Represent this sentence: {query}"
    return embed_texts([query])[0]
