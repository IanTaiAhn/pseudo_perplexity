import os
import chromadb
from chromadb.config import Settings
from api.schemas import Chunk
from ingestion.embedder import embed_texts

_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
_COLLECTION_NAME = "documents"

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def index_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    collection = _get_collection()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source": c.source,
                "source_type": c.source_type,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def get_collection() -> chromadb.Collection:
    return _get_collection()
