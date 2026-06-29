import importlib.resources
import pytest
from ingestion.chunker import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP
from tokenizers import Tokenizer

_TOKENIZER_PATH = str(
    importlib.resources.files("anthropic").joinpath("tokenizer.json")
)
_tokenizer = Tokenizer.from_file(_TOKENIZER_PATH)


def test_empty_text_returns_no_chunks():
    assert chunk_text("", source="test.txt") == []


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Hello world.", source="test.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "test.txt"
    assert chunks[0].source_type == "document"
    assert chunks[0].chunk_index == 0


def test_long_text_produces_multiple_chunks():
    text = "word " * 600  # well over 512 tokens
    chunks = chunk_text(text, source="long.txt")
    assert len(chunks) > 1


def test_chunk_token_length_within_limit():
    text = "token " * 1000
    chunks = chunk_text(text, source="big.txt")
    for chunk in chunks:
        token_count = len(_tokenizer.encode(chunk.text).ids)
        assert token_count <= CHUNK_SIZE


def test_chunk_ids_are_unique():
    text = "sentence. " * 600
    chunks = chunk_text(text, source="dup.txt")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_source_type_propagated():
    chunks = chunk_text("some text", source="page.html", source_type="web")
    assert chunks[0].source_type == "web"


def test_chunk_index_sequential():
    text = "word " * 1000
    chunks = chunk_text(text, source="seq.txt")
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
