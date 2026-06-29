import importlib.resources
import uuid
from tokenizers import Tokenizer
from api.schemas import Chunk


CHUNK_SIZE = 512      # tokens
CHUNK_OVERLAP = 50   # tokens

# Use Anthropic's bundled tokenizer — available locally without network access.
# This is a good match since we're synthesizing with Claude.
_TOKENIZER_PATH = str(
    importlib.resources.files("anthropic").joinpath("tokenizer.json")
)
_tokenizer = Tokenizer.from_file(_TOKENIZER_PATH)


def chunk_text(text: str, source: str, source_type: str = "document") -> list[Chunk]:
    if not text.strip():
        return []

    encoding = _tokenizer.encode(text)
    token_ids = encoding.ids
    offsets = encoding.offsets  # (char_start, char_end) per token

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(token_ids):
        end = min(start + CHUNK_SIZE, len(token_ids))

        # Recover text via character offsets to avoid detokenization artifacts
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_str = text[char_start:char_end]

        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_str,
            source=source,
            source_type=source_type,
            chunk_index=idx,
        ))

        idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
