import io
import fitz  # pymupdf
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas import IngestResponse
from ingestion.chunker import chunk_text
from ingestion.indexer import index_chunks

router = APIRouter()


def _extract_text_from_pdf(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "unknown"
    data = await file.read()

    if filename.lower().endswith(".pdf"):
        try:
            text = _extract_text_from_pdf(data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="File must be UTF-8 text or PDF.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Document appears to be empty.")

    chunks = chunk_text(text, source=filename, source_type="document")
    count = index_chunks(chunks)

    return IngestResponse(
        chunks_indexed=count,
        source=filename,
        status="ok",
    )
