import httpx
from bs4 import BeautifulSoup
from ingestion.chunker import chunk_text
from ingestion.http_client import get_ssl_verify
from api.schemas import Chunk


async def fetch_and_chunk(url: str) -> list[Chunk]:
    """Fetch a URL, extract clean prose, and return as chunks."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, verify=get_ssl_verify()) as client:
        response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove boilerplate elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text.strip():
        return []

    return chunk_text(text, source=url, source_type="web")
