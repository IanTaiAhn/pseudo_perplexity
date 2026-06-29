from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.routes import ingest, query

app = FastAPI(
    title="Pseudo-Perplexity",
    description="AI research assistant with grounded answers and inline citations.",
    version="0.1.0",
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
