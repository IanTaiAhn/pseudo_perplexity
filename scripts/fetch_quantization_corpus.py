"""
Downloads a small family of LLM-quantization papers (same subfield as the
existing QRazor eval set) into corpus/quantization_papers/, then optionally
POSTs each one to a running app's /ingest endpoint.

Run this on a machine with normal network access — it will NOT work from a
sandboxed environment with restricted egress (arxiv.org gets blocked there).

Usage:
    python scripts/fetch_quantization_corpus.py              # download only
    python scripts/fetch_quantization_corpus.py --ingest      # download + ingest into localhost:8000
    python scripts/fetch_quantization_corpus.py --ingest --api-url http://localhost:8000
"""
import argparse
from pathlib import Path

import httpx

# Same narrow subfield as QRazor (LLM quantization) — chosen so questions like
# "what group size does QRazor use?" have hard negatives from papers that use
# similar vocabulary but different specific numbers. See ARCHITECTURE.md.
PAPERS = [
    ("gptq_2210.17323", "https://arxiv.org/pdf/2210.17323", "GPTQ: Accurate Post-Training Quantization for GPTs"),
    ("smoothquant_2211.10438", "https://arxiv.org/pdf/2211.10438", "SmoothQuant"),
    ("awq_2306.00978", "https://arxiv.org/pdf/2306.00978", "AWQ: Activation-aware Weight Quantization"),
    ("omniquant_2308.13137", "https://arxiv.org/pdf/2308.13137", "OmniQuant"),
    ("quarot_2404.00456", "https://arxiv.org/pdf/2404.00456", "QuaRot"),
    ("qserve_2405.04532", "https://arxiv.org/pdf/2405.04532", "QServe"),
    ("spinquant_2405.16406", "https://arxiv.org/pdf/2405.16406", "SpinQuant"),
]

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "quantization_papers"


def download_papers() -> list[Path]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(follow_redirects=True, timeout=30.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        for name, url, title in PAPERS:
            dest = CORPUS_DIR / f"{name}.pdf"
            if dest.exists():
                print(f"skip (already downloaded): {dest.name}")
                paths.append(dest)
                continue
            print(f"downloading {title} <- {url}")
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            paths.append(dest)
    return paths


def ingest_papers(paths: list[Path], api_url: str) -> None:
    with httpx.Client(timeout=120.0) as client:
        for path in paths:
            print(f"ingesting {path.name} into {api_url}/ingest")
            with open(path, "rb") as f:
                resp = client.post(f"{api_url}/ingest", files={"file": (path.name, f, "application/pdf")})
            resp.raise_for_status()
            print(f"  -> {resp.json()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="POST each PDF to a running app's /ingest endpoint")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the running app")
    args = parser.parse_args()

    downloaded = download_papers()
    print(f"\n{len(downloaded)} papers in {CORPUS_DIR}")

    if args.ingest:
        ingest_papers(downloaded, args.api_url)
