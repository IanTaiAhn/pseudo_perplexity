# Manual Testing Guide — Layers 1-5

Steps to manually exercise the system end-to-end, using **local models** (no
Anthropic API key needed for the main path) and the **Tavily free tier**
(1,000 searches/month) for web search. Run through these in order — later
steps assume the app is already up and a document has been ingested.

## 0. Prerequisites

- Python 3.11, a virtualenv, `pip install -r requirements.txt`
- A Tavily API key (free tier): https://tavily.com
- Enough disk/RAM to download two small HF models on first run:
  - `sentence-transformers/all-MiniLM-L6-v2` (embeddings + reranker base, ~90MB)
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker, ~90MB)
  - A local generation model, e.g. `Qwen/Qwen2.5-1.5B-Instruct` (~3GB) — CPU-only works, just slower (10-60s/query depending on hardware)

## 1. Configure `.env` for a fully local run

Copy `.env.example` to `.env` and set:

```
TAVILY_API_KEY=<your real key>
LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LOG_LEVEL=INFO
```

Leave `ANTHROPIC_API_KEY` unset/blank. Leave `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
blank too unless you specifically want to test Langfuse tracing — `generator.py`
no-ops tracing when they're unset (see `ARCHITECTURE.md` decision log, 2026-07-08).

**Note:** the Layer 4 RAGAS eval pipeline (`evaluation/ragas_runner.py`) always
calls real Claude as its judge LLM regardless of `LOCAL_LLM_MODEL` — that's a
separate, optional step (see step 9). Everything else (ingest, query, retrieval,
observability) runs fully local + Tavily.

## 2. Start the app

Without Docker (fastest iteration):
```
uvicorn api.main:app --reload
```

Or with Docker (also brings up Prometheus + Grafana):
```
docker-compose up --build
```

First local-model query will be slow (model download + load into memory) —
that's expected, not a bug.

## 3. Smoke test

```
curl localhost:8000/health
```
Expect `{"status": "ok"}`.

## 4. Ingest a document

Grab any PDF or `.txt` you have (e.g. a paper) and:
```
curl -F "file=@yourdoc.pdf" localhost:8000/ingest
```
Expect `{"chunks_indexed": N, "source": "yourdoc.pdf", "status": "ok"}` with `N > 0`.

Try an edge case too: an empty file (expect HTTP 422 "Document appears to be empty")
and a non-UTF8/non-PDF file (expect 422 "File must be UTF-8 text or PDF").

## 5. Inspect what got indexed

```
curl localhost:8000/debug/collection
```
Confirm `chunks_in_db` matches what step 4 reported, and `sample_sources`
shows your filename.

```
curl "localhost:8000/debug/query?q=<something your doc actually discusses>"
```
Look at the `score` values returned — these are raw cosine similarities
(pre-hybrid, pre-rerank), so a good match should be noticeably higher than
a bad one. Try a query totally unrelated to the document's content and
confirm scores drop.

## 6. End-to-end query — documents only

```
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "<a question your ingested doc actually answers>", "use_web_search": false}'
```
Check:
- `answer` contains `[1]`/`[2]`-style citation markers
- `citations` list is non-empty and each `citation_number` referenced in the
  answer text actually appears in `citations`
- `model_used` is your `LOCAL_LLM_MODEL` string
- `estimated_cost_usd` is `0.0` (local inference — no API cost)
- `latency_ms` seems plausible for your hardware

## 7. End-to-end query — web search only (Tavily)

```
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "<something current-events-y your doc would not know>", "use_documents": false}'
```
Confirm citations point to real URLs (`source_type` web) and the answer
reads like it's grounded in fresh web content, not the ingested doc.

If `TAVILY_API_KEY` is missing/invalid, `web_search_agent.search()` silently
returns `[]` (see `agents/web_search_agent.py`) rather than erroring — so if
this comes back empty, first confirm the key is actually set before assuming
retrieval is broken.

## 8. Combined retrieval

Run a query where both the doc and the web have relevant info
(`use_web_search`/`use_documents` both default `true`). Confirm the final
top results plausibly interleave `document` and `web` source types — this
exercises the merge + cross-encoder rerank in `agents/orchestrator.py`.

## 9. Low-confidence / no-match behavior

Ask something neither your doc nor the web can answer well, or turn off both
sources' realistic chances (e.g. a nonsense query). Confirm you get the fixed
`"I don't have enough information..."` response with empty `citations` and
`estimated_cost_usd: 0.0` (no LLM call should happen — check logs for absence
of an `llm_call` event).

**Worth specifically checking:** the confidence gate in `api/routes/query.py`
compares the top chunk's `score` against a fixed `0.3` threshold — but after
Layer 3, `score` is set from the cross-encoder reranker (`retrieval/reranker.py`),
whose raw output is an *unbounded* logit (often ranges roughly -10 to +10, not
0-1). Compare a few `/debug/query` cosine scores (bounded 0-1) against the
`score` your `/query` responses actually show for the same question — if
reranked scores rarely cross 0.3 for good matches (or routinely cross it for
bad ones), the gate isn't calibrated to the post-rerank scale and you may want
to flag/adjust the threshold.

## 10. Guardrails (currently stubs — don't expect behavior)

`guardrails/input_validator.py` and `guardrails/output_validator.py` are empty
placeholder files (just a comment, no code). Don't spend time testing prompt
injection resistance or toxicity filtering yet — there's nothing there to test.

## 11. Observability

- `curl localhost:8000/metrics` — confirm `query_total`, `llm_latency_seconds`,
  `retrieval_latency_seconds`, `retrieval_hit_rate`, `llm_tokens_total` all show
  non-zero activity after a few queries from steps 6-9.
- Watch stdout while querying — should be single-line JSON logs (structlog),
  each with `query_id` bound across `query_received` → `retrieval_completed` →
  `llm_call` (or `low_retrieval_confidence`) for the same request.
- If running via `docker-compose up`, open `http://localhost:9090` (Prometheus,
  confirm the `app` target is `UP`) and `http://localhost:3000` (Grafana,
  admin/admin, dashboard should auto-load from `infra/grafana/provisioning/`
  with panels populating as you send queries).
- Drift detector — run 30+ varied queries, then:
  ```
  python -c "from monitoring.drift_detector import snapshot_baseline; print(snapshot_baseline())"
  ```
  Run a batch of on-topic queries and a batch of deliberately off-topic ones,
  then:
  ```
  python -c "from monitoring.drift_detector import check_drift; print(check_drift())"
  ```
  Confirm `drifted=True` shows up once you've fed it a visibly different
  query topic mix, and `drifted=False` on similar-topic queries.

## 12. Automated test suite

```
pytest tests/ -v
```
All should pass without any API keys set — they mock the reranker/LLM/Tavily
calls (see `tests/test_reranker.py`, `tests/test_orchestrator.py`, etc.).

## 13. (Optional, costs a few cents) RAGAS regression gate

Only if you want to test Layer 4's eval gate — this genuinely requires
`ANTHROPIC_API_KEY` since RAGAS uses Claude as judge regardless of
`LOCAL_LLM_MODEL`:
```
python -m evaluation.regression_test
```
Expect PASS/WARN/FAIL lines per metric (faithfulness, context_relevance,
answer_relevance, context_recall) against the thresholds in
`perplexity_clone_spec.md` section 8.

## Known rough edges to keep in mind while testing

- Guardrails are unimplemented stubs (step 10).
- Confidence-gate threshold may be miscalibrated against real cross-encoder
  score ranges (step 9) — worth deciding whether to fix in this pass.
- Tavily free tier caps at 1,000 searches/month — each `/query` with
  `use_web_search: true` burns one credit.
- First query after a fresh boot is slow while models load into memory —
  expected, not a regression.
