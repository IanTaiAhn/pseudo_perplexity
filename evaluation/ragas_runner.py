"""
Runs the eval dataset through the real retrieval + generation pipeline and
scores the results with RAGAS. This is Layer 4's "offline evaluation" step —
see perplexity_clone_spec.md section 8 for metric definitions and thresholds.

RAGAS defaults to OpenAI for its judge LLM and embeddings. This project has no
OpenAI integration (see decision log), so both are wired to what the rest of
the system already uses: Claude as the judge LLM, and the project's own
sentence-transformers model as the judge embedder for answer_relevancy.
"""
import asyncio
import json
import os
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from agents.orchestrator import retrieve
from api.schemas import QueryRequest
from ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE
from synthesis.generator import generate

load_dotenv()

_EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
_RESULTS_PATH = Path(__file__).parent / "results" / "latest.json"

# The judge LLM is intentionally independent of LOCAL_LLM_MODEL (used to run
# the app's own generation step for free during dev) — a small local model is
# a weak judge for faithfulness/relevance NLI tasks, so eval always spends a
# little real API budget rather than trusting a cheap judge's scores.
_RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", os.getenv("LLM_MODEL", "claude-sonnet-4-6"))
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_TOP_K = 5

# Mirrors the confidence gate in api/routes/query.py so eval sees the same
# fallback behavior real users would get instead of forcing generation on
# junk retrieval.
_LOW_CONFIDENCE_THRESHOLD = 0.3
_LOW_CONFIDENCE_RESPONSE = (
    "I don't have enough information in the available sources to answer this question confidently."
)


def _load_eval_dataset() -> list[dict]:
    with open(_EVAL_DATASET_PATH) as f:
        return json.load(f)


async def _run_pipeline(question: str) -> tuple[str, list[str]]:
    """Runs one question through real retrieval + generation. Web search is
    disabled: the eval dataset's ground truth is scoped to a fixed document
    corpus, so pulling in live web results would make scores non-reproducible."""
    request = QueryRequest(query=question, top_k=_TOP_K, use_web_search=False, use_documents=True)
    chunks = await retrieve(request)
    chunks = chunks[:_TOP_K]

    if not chunks or (chunks[0].score is not None and chunks[0].score < _LOW_CONFIDENCE_THRESHOLD):
        return _LOW_CONFIDENCE_RESPONSE, [c.text for c in chunks]

    response = generate(query=question, chunks=chunks)
    return response.answer, [c.text for c in chunks]


def _build_ragas_dataset(eval_questions: list[dict]) -> EvaluationDataset:
    samples = []
    for item in eval_questions:
        answer, contexts = asyncio.run(_run_pipeline(item["question"]))
        samples.append({
            "user_input": item["question"],
            "response": answer,
            "retrieved_contexts": contexts or [""],  # RAGAS requires non-empty contexts
            "reference": item["ground_truth_answer"],
        })
    return EvaluationDataset.from_list(samples)


def _get_metrics() -> list:
    return [
        Faithfulness(),
        LLMContextPrecisionWithReference(name="context_relevance"),
        LLMContextRecall(),  # already named "context_recall" by default
        ResponseRelevancy(name="answer_relevance"),
    ]


def run() -> dict:
    eval_questions = _load_eval_dataset()
    dataset = _build_ragas_dataset(eval_questions)
    metrics = _get_metrics()

    judge_llm = ChatAnthropic(model=_RAGAS_LLM_MODEL)
    judge_embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)

    mlflow.set_experiment("ragas_eval")
    with mlflow.start_run():
        mlflow.log_param("num_questions", len(eval_questions))
        mlflow.log_param("embedding_model", _EMBEDDING_MODEL)
        mlflow.log_param("ragas_llm_model", _RAGAS_LLM_MODEL)
        mlflow.log_param("top_k", _TOP_K)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        result = evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)
        df = result.to_pandas()

        metric_names = ["faithfulness", "context_relevance", "context_recall", "answer_relevance"]
        aggregate = {}
        for metric_name in metric_names:
            if metric_name in df.columns:
                mean_score = float(df[metric_name].mean())
                aggregate[metric_name] = mean_score
                mlflow.log_metric(metric_name, mean_score)

        per_question = df[["user_input"] + [m for m in metric_names if m in df.columns]].to_dict(
            orient="records"
        )

        output = {
            "timestamp": time.time(),
            "num_questions": len(eval_questions),
            "aggregate": aggregate,
            "per_question": per_question,
        }

        _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        mlflow.log_artifact(str(_RESULTS_PATH))

    return output


if __name__ == "__main__":
    scores = run()
    print(json.dumps(scores["aggregate"], indent=2))
