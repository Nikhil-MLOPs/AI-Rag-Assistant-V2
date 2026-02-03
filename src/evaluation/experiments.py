# src/evaluation/experiments.py

import os
import json
import time
import itertools
from pathlib import Path

from dotenv import load_dotenv
import dagshub
import mlflow

from src.evaluation.metrics import compute_metrics
from src.rag.chain import rag_chain

load_dotenv()


# DAGSHUB + MLFLOW

dagshub.init(
    repo_owner="Nikhil-MLOPs",
    repo_name="AI-Rag-Assistant-V2",
    mlflow=True,
)

mlflow.set_experiment(
    os.getenv("MLFLOW_EXPERIMENT_NAME", "Experiment Evaluations")
)

# PATHS

GOLDEN_PATH = Path(
    os.getenv("GOLDEN_DATASET_PATH", "data/golden/golden_dataset.jsonl")
)

# EXPERIMENT GRID (CONFIGURABLE)

TEMPERATURES = [0.1, 0.2]
TOP_KS = [1, 2, 3, 4]
CHUNK_SIZES = [800, 900, 1000, 1100]
CHUNK_OVERLAPS = [100, 150, 200]

# EXECUTION CAPS

MAX_EXPERIMENT_SECONDS = int(os.getenv("MAX_EXPERIMENT_SECONDS", 120))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 7))
MAX_QUESTION_SECONDS = int(os.getenv("MAX_QUESTION_SECONDS", 30))

# LOAD GOLDEN DATASET

def load_golden():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# EVALUATION-SAFE RAG RUNNER

def run_rag_for_evaluation(question: str, timeout: int) -> str:
    """
    Runs rag_chain and attempts to collect the FULL final answer
    (including validation + sources), within a time budget.
    Never raises.
    """
    answer = ""
    start = time.time()

    try:
        for token in rag_chain(question):
            answer += token
            if time.time() - start > timeout:
                break
    except Exception:
        pass

    return answer

# MAIN EXPERIMENT LOOP

def run():
    golden_data = load_golden()[:MAX_QUESTIONS]

    grid = list(itertools.product(
        TEMPERATURES,
        TOP_KS,
        CHUNK_SIZES,
        CHUNK_OVERLAPS,
    ))

    assert len(grid) >= 50, "Less than 50 experiments!"

    for idx, (temp, top_k, chunk_size, chunk_overlap) in enumerate(grid, 1):

        experiment_start = time.time()

        with mlflow.start_run(run_name=f"exp_{idx:03d}"):

            
            # Log experiment params
            mlflow.log_params({
                "temperature": temp,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            })

            agg = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "faithfulness": 0.0,
                "refusal_penalty": 0.0,
            }

            evaluated = 0

            for golden in golden_data:

                
                # Experiment-level timeout
                if time.time() - experiment_start > MAX_EXPERIMENT_SECONDS:
                    mlflow.set_tag("early_stop", "experiment_timeout")
                    break

                question = golden["question"]

                answer = run_rag_for_evaluation(
                    question=question,
                    timeout=MAX_QUESTION_SECONDS,
                )

                metrics = compute_metrics(
                    question=question,
                    answer=answer,
                    golden=golden,
                )

                for k in agg:
                    agg[k] += metrics[k]

                evaluated += 1

            
            # Log aggregated metrics
            evaluated = max(evaluated, 1)

            for k, v in agg.items():
                mlflow.log_metric(f"avg_{k}", v / evaluated)

            mlflow.log_metric("questions_evaluated", evaluated)
            mlflow.set_tag("status", "success")


if __name__ == "__main__":
    run()
