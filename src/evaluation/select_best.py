import os
import json
import mlflow
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Experiment Evaluations")
OUTPUT_PATH = "configs/selected_experiment.yaml"


def main():
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[
            "metrics.avg_accuracy DESC",
            "metrics.avg_faithfulness DESC",
            "metrics.avg_recall DESC",
            "metrics.avg_precision DESC",
            "metrics.avg_refusal_penalty ASC",
            "attributes.start_time ASC",
        ],
    )

    best = runs[0]

    selected = {
        "experiment": {
            "run_id": best.info.run_id,
            "experiment_name": EXPERIMENT_NAME,
        },
        "llm": {
            "temperature": float(best.data.params["temperature"]),
        },
        "retrieval": {
            "top_k": int(best.data.params["top_k"]),
        },
        "chunking": {
            "chunk_size": int(best.data.params["chunk_size"]),
            "chunk_overlap": int(best.data.params["chunk_overlap"]),
        },
        "metrics": best.data.metrics,
    }

    # Write YAML-compatible dict (simple, no fancy tags)
    import yaml
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(selected, f, sort_keys=False)

    print("\n🏆 BEST EXPERIMENT SELECTED")
    print("Run ID:", best.info.run_id)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
