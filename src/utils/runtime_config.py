from pathlib import Path
import yaml

DEFAULT_CFG_PATH = Path("configs/rag.yaml")
SELECTED_CFG_PATH = Path("configs/selected_experiment.yaml")

def load_runtime_config() -> dict:
    """
    Loads base RAG config and overrides it with
    selected_experiment.yaml if present.
    """
    with open(DEFAULT_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if SELECTED_CFG_PATH.exists():
        with open(SELECTED_CFG_PATH, "r", encoding="utf-8") as f:
            selected = yaml.safe_load(f) or {}

        # Deep merge (only what exists)
        for section, values in selected.items():
            if section not in cfg:
                cfg[section] = values
            elif isinstance(values, dict):
                cfg[section].update(values)

    return cfg
