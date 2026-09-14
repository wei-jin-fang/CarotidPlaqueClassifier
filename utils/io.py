from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def create_experiment_dir(output_root: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_root) / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def save_config(config: Dict[str, Any], save_path: str | Path) -> None:
    with Path(save_path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def save_history(history: Dict[str, Any], save_path: str | Path) -> None:
    with Path(save_path).open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger(f"train_logger_{Path(log_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
