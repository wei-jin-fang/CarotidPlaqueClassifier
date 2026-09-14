from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .training import prepare_batch


@torch.no_grad()
def export_attention_artifacts(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str | Path,
    max_batches: int = 1,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    exported = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images, _ = prepare_batch(batch, device)
        _ = model(images)

        if not hasattr(model, "get_attention_outputs"):
            raise AttributeError("Model does not provide get_attention_outputs().")

        attention_outputs = model.get_attention_outputs()
        relation_matrix = attention_outputs["relation_matrix"]
        feature_sequence = attention_outputs["feature_sequence"]
        feature_metadata = attention_outputs["feature_metadata"]

        if relation_matrix is None or feature_sequence is None or feature_metadata is None:
            continue

        names = batch["name"]
        relation_matrix = relation_matrix.cpu().numpy()
        feature_sequence = feature_sequence.cpu().numpy()

        for sample_idx, sample_name in enumerate(names):
            sample_dir = save_dir / f"batch_{batch_idx:03d}_{sample_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            np.save(sample_dir / "relation_matrix.npy", relation_matrix[sample_idx])
            np.save(sample_dir / "feature_sequence.npy", feature_sequence[sample_idx])

            metadata: Dict[str, Any] = {
                "name": sample_name,
                "stage": feature_metadata["stage"],
                "num_slices_after_stage": int(feature_sequence[sample_idx].shape[1]),
                "channels": int(feature_sequence[sample_idx].shape[0]),
                "height": int(feature_sequence[sample_idx].shape[2]),
                "width": int(feature_sequence[sample_idx].shape[3]),
                "relation_matrix_shape": list(relation_matrix[sample_idx].shape),
                "feature_sequence_shape": list(feature_sequence[sample_idx].shape),
            }
            with (sample_dir / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)

            exported += 1

    summary = {"exported_samples": exported, "save_dir": str(save_dir)}
    with (save_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
