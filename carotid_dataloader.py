from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def natural_key(text: str) -> List[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def default_transform(image: Image.Image) -> Tensor:
    array = torch.from_numpy(np.array(image, copy=True))
    if array.ndim == 2:
        array = array.unsqueeze(0)
    else:
        array = array.permute(2, 0, 1)
    return array.float() / 255.0


def select_middle_n(paths: Sequence[Path], n_images: int) -> List[Path]:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}.")
    if len(paths) < n_images:
        raise ValueError(f"Need at least {n_images} images, but only found {len(paths)}.")

    start = (len(paths) - n_images) // 2
    end = start + n_images
    return list(paths[start:end])


@dataclass
class SampleInfo:
    name: str
    label: int
    image_paths: List[Path]


class CarotidPlaqueDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        excel_path: str | Path,
        n_images: int,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        image_mode: str = "L",
        label_conflict: str = "skip",
        verbose: bool = True,
    ) -> None:
        self.image_root = Path(image_root)
        self.excel_path = Path(excel_path)
        self.n_images = n_images
        self.transform = transform or default_transform
        self.image_mode = image_mode
        self.label_conflict = label_conflict
        self.verbose = verbose

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        if self.label_conflict not in {"skip", "first", "error"}:
            raise ValueError("label_conflict must be one of: 'skip', 'first', 'error'.")

        self.samples, self.stats = self._build_samples()

        if self.verbose:
            self._print_stats()

    def _build_label_mapping(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        df = pd.read_excel(self.excel_path, usecols=["name", "label"])
        df = df.dropna(subset=["name", "label"]).copy()
        df["name"] = df["name"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

        label_mapping: Dict[str, int] = {}
        stats = {
            "excel_rows": len(df),
            "duplicate_same_label": 0,
            "duplicate_conflict": 0,
        }

        for name, group in df.groupby("name"):
            labels = sorted(group["label"].unique().tolist())
            if len(labels) == 1:
                label_mapping[name] = labels[0]
                if len(group) > 1:
                    stats["duplicate_same_label"] += 1
                continue

            stats["duplicate_conflict"] += 1
            if self.label_conflict == "skip":
                continue
            if self.label_conflict == "first":
                label_mapping[name] = int(group["label"].iloc[0])
                continue
            raise ValueError(f"Conflicting labels found for {name}: {labels}")

        return label_mapping, stats

    def _build_samples(self) -> Tuple[List[SampleInfo], Dict[str, object]]:
        label_mapping, excel_stats = self._build_label_mapping()
        patient_dirs = sorted([path for path in self.image_root.iterdir() if path.is_dir()], key=lambda p: natural_key(p.name))
        samples: List[SampleInfo] = []
        skipped_missing_label = 0
        skipped_too_short = 0
        image_counts: List[int] = []

        for patient_dir in patient_dirs:
            name = patient_dir.name
            if name not in label_mapping:
                skipped_missing_label += 1
                continue

            image_paths = sorted(
                [path for path in patient_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
                key=lambda p: natural_key(p.name),
            )
            if len(image_paths) < self.n_images:
                skipped_too_short += 1
                continue

            selected_paths = select_middle_n(image_paths, self.n_images)
            samples.append(SampleInfo(name=name, label=label_mapping[name], image_paths=selected_paths))
            image_counts.append(len(image_paths))

        label_counter = Counter(sample.label for sample in samples)
        stats: Dict[str, object] = {
            "image_root": str(self.image_root),
            "excel_path": str(self.excel_path),
            "requested_n_images": self.n_images,
            "patient_dirs_found": len(patient_dirs),
            "valid_samples": len(samples),
            "skipped_missing_or_conflict_label": skipped_missing_label,
            "skipped_too_short": skipped_too_short,
            "label_0_count": label_counter.get(0, 0),
            "label_1_count": label_counter.get(1, 0),
            "min_raw_images": min(image_counts) if image_counts else 0,
            "max_raw_images": max(image_counts) if image_counts else 0,
            "avg_raw_images": round(sum(image_counts) / len(image_counts), 2) if image_counts else 0.0,
        }
        stats.update(excel_stats)
        return samples, stats

    def _print_stats(self) -> None:
        print("=" * 60)
        print("CarotidPlaqueDataset loaded")
        print(f"image_root: {self.stats['image_root']}")
        print(f"excel_path: {self.stats['excel_path']}")
        print(f"requested_n_images: {self.stats['requested_n_images']}")
        print(f"patient_dirs_found: {self.stats['patient_dirs_found']}")
        print(f"valid_samples: {self.stats['valid_samples']}")
        print(f"label_0_count (stable): {self.stats['label_0_count']}")
        print(f"label_1_count (unstable): {self.stats['label_1_count']}")
        print(f"skipped_missing_or_conflict_label: {self.stats['skipped_missing_or_conflict_label']}")
        print(f"skipped_too_short: {self.stats['skipped_too_short']}")
        print(f"duplicate_same_label_in_excel: {self.stats['duplicate_same_label']}")
        print(f"duplicate_conflict_in_excel: {self.stats['duplicate_conflict']}")
        print(
            "raw_images_per_patient: "
            f"min={self.stats['min_raw_images']}, "
            f"max={self.stats['max_raw_images']}, "
            f"avg={self.stats['avg_raw_images']}"
        )
        print("=" * 60)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        images: List[Tensor] = []
        for image_path in sample.image_paths:
            with Image.open(image_path) as image:
                images.append(self.transform(image.convert(self.image_mode)))

        image_tensor = torch.stack(images, dim=0)
        label_tensor = torch.tensor(sample.label, dtype=torch.long)

        return {
            "name": sample.name,
            "images": image_tensor,
            "label": label_tensor,
        }


def create_carotid_dataloader(
    image_root: str | Path,
    excel_path: str | Path,
    n_images: int,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable[[Image.Image], Tensor]] = None,
    image_mode: str = "L",
    label_conflict: str = "skip",
    verbose: bool = True,
) -> DataLoader:
    dataset = CarotidPlaqueDataset(
        image_root=image_root,
        excel_path=excel_path,
        n_images=n_images,
        transform=transform,
        image_mode=image_mode,
        label_conflict=label_conflict,
        verbose=verbose,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    image_root = "/media/data/wjf/data/Carotid_artery_crop_old"
    # image_root = "/media/data/wjf/data/Carotid_artery_ps/img_crop"
    
    excel_path = "/media/data/wjf/data/Carotid_artery_excel.xlsx"

    dataloader = create_carotid_dataloader(
        image_root=image_root,
        excel_path=excel_path,
        n_images=100,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        image_mode="L",
        label_conflict="skip",
        verbose=True,
    )

    first_batch = next(iter(dataloader))
    print(f"batch names: {first_batch['name']}")
    print(f"batch images shape: {tuple(first_batch['images'].shape)}")
    print(f"batch labels: {first_batch['label'].tolist()}")
