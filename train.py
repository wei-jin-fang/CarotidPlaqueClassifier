from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from carotid_dataloader import CarotidPlaqueDataset
from models import build_model
from utils import export_attention_artifacts
from utils.io import create_experiment_dir, save_config, save_history, setup_logger
from utils.metrics import format_metrics
from utils.plotting import plot_loss_curves
from utils.training import evaluate_one_epoch, set_seed, train_one_epoch
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train carotid plaque stability classifier.")
    parser.add_argument("--image_root", type=str, default="/media/data/wjf/data/Carotid_artery_crop_old")
    parser.add_argument("--excel_path", type=str, default="/media/data/wjf/data/Carotid_artery_excel.xlsx")
    parser.add_argument("--output_root", type=str, default="runs")
    parser.add_argument("--model_name", type=str, default="simple_3dcnn")
    parser.add_argument("--n_images", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label_conflict", type=str, default="skip", choices=["skip", "first", "error"])
    parser.add_argument("--export_attention_artifacts", action="store_true")
    parser.add_argument("--attention_export_batches", type=int, default=1)

    parser.add_argument("--simple_3dcnn_base_channels", type=int, default=8)

    parser.add_argument("--resnet3d_base_channels", type=int, default=32)
    parser.add_argument(
        "--resnet3d_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
    )
    parser.add_argument("--resnet3d_attention_reduction", type=int, default=4)
    parser.add_argument("--resnet3d_attention_stage", type=int, default=3, choices=[1, 2, 3, 4])

    parser.add_argument("--ultra_light_3dcnn_base_channels", type=int, default=8)
    parser.add_argument("--classifier_hidden_dim", type=int, default=32)
    parser.add_argument("--classifier_dropout", type=float, default=0.3)
    return parser.parse_args()


def validate_split_ratios(args: argparse.Namespace) -> None:
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio:.6f}."
        )


def build_base_dataset(args: argparse.Namespace) -> CarotidPlaqueDataset:
    return CarotidPlaqueDataset(
        image_root=args.image_root,
        excel_path=args.excel_path,
        n_images=args.n_images,
        image_mode="L",
        label_conflict=args.label_conflict,
        verbose=True,
    )


def build_cv_splits(args: argparse.Namespace) -> Tuple[CarotidPlaqueDataset, List[Dict[str, List[int]]]]:
    dataset = build_base_dataset(args)
    labels = [sample.label for sample in dataset.samples]
    indices = list(range(len(dataset)))

    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_state)
    val_fraction = args.val_ratio / (args.train_ratio + args.val_ratio)

    folds: List[Dict[str, List[int]]] = []
    for fold_id, (trainval_indices, test_indices) in enumerate(splitter.split(indices, labels), start=1):
        trainval_indices = list(trainval_indices)
        test_indices = list(test_indices)

        trainval_labels = [labels[index] for index in trainval_indices]
        train_indices, val_indices = train_test_split(
            trainval_indices,
            test_size=val_fraction,
            random_state=args.random_state + fold_id,
            stratify=trainval_labels,
        )

        folds.append(
            {
                "fold_id": fold_id,
                "train_indices": list(train_indices),
                "val_indices": list(val_indices),
                "test_indices": test_indices,
            }
        )

    return dataset, folds


def build_fold_dataloaders(
    dataset: CarotidPlaqueDataset,
    fold_split: Dict[str, List[int]],
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_dataset = Subset(dataset, fold_split["train_indices"])
    val_dataset = Subset(dataset, fold_split["val_indices"])
    test_dataset = Subset(dataset, fold_split["test_indices"])

    train_sampler = None
    train_shuffle = True
    if args.model_name in {
        "resnet3d",
        "resnet2d_attention",
        "resnet2d",
        "resnet3d_attention",
        "resnet3d_attention_scalingL2",
        "ultra_light_3dcnn",
    }:
        train_labels = [dataset.samples[index].label for index in fold_split["train_indices"]]
        label_counts = {
            0: sum(label == 0 for label in train_labels),
            1: sum(label == 1 for label in train_labels),
        }
        sample_weights = [1.0 / label_counts[label] for label in train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    labels = [sample.label for sample in dataset.samples]
    split_stats = {
        "train_samples": len(fold_split["train_indices"]),
        "val_samples": len(fold_split["val_indices"]),
        "test_samples": len(fold_split["test_indices"]),
        "train_label_0": sum(labels[index] == 0 for index in fold_split["train_indices"]),
        "train_label_1": sum(labels[index] == 1 for index in fold_split["train_indices"]),
        "val_label_0": sum(labels[index] == 0 for index in fold_split["val_indices"]),
        "val_label_1": sum(labels[index] == 1 for index in fold_split["val_indices"]),
        "test_label_0": sum(labels[index] == 0 for index in fold_split["test_indices"]),
        "test_label_1": sum(labels[index] == 1 for index in fold_split["test_indices"]),
    }
    return train_loader, val_loader, test_loader, split_stats


def build_class_weights(train_dataset: Subset, device: torch.device) -> torch.Tensor | None:
    labels = [train_dataset.dataset.samples[index].label for index in train_dataset.indices]
    label_0 = sum(label == 0 for label in labels)
    label_1 = sum(label == 1 for label in labels)
    imbalance_ratio = max(label_0, label_1) / min(label_0, label_1)
    if imbalance_ratio < 1.5:
        return None

    total = label_0 + label_1
    weight_0 = total / (2 * label_0)
    weight_1 = total / (2 * label_1)
    return torch.tensor([weight_0, weight_1], dtype=torch.float32, device=device)


def save_cv_losses_csv(
    fold_histories: Sequence[Dict[str, List[float]]],
    save_path: Path,
    num_folds: int,
) -> None:
    max_len = max(len(history["train_loss"]) for history in fold_histories)
    headers = ["epoch"]
    headers.extend([f"train_loss_fold{fold_id}" for fold_id in range(1, num_folds + 1)])
    headers.extend([f"val_loss_fold{fold_id}" for fold_id in range(1, num_folds + 1)])

    with save_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for epoch in range(max_len):
            row = [epoch]
            for fold_id in range(num_folds):
                train_losses = fold_histories[fold_id]["train_loss"]
                row.append(train_losses[epoch] if epoch < len(train_losses) else "")
            for fold_id in range(num_folds):
                val_losses = fold_histories[fold_id]["val_loss"]
                row.append(val_losses[epoch] if epoch < len(val_losses) else "")
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_split_ratios(args)
    if args.num_folds != 10:
        raise ValueError(f"This training mode requires 10 folds, got num_folds={args.num_folds}.")
    set_seed(args.seed)

    experiment_dir = create_experiment_dir(args.output_root)
    logger = setup_logger(experiment_dir / "train.log")

    config = vars(args).copy()
    config["image_mode"] = "L"
    save_config(config, experiment_dir / "config.yaml")

    logger.info("Experiment directory: %s", experiment_dir)
    logger.info("Configuration saved to %s", experiment_dir / "config.yaml")

    dataset, cv_splits = build_cv_splits(args)
    logger.info("Prepared %d-fold split for %d samples.", args.num_folds, len(dataset))

    device = torch.device(args.device)
    logger.info("Model name: %s", args.model_name)

    loss_curve_dir = experiment_dir / "loss_curves_10fold"
    loss_curve_dir.mkdir(parents=True, exist_ok=True)

    fold_histories: List[Dict[str, List[float]]] = []
    fold_results: List[Dict[str, object]] = []

    for fold_split in cv_splits:
        fold_id = int(fold_split["fold_id"])
        logger.info("========== Fold %d/%d =========", fold_id, args.num_folds)

        train_loader, val_loader, test_loader, split_stats = build_fold_dataloaders(dataset, fold_split, args)
        logger.info("Fold %d split stats: %s", fold_id, split_stats)

        model = build_model(args.model_name, args).to(device)

        class_weights = build_class_weights(train_loader.dataset, device)
        if class_weights is None:
            logger.info("Fold %d class weights: disabled", fold_id)
            criterion = nn.CrossEntropyLoss()
        else:
            logger.info("Fold %d class weights: %s", fold_id, class_weights.tolist())
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_acc = float("-inf")
        best_val_metrics = None
        model_path = experiment_dir / f"best_model_fold{fold_id}.pth"

        epoch0_train_loss, _ = evaluate_one_epoch(model, train_loader, criterion, device)
        epoch0_val_loss, epoch0_val_metrics = evaluate_one_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(epoch0_train_loss)
        history["val_loss"].append(epoch0_val_loss)
        logger.info(
            "Fold %d Epoch [0/%d] lr=%.6f train_loss=%.4f val_loss=%.4f %s",
            fold_id,
            args.epochs,
            optimizer.param_groups[0]["lr"],
            epoch0_train_loss,
            epoch0_val_loss,
            format_metrics(epoch0_val_metrics),
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = evaluate_one_epoch(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            scheduler.step()

            logger.info(
                "Fold %d Epoch [%d/%d] lr=%.6f train_loss=%.4f val_loss=%.4f %s",
                fold_id,
                epoch,
                args.epochs,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
                format_metrics(val_metrics),
            )

            current_acc = val_metrics["accuracy"]
            comparable_acc = current_acc if current_acc == current_acc else float("-inf")
            if comparable_acc > best_val_acc:
                best_val_acc = comparable_acc
                best_val_metrics = val_metrics.copy()
                torch.save(model.state_dict(), model_path)

        if best_val_metrics is None:
            _, best_val_metrics = evaluate_one_epoch(model, val_loader, criterion, device)
            torch.save(model.state_dict(), model_path)

        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_metrics = evaluate_one_epoch(model, test_loader, criterion, device)

        logger.info("Fold %d best validation metrics: %s", fold_id, format_metrics(best_val_metrics))
        logger.info("Fold %d final test loss=%.4f", fold_id, test_loss)
        logger.info(
            "Fold %d test metrics: accuracy=%.4f auc=%.4f recall=%.4f f1=%.4f precision=%.4f",
            fold_id,
            test_metrics["accuracy"],
            test_metrics["auc"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["precision"],
        )

        plot_path = loss_curve_dir / f"loss_curve_fold{fold_id}.png"
        plot_loss_curves(history["train_loss"], history["val_loss"], plot_path)
        logger.info("Fold %d saved model to %s", fold_id, model_path)
        logger.info("Fold %d saved curve to %s", fold_id, plot_path)

        fold_histories.append(history)
        fold_results.append(
            {
                "fold_id": fold_id,
                "split_stats": split_stats,
                "best_val_metrics": best_val_metrics,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "model_path": str(model_path),
                "loss_curve_path": str(plot_path),
            }
        )

        if args.model_name in {
            "resnet2d_attention",
            "resnet3d_attention",
            "resnet3d_attention_scalingL2",
        } and args.export_attention_artifacts:
            attention_dir = experiment_dir / "attention_artifacts" / f"fold_{fold_id}"
            export_attention_artifacts(
                model=model,
                dataloader=test_loader,
                device=device,
                save_dir=attention_dir,
                max_batches=args.attention_export_batches,
            )
            logger.info("Fold %d saved attention artifacts to %s", fold_id, attention_dir)

    loss_csv_path = experiment_dir / "losses_10fold.csv"
    save_cv_losses_csv(fold_histories, loss_csv_path, args.num_folds)

    history = {
        "num_folds": args.num_folds,
        "epochs": args.epochs,
        "fold_results": fold_results,
        "fold_histories": fold_histories,
    }
    save_history(history, experiment_dir / "history.json")

    logger.info("Saved 10-fold losses CSV to %s", loss_csv_path)
    logger.info("Saved 10-fold curves to %s", loss_curve_dir)


if __name__ == "__main__":
    main()
