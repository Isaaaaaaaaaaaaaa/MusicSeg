import argparse
import json
import os
import random
from typing import List, Tuple

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from .config import AudioConfig, BoundaryConfig, TransformerConfig
from .data import BoundaryDataset, list_pairs
from .model import BoundaryNet, TransformerBoundaryNet, save_boundary


def split_pairs(pairs: List[Tuple[str, str]], val_ratio: float, test_ratio: float, seed: int):
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test : n_test + n_val]
    train_pairs = pairs[n_test + n_val :]
    return train_pairs, val_pairs, test_pairs


def eval_boundary(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device, threshold: float = 0.3):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_elems = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            labels = batch["labels"].to(device)
            logits = model(mel)
            loss = loss_fn(logits, labels)
            # 使用更低的阈值来评估，因为正样本非常稀疏
            preds = (torch.sigmoid(logits) > threshold).to(labels.dtype)
            total_correct += (preds == labels).sum().item()
            total_elems += labels.numel()
            total_loss += loss.item() * labels.numel()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    if total_elems == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
        }
    loss = total_loss / total_elems
    acc = total_correct / total_elems
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(total_elems),
    }


def train(
    data_dir: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    arch: str = "transformer",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    audio_cfg = AudioConfig()
    boundary_cfg = BoundaryConfig()
    pairs = list_pairs(data_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, val_ratio, test_ratio, seed)
    train_dataset = BoundaryDataset(data_dir, audio_cfg, boundary_cfg, pairs=train_pairs)
    workers = max(1, min(os.cpu_count() or 1, 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    metrics_path = out_path + ".metrics.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    val_loader = None
    if val_pairs:
        val_dataset = BoundaryDataset(data_dir, audio_cfg, boundary_cfg, pairs=val_pairs)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )
    test_loader = None
    if test_pairs:
        test_dataset = BoundaryDataset(data_dir, audio_cfg, boundary_cfg, pairs=test_pairs)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )
    if arch == "lstm":
        model = BoundaryNet(audio_cfg.n_mels)
        arch_cfg = {"type": "lstm"}
    else:
        tcfg = TransformerConfig()
        model = TransformerBoundaryNet(
            n_mels=audio_cfg.n_mels,
            d_model=128,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
        )
        arch_cfg = {"type": "transformer", "tcfg": tcfg.__dict__}
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 计算类别不平衡比例，动态设置pos_weight
    # 边界点是极度稀疏的（通常<1%），需要更大的pos_weight
    pos_weight_value = 50.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_elems = 0.0
        tp = 0.0
        fp = 0.0
        fn = 0.0
        for batch in tqdm(train_loader, desc=f"boundary epoch {epoch+1}/{epochs}"):
            mel = batch["mel"].to(device)
            labels = batch["labels"].to(device)
            logits = model(mel)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                # 训练时使用较低阈值评估，因为正样本非常稀疏
                preds = (torch.sigmoid(logits) > 0.3).to(labels.dtype)
                total_correct += (preds == labels).sum().item()
                total_elems += labels.numel()
                total_loss += loss.item() * labels.numel()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
        if total_elems > 0:
            epoch_loss = total_loss / total_elems
            epoch_acc = total_correct / total_elems
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            line = {
                "phase": "train",
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "acc": epoch_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(total_elems),
                "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                "seed": seed,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(
                f"boundary train epoch {epoch+1}/{epochs} loss={epoch_loss:.6f} acc={epoch_acc:.6f} "
                f"precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}"
            )
        if val_loader is not None:
            val_metrics = eval_boundary(model, val_loader, loss_fn, device)
            val_line = {
                "phase": "val",
                "epoch": epoch + 1,
                **val_metrics,
                "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                "seed": seed,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_line, ensure_ascii=False) + "\n")
            print(
                f"boundary val epoch {epoch+1}/{epochs} loss={val_metrics['loss']:.6f} acc={val_metrics['acc']:.6f} "
                f"precision={val_metrics['precision']:.6f} recall={val_metrics['recall']:.6f} f1={val_metrics['f1']:.6f}"
            )

    save_boundary(
        out_path,
        model,
        {
            "audio_cfg": audio_cfg.__dict__,
            "boundary_cfg": boundary_cfg.__dict__,
            "arch": arch_cfg,
        },
    )
    if test_loader is not None:
        test_metrics = eval_boundary(model, test_loader, loss_fn, device)
        test_line = {
            "phase": "test",
            "epoch": epochs,
            **test_metrics,
            "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
            "seed": seed,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_line, ensure_ascii=False) + "\n")
        print(
            f"boundary test loss={test_metrics['loss']:.6f} acc={test_metrics['acc']:.6f} "
            f"precision={test_metrics['precision']:.6f} recall={test_metrics['recall']:.6f} f1={test_metrics['f1']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_path", default="checkpoints/boundary.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch", choices=["lstm", "transformer"], default="transformer")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        args.data_dir,
        args.out_path,
        args.epochs,
        args.batch_size,
        args.lr,
        args.arch,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
