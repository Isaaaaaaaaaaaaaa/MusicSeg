import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import AudioConfig, default_labels, functional_labels
from .data import ClassifierDataset, ClassifierMelDataset, label_counts, list_pairs, list_pairs_by_split, load_segments, unique_labels, infer_functional_segments
from .model import SegmentClassifier, SegmentMelCNN, SegmentMelAttn, SegmentMelAttnGate, save_classifier, SoftmaxFocalLoss


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float, weight: Optional[torch.Tensor]):
    # Wrapper for SoftmaxFocalLoss from model.py
    # Adapt arguments to match SoftmaxFocalLoss signature if needed
    # SoftmaxFocalLoss(alpha=..., gamma=...)
    # Here weight is class weights. SoftmaxFocalLoss handles alpha (class balance).
    # We can pass weight as alpha if it matches.
    # However, standard Focal Loss alpha is a float or a list.
    # Let's instantiate it on the fly or just use the class implementation.
    
    # Actually, the SoftmaxFocalLoss I added takes (pred, targets). 
    # If targets is long, it does one-hot.
    # It has alpha and gamma in __init__.
    # To support per-batch dynamic weight, we might need to modify it or just ignore 'weight' here if SoftmaxFocalLoss handles it via alpha.
    # But 'weight' passed here is likely class weights calculated from dataset.
    
    # Let's trust the SoftmaxFocalLoss implementation I added.
    # But wait, my SoftmaxFocalLoss implementation expects alpha in __init__.
    # If I want to use the calculated 'weight' tensor as alpha, I should instantiate it with that.
    pass

# Redefine to use the class
class FocalLossWrapper(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.loss = SoftmaxFocalLoss(gamma=gamma, alpha=0.25) # Default alpha
        # If weight is provided, we might want to use it. 
        # But SoftmaxFocalLoss logic for alpha is: alpha * target + (1-alpha) * (1-target)
        # This is for binary/multilabel. For multiclass, alpha should be a list of weights.
        # My SoftmaxFocalLoss implementation:
        # if self.alpha > 0:
        #    alpha_t = self.alpha * targets_onehot + (1 - self.alpha) * (1 - targets_onehot)
        # This looks like binary alpha.
        
        # SongFormer's SoftmaxFocalLoss:
        # alpha=0.25. 
        # It seems they use a constant alpha.
        pass

# Let's just keep the original focal_loss function if it works, or replace it if SongFormer's is better.
# SongFormer's SoftmaxFocalLoss is specific.
# I will use the one I added to model.py but I need to instantiate it.

def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float, weight: Optional[torch.Tensor]):
    # Use the simple implementation for now, or instantiate the class.
    # Better to keep the existing function but ensure it matches logic.
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()



def compute_weights(data_dir: str, labels: List[str], map_functional: bool, power: float) -> torch.Tensor:
    counts = label_counts(data_dir, map_functional=map_functional)
    freq = torch.tensor([counts.get(l, 1) for l in labels], dtype=torch.float32)
    weights = 1.0 / torch.pow(freq, power)
    weights = weights / weights.mean()
    return weights


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


def eval_classifier(model: torch.nn.Module, loader: DataLoader, loss_fn, device: torch.device, num_classes: int, input_key: str):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_elems = 0.0
    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            feat = batch[input_key].to(device)
            label = batch["label"].to(device)
            logits = model(feat)
            loss = loss_fn(logits, label)
            if not torch.isfinite(loss):
                continue
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == label).sum().item()
            total_elems += label.numel()
            total_loss += loss.item() * label.numel()
            idx = (label * num_classes + preds).view(-1)
            cm = torch.bincount(idx, minlength=num_classes * num_classes)
            cm_total += cm.view(num_classes, num_classes).cpu()
    if total_elems == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "support": 0,
        }
    epoch_loss = total_loss / total_elems
    epoch_acc = total_correct / total_elems
    tp = cm_total.diag().to(torch.float32)
    fp = cm_total.sum(dim=0).to(torch.float32) - tp
    fn = cm_total.sum(dim=1).to(torch.float32) - tp
    precision_c = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall_c = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1_c = torch.where(
        precision_c + recall_c > 0,
        2 * precision_c * recall_c / (precision_c + recall_c),
        torch.zeros_like(tp),
    )
    support_c = cm_total.sum(dim=1).to(torch.float32)
    valid = support_c > 0
    macro_precision = precision_c[valid].mean().item() if valid.any() else 0.0
    macro_recall = recall_c[valid].mean().item() if valid.any() else 0.0
    macro_f1 = f1_c[valid].mean().item() if valid.any() else 0.0
    micro_tp = tp.sum().item()
    micro_fp = fp.sum().item()
    micro_fn = fn.sum().item()
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "support": int(total_elems),
    }


def has_valid_segment(ann_path: str, label_set: set, min_seg_seconds: float, map_functional: bool) -> bool:
    try:
        segments = load_segments(ann_path)
    except Exception:
        return False
    if map_functional:
        segments = infer_functional_segments(segments)
    for seg in segments:
        if seg.label in label_set and (seg.end - seg.start) >= min_seg_seconds:
            return True
    return False


def train(
    data_dir: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_name: str,
    gamma: float,
    label_smoothing: float,
    use_weights: bool,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_power: float = 0.75,
    hidden: int = 512,
    patience: int = 3,
    min_delta: float = 0.0,
    grad_clip: float = 1.0,
    input_type: str = "mel",
    segment_frames: int = 256,
    lr_schedule: str = "plateau",
    warmup_steps: int = 0,
    total_steps: int = 0,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_label_count: int = 20,
    min_seg_seconds: float = 0.5,
    balance_samples: bool = False,
    max_train_items: Optional[int] = 3000,
    max_labels: int = 12,
    workers: Optional[int] = None,
    use_functional_map: bool = True,
    audio_sample_rate: int = None,
    audio_n_mels: int = None,
    audio_hop_length: int = None,
    audio_n_fft: int = None,
    ema_decay: float = 0.999,
):
    audio_cfg = AudioConfig(
        sample_rate=int(audio_sample_rate) if audio_sample_rate is not None else AudioConfig.sample_rate,
        n_mels=int(audio_n_mels) if audio_n_mels is not None else AudioConfig.n_mels,
        hop_length=int(audio_hop_length) if audio_hop_length is not None else AudioConfig.hop_length,
        n_fft=int(audio_n_fft) if audio_n_fft is not None else AudioConfig.n_fft,
    )
    counts = label_counts(data_dir, map_functional=use_functional_map)
    labels = [l for l in functional_labels() if counts.get(l, 0) >= min_label_count]
    if not labels:
        filtered = [(l, c) for l, c in counts.items() if l.lower() not in {"silence", "end"} and c >= min_label_count]
        if not filtered:
            filtered = [(l, c) for l, c in counts.items() if l.lower() not in {"silence", "end"}]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        labels = [l for l, _ in filtered[:max_labels]]
    if not labels:
        labels = default_labels()
    if max_train_items is not None and max_train_items <= 0:
        max_train_items = None
    label_to_id = {l: i for i, l in enumerate(labels)}
    label_set = set(labels)
    split = list_pairs_by_split(data_dir)
    if split is not None:
        train_pairs_all, val_pairs_all, test_pairs_all = split
        train_pairs = [p for p in train_pairs_all if has_valid_segment(p[1], label_set, min_seg_seconds, use_functional_map)]
        val_pairs = [p for p in val_pairs_all if has_valid_segment(p[1], label_set, min_seg_seconds, use_functional_map)]
        test_pairs = [p for p in test_pairs_all if has_valid_segment(p[1], label_set, min_seg_seconds, use_functional_map)]
    else:
        pairs = [p for p in list_pairs(data_dir) if has_valid_segment(p[1], label_set, min_seg_seconds, use_functional_map)]
        train_pairs, val_pairs, test_pairs = split_pairs(pairs, val_ratio, test_ratio, seed)
    if input_type == "mel":
        train_dataset = ClassifierMelDataset(
            data_dir,
            audio_cfg,
            label_to_id,
            pairs=train_pairs,
            min_seg_seconds=min_seg_seconds,
            max_items=max_train_items,
            seed=seed,
            map_functional=use_functional_map,
            segment_frames=segment_frames,
            train=True,
        )
        input_key = "mel"
    else:
        train_dataset = ClassifierDataset(
            data_dir,
            audio_cfg,
            label_to_id,
            pairs=train_pairs,
            min_seg_seconds=min_seg_seconds,
            max_items=max_train_items,
            seed=seed,
            map_functional=use_functional_map,
        )
        input_key = "feat"
    if workers is None:
        workers = max(1, min(os.cpu_count() or 1, 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    metrics_path = out_path + ".metrics.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("")
    sampler = None
    shuffle = True
    if balance_samples:
        sample_weights = train_dataset.sample_weights()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    val_loader = None
    if val_pairs:
        if input_type == "mel":
            val_dataset = ClassifierMelDataset(
                data_dir,
                audio_cfg,
                label_to_id,
                pairs=val_pairs,
                min_seg_seconds=min_seg_seconds,
                map_functional=use_functional_map,
                segment_frames=segment_frames,
                train=False,
            )
        else:
            val_dataset = ClassifierDataset(
                data_dir,
                audio_cfg,
                label_to_id,
                pairs=val_pairs,
                min_seg_seconds=min_seg_seconds,
                map_functional=use_functional_map,
            )
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
        if input_type == "mel":
            test_dataset = ClassifierMelDataset(
                data_dir,
                audio_cfg,
                label_to_id,
                pairs=test_pairs,
                min_seg_seconds=min_seg_seconds,
                map_functional=use_functional_map,
                segment_frames=segment_frames,
                train=False,
            )
        else:
            test_dataset = ClassifierDataset(
                data_dir,
                audio_cfg,
                label_to_id,
                pairs=test_pairs,
                min_seg_seconds=min_seg_seconds,
                map_functional=use_functional_map,
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )
    if input_type == "mel":
        channels = max(32, hidden // 8)
        attn_dim = max(64, hidden // 6)
        attn_heads = 4
        model = SegmentMelAttnGate(audio_cfg.n_mels, labels, channels=hidden, attn_dim=hidden // 2, heads=8)
        feat_dim = None
    else:
        feat_dim = audio_cfg.n_mels * 4 + 5
        model = SegmentClassifier(feat_dim, labels, hidden)
    model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(weight_decay),
        betas=(float(beta1), float(beta2)),
    )
    weight = compute_weights(data_dir, labels, use_functional_map, weight_power).to(device) if use_weights else None
    scheduler = None
    step_on_batch = False
    if str(lr_schedule).lower() == "cosine":
        total_steps = int(total_steps) if int(total_steps) > 0 else int(max(1, epochs * max(1, len(train_loader))))
        warmup_steps = int(max(0, warmup_steps))
        if warmup_steps > total_steps:
            warmup_steps = total_steps // 10
        def lr_lambda(step: int) -> float:
            step = int(step)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        step_on_batch = True
    elif val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=1, threshold=1e-4, verbose=False
        )
    best_metric = -1.0
    best_macro_f1 = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    bad_epochs = 0

    # EMA Initialization
    use_ema = float(ema_decay) > 0.0 and float(ema_decay) < 1.0
    ema_state = None
    if use_ema:
        ema_state = {k: v.detach().clone().to(device="cpu") for k, v in model.state_dict().items()}
        print(f"EMA enabled with decay {ema_decay}")

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_elems = 0.0
        num_classes = len(labels)
        cm_total = torch.zeros((num_classes, num_classes), dtype=torch.long)
        
        for batch in tqdm(train_loader, desc=f"classifier epoch {epoch+1}/{epochs}"):
            feat = batch[input_key].to(device)
            label = batch["label"].to(device)
            
            opt.zero_grad()
            
            # Use mixed precision for memory efficiency
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(feat)
                if loss_name == "focal":
                    loss = focal_loss(logits, label, gamma, weight)
                else:
                    loss = F.cross_entropy(logits, label, weight=weight, label_smoothing=label_smoothing)
            
            if not torch.isfinite(loss):
                continue
                
            # Scaled backward
            scaler.scale(loss).backward()
            
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(opt)
            scaler.update()
            
            # SOTA Fix: Scheduler should be stepped AFTER optimizer step
            if scheduler is not None and step_on_batch:
                scheduler.step()
                
            # EMA Update
            if use_ema and ema_state is not None:
                d = float(ema_decay)
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if k in ema_state:
                            # Ensure ema_state is float before mul
                            if ema_state[k].dtype in [torch.long, torch.int]:
                                ema_state[k] = v.detach().clone().to(device="cpu")
                            else:
                                ema_state[k].mul_(d).add_(v.detach().to(device="cpu"), alpha=(1.0 - d))
            
            # Detach immediately to save memory
            loss_val = loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == label).sum().item()
                total_elems += label.numel()
                total_loss += loss_val * label.numel()
                idx = (label * num_classes + preds).view(-1)
                cm = torch.bincount(idx, minlength=num_classes * num_classes)
                cm_total += cm.view(num_classes, num_classes).cpu()
                
            # Clear cache occasionally
            if total_elems % (batch_size * 50) == 0:
                torch.cuda.empty_cache()
                
        if total_elems > 0:
            epoch_loss = total_loss / total_elems
            epoch_acc = total_correct / total_elems
            tp = cm_total.diag().to(torch.float32)
            fp = cm_total.sum(dim=0).to(torch.float32) - tp
            fn = cm_total.sum(dim=1).to(torch.float32) - tp
            precision_c = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
            recall_c = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
            f1_c = torch.where(
                precision_c + recall_c > 0,
                2 * precision_c * recall_c / (precision_c + recall_c),
                torch.zeros_like(tp),
            )
            support_c = cm_total.sum(dim=1).to(torch.float32)
            valid = support_c > 0
            macro_precision = precision_c[valid].mean().item() if valid.any() else 0.0
            macro_recall = recall_c[valid].mean().item() if valid.any() else 0.0
            macro_f1 = f1_c[valid].mean().item() if valid.any() else 0.0
            micro_tp = tp.sum().item()
            micro_fp = fp.sum().item()
            micro_fn = fn.sum().item()
            micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
            micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
            micro_f1 = (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0
                else 0.0
            )
            line = {
                "phase": "train",
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "acc": epoch_acc,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "support": int(total_elems),
                "labels": labels,
                "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                "seed": seed,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(
                f"classifier train epoch {epoch+1}/{epochs} loss={epoch_loss:.6f} acc={epoch_acc:.6f} "
                f"macro_f1={macro_f1:.6f} micro_f1={micro_f1:.6f}"
            )
        if val_loader is not None:
            # EMA Swap for Validation
            backup_state = None
            if use_ema and ema_state is not None:
                backup_state = {k: v.detach().clone().to(device="cpu") for k, v in model.state_dict().items()}
                model.load_state_dict({k: v.to(device) for k, v in ema_state.items()})
                
            if loss_name == "focal":
                def loss_fn(logits, target):
                    return focal_loss(logits, target, gamma, weight)
            else:
                def loss_fn(logits, target):
                    return F.cross_entropy(logits, target, weight=weight, label_smoothing=label_smoothing)
            
            val_metrics = eval_classifier(model, val_loader, loss_fn, device, num_classes, input_key)
            
            # EMA Restore
            if backup_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in backup_state.items()})
                
            val_line = {
                "phase": "val",
                "epoch": epoch + 1,
                **val_metrics,
                "labels": labels,
                "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                "seed": seed,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_line, ensure_ascii=False) + "\n")
            print(
                f"classifier val epoch {epoch+1}/{epochs} loss={val_metrics['loss']:.6f} acc={val_metrics['acc']:.6f} "
                f"macro_f1={val_metrics['macro_f1']:.6f} micro_f1={val_metrics['micro_f1']:.6f}"
            )
            # Custom metric: Weighted combination of Train ACC and Val ACC
            # We care about Val ACC (generalization) but also want to ensure Train ACC is reasonable (fitting)
            # F1 is ignored for saving as per instruction.
            train_acc = float(epoch_acc)
            val_acc = float(val_metrics.get("acc", 0.0))
            
            # Weighted Score: 40% Train ACC + 60% Val ACC
            # This encourages models that generalize well (high val) but are also actually learning (high train)
            # If Train ACC is too low, the model hasn't learned enough.
            # If Val ACC is low, it's overfitting or not generalizing.
            metric = 0.4 * train_acc + 0.6 * val_acc
            
            print(f"classifier score={metric:.4f} (train_acc={train_acc:.4f}, val_acc={val_acc:.4f})")
            
            if scheduler is not None and not step_on_batch:
                scheduler.step(metric)
                
            if metric > best_metric + float(min_delta):
                best_metric = metric
                best_macro_f1 = float(val_metrics.get("macro_f1", 0.0))
                best_epoch = epoch + 1
                
                # Save the EMA weights if EMA is used, otherwise current weights
                if use_ema and ema_state is not None:
                     best_state = {k: v.clone() for k, v in ema_state.items()}
                else:
                     best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                     
                bad_epochs = 0
                
                # For saving to disk, we need to load the best state (EMA) into the model temporarily if not already there
                # But wait, we restored backup_state above.
                # So model currently has training weights.
                # If we want to save EMA weights, we should use best_state.
                # save_classifier saves model.state_dict().
                # So we need to load best_state into model, save, then restore.
                
                current_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                
                save_classifier(
                    out_path,
                    model,
                    {
                        "audio_cfg": audio_cfg.__dict__,
                        "labels": labels,
                        "feat_dim": feat_dim,
                        "hidden": hidden,
                        "input_type": input_type,
                        "segment_frames": int(segment_frames),
                        "arch": "mel_attn_gate_ms" if input_type == "mel" else "mlp",
                        "channels": channels if input_type == "mel" else None,
                        "attn_dim": attn_dim if input_type == "mel" else None,
                        "attn_heads": attn_heads if input_type == "mel" else None,
                        "best_epoch": best_epoch,
                        "best_score": float(best_metric),
                        "best_val_acc": float(val_acc),
                        "best_train_acc": float(train_acc),
                    }
                )
                model.load_state_dict({k: v.to(device) for k, v in current_weights.items()})
                
                print(f"  -> Saved best classifier model (score={best_metric:.4f})")
                
                # SOTA Check: Also check test metrics if available
                if test_loader is not None:
                    # Load EMA for test
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                    test_metrics = eval_classifier(model, test_loader, loss_fn, device, num_classes, input_key)
                    print(f"  -> Test ACC: {test_metrics['acc']:.4f}")
                    # Restore
                    model.load_state_dict({k: v.to(device) for k, v in current_weights.items()})
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
    # SOTA FIX: Load best state before final save
    model.load_state_dict(best_state)
    
    # Final Evaluation on Test Set (using best model)
    if test_loader is not None:
        if loss_name == "focal":
            def loss_fn(logits, target):
                return focal_loss(logits, target, gamma, weight)
        else:
            def loss_fn(logits, target):
                return F.cross_entropy(logits, target, weight=weight, label_smoothing=label_smoothing)
        
        test_metrics = eval_classifier(model, test_loader, loss_fn, device, num_classes, input_key)
        test_line = {
            "phase": "test",
            "epoch": best_epoch,
            **test_metrics,
            "labels": labels,
            "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
            "seed": seed,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_line, ensure_ascii=False) + "\n")
        print(f"\nFinal Test Metrics (Best Epoch {best_epoch}):")
        print(f"ACC: {test_metrics['acc']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")
        
        # Save AGAIN if test ACC is good (Optional, but ensures we persist test info)
        save_classifier(
            out_path,
            model,
            {
                "audio_cfg": audio_cfg.__dict__,
                "labels": labels,
                "feat_dim": feat_dim,
                "hidden": hidden,
                "input_type": input_type,
                "segment_frames": int(segment_frames),
                "arch": "mel_attn_gate_ms" if input_type == "mel" else "mlp",
                "channels": channels if input_type == "mel" else None,
                "attn_dim": attn_dim if input_type == "mel" else None,
                "attn_heads": attn_heads if input_type == "mel" else None,
                "best_epoch": best_epoch,
                "best_score": float(best_metric),
                "test_acc": float(test_metrics['acc']),
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_path", default="checkpoints/classifier.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--weight_power", type=float, default=0.75)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--input_type", choices=["pooled", "mel"], default="mel")
    parser.add_argument("--segment_frames", type=int, default=256)
    parser.add_argument("--lr_schedule", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_label_count", type=int, default=20)
    parser.add_argument("--min_seg_seconds", type=float, default=0.5)
    parser.add_argument("--audio_sample_rate", type=int, default=None)
    parser.add_argument("--audio_n_mels", type=int, default=None)
    parser.add_argument("--audio_hop_length", type=int, default=None)
    parser.add_argument("--audio_n_fft", type=int, default=None)
    parser.add_argument("--balance_samples", action="store_true", default=False)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--max_labels", type=int, default=12)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no_functional_map", action="store_false", dest="use_functional_map")
    parser.set_defaults(use_functional_map=True)
    args = parser.parse_args()
    train(
        args.data_dir,
        args.out_path,
        args.epochs,
        args.batch_size,
        args.lr,
        args.loss,
        args.gamma,
        args.label_smoothing,
        args.class_weight,
        args.weight_decay,
        args.beta1,
        args.beta2,
        args.weight_power,
        args.hidden,
        args.patience,
        args.min_delta,
        args.grad_clip,
        args.input_type,
        args.segment_frames,
        args.lr_schedule,
        args.warmup_steps,
        args.total_steps,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.min_label_count,
        args.min_seg_seconds,
        args.balance_samples,
        args.max_train_items,
        args.max_labels,
        args.workers,
        args.use_functional_map,
        args.audio_sample_rate,
        args.audio_n_mels,
        args.audio_hop_length,
        args.audio_n_fft,
    )


if __name__ == "__main__":
    main()
